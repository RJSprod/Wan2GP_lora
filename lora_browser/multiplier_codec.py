"""Parse and re-serialise WanGP's ``loras_multipliers`` string.

WanGP's own parser lives in ``shared/utils/loras_mutipliers.py``.  The rules
this module mirrors, in the order the native parser applies them:

* Lines whose first non-space character is ``#`` are comments and are dropped.
* A single ``|`` marks the boundary between accelerator-managed multipliers and
  user multipliers.  It is replaced by a space before tokenising, so the number
  of tokens to its left is the only thing that carries meaning.
* Remaining tokens are whitespace separated and positional: token *i* belongs
  to ``activated_loras[i]``.  A missing token defaults to ``1``.
* Within a token, ``;`` separates guidance phases, ``,`` separates step/time
  varying values, and ``:`` separates LoRA multiplier branches.
* A token with fewer ``;`` parts than the model's phase count is padded by
  repeating its last value, so ``"0.8"`` and ``"0.8;0.8"`` are equivalent.
  A token with *more* parts than the phase count is rejected by WanGP, so this
  codec never emits one.

Tokens are classified three ways:

* ``SIMPLE`` -- one scalar per phase, editable by the sliders.
* ``SCHEDULED`` -- ``,`` separated step values whose phase structure the editor
  can map safely, so the timeline can edit them.
* ``ADVANCED`` -- ``:`` branch syntax, malformed values, or a phase structure
  this plugin refuses to guess at.  Carried through verbatim and only ever
  replaced on an explicit user conversion.

The phase structure of a scheduled token is what decides between the last two.
A comma-only token is time-only: WanGP spreads it across the whole inference
run regardless of phases, so it is one *shared* schedule.  A token that
declares exactly as many ``;`` parts as the model's phase capacity maps one
list per phase.  Anything in between -- two expressions for a three-phase
model, say -- expands according to ``model_switch_phase`` inside WanGP's own
parser, which this plugin does not replicate: those tokens stay ADVANCED and
read-only rather than being re-serialised from a guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .utils import format_number, is_finite_number

SIMPLE = "simple"
SCHEDULED = "scheduled"
ADVANCED = "advanced"

DEFAULT_TOKEN = "1"


@dataclass
class ParsedMultipliers:
    """Raw tokens plus the position of the accelerator boundary."""

    tokens: list[str] = field(default_factory=list)
    #: Number of tokens that sit before the ``|``; ``-1`` when there is no bar.
    separator_index: int = -1

    @property
    def has_separator(self) -> bool:
        return self.separator_index >= 0


@dataclass
class TokenInfo:
    """One LoRA's multiplier, classified for the editor."""

    raw: str
    kind: str
    #: Phase scalars, already padded/truncated to the requested phase count.
    #: ``None`` for advanced tokens.
    values: list[float] | None = None
    #: Values beyond the requested phase count, kept so that switching a model
    #: back to two phases does not lose a carefully chosen phase 2.
    overflow: list[float] = field(default_factory=list)
    #: The scalars the token actually spelled out, before any padding.  ``"0.8"``
    #: and ``"0.8;0.8"`` mean the same thing to WanGP but not to the editor: only
    #: the second one states a phase 2, so only it may overwrite a remembered value.
    declared: list[float] = field(default_factory=list)
    #: Set for ``SCHEDULED`` tokens; ``None`` otherwise.
    schedule: "ScheduledToken | None" = None
    #: Why an ``ADVANCED`` token is read-only, phrased for the panel.
    reason: str = ""


@dataclass
class ScheduledToken:
    """A comma-scheduled token whose phase structure the editor can map.

    ``phase_values`` holds one value list per *declared* phase, exactly as the
    token spelled them out -- no padding, so nothing is invented.  ``shared`` is
    the comma-only case: a single list that WanGP spreads across the whole run
    rather than inside one phase.
    """

    raw: str
    phase_values: list[list[float]] = field(default_factory=list)
    shared: bool = False
    declared_phase_count: int = 0
    editable: bool = True
    reason: str = ""

    def values_for(self, phase: int) -> list[float]:
        """The value list that drives ``phase``, or ``[]`` if it has none."""
        if self.shared:
            return list(self.phase_values[0]) if self.phase_values else []
        index = int(phase)
        if 0 <= index < len(self.phase_values):
            return list(self.phase_values[index])
        return []


def strip_comments(text: str) -> str:
    """Drop comment lines the way ``preparse_loras_multipliers`` does."""
    lines = str(text or "").replace("\r", "").split("\n")
    kept = [line.strip() for line in lines]
    return " ".join(line for line in kept if line and not line.startswith("#"))


def parse_multipliers(text: str) -> ParsedMultipliers:
    """Tokenise a multiplier string, remembering where the ``|`` was."""
    flat = strip_comments(text)
    if not flat:
        return ParsedMultipliers()

    bar = flat.find("|")
    if bar < 0:
        return ParsedMultipliers(tokens=flat.split(), separator_index=-1)

    left = flat[:bar].split()
    right = flat[bar + 1:].replace("|", " ").split()
    return ParsedMultipliers(tokens=left + right, separator_index=len(left))


def align_tokens(tokens: list[str], count: int) -> list[str]:
    """Make the token list exactly as long as the selected LoRA list.

    Missing multipliers default to ``1``; surplus tokens belong to LoRAs that
    are no longer selected and are dropped.
    """
    aligned = list(tokens[:count])
    aligned.extend([DEFAULT_TOKEN] * (count - len(aligned)))
    return aligned


def classify(raw: str, phases: int, model_switch_phase: int | None = None) -> TokenInfo:
    """Decide how ``raw`` may be edited: sliders, timeline, or not at all.

    ``model_switch_phase`` is accepted because it is what WanGP's own parser
    consults to expand an under-declared multi-phase token.  The plugin does not
    reimplement that expansion, so the argument only ever makes classification
    *more* conservative; it is recorded in the reason text.
    """
    token = str(raw or "").strip() or DEFAULT_TOKEN
    phases = max(1, int(phases or 1))

    if ":" in token:
        # Branch syntax. WanGP understands it; the visual editor does not.
        return TokenInfo(raw=token, kind=ADVANCED, reason="Uses ':' branch syntax.")

    if "," in token:
        return _classify_scheduled(token, phases, model_switch_phase)

    parts = [part.strip() for part in token.split(";")]
    if not parts or not all(is_finite_number(part) for part in parts):
        return TokenInfo(raw=token, kind=ADVANCED, reason="Not a list of numbers.")

    numbers = [float(part) for part in parts]
    visible = numbers[:phases]
    while len(visible) < phases:
        # WanGP pads a short token by repeating its last value.
        visible.append(visible[-1])
    return TokenInfo(
        raw=token,
        kind=SIMPLE,
        values=visible,
        overflow=numbers[phases:],
        declared=numbers,
    )


def _classify_scheduled(token: str, phases: int, model_switch_phase: int | None) -> TokenInfo:
    """Classify a comma token, refusing any phase mapping that is a guess."""
    parts = [part.strip() for part in token.split(";")]
    lists: list[list[float]] = []
    for part in parts:
        entries = [entry.strip() for entry in part.split(",")]
        if not entries or not all(is_finite_number(entry) for entry in entries):
            return TokenInfo(raw=token, kind=ADVANCED, reason="Not a list of numbers.")
        lists.append([float(entry) for entry in entries])

    declared = len(lists)
    shared = declared == 1

    if declared > phases:
        return TokenInfo(
            raw=token,
            kind=ADVANCED,
            reason=(
                f"Declares {declared} phases; this model accepts {phases}. "
                "Kept exactly as imported."
            ),
        )

    if not shared and declared != phases:
        # WanGP expands this using model_switch_phase; guessing which phase each
        # list lands in could silently change what renders.
        hint = "" if model_switch_phase is None else f" (model_switch_phase={model_switch_phase})"
        return TokenInfo(
            raw=token,
            kind=ADVANCED,
            reason=(
                f"Declares {declared} phase schedules for a {phases}-phase model{hint}; "
                "WanGP decides how those expand, so it is kept exactly as imported."
            ),
        )

    return TokenInfo(
        raw=token,
        kind=SCHEDULED,
        schedule=ScheduledToken(
            raw=token,
            phase_values=lists,
            shared=shared,
            declared_phase_count=declared,
            editable=True,
        ),
    )


def parse_schedule(raw: str, phases: int, model_switch_phase: int | None = None) -> ScheduledToken | None:
    """The scheduled view of ``raw``, or ``None`` when it is not one.

    Parsing never mutates the token: the returned object carries ``raw`` so a
    caller that makes no edit can put the original back byte-for-byte.
    """
    info = classify(raw, phases, model_switch_phase)
    return info.schedule if info.kind == SCHEDULED else None


def build_schedule(phase_values: list[list[float]], shared: bool = False) -> str:
    """Serialise value lists phase-major: ``,`` inside a phase, ``;`` between.

    ``shared=True`` emits only the first list, i.e. a comma-only token that
    WanGP spreads across the whole run instead of inside one phase.
    """
    lists = [list(values) for values in phase_values if list(values)]
    if not lists:
        return DEFAULT_TOKEN
    if shared:
        lists = lists[:1]
    return ";".join(
        ",".join(format_number(value) for value in values) for values in lists
    )


def build_token(values: list[float], phases: int) -> str:
    """Serialise phase scalars into a native token.

    Only ``phases`` values are emitted: WanGP rejects a token that declares
    more phases than the model currently supports.
    """
    phases = max(1, int(phases or 1))
    numbers = [float(value) for value in values][:phases]
    if not numbers:
        numbers = [1.0]
    while len(numbers) < phases:
        numbers.append(numbers[-1])
    return ";".join(format_number(number) for number in numbers)


def default_token(phases: int) -> str:
    """A freshly included LoRA starts at 1.0 on every editable phase."""
    return build_token([1.0] * max(1, int(phases or 1)), phases)


def serialize(tokens: list[str], separator_index: int = -1) -> str:
    """Rebuild the native multiplier string.

    No space is placed around the ``|``: WanGP replaces the bar with a space
    and then splits on single spaces, so ``"1 | 1"`` would tokenise into empty
    strings and fail validation.
    """
    cleaned = [str(token).strip() or DEFAULT_TOKEN for token in tokens]
    if separator_index is None or separator_index < 0:
        return " ".join(cleaned)

    index = min(max(int(separator_index), 0), len(cleaned))
    return " ".join(cleaned[:index]) + "|" + " ".join(cleaned[index:])


def separator_after_removal(separator_index: int, removed_positions: list[int]) -> int:
    """Keep the accelerator boundary pointing at the same LoRAs after removals."""
    if separator_index is None or separator_index < 0:
        return -1
    removed_before = sum(1 for position in removed_positions if position < separator_index)
    return max(0, separator_index - removed_before)


def rescale_values(values: list[float], phases: int, remembered: list[float] | None = None) -> list[float]:
    """Grow or shrink a phase vector when the model's phase count changes.

    Growing prefers previously remembered values for the phases that were
    hidden, so One Phase -> Two Phases restores the phase 2 the user chose
    rather than resetting it to 1.0.
    """
    phases = max(1, int(phases or 1))
    result = list(values[:phases])
    memory = list(remembered or [])
    while len(result) < phases:
        index = len(result)
        if index < len(memory):
            result.append(float(memory[index]))
        elif result:
            result.append(result[-1])
        else:
            result.append(1.0)
    return result
