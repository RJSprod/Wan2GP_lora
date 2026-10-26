"""The state model shared by the panel and WanGP's native components.

Everything in here is pure: it takes native LoRA state in, applies a typed
frontend action, and returns the native state to write back.  Keeping it free
of Gradio makes the reconciliation rules -- which are where multiplier
corruption would come from -- directly testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import multiplier_codec as codec
from . import schedule as sch
from .inventory import Inventory
from .utils import format_number, is_finite_number, normalize_id

PHASE_2_TILING_GUIDANCE_VALUE = "2~"

#: The slider is the coarse control: it covers 0..1 on a 0.05 grid, which is
#: what a thumb can actually hit. It never rewrites a value it cannot represent
#: -- an imported 0.71 keeps its thumb near 0.70 and stays 0.71 until the slider
#: is actually moved, and an imported 1.23 pins the thumb at 1.00 and stays 1.23.
SLIDER_MIN = 0.0
SLIDER_MAX = 1.0
SLIDER_STEP = 0.05

#: The +/- buttons step the exact value, and are not confined to the slider's
#: 0..1 window.
NUDGE_STEP = 0.01

#: Hard bounds for the numeric field. Negative multipliers are legitimate
#: (they invert a LoRA's effect) and WanGP does not cap the upper end either.
VALUE_MIN = -10.0
VALUE_MAX = 10.0

#: Direct numeric entry keeps this many decimals. Two would quietly round a
#: deliberate 0.125 to 0.13; four is the same precision the serialiser emits.
VALUE_DECIMALS = 4

#: How schedule slot numbers may be labelled.
#:
#: ``GLOBAL_EXACT``  - the timeline really is inference steps 1..N.
#: ``STAGE_EXACT``   - the timeline is one pipeline stage's steps 1..N, and the
#:                      plugin knows that stage's length (LTX-2, MiniMax H3).
#: ``PHASE_RELATIVE`` - slots inside one phase, whose global step numbers depend
#:                      on runtime phase boundaries the plugin cannot see.
#: ``NORMALIZED_READONLY`` - preserved, but not offered for dragging.
COORD_GLOBAL_EXACT = "global_exact"
COORD_STAGE_EXACT = "stage_exact"
COORD_PHASE_RELATIVE = "phase_relative"
COORD_NORMALIZED_READONLY = "normalized_readonly"

#: LTX-2's sigma tables, as WanGP's ``models/ltx2/ltx_pipelines/utils/
#: constants.py`` defines them.  A table of n + 1 sigmas runs n steps, step i
#: evaluated at the i-th sigma.  The first stage runs the distilled table on a
#: distilled checkpoint -- or on the Dev checkpoint with a distilled sampler --
#: and the step counter otherwise.  The refinement stage after the upscale
#: always runs one of the two stage-2 tables, whatever the counter says: the
#: second on the 22B checkpoints (LTX-2.3, 2.5), whose handler sets
#: ``ltx2_22B_class``, and on Dev after a distilled sampler.
LTX2_DISTILLED_SIGMAS = (1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0)
LTX2_STAGE_2_SIGMAS = (0.909375, 0.725, 0.421875, 0.0)
LTX2_STAGE_2_SIGMAS_8_STEPS = (0.85, 0.725, 0.4219, 0.0)
LTX2_DISTILLED_STEPS = len(LTX2_DISTILLED_SIGMAS) - 1
LTX2_REFINE_STEPS = len(LTX2_STAGE_2_SIGMAS) - 1
LTX2_DISTILLED_SAMPLERS = frozenset({"distilled_8_steps", "distilled_8_steps_ancestral"})

#: WanGP's default for "Phase 2 Steps" (``num_inference_steps2``), used when the
#: plugin could not obtain that component.  MiniMax H3 sets it to 3.
PHASE_2_STEPS_DEFAULT = 3

#: Field separator used only inside the canonical fingerprint below.
_SIGNATURE_SEP = "\x1f"


@dataclass(frozen=True)
class PhaseConfig:
    """How many multiplier phases exist, and how many are editable right now.

    These are deliberately two numbers.  MiniMax H3 declares
    ``lora_multiplier_phases = 2`` in its model definition whatever the
    guidance mode is, so in One Phase mode the token may still legitimately
    carry two values -- which is how a phase 2 the user tuned survives a trip
    through One Phase without the plugin having to hide it somewhere.
    """

    #: Values a native token may declare; WanGP rejects a token with more.
    capacity: int = 1
    #: Sliders the editor shows for the current guidance mode.
    effective: int = 1
    #: True when every phase is a pipeline stage that runs a known number of
    #: steps -- LTX-2, where phase 1 is the first denoising pass and phase 2 the
    #: refinement after the upscale, and MiniMax H3, whose phase 2 is a separate
    #: enhancement pass with a step count of its own.  Elsewhere a second phase
    #: starts at a switch step the sampler only settles at generation time.
    staged: bool = False

    @property
    def hidden(self) -> int:
        return max(0, self.capacity - self.effective)


def is_ltx2(model_def: dict | None) -> bool:
    """WanGP's LTX-2 family: LTX-2, 2.3, 2.5 and the checkpoints built on them.

    Recognised by architecture, with the key the LTX-2 handler adds to every
    model definition it builds as the backstop -- JoyAI Echo runs the same
    pipelines under an architecture of its own.
    """
    model_def = model_def or {}
    architecture = str(model_def.get("architecture", "") or "")
    return (
        architecture.startswith("ltx2")
        or architecture == "joyai_echo"
        or "ltx2_22B_class" in model_def
    )


def has_own_phase_2_steps(model_def: dict | None) -> bool:
    """A model whose phase 2 is a pass of its own, run for "Phase 2 Steps".

    WanGP shows that slider (``num_inference_steps2``) for a model that sets
    ``inference_steps2``, and such a model runs each phase as its own stage:
    MiniMax H3 expands phase 1's list over the full ``num_inference_steps``
    denoise and phase 2's over the enhancement pass after the latent upscale
    (``activate_lora_phase`` in its pipeline).  So in either guidance mode
    phase 1 is the same run, and a schedule on it means the same thing.
    """
    return bool((model_def or {}).get("inference_steps2", False))


def is_staged(model_def: dict | None) -> bool:
    """Whether every LoRA phase is a pipeline stage of known length."""
    return is_ltx2(model_def) or has_own_phase_2_steps(model_def)


def ltx2_runs_distilled_steps(model_def: dict | None, sampler: Any = "") -> bool:
    """Whether an LTX-2 first stage runs the fixed distilled table.

    Mirrors the handler: a distilled checkpoint (``ltx2_pipeline`` set to
    ``distilled``, which JoyAI Echo always is) ignores the step counter, and so
    does the Dev checkpoint on one of the distilled samplers.
    """
    return _ltx2_distilled_checkpoint(model_def) or _distilled_sampler(sampler)


def _ltx2_distilled_checkpoint(model_def: dict | None) -> bool:
    model_def = model_def or {}
    return (
        model_def.get("ltx2_pipeline") == "distilled"
        or str(model_def.get("architecture", "") or "") == "joyai_echo"
        or bool(model_def.get("joyai_echo", False))
    )


def _distilled_sampler(sampler: Any) -> bool:
    return str(sampler or "").strip().lower() in LTX2_DISTILLED_SAMPLERS


def ltx2_stage_2_sigmas(model_def: dict | None, sampler: Any = "") -> tuple[float, ...]:
    """The table an LTX-2 refinement stage runs, as both pipelines choose it.

    WanGP's distilled pipeline also falls back to the first table while
    rendering a layout; both tables match the same stage-1 steps, so nothing
    here depends on telling those runs apart.
    """
    model_def = model_def or {}
    if model_def.get("ltx2_22B_class", False):
        return LTX2_STAGE_2_SIGMAS_8_STEPS
    if not _ltx2_distilled_checkpoint(model_def) and _distilled_sampler(sampler):
        return LTX2_STAGE_2_SIGMAS_8_STEPS
    return LTX2_STAGE_2_SIGMAS


def nearest_steps(stage_1: tuple[float, ...], stage_2: tuple[float, ...]) -> tuple[int, ...]:
    """For each step of ``stage_2``, the step of ``stage_1`` run at the nearest
    noise level, 1-based; a tie goes to the later, quieter step."""
    starts = stage_1[:-1]
    return tuple(
        min(range(len(starts)), key=lambda index: (abs(starts[index] - sigma), -index)) + 1
        for sigma in stage_2[:-1]
    )


def ltx2_stage_2_match(model_def: dict | None, sampler: Any = "") -> tuple[int, ...]:
    """The stage-1 steps whose noise levels the refinement re-runs, one per
    refinement step -- or empty when stage 1's noise levels are not known.

    A distilled stage 1 ends where stage 2 works: the first stage-2 table is
    the distilled table's last three steps exactly, and the second differs only
    in starting at 0.85, inside step 6 (0.909 -> 0.725).  Either way that is
    steps 6, 7 and 8.  A Dev stage 1 on its own scheduler runs a table this
    does not reproduce, so nothing is matched there.
    """
    if not ltx2_runs_distilled_steps(model_def, sampler):
        return ()
    return nearest_steps(LTX2_DISTILLED_SIGMAS, ltx2_stage_2_sigmas(model_def, sampler))


def phase_step_counts(
    model_def: dict | None, phases: PhaseConfig, steps: int, sampler: Any = "", steps2: Any = None
) -> tuple[int, ...]:
    """How many steps each LoRA phase runs, when the model makes that knowable.

    Empty unless the phases are stages.  There WanGP expands phase 1's list
    over the first stage and phase 2's over the second, each at its own length
    -- see ``update_loras_slists`` in the LTX-2 and MiniMax H3 pipelines -- so a
    timeline per phase can be exact.  One Phase runs the first stage only.

    LTX-2's refinement always runs three steps.  An H3-style phase 2 runs
    "Phase 2 Steps" (``steps2``), WanGP's default where that is not known.
    """
    if not phases.staged:
        return ()
    if is_ltx2(model_def):
        first = LTX2_DISTILLED_STEPS if ltx2_runs_distilled_steps(model_def, sampler) else int(steps or 0)
        second = LTX2_REFINE_STEPS
    else:
        first = int(steps or 0)
        second = _step_count(steps2) or PHASE_2_STEPS_DEFAULT
    if first <= 0:
        return ()
    return (first,) if phases.effective <= 1 else (first, second)


def _step_count(value: Any) -> int:
    try:
        return max(0, int(float(value)))
    except (TypeError, ValueError):
        return 0


def resolve_phases(model_def: dict | None, guidance_phases_value: Any) -> PhaseConfig:
    """Resolve phase capability from current WanGP data, not from a model name.

    Mirrors ``wgp.py``: the guidance dropdown can hold the string ``"2~"``
    ("Two Phases with Tiling"), which is still two multiplier phases -- tiling
    changes how phase 2 is processed, it does not add a third phase.
    """
    model_def = model_def or {}

    if guidance_phases_value == PHASE_2_TILING_GUIDANCE_VALUE:
        guidance_count = 2
    else:
        try:
            guidance_count = int(guidance_phases_value or 0)
        except (TypeError, ValueError):
            guidance_count = 0

    max_phases = int(model_def.get("guidance_max_phases", 0) or 0)
    if model_def.get("lock_guidance_phases", False):
        guidance_count = max_phases
    elif max_phases:
        guidance_count = min(guidance_count, max_phases)

    capacity = model_def.get("lora_multiplier_phases", guidance_count)
    try:
        capacity = int(capacity or guidance_count or 1)
    except (TypeError, ValueError):
        capacity = guidance_count or 1
    capacity = max(1, capacity)

    return PhaseConfig(
        capacity=capacity,
        effective=max(1, min(capacity, guidance_count or 1)),
        staged=is_staged(model_def),
    )


@dataclass(frozen=True)
class ScheduleContext:
    """What the editor knows about schedule time coordinates.

    ``steps`` is WanGP's ``num_inference_steps`` when the plugin could obtain
    the component, else 0.  ``phase_steps`` is how many steps each phase runs
    where the model makes that knowable (``phase_step_counts``: the stages of
    LTX-2 and MiniMax H3); elsewhere it is empty, a phase-specific schedule is
    expanded inside a runtime interval that depends on model switching, and
    labelling its slots with step numbers the plugin cannot verify would be a
    lie.  ``boundaries_known`` says which of the two this is.

    ``stage_2_match`` is, for each refinement step of an LTX-2 run in Two
    Phases, the stage-1 step at the nearest noise level, where stage 1's noise
    levels are known (``ltx2_stage_2_match``); "Fill from stage 1" reads those.
    """

    steps: int = 0
    boundaries_known: bool = False
    phase_steps: tuple[int, ...] = ()
    stage_2_match: tuple[int, ...] = ()

    @classmethod
    def from_native(
        cls,
        steps_value: Any,
        model_def: dict | None = None,
        phases: PhaseConfig | None = None,
        sampler: Any = "",
        steps2_value: Any = None,
    ) -> "ScheduleContext":
        steps = _step_count(steps_value)
        phase_steps = phase_step_counts(model_def, phases, steps, sampler, steps2_value) if phases else ()
        # Noise levels are matched across LTX-2's stages only: its sigma tables
        # are the ones mirrored here.
        matched = len(phase_steps) > 1 and is_ltx2(model_def)
        return cls(
            steps=steps,
            boundaries_known=bool(phase_steps),
            phase_steps=phase_steps,
            stage_2_match=ltx2_stage_2_match(model_def, sampler) if matched else (),
        )

    def phase_length(self, phase: int = 0) -> int:
        """Steps ``phase`` runs, or 0 when nobody can say.

        Without stages only phase 1 has a length, the run's -- which it is only
        in One Phase, the one mode that schedules then.
        """
        if self.phase_steps:
            return self.phase_steps[phase] if 0 <= phase < len(self.phase_steps) else 0
        return self.steps if phase == 0 else 0

    def target_slots(self, phase: int = 0) -> int:
        """One slot per step of ``phase``, or 0 when there is no count to follow.

        Clamped, so a step count beyond what a timeline can hold settles at the
        ceiling instead of never matching and asking to be refitted forever.

        A single step is not followed either.  Cutting a schedule to one step
        would leave a scalar in the token, which nothing could tell from a plain
        strength, so the next step count could not bring the rest back -- and
        WanGP runs a one-step job on the first value of any list, so leaving the
        list as it is changes nothing that renders.  It matters more than it
        sounds: typing ``12`` into the step box passes through ``1``.
        """
        length = self.phase_length(phase)
        return sch.clamp_slots(length) if length >= sch.MIN_WINDOW else 0

    def default_slots(self, shared: bool = False, phase: int = 0) -> int:
        """Timeline length for a freshly opened schedule.

        A shared list is stretched over every stage it runs in, so it follows
        the first: that is where it runs one value per step.
        """
        length = self.phase_length(0 if shared else phase)
        return sch.clamp_slots(length) if length else sch.DEFAULT_SLOTS


def scheduling_allowed(phases: PhaseConfig) -> bool:
    """Whether a step schedule can say what it appears to say.

    WanGP's ``expand_slist`` stretches each phase's comma list to fill that
    phase's interval, so a list is step-exact exactly when its length equals the
    number of steps in the interval it covers.

    In One Phase mode the switch points sit at the end of the run, so phase 1
    covers every step: a list of ``num_inference_steps`` values runs one value
    per step, exactly as drawn.

    With two or more guidance phases, phase 1 usually covers only up to
    ``model_switch_step`` -- derived at generation time from the sampler's
    timesteps and the switch threshold, and not knowable while editing.  A
    four-slot schedule drawn against a four-step run would be squeezed into
    however many steps phase 1 turns out to be, so the timeline would be showing
    something WanGP is not going to do.  Rather than draw that, the editor does
    not offer scheduling there -- and does not touch a schedule already in a
    token either: switching the mode back brings it back exactly as it was.

    LTX-2 and MiniMax H3 are the exception, and they are why ``staged``
    exists.  Their phases are not cut out of one run at a switch step; each is
    a pipeline stage with a length known while editing.  A list as long as its
    stage runs one value per step there, in either mode.
    """
    return phases.effective <= 1 or phases.staged


SCHEDULING_DISABLED_REASON = (
    "Kept exactly as it is. With more than one phase, this model only settles "
    "where each phase starts while generating, so a timeline could not show what "
    "runs - switch to One Phase to edit it."
)


def phase_labels(phases: int) -> list[str]:
    """Generic labels only -- mislabelling a phase is worse than not naming it."""
    return ["Strength"] if phases <= 1 else [f"Phase {index + 1}" for index in range(phases)]


def sanitize_value(value: Any, fallback: float = 1.0) -> float:
    """Reject NaN/Infinity and out-of-range values coming from the frontend.

    Anything outside the supported range falls back to the previous value
    rather than being clamped, so a typo cannot quietly become -10.
    """
    if not is_finite_number(value):
        return fallback
    number = round(float(value), VALUE_DECIMALS)
    if number < VALUE_MIN or number > VALUE_MAX:
        return fallback
    return number


@dataclass
class Stack:
    """The selected LoRAs and their multiplier tokens, in native order.

    ``ids``/``tokens`` are what WanGP applies.  ``disabled`` are LoRAs switched
    off in the panel: kept with their multipliers, as comment lines in the
    multiplier text (``codec.DISABLED_MARK``), and applied by nothing.
    """

    ids: list[str] = field(default_factory=list)
    tokens: list[str] = field(default_factory=list)
    separator_index: int = -1
    disabled: list[codec.DisabledLora] = field(default_factory=list)

    @classmethod
    def from_native(cls, selected, multipliers: str) -> "Stack":
        ids = [normalize_id(value) for value in (selected or []) if normalize_id(value)]
        parsed = codec.parse_multipliers(multipliers)
        stack = cls(
            ids=ids,
            tokens=codec.align_tokens(parsed.tokens, len(ids)),
            separator_index=min(parsed.separator_index, len(ids)) if parsed.has_separator else -1,
        )
        seen = set(ids)
        for entry in codec.parse_disabled(multipliers):
            lora_id = normalize_id(entry.id)
            # Selected again -- by a preset, say, or WanGP's own picker: the
            # selection is what runs, so it wins over the switched-off copy.
            if lora_id and lora_id not in seen:
                seen.add(lora_id)
                stack.disabled.append(codec.DisabledLora(id=lora_id, token=entry.token, at=entry.at))
        return stack

    def index_of(self, lora_id: str) -> int:
        target = normalize_id(lora_id)
        for position, value in enumerate(self.ids):
            if value == target:
                return position
        return -1

    def is_system(self, position: int) -> bool:
        """LoRAs left of the bar were placed by an accelerator profile."""
        return self.separator_index >= 0 and 0 <= position < self.separator_index

    def token_for(self, lora_id: str) -> str:
        position = self.index_of(lora_id)
        return self.tokens[position] if position >= 0 else codec.DEFAULT_TOKEN

    def add(self, lora_id: str, capacity: int) -> bool:
        """Include a LoRA on the user side of the stack, at 1.0 per phase.

        A newly included LoRA never inherits the strength of whichever row
        previously occupied its index.  One that is switched off is already
        included: it comes back on, as it was.
        """
        lora_id = normalize_id(lora_id)
        if not lora_id or self.index_of(lora_id) >= 0:
            return False
        if self.disabled_entry(lora_id) is not None:
            return self.set_enabled(lora_id, True)
        self.ids.append(lora_id)
        self.tokens.append(codec.default_token(capacity))
        return True

    def remove(self, lora_id: str) -> bool:
        """Take a LoRA out of the stack, switched on or off."""
        position = self.index_of(lora_id)
        if position >= 0 and self.is_system(position):
            self.ids.pop(position)
            self.tokens.pop(position)
            self.separator_index = codec.separator_after_removal(self.separator_index, [position])
            return True
        target = normalize_id(lora_id)
        entries = self.user_entries()
        kept = [entry for entry in entries if entry[0] != target]
        if len(kept) == len(entries):
            return False
        self.set_user_entries(kept)
        return True

    def disabled_entry(self, lora_id: str) -> codec.DisabledLora | None:
        target = normalize_id(lora_id)
        return next((entry for entry in self.disabled if entry.id == target), None)

    def user_entries(self) -> list[tuple[str, bool, str]]:
        """``(id, on, token)`` for the user's LoRAs, in the order the panel
        lists them: a switched-off LoRA stays where it was."""
        entries = [
            (lora_id, True, self.tokens[position])
            for position, lora_id in enumerate(self.ids) if not self.is_system(position)
        ]
        for entry in sorted(self.disabled, key=lambda entry: entry.at):
            entries.insert(max(0, min(int(entry.at), len(entries))), (entry.id, False, entry.token))
        return entries

    def set_user_entries(self, entries: list[tuple[str, bool, str]]) -> None:
        """Write the user's side back: on ones into the selection, in order,
        off ones as comment lines that remember their place."""
        keep = self.separator_index if self.separator_index >= 0 else 0
        self.ids = self.ids[:keep] + [lora_id for lora_id, on, _ in entries if on]
        self.tokens = self.tokens[:keep] + [token for _, on, token in entries if on]
        self.disabled = [
            codec.DisabledLora(id=lora_id, token=token, at=index)
            for index, (lora_id, on, token) in enumerate(entries) if not on
        ]

    def set_enabled(self, lora_id: str, on: bool) -> bool:
        """Switch one of the user's LoRAs on or off, keeping its multiplier and
        its place.  WanGP-managed LoRAs are left to WanGP."""
        target = normalize_id(lora_id)
        if not on and ("\n" in target or "\r" in target):
            return False                 # a comment line could not hold it
        entries = self.user_entries()
        for index, (entry_id, entry_on, token) in enumerate(entries):
            if entry_id == target:
                if entry_on == bool(on):
                    return False
                entries[index] = (entry_id, bool(on), token)
                self.set_user_entries(entries)
                return True
        return False

    def set_all_enabled(self, on: bool) -> int:
        """Switch every user LoRA on or off; returns how many changed."""
        entries = self.user_entries()
        changed = [
            entry for entry in entries
            if entry[1] != bool(on) and (on or not ("\n" in entry[0] or "\r" in entry[0]))
        ]
        if changed:
            moved = {entry[0] for entry in changed}
            self.set_user_entries([
                (lora_id, bool(on) if lora_id in moved else entry_on, token)
                for lora_id, entry_on, token in entries
            ])
        return len(changed)

    def set_token(self, lora_id: str, token: str) -> bool:
        position = self.index_of(lora_id)
        if position < 0:
            return False
        self.tokens[position] = str(token or codec.DEFAULT_TOKEN).strip() or codec.DEFAULT_TOKEN
        return True

    def to_native(self, inventory: Inventory | None = None) -> tuple[list[str], str]:
        """Selections and multiplier tokens, emitted in exactly the same order.

        A selected LoRA that the inventory cannot resolve keeps its original
        value: imported settings may legitimately reference a file that is not
        installed locally, and dropping it would silently damage them.
        """
        values = []
        for lora_id in self.ids:
            entry = inventory.get(lora_id) if inventory else None
            values.append(entry.native_value if entry else lora_id)
        return values, codec.serialize(self.tokens, self.separator_index, self.disabled)


def build_items(
    inventory: Inventory,
    stack: "Stack",
    phases: PhaseConfig,
    metadata=None,
    model_key: str = "",
    memory: dict[str, list[float]] | None = None,
    catalogue: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Browser tiles for everything WanGP can activate for this model.

    ``catalogue`` maps LoRA id -> CatalogueIndex, supplying the Civitai name and
    trigger words. Those travel with the payload so naming, search and sort all
    happen client-side with no round trip.
    """
    memory = memory or {}
    catalogue = catalogue or {}
    items = []
    off = {lora.id for lora in stack.disabled}
    for entry in inventory.entries:
        position = stack.index_of(entry.id)
        index = catalogue.get(entry.id)
        item: dict[str, Any] = {
            "id": entry.id,
            "name": entry.name,
            "label": entry.label,
            "parent": entry.parent,
            "active": position >= 0,
            # In the stack, switched off: a tile click switches it back on.
            "disabled": entry.id in off,
            "favorite": bool(metadata.is_favorite(model_key, entry.id)) if metadata else False,
            "tag": metadata.tag(model_key, entry.id) if metadata else "",
            "system_managed": stack.is_system(position),
            "missing": False,
            "civitai_name": getattr(index, "civitai_name", "") if index else "",
            "words": list(getattr(index, "trained_words", []) or []) if index else [],
            "has_catalogue": bool(getattr(index, "has_sidecar", False)) if index else False,
            "has_video": bool(entry.video_preview) if entry else False,
            "mtime": entry.mtime,
        }
        if position >= 0:
            item.update(
                multiplier_fields(
                    stack.tokens[position], phases, entry.id, memory, with_schedules=False
                )
            )
        items.append(item)
    return items


def build_active_rows(
    inventory: Inventory,
    stack: "Stack",
    phases: PhaseConfig,
    memory: dict[str, list[float]] | None = None,
    catalogue: dict[str, Any] | None = None,
    schedules: dict[tuple[str, int], sch.PhaseSchedule] | None = None,
    context: ScheduleContext | None = None,
) -> list[dict[str, Any]]:
    """Editable rows, including LoRAs missing from disk.

    WanGP-managed LoRAs first, then the user's in the panel's order, with the
    switched-off ones where they were: ``enabled`` is off for those, and they
    carry only a ``summary`` of what switching them back on brings back.
    """
    memory = memory or {}
    catalogue = catalogue or {}
    schedules = schedules if schedules is not None else {}

    def base(lora_id: str, system: bool, enabled: bool) -> dict[str, Any]:
        entry = inventory.get(lora_id)
        index = catalogue.get(lora_id)
        fallback_name = lora_id.rsplit("/", 1)[-1]
        return {
            "id": lora_id,
            "name": entry.name if entry else fallback_name,
            "label": entry.label if entry else fallback_name,
            "civitai_name": getattr(index, "civitai_name", "") if index else "",
            "has_catalogue": bool(getattr(index, "has_sidecar", False)) if index else False,
            "system_managed": system,
            "missing": entry is None,
            "enabled": enabled,
        }

    rows = []
    listed = [
        (lora_id, True, stack.tokens[position], True)
        for position, lora_id in enumerate(stack.ids) if stack.is_system(position)
    ] + [(lora_id, on, token, False) for lora_id, on, token in stack.user_entries()]
    for lora_id, on, token, system in listed:
        row = base(lora_id, system, on)
        if on:
            row.update(multiplier_fields(token, phases, lora_id, memory, schedules, context))
        else:
            # Nothing to edit until it is back on: what it would bring back.
            row.update(multiplier_fields(token, phases, lora_id, {}, with_schedules=False))
            row["summary"] = describe_token(token, phases)
        rows.append(row)
    return rows


def multiplier_fields(
    raw: str,
    phases: PhaseConfig,
    lora_id: str,
    memory: dict[str, list[float]] | None = None,
    schedules: dict[tuple[str, int], sch.PhaseSchedule] | None = None,
    context: ScheduleContext | None = None,
    with_schedules: bool = True,
) -> dict[str, Any]:
    """Editor fields for one token: sliders, timeline, or preserved raw text.

    ``with_schedules`` is off for browser tiles, which only need to know that a
    LoRA is active -- sending every tile a region list would bloat the payload
    for nothing.
    """
    memory = memory if memory is not None else {}
    info = codec.classify(raw, phases.capacity)

    if info.kind == codec.ADVANCED:
        return {
            "multiplier_raw": info.raw,
            "multiplier_kind": codec.ADVANCED,
            "phase_values": [],
            "hidden_values": [],
            "linked": False,
            "phase_schedules": [],
            "schedule_shared": False,
            "schedule_reason": info.reason,
            "advanced_preserved": True,
            "stage_2_fill": None,
        }

    if info.kind == codec.SCHEDULED and scheduling_allowed(phases):
        return _scheduled_fields(info, phases, lora_id, schedules or {}, context, with_schedules)

    if info.kind == codec.SCHEDULED:
        # A mode with more than one guidance phase on a model whose phases are
        # not stages: the schedule is kept as it is, and is shown -- not edited
        # -- until One Phase can edit it truthfully again.
        return {
            "multiplier_raw": info.raw,
            "multiplier_kind": codec.SCHEDULED,
            "phase_values": [],
            "hidden_values": [],
            "linked": False,
            "phase_schedules": [],
            "schedule_shared": False,
            "schedule_reason": SCHEDULING_DISABLED_REASON,
            "advanced_preserved": False,
            "stage_2_fill": None,
            "summary": describe_token(info.raw, phases),
        }

    values = full_values(info, phases, lora_id, memory)
    visible = values[: phases.effective]
    return {
        "multiplier_raw": info.raw,
        "multiplier_kind": codec.SIMPLE,
        "phase_values": [round(value, VALUE_DECIMALS) for value in visible],
        "hidden_values": [round(value, VALUE_DECIMALS) for value in values[phases.effective:]],
        # Linking is a UI preference the user sets, never inferred from the
        # values happening to be equal -- otherwise a freshly added LoRA (1;1)
        # would start linked and dragging phase 1 would silently move phase 2.
        "linked": False,
        # A scalar phase can still have an *open* timeline: the editor holds the
        # schedule until the first region makes it real, so opening the panel
        # cannot change what WanGP renders.
        "phase_schedules": (
            _visible_schedules(
                lora_id, phases, schedules or {}, context,
                shared=token_is_shared(info, phases, lora_id, schedules),
            )
            if with_schedules and scheduling_allowed(phases) else []
        ),
        "schedule_shared": token_is_shared(info, phases, lora_id, schedules),
        "schedule_reason": "",
        "advanced_preserved": False,
        "stage_2_fill": None,
    }


def _scheduled_fields(
    info: codec.TokenInfo,
    phases: PhaseConfig,
    lora_id: str,
    schedules: dict[tuple[str, int], sch.PhaseSchedule],
    context: ScheduleContext | None,
    with_schedules: bool,
) -> dict[str, Any]:
    """Row fields for a token the timeline can edit."""
    token = info.schedule
    shared = token.shared
    bases = []
    for phase in range(phases.capacity):
        stored = schedules.get(schedule_key(lora_id, phase, shared))
        if stored is not None:
            bases.append(float(stored.base))
            continue
        values = token.values_for(phase)
        bases.append(float(values[0]) if values else 1.0)

    return {
        "multiplier_raw": info.raw,
        "multiplier_kind": codec.SCHEDULED,
        "phase_values": [round(value, VALUE_DECIMALS) for value in bases[: phases.effective]],
        "hidden_values": [round(value, VALUE_DECIMALS) for value in bases[phases.effective:]],
        "linked": False,
        "phase_schedules": (
            _visible_schedules(lora_id, phases, schedules, context, shared=shared)
            if with_schedules else []
        ),
        "schedule_shared": shared,
        "schedule_reason": "",
        "advanced_preserved": False,
        # What "Fill from stage 1" would draw on an empty stage 2.
        "stage_2_fill": (
            stage_2_fill_payload(stage_2_fill(info, phases, context)) if with_schedules else None
        ),
    }


def _visible_schedules(
    lora_id: str,
    phases: PhaseConfig,
    schedules: dict[tuple[str, int], sch.PhaseSchedule],
    context: ScheduleContext | None,
    shared: bool,
) -> list[dict[str, Any] | None]:
    """One entry per editable phase: its schedule payload, or ``None``."""
    result: list[dict[str, Any] | None] = []
    for phase in range(max(1, phases.effective)):
        stored = schedules.get(schedule_key(lora_id, phase, shared))
        result.append(
            schedule_payload(stored, shared=shared, phases=phases, context=context, phase=phase)
            if stored is not None else None
        )
    return result


def schedule_payload(
    schedule: sch.PhaseSchedule,
    *,
    shared: bool,
    phases: PhaseConfig,
    context: ScheduleContext | None,
    phase: int = 0,
) -> dict[str, Any]:
    """One phase's schedule, as the frontend consumes it.

    Region ids travel with it; they are an editor concept and never appear in
    the native token.

    Only what the run reaches is sent: a region crossing the end of the run
    arrives cut there, and one past it not at all, so the timeline shows
    exactly what WanGP is given.  ``drawn_past_end`` says how far the rest
    goes, so the panel can say it is kept.

    On a staged model the schedule also says which stage it drives and how
    long that stage is, so the timeline can be labelled with what runs.
    """
    shown = sch.visible_regions(schedule)
    selected = schedule.selected_region_id
    if not any(region.id == selected for region in shown):
        selected = shown[0].id if shown else None
    last = sch.last_drawn_step(schedule)
    stage = 0 if shared else phase
    staged = bool(phases.staged and context and context.phase_steps)
    return {
        "base": round(float(schedule.base), VALUE_DECIMALS),
        # What a region drawn now starts at, so the drawing ghost shows it.
        "seed_strength": round(seed_strength(schedule), VALUE_DECIMALS),
        "slots": sch.held_slots(schedule.slots),
        "coordinate_mode": coordinate_mode(
            schedule, shared=shared, phases=phases, context=context, phase=phase
        ),
        "steps": int(context.steps) if context else 0,
        # 1-based stage this timeline drives, the stages that run, and how many
        # steps this one is; all 0 where the model has no stages to speak of.
        "stage": stage + 1 if staged else 0,
        "stage_count": len(context.phase_steps) if staged else 0,
        "stage_steps": context.phase_length(stage) if staged else 0,
        "regions": [
            {
                "id": region.id,
                "start": int(region.start),
                "end": int(region.end),
                "strength": round(float(region.strength), VALUE_DECIMALS),
            }
            for region in shown
        ],
        "selected_region_id": selected,
        # The last step drawn on when that is past the end of the run, else 0.
        "drawn_past_end": last if last > sch.held_slots(schedule.slots) else 0,
        # Every slot no region covers is worth this. It is not a fill the base
        # hides behind: a scheduled phase applies the LoRA only where it is drawn.
        "gap_strength": sch.GAP_STRENGTH,
        "editable": bool(schedule.editable),
        "normalization_required": bool(schedule.normalization_required),
        "dirty": bool(schedule.dirty),
        "shared": bool(shared),
        # True once the native token actually carries comma syntax for it; an
        # open-but-empty schedule is not yet part of what WanGP is told.
        "active": sch.is_materialized(schedule),
        "reason": schedule.reason,
    }


def coordinate_mode(
    schedule: sch.PhaseSchedule,
    *,
    shared: bool,
    phases: PhaseConfig,
    context: ScheduleContext | None,
    phase: int = 0,
) -> str:
    """Decide what slot numbers may claim to be.

    Exact global steps are only claimed for a schedule that really does span the
    whole run at one slot per step.  Everything else says so: guessing phase
    boundaries from equal percentages would mislabel every model whose switch
    point is not exactly halfway.

    A staged model's phase is a stage of known length, so a timeline exactly
    that long is exact within its stage -- and when the stage is the whole run
    (One Phase), within the run.
    """
    if not schedule.editable:
        return COORD_NORMALIZED_READONLY
    slots = sch.held_slots(schedule.slots)
    if phases.staged and context and context.phase_steps:
        if slots != context.phase_length(0 if shared else phase):
            return COORD_PHASE_RELATIVE
        return COORD_GLOBAL_EXACT if len(context.phase_steps) == 1 else COORD_STAGE_EXACT
    steps = int(context.steps) if context else 0
    if shared and steps and slots == steps:
        return COORD_GLOBAL_EXACT
    return COORD_PHASE_RELATIVE


def describe_token(raw: str, phases: PhaseConfig) -> str:
    """One line saying what a multiplier does, for a row that only shows it.

    ``0.7``, ``Phase 1 0.7 · Phase 2 1``, ``∿ 1 at steps 1–4, 0.5 at step 6``:
    a scheduled phase is read as its regions, the way the timeline draws it.
    """
    info = codec.classify(raw, phases.capacity)
    if info.kind == codec.ADVANCED:
        return info.raw
    if info.kind == codec.SIMPLE:
        parts = [format_number(value) for value in (info.values or [1.0])[: phases.effective]]
    else:
        token = info.schedule
        count = 1 if token.shared else max(1, min(phases.effective, phases.capacity))
        parts = [_describe_values(token.values_for(phase)) for phase in range(count)]
    if len(parts) == 1:
        return parts[0]
    labels = phase_labels(len(parts))
    return " · ".join(f"{labels[index]} {part}" for index, part in enumerate(parts))


def _describe_values(values: list[float]) -> str:
    if len(values) <= 1:
        return format_number(values[0] if values else 1.0)
    regions = sch.reconstruct_schedule(values).regions
    if not regions:
        return "∿ off"
    return "∿ " + ", ".join(
        f"{format_number(region.strength)} at "
        + (f"step {region.start}" if region.start == region.end else f"steps {region.start}\u2013{region.end}")
        for region in regions
    )


def full_values(
    info: codec.TokenInfo,
    phases: PhaseConfig,
    lora_id: str,
    memory: dict[str, list[float]],
) -> list[float]:
    """Capacity-length phase vector, filling gaps from remembered values.

    Only the phases the token actually declared count as known; a token that
    was padded up to capacity must still defer to a remembered value.
    """
    return codec.rescale_values(list(info.declared), phases.capacity, memory.get(lora_id))


def remember_hidden_phases(stack: "Stack", phases: PhaseConfig, memory: dict[str, list[float]]) -> None:
    """Keep phase values the current mode cannot show.

    When a model's capacity is wide enough (H3 keeps two even in One Phase),
    the hidden value simply stays in the native token.  When capacity shrinks
    with the guidance mode, the token has to lose it -- WanGP rejects a token
    declaring more phases than the model supports -- so it is held here and
    restored the moment the mode comes back.
    """
    for position, lora_id in enumerate(stack.ids):
        info = codec.classify(stack.tokens[position], phases.capacity)
        if info.kind != codec.SIMPLE:
            continue
        values = list(info.declared)
        remembered = memory.get(lora_id, [])
        if len(remembered) > len(values):
            # Keep the remembered tail; refresh the head from the live token.
            values = values + remembered[len(values):]
        memory[lora_id] = values


def normalize_stack_tokens(stack: "Stack", phases: PhaseConfig, memory: dict[str, list[float]]) -> bool:
    """Rewrite simple tokens so they declare exactly ``phases.capacity`` values.

    Advanced tokens are left byte-for-byte alone.  Returns ``True`` when
    anything actually changed, so the caller can avoid a pointless native write
    (and the change-event loop it would start).
    """
    changed = False
    for position, lora_id in enumerate(stack.ids):
        info = codec.classify(stack.tokens[position], phases.capacity)
        if info.kind != codec.SIMPLE:
            continue
        values = full_values(info, phases, lora_id, memory)
        token = codec.build_token(values, phases.capacity)
        if token != stack.tokens[position]:
            stack.tokens[position] = token
            changed = True
    return changed


# ------------------------------------------------------- phase structure
#
# On a model whose LoRA phases follow the guidance mode -- LTX-2 among them --
# switching Two Phases to One leaves every "a;b" token declaring a phase the
# run no longer has, and WanGP refuses the whole multiplier string for it
# ("there should be at most 1 phases").  normalize_stack_tokens fixes simple
# tokens, but only on the next edit, and it never touches a schedule.  These
# fit every token to the mode as soon as the mode changes, and keep what they
# cut so the mode coming back restores it.
#
# Schedules on a staged model are the exception to the keeping.  Its phases
# are stages of different lengths -- eight distilled steps, then three of
# refinement, on LTX-2 -- so a list written for one stage means nothing in the
# other, and WanGP squeezing a comma-only list into both is not a schedule
# anyone drew.  There, a stage-1 schedule entering Two Phases gets an empty
# stage 2 of its own, and a stage 2 a narrower mode cut is not held: it starts
# empty again when the mode widens.  The switch then fills that empty stage 2
# from stage 1, as "Fill from stage 1" does (``fill_stages_from_stage_1``).
#
# MiniMax H3 keeps two phases in every mode, so One Phase cuts nothing from it:
# both stages stay in the token, and the mode only decides whether the second
# runs.  Phase 1 is the same run either way, so a switch never changes it.


def _value_list(part: str) -> bool:
    """A phase part WanGP reads as numbers: one value or a comma list."""
    entries = [entry.strip() for entry in str(part).split(",")]
    return bool(entries) and all(is_finite_number(entry) for entry in entries)


#: lora id -> (parts the cut kept, the parts it removed, in order).
HiddenPhases = dict[str, tuple[int, list[str]]]


def _phase_parts(token: str) -> list[str] | None:
    """The ``;`` parts of a token this may re-cut, or ``None`` to leave it be.

    Branch syntax and anything malformed are never re-cut: those stay exactly
    as imported.
    """
    raw = str(token or "").strip()
    if not raw or ":" in raw:
        return None
    parts = [part.strip() for part in raw.split(";")]
    return parts if all(_value_list(part) for part in parts) else None


def _remembered_phases(parts: list[str] | None, memory) -> list[str] | None:
    """Every phase ``parts`` stands for, given what a cut took -- or ``None``
    when the memory is not about this token.

    It is about the token the cut left (exactly ``kept`` parts) and about any
    longer form of it whose extra parts are the ones the cut removed: the
    token the cut was made from, arriving late in some event.
    """
    if parts is None or memory is None:
        return None
    kept, removed = memory
    extra = len(parts) - kept
    if extra < 0 or parts[kept:] != list(removed[:extra]):
        return None
    return parts + list(removed[extra:])


def _fitted_token(token: str, lora_id: str, phases: PhaseConfig, hidden: HiddenPhases):
    """``(token, memory)`` for this phase mode, or ``None`` when it fits.

    Parts beyond the mode are cut, and the memory says how many parts were
    kept and what went -- including anything an earlier, narrower cut took.
    A token the memory is about gets the cut parts back once the mode is wide
    enough for them.  Anything else with fewer parts than the mode is left
    alone: WanGP spreads its last part over the phases it does not name, which
    is valid and was someone's choice.

    A token that schedules any stage of a staged model is the exception: it
    follows ``_fitted_stages``, which holds nothing and restores nothing.
    """
    parts = _phase_parts(token)
    if parts is None:
        return None

    capacity = max(1, phases.capacity)
    if phases.staged and any(_is_list(part) for part in parts):
        return _fitted_stages(parts, phases)
    full = _remembered_phases(parts, hidden.get(normalize_id(lora_id))) or parts
    if len(parts) > capacity:
        return ";".join(parts[:capacity]), (capacity, full[capacity:])
    if len(full) > len(parts) and capacity > len(parts):
        restored, rest = full[:capacity], full[capacity:]
        return ";".join(restored), ((len(restored), rest) if rest else None)
    return None


def _is_list(part: str) -> bool:
    """A phase part that is a step schedule rather than one strength."""
    return "," in part


#: What a stage with nothing drawn on it is worth in a scheduled token: the
#: LoRA does not run there until a region says it does.
EMPTY_STAGE = format_number(sch.GAP_STRENGTH)


def _fitted_stages(parts: list[str], phases: PhaseConfig):
    """``_fitted_token`` for a staged token that schedules some stage.

    Stages beyond what the model accepts are cut and not remembered.  A
    stage-1 schedule with no stage 2 -- a One Phase token entering Two Phases
    -- gets an empty one, never a copy of itself: WanGP would otherwise expand
    that one list over both stages, squeezing eight steps' worth into the
    refinement's three.  The resync then fills it from stage 1
    (``fill_stages_from_stage_1``).

    Only a stage the mode runs is added.  H3 accepts two in One Phase too, and
    a list with no ';' runs exactly as drawn there, so it is left alone.
    """
    capacity = max(1, phases.capacity)
    running = max(1, min(phases.effective, capacity))
    if len(parts) > capacity:
        return ";".join(parts[:capacity]), None
    if len(parts) < running and _is_list(parts[-1]):
        return ";".join(parts + [EMPTY_STAGE] * (running - len(parts))), None
    return None


def has_step_schedule(stack: "Stack") -> bool:
    """True when some token schedules a phase, in a form the fit may re-cut."""
    return any(
        any(_is_list(part) for part in _phase_parts(token) or [])
        for token in stack.tokens
    )


def phase_structure_needs_fit(stack: "Stack", phases: PhaseConfig, hidden: HiddenPhases) -> bool:
    """True when some token declares phases the mode does not run, or could
    get back what a narrower mode took from it."""
    return any(
        _fitted_token(token, lora_id, phases, hidden) is not None
        for lora_id, token in zip(stack.ids, stack.tokens)
    )


def fit_phase_structure(stack: "Stack", phases: PhaseConfig, hidden: HiddenPhases) -> bool:
    """Make every token declare a phase structure WanGP accepts in this mode.

    Returns ``True`` when any token changed.  What a cut removes is kept in
    ``hidden`` verbatim and handed back when the mode widens again -- except
    from a staged token that schedules a stage, whose stage 2 starts empty in
    every Two Phases it enters (``_fitted_stages``).
    """
    changed = False
    for position, lora_id in enumerate(stack.ids):
        fitted = _fitted_token(stack.tokens[position], lora_id, phases, hidden)
        if fitted is None:
            continue
        token, memory = fitted
        key = normalize_id(lora_id)
        if memory is not None:
            hidden[key] = memory
        else:
            hidden.pop(key, None)
        if token != stack.tokens[position]:
            stack.tokens[position] = token
            changed = True
    return changed


def forget_superseded_phases(stack: "Stack", hidden: HiddenPhases) -> None:
    """Drop the memory of a cut whose token no longer looks as the cut left it.

    Something else rewrote it -- an edit filling every phase, a preset, a
    profile -- and parts cut from the old token must not be appended to a new
    one that has nothing to do with them.

    The token the cut was made from is not "something else": an event sent
    before the cut landed carries it, and it is recognisable -- its extra parts
    are exactly the ones removed.  Forgetting on that would lose a phase 2
    to nothing more than event timing.
    """
    for lora_id, token in zip(stack.ids, stack.tokens):
        key = normalize_id(lora_id)
        memory = hidden.get(key)
        if memory is not None and _remembered_phases(_phase_parts(token), memory) is None:
            del hidden[key]


def set_phase_value(
    stack: "Stack",
    lora_id: str,
    phase: int,
    value: Any,
    phases: PhaseConfig,
    memory: dict[str, list[float]],
    linked: bool = False,
    schedules: dict[tuple[str, int], sch.PhaseSchedule] | None = None,
) -> bool:
    """Write one strength edit back into the token.

    Which thing it writes depends on the selected phase:

    * an unscheduled phase -> the plain scalar,
    * a phase whose timeline is open but empty -> the value it is worth until
      something is drawn,
    * a phase that has regions -> nothing at all.

    The last rule is the point.  Once a phase is scheduled, every slot is either
    inside a region or zero, so there is no plain strength left for a slider to
    mean; the editor removes that control rather than leaving one that looks
    like it does something.  A stray edit from anywhere else is refused here for
    the same reason.

    Editing a LoRA whose token is advanced is refused too: the frontend has to
    ask for an explicit conversion first, so branch syntax is never flattened by
    an accidental drag.
    """
    position = stack.index_of(lora_id)
    if position < 0:
        return False

    info = codec.classify(stack.tokens[position], phases.capacity)
    if info.kind == codec.SCHEDULED or (
        info.kind == codec.SIMPLE and schedules and _has_schedule(lora_id, phases, schedules)
    ):
        if schedules is None:
            return False
        return _set_base(stack, info, lora_id, phase, value, phases, memory, schedules, linked)
    if info.kind != codec.SIMPLE:
        return False

    values = full_values(info, phases, lora_id, memory)
    number = sanitize_value(value, values[0] if values else 1.0)

    if linked and phases.effective > 1:
        for index in range(phases.effective):
            values[index] = number
    elif 0 <= int(phase) < len(values):
        values[int(phase)] = number
    else:
        return False

    stack.tokens[position] = codec.build_token(values, phases.capacity)
    memory[lora_id] = list(values)
    return True


def convert_to_simple(stack: "Stack", lora_id: str, phases: PhaseConfig) -> bool:
    """Explicitly replace an advanced schedule with a plain multiplier."""
    position = stack.index_of(lora_id)
    if position < 0:
        return False
    stack.tokens[position] = codec.default_token(phases.capacity)
    return True


# --------------------------------------------------------------- schedules
#
# Native tokens stay the truth.  Regions are the editor's decomposition of one
# phase's value list, held per media-generator tab and re-derived from the token
# whenever the two disagree -- so a preset, an .lset import or a queue edit can
# never leave the timeline showing something WanGP is not going to render.

#: Phase index used for a shared (comma-only) schedule, which belongs to no
#: single phase.
SHARED_PHASE = -1


def schedule_key(lora_id: str, phase: Any, shared: bool) -> tuple[str, int]:
    """Editor-state key for one phase's schedule."""
    try:
        index = int(phase)
    except (TypeError, ValueError):
        index = 0
    return (normalize_id(lora_id), SHARED_PHASE if shared else max(0, index))


def token_is_shared(
    info: codec.TokenInfo,
    phases: PhaseConfig,
    lora_id: str = "",
    schedules: dict[tuple[str, int], sch.PhaseSchedule] | None = None,
) -> bool:
    """Whether this LoRA's schedule spans the run rather than one phase.

    An existing comma-only token is shared by definition.  A token with no
    schedule yet becomes shared only on a single-phase model, where a comma list
    cannot mean anything else.

    A shared timeline whose regions currently work out flat collapses back to a
    scalar in the token, so the token alone would say "not shared" and the
    editor would lose the timeline.  An already-open shared schedule therefore
    keeps its answer.

    Except across a staged model's two stages, which are never shared: they run
    different numbers of steps, so each gets a timeline of its own, and a
    comma-only token there is only waiting for ``fit_phase_structure`` to give
    it an empty stage 2.  A shared timeline left open by One Phase is not
    carried into Two Phases.
    """
    if info.kind == codec.SCHEDULED and info.schedule is not None:
        return info.schedule.shared
    if phases.staged and phases.capacity > 1:
        return False
    if lora_id and schedules and schedule_key(lora_id, 0, True) in schedules:
        return True
    return phases.capacity <= 1


def sync_schedules(
    stack: "Stack",
    phases: PhaseConfig,
    schedules: dict[tuple[str, int], sch.PhaseSchedule],
    context: ScheduleContext | None = None,
) -> None:
    """Reconcile held region state with the native tokens.

    Kept when the token is what the stored regions say -- that is what makes
    region ids stable across a round trip -- and rebuilt from the token
    otherwise.  Nothing here writes to the token: an imported schedule is
    rendered, never re-serialised.

    "What the stored regions say" is read at the token's own length, not the
    schedule's, and against the versions the panel recently wrote as well as
    the current one.  Both are about the same fact: every event Gradio sends
    carries the multiplier text as it stood when the event was *sent*.  The step
    slider fires on every value it passes through, so while it is dragged, or
    while a step count is typed, the token in hand is routinely a step or a
    write behind.  Rebuilding the schedule from such a token is how a step
    change used to stretch it: the rebuild could not know the schedule had been
    drawn one slot per step, judged it an import because its length disagreed
    with the counter, and the resync that followed resampled it.  A token that
    is a window onto this schedule, or onto a version of it, is that schedule.
    """
    live: set[tuple[str, int]] = set()
    if not scheduling_allowed(phases):
        # Nothing here could be edited truthfully.  The tokens are left exactly
        # as they are, so One Phase picks every schedule up again.
        schedules.clear()
        return

    context = context or ScheduleContext()
    for position, lora_id in enumerate(stack.ids):
        info = codec.classify(stack.tokens[position], phases.capacity)
        if info.kind == codec.ADVANCED:
            continue

        shared = token_is_shared(info, phases, lora_id, schedules)
        for phase in range(1 if shared else phases.capacity):
            key = schedule_key(lora_id, phase, shared)
            values = _phase_values(info, phases, lora_id, phase)
            stored = schedules.get(key)
            held = _recognise(stored, values) if stored is not None else None
            target = context.target_slots(phase)

            if held is not None:
                live.add(key)
                schedules[key] = held
                if not held.regions:
                    # Nothing has been drawn yet, so the timeline is free to
                    # follow the step counter the user is currently looking at.
                    held.slots = context.default_slots(shared, phase)
                    continue
                # The window is wherever the token says it is.  When that is a
                # step behind the counter, schedules_need_resync says so and
                # the resync moves it -- from these regions, not this token.
                held.slots = sch.held_slots(len(values))
                if target and sch.clamp_slots(held.slots) == target:
                    # One value per step, on screen right now: whatever this
                    # schedule's history, slot i is step i from here on.
                    held.step_aligned = True
                continue

            if len(values) > 1:
                # The token carries a schedule this editor state does not match:
                # WanGP is the truth, so derive the timeline from the token.
                live.add(key)
                rebuilt = sch.reconstruct_schedule(values, raw=info.raw)
                # A token the panel did not write: one value per step means it
                # runs one value per step, so a later change to the counter
                # moves the end of its timeline and leaves every region where
                # it is.  A token of some other length was authored at its own
                # resolution and is spread across the whole run by WanGP, so it
                # is resampled into alignment instead, once.
                #
                # The panel's own recent tokens do not get here -- _recognise
                # knows them at any length -- and that is the point: judging
                # those by their length is what stretched a drawn schedule
                # whenever a step change arrived in more than one event.
                rebuilt.step_aligned = len(values) == target
                if stored is not None:
                    # Carry the memory over, so a late event with a token the
                    # panel wrote before this one is still recognised.
                    rebuilt.history = stored.history
                schedules[key] = rebuilt

    # A switched-off LoRA's timelines wait for it, so switching it back on finds
    # the same regions -- anything kept past the end of the run included.
    off = {entry.id for entry in stack.disabled}
    for key in [key for key in schedules if key not in live and key[0] not in off]:
        del schedules[key]


def _drawn_schedules(
    stack: "Stack",
    phases: PhaseConfig,
    schedules: dict[tuple[str, int], sch.PhaseSchedule],
):
    """Every phase that has something drawn on its timeline, in stack order.

    An empty timeline is skipped: it has no values in the token, so its length
    is a view preference that ``sync_schedules`` already keeps current.
    """
    for position, lora_id in enumerate(stack.ids):
        info = codec.classify(stack.tokens[position], phases.capacity)
        if info.kind == codec.ADVANCED:
            continue
        shared = token_is_shared(info, phases, lora_id, schedules)
        for phase in range(1 if shared else phases.capacity):
            key = schedule_key(lora_id, phase, shared)
            schedule = schedules.get(key)
            if schedule is None or not schedule.regions or not schedule.editable:
                continue
            yield lora_id, phase, key, schedule


def schedules_need_resync(
    stack: "Stack",
    phases: PhaseConfig,
    schedules: dict[tuple[str, int], sch.PhaseSchedule],
    context: ScheduleContext | None,
) -> bool:
    """True when some timeline is not the length its phase runs."""
    if not scheduling_allowed(phases):
        # Nothing is edited in this mode, so nothing follows the counter
        # either; a schedule waits in its token for One Phase.
        return False

    context = context or ScheduleContext()
    for _, phase, _, schedule in _drawn_schedules(stack, phases, schedules):
        target = context.target_slots(phase)
        if target and sch.clamp_slots(schedule.slots) != target:
            return True
    return False


def refit_schedules(
    stack: "Stack",
    phases: PhaseConfig,
    schedules: dict[tuple[str, int], sch.PhaseSchedule],
    memory: dict[str, list[float]],
    context: ScheduleContext | None,
) -> bool:
    """Bring every timeline to the current step count.

    Two different operations, because a slot means two different things:

    * a schedule drawn in the panel is step-aligned -- slot *i* is step *i* --
      so only the **end of its window moves**.  Steps that still exist keep
      exactly what they had, a new step arrives empty unless something was
      drawn there before, and steps cut off are hidden rather than deleted.
    * a schedule that arrived from a preset, an .lset file or a hand edit was
      authored at its own resolution, and WanGP spreads it across the whole run.
      Truncating that would change what it renders, so it is **resampled** into
      step alignment once, and is step-aligned from then on.

    In a guidance mode that cannot show a schedule truthfully, nothing is
    refitted and nothing is cleared: a phase switch never rewrites a
    schedule, so switching back finds it exactly as it was.

    ``step_aligned`` is what tells them apart; ``sync_schedules`` decides it.
    """
    if not scheduling_allowed(phases):
        return False

    context = context or ScheduleContext()
    changed = False
    for lora_id, phase, key, schedule in list(_drawn_schedules(stack, phases, schedules)):
        # Each phase follows its own length: on LTX-2 the refinement stage is
        # three steps whatever the counter says.
        target = context.target_slots(phase)
        if not target or sch.clamp_slots(schedule.slots) == target:
            continue
        schedules[key] = (
            sch.refit_schedule(schedule, target) if schedule.step_aligned
            else sch.normalize_schedule(schedule, target)
        )
        target_state = _resolve(stack, lora_id, phase, phases, memory, schedules, context, create=False)
        changed = _commit(stack, lora_id, target_state, phases, memory) or changed
    return changed


def _phase_values(info: codec.TokenInfo, phases: PhaseConfig, lora_id: str, phase: int) -> list[float]:
    """The native value list driving one phase, without inventing anything."""
    if info.kind == codec.SCHEDULED and info.schedule is not None:
        return info.schedule.values_for(phase)
    if info.kind == codec.SIMPLE and info.values:
        index = min(max(0, int(phase)), len(info.values) - 1)
        return [float(info.values[index])]
    return []


def _recognise(stored: sch.PhaseSchedule, values: list[float]) -> sch.PhaseSchedule | None:
    """The schedule ``values`` came from: ``stored``, a recent version, or None.

    The newest version that fits wins.  A version is put back as a copy that
    still shares the history, so recognising an old token costs nothing that a
    newer one would need.
    """
    if _tracks(stored, values):
        return stored
    # A copy: Gradio runs handlers on worker threads, and an edit appending a
    # version mid-loop is no reason to miss one.
    for version in reversed(list(stored.history)):
        if _tracks(version, values):
            restored = version.copy()
            restored.history = stored.history
            return restored
    return None


def _tracks(schedule: sch.PhaseSchedule, values: list[float]) -> bool:
    """True when the token says what ``schedule`` says, at the token's length.

    Compared against what the schedule would *emit*, so a timeline that is open
    but not yet doing anything keeps its slot count and regions instead of being
    rebuilt from the scalar it currently compiles to.

    Compared at the token's own length because the step count is only a window
    onto a drawn schedule: the same regions read through one step more or less
    are still the same schedule, and a token a step behind the counter is the
    normal state of affairs while the step slider moves.  Other lengths stop at
    what the panel could have written: the editor never cuts a drawn schedule
    down to one step, so a scalar is somebody's plain strength, and nothing it
    writes is longer than a timeline can be.
    """
    if not values:
        return False
    own = len(values) == sch.held_slots(schedule.slots)
    if not schedule.regions:
        emitted = [float(schedule.base)]
    elif own or sch.MIN_WINDOW <= len(values) <= sch.MAX_SLOTS:
        emitted = sch.compile_window(schedule, len(values))
    else:
        return False
    if len(emitted) != len(values):
        return False
    return all(_close(left, right) for left, right in zip(emitted, values))


def _close(left: float, right: float) -> bool:
    return abs(float(left) - float(right)) <= 1e-9


def _has_schedule(
    lora_id: str, phases: PhaseConfig, schedules: dict[tuple[str, int], sch.PhaseSchedule]
) -> bool:
    if schedule_key(lora_id, 0, True) in schedules:
        return True
    return any(schedule_key(lora_id, phase, False) in schedules for phase in range(phases.capacity))


@dataclass
class _Target:
    """Everything an edit needs about one (LoRA, phase) schedule."""

    info: codec.TokenInfo
    key: tuple[str, int]
    schedule: sch.PhaseSchedule
    lists: list[list[float]]
    index: int
    shared: bool


def _resolve(
    stack: "Stack",
    lora_id: str,
    phase: Any,
    phases: PhaseConfig,
    memory: dict[str, list[float]],
    schedules: dict[tuple[str, int], sch.PhaseSchedule],
    context: ScheduleContext | None,
    create: bool = False,
) -> _Target:
    """Locate one phase's schedule, raising a message the panel can show."""
    position = stack.index_of(lora_id)
    if position < 0:
        raise sch.ScheduleError("That LoRA is not active.")

    if not scheduling_allowed(phases):
        raise sch.ScheduleError(SCHEDULING_DISABLED_REASON)

    info = codec.classify(stack.tokens[position], phases.capacity)
    if info.kind == codec.ADVANCED:
        raise sch.ScheduleError(
            info.reason or "This multiplier is preserved as imported and is not editable here."
        )

    shared = token_is_shared(info, phases, lora_id, schedules)
    index = 0 if shared else max(0, int(phase or 0))
    if index >= phases.capacity:
        raise sch.ScheduleError("That phase does not exist for this model.")

    key = schedule_key(lora_id, index, shared)
    schedule = schedules.get(key)
    if schedule is None:
        if not create:
            raise sch.ScheduleError("No schedule is open for that phase.")
        values = _phase_values(info, phases, lora_id, index)
        base = float(values[0]) if values else 1.0
        schedule = sch.PhaseSchedule(
            base=base,
            slots=(context or ScheduleContext()).default_slots(shared, index),
            regions=[],
            # Opened at the step count, so it is step-aligned before anything
            # is drawn on it.
            step_aligned=True,
        )
        schedules[key] = schedule

    if not schedule.editable:
        raise sch.ScheduleError(
            schedule.reason or "This schedule is preserved as imported and is read-only."
        )

    lists = _token_lists(info, phases, lora_id, memory, schedules, shared)
    return _Target(info=info, key=key, schedule=schedule, lists=lists, index=index, shared=shared)


def _token_lists(
    info: codec.TokenInfo,
    phases: PhaseConfig,
    lora_id: str,
    memory: dict[str, list[float]],
    schedules: dict[tuple[str, int], sch.PhaseSchedule],
    shared: bool,
) -> list[list[float]]:
    """Every phase's value list, ready to be re-serialised.

    Phases the current guidance mode hides are included exactly as the token
    spelled them, so editing phase 1 cannot drop a phase 2 the user tuned.
    """
    if shared:
        values = _phase_values(info, phases, lora_id, 0)
        return [list(values) if values else [1.0]]

    if info.kind == codec.SCHEDULED and info.schedule is not None:
        lists = [list(values) for values in info.schedule.phase_values]
        while len(lists) < phases.capacity:
            lists.append(list(lists[-1]) if lists else [1.0])
        return lists[: phases.capacity]

    scalars = full_values(info, phases, lora_id, memory)
    return [[value] for value in scalars]


def _commit(
    stack: "Stack",
    lora_id: str,
    target: _Target,
    phases: PhaseConfig,
    memory: dict[str, list[float]],
) -> bool:
    """Serialise the edited phase back into the native token."""
    target.schedule.dirty = True
    # Drawn in the panel means drawn against the step counter: slot i is step
    # i, and a later change to the step count moves the end of the timeline
    # rather than sliding every region along it.
    target.schedule.step_aligned = True
    lists = [list(values) for values in target.lists]
    lists[target.index] = sch.native_values(target.schedule)

    if target.shared:
        lists = lists[:1]

    if all(len(values) <= 1 for values in lists):
        scalars = [values[0] if values else 1.0 for values in lists]
        if target.shared and phases.capacity > 1:
            # A shared scalar is simply a scalar; pad it the way a simple token
            # is padded so the rest of the editor sees one value per phase.
            scalars = codec.rescale_values(scalars, phases.capacity, memory.get(lora_id))
        token = codec.build_token(scalars, phases.capacity)
        memory[normalize_id(lora_id)] = list(scalars)
    else:
        token = codec.build_schedule(lists, shared=target.shared)

    changed = stack.token_for(lora_id) != token
    stack.set_token(lora_id, token)
    # So that this token, arriving late in some event after a newer one, is
    # known for the schedule it came from.
    sch.remember(target.schedule)
    return changed


def _set_base(
    stack: "Stack",
    info: codec.TokenInfo,
    lora_id: str,
    phase: Any,
    value: Any,
    phases: PhaseConfig,
    memory: dict[str, list[float]],
    schedules: dict[tuple[str, int], sch.PhaseSchedule],
    linked: bool,
) -> bool:
    """Strength edit on a LoRA that has at least one schedule.

    Linking moves the *base* of every visible phase, as the user asked.  It
    never copies regions between phases: schedules stay independent unless the
    user explicitly asks for one to be duplicated.
    """
    shared = token_is_shared(info, phases, lora_id, schedules)
    targets = range(phases.effective) if (linked and phases.effective > 1 and not shared) else [int(phase or 0)]
    changed = False

    for index in targets:
        try:
            target = _resolve(stack, lora_id, index, phases, memory, schedules, None, create=False)
        except sch.ScheduleError:
            # An unscheduled phase keeps its plain scalar behaviour.
            changed = _set_scalar_phase(stack, lora_id, index, value, phases, memory, schedules) or changed
            continue
        if target.schedule.regions:
            # Scheduled: the timeline owns every slot, so there is nothing here
            # for a plain strength edit to change.
            continue
        target.schedule.base = sanitize_value(value, target.schedule.base)
        changed = _commit(stack, lora_id, target, phases, memory) or changed

    return changed


def _set_scalar_phase(
    stack: "Stack",
    lora_id: str,
    phase: int,
    value: Any,
    phases: PhaseConfig,
    memory: dict[str, list[float]],
    schedules: dict[tuple[str, int], sch.PhaseSchedule] | None = None,
) -> bool:
    """Set one unscheduled phase of a token whose other phases are scheduled."""
    position = stack.index_of(lora_id)
    if position < 0:
        return False
    info = codec.classify(stack.tokens[position], phases.capacity)
    if info.kind == codec.ADVANCED:
        return False

    shared = token_is_shared(info, phases, lora_id, schedules)
    lists = _token_lists(info, phases, lora_id, memory, schedules or {}, shared)
    index = max(0, min(int(phase), len(lists) - 1))
    if len(lists[index]) > 1:
        return False
    number = sanitize_value(value, lists[index][0] if lists[index] else 1.0)
    lists[index] = [number]

    if all(len(values) <= 1 for values in lists):
        scalars = [values[0] for values in lists]
        token = codec.build_token(scalars, phases.capacity)
        memory[normalize_id(lora_id)] = list(scalars)
    else:
        token = codec.build_schedule(lists, shared=shared)

    changed = stack.tokens[position] != token
    stack.tokens[position] = token
    return changed


def schedule_enable(
    stack: "Stack",
    lora_id: str,
    phase: Any,
    phases: PhaseConfig,
    memory: dict[str, list[float]],
    schedules: dict[tuple[str, int], sch.PhaseSchedule],
    context: ScheduleContext | None = None,
) -> bool:
    """Open a timeline for one phase, starting from its current strength.

    Nothing is written to WanGP: the schedule has no regions yet, so it means
    exactly the scalar it came from.  That is what makes opening the panel safe.
    """
    _resolve(stack, lora_id, phase, phases, memory, schedules, context, create=True)
    return False


def schedule_set_base(
    stack: "Stack",
    lora_id: str,
    phase: Any,
    value: Any,
    phases: PhaseConfig,
    memory: dict[str, list[float]],
    schedules: dict[tuple[str, int], sch.PhaseSchedule],
    context: ScheduleContext | None = None,
    linked: bool = False,
) -> bool:
    """Set what a phase is worth while its timeline is still empty.

    Refused once regions exist: their gaps are zero, not this value, so writing
    it would only change what a future region seeds at without changing a single
    slot the user can see.
    """
    target = _resolve(stack, lora_id, phase, phases, memory, schedules, context, create=True)
    if target.schedule.regions:
        raise sch.ScheduleError(
            "This phase is scheduled - edit its regions on the timeline instead."
        )
    target.schedule.base = sanitize_value(value, target.schedule.base)
    changed = _commit(stack, lora_id, target, phases, memory)

    if linked and phases.effective > 1 and not target.shared:
        for index in range(phases.effective):
            if index == target.index:
                continue
            try:
                other = _resolve(stack, lora_id, index, phases, memory, schedules, context, create=False)
            except sch.ScheduleError:
                changed = _set_scalar_phase(stack, lora_id, index, value, phases, memory, schedules) or changed
                continue
            if other.schedule.regions:
                continue
            other.schedule.base = sanitize_value(value, other.schedule.base)
            changed = _commit(stack, lora_id, other, phases, memory) or changed
    return changed


def seed_strength(schedule: sch.PhaseSchedule) -> float:
    """The strength a newly drawn region starts at: the schedule's base.

    Unless the base is what an empty slot is already worth -- an empty
    stage 2, an all-zero import -- when a region seeded at it would draw
    nothing at all.  It starts at full strength then, as a new LoRA does.
    """
    base = sanitize_value(schedule.base, 1.0)
    return 1.0 if _close(base, sch.GAP_STRENGTH) else base


def schedule_add_region(
    stack: "Stack",
    lora_id: str,
    phase: Any,
    phases: PhaseConfig,
    memory: dict[str, list[float]],
    schedules: dict[tuple[str, int], sch.PhaseSchedule],
    context: ScheduleContext | None = None,
    start: Any = None,
    end: Any = None,
) -> bool:
    """Add one region at the schedule's base strength.

    With ``start`` (and optionally ``end``) the user drew it on the timeline, so
    it goes where they put it -- clipped to the free run their gesture started
    in, never over a neighbour.  Without them it is placed automatically: the
    free tail, else the first hole that fits, else the largest.

    Either way the timeline being full is a refusal, not an overlap: creating a
    region on top of another and letting collision resolution sort it out would
    delete someone's work as a side effect of one tap.
    """
    target = _resolve(stack, lora_id, phase, phases, memory, schedules, context, create=True)
    schedule = target.schedule
    span = None
    if start is not None:
        span = sch.fit_region(schedule.regions, schedule.slots, start, end)
    if span is None:
        span = sch.find_region_slot(schedule.regions, schedule.slots)
    if span is None:
        raise sch.ScheduleError("The schedule is full - delete or shrink a region first.")

    region = sch.Region(
        id=sch.next_region_id(schedule.regions),
        start=span[0],
        end=span[1],
        strength=seed_strength(schedule),
    )
    schedule.regions = sch.sort_regions(schedule.regions + [region])
    schedule.selected_region_id = region.id
    sch.validate_regions(schedule.regions, schedule.slots)
    return _commit(stack, lora_id, target, phases, memory)


def schedule_commit_region(
    stack: "Stack",
    lora_id: str,
    phase: Any,
    region_id: str,
    start: Any,
    end: Any,
    phases: PhaseConfig,
    memory: dict[str, list[float]],
    schedules: dict[tuple[str, int], sch.PhaseSchedule],
    context: ScheduleContext | None = None,
    keep_width: bool = False,
) -> bool:
    """Accept the final bounds of a move or resize and settle the collisions.

    The frontend sends where the region ended up, not how it got there: the
    gesture is presentation, the bounds are the contract.

    A region crossing the end of the run is dragged as the panel shows it, cut
    at the end.  If it still reaches the end afterwards -- its left edge moved,
    or it was put back where it was -- the part past the end, which the gesture
    could not see, stays as it was.  Moved away from the end or shortened from
    the right, it is the region the user drew and nothing more.
    """
    target = _resolve(stack, lora_id, phase, phases, memory, schedules, context, create=False)
    schedule = target.schedule
    region = schedule.region(region_id)
    if region is None:
        raise sch.ScheduleError("That region no longer exists.")
    slots = sch.clamp_slots(schedule.slots)
    if int(region.start) > slots:
        raise sch.ScheduleError(
            "That region is past the last step; raise the step count to edit it."
        )

    shown_width = min(int(region.end), slots) - int(region.start) + 1
    first, last = sch.clamp_region(
        start, end, slots, keep_width=keep_width, width=shown_width
    )
    if int(region.end) > slots and last == slots:
        last = int(region.end)
    winner = sch.Region(id=region.id, start=first, end=last, strength=region.strength)
    schedule.regions = sch.resolve_collision(schedule.regions, winner)
    schedule.selected_region_id = winner.id
    sch.validate_regions(schedule.regions, schedule.slots)
    return _commit(stack, lora_id, target, phases, memory)


def schedule_set_region_strength(
    stack: "Stack",
    lora_id: str,
    phase: Any,
    region_id: str,
    value: Any,
    phases: PhaseConfig,
    memory: dict[str, list[float]],
    schedules: dict[tuple[str, int], sch.PhaseSchedule],
    context: ScheduleContext | None = None,
) -> bool:
    target = _resolve(stack, lora_id, phase, phases, memory, schedules, context, create=False)
    region = target.schedule.region(region_id)
    if region is None:
        raise sch.ScheduleError("That region no longer exists.")
    region.strength = sanitize_value(value, region.strength)
    target.schedule.selected_region_id = region.id
    return _commit(stack, lora_id, target, phases, memory)


def schedule_delete_region(
    stack: "Stack",
    lora_id: str,
    phase: Any,
    region_id: str,
    phases: PhaseConfig,
    memory: dict[str, list[float]],
    schedules: dict[tuple[str, int], sch.PhaseSchedule],
    context: ScheduleContext | None = None,
) -> bool:
    target = _resolve(stack, lora_id, phase, phases, memory, schedules, context, create=False)
    schedule = target.schedule
    if schedule.region(region_id) is None:
        raise sch.ScheduleError("That region no longer exists.")
    schedule.regions = [region for region in schedule.regions if region.id != str(region_id)]
    schedule.selected_region_id = schedule.regions[0].id if schedule.regions else None
    return _commit(stack, lora_id, target, phases, memory)


def schedule_clear_phase(
    stack: "Stack",
    lora_id: str,
    phase: Any,
    phases: PhaseConfig,
    memory: dict[str, list[float]],
    schedules: dict[tuple[str, int], sch.PhaseSchedule],
    context: ScheduleContext | None = None,
) -> bool:
    """Return one phase to its schedule base, leaving other phases alone.

    Deliberately different from closing the panel, which only collapses it.
    """
    target = _resolve(stack, lora_id, phase, phases, memory, schedules, context, create=False)
    target.schedule.regions = []
    target.schedule.selected_region_id = None
    changed = _commit(stack, lora_id, target, phases, memory)
    schedules.pop(target.key, None)
    return changed


def schedule_normalize(
    stack: "Stack",
    lora_id: str,
    phase: Any,
    slots: Any,
    phases: PhaseConfig,
    memory: dict[str, list[float]],
    schedules: dict[tuple[str, int], sch.PhaseSchedule],
    context: ScheduleContext | None = None,
) -> bool:
    """Re-grid one schedule onto a different number of slots.

    Only ever from an explicit user action: silently regridding an imported
    schedule would change what WanGP renders without anyone asking.
    """
    position = stack.index_of(lora_id)
    if position < 0:
        raise sch.ScheduleError("That LoRA is not active.")
    info = codec.classify(stack.tokens[position], phases.capacity)
    if info.kind == codec.ADVANCED:
        raise sch.ScheduleError(
            info.reason or "This multiplier is preserved as imported and is not editable here."
        )

    shared = token_is_shared(info, phases, lora_id, schedules)
    index = 0 if shared else max(0, int(phase or 0))
    key = schedule_key(lora_id, index, shared)
    existing = schedules.get(key)
    if existing is None:
        raise sch.ScheduleError("No schedule is open for that phase.")

    target_slots = sch.clamp_slots(slots)
    if target_slots == sch.clamp_slots(existing.slots) and existing.editable:
        return False

    schedules[key] = sch.normalize_schedule(existing, target_slots)
    target = _resolve(stack, lora_id, index, phases, memory, schedules, context, create=False)
    return _commit(stack, lora_id, target, phases, memory)


# ----------------------------------------------------------- stage 2 fill
#
# A stage 2 is a timeline of its own: nothing keeps it in step with
# stage 1.  It is drawn from stage 1 at two moments only -- entering Two
# Phases, and "Fill from stage 1" on an empty stage 2 -- each time once, from
# what stage 1 applies where the two stages share noise levels, and it is
# plain regions from then on.


@dataclass(frozen=True)
class Stage2Fill:
    """What "Fill from stage 1" would draw on an empty stage 2, and why."""

    #: One value per refinement step.
    values: tuple[float, ...]
    #: The stage-1 steps they were read from, matched by noise level -- or
    #: empty when stage 2 holds stage 1's final strength instead.
    steps: tuple[int, ...] = ()

    @property
    def empty(self) -> bool:
        """Nothing to carry: every value is what an empty slot is worth."""
        return all(_close(value, sch.GAP_STRENGTH) for value in self.values)

    @property
    def note(self) -> str:
        """What the button says it will do, or why it cannot."""
        if self.steps:
            if self.empty:
                return (
                    f"Stage 1 is off at steps {step_span(self.steps)}, where it runs stage 2's "
                    "noise levels - there is nothing to carry."
                )
            return (
                f"Copy stage 1's steps {step_span(self.steps)}, matched by noise level: "
                + ", ".join(format_number(value) for value in self.values)
            )
        if self.empty:
            return "Stage 1 ends at 0 - there is nothing to carry."
        return (
            f"Hold stage 1's final strength, {format_number(self.values[0])}, for all "
            f"{len(self.values)} steps: this sampler's steps do not line up with stage 2's."
        )

    @property
    def done(self) -> str:
        """What the status bar says once it has."""
        if self.steps:
            return f"Stage 2 filled from stage 1's steps {step_span(self.steps)}, matched by noise level."
        return (
            f"Stage 2 holds stage 1's final strength, {format_number(self.values[0])}: "
            "this sampler's steps do not line up with stage 2's."
        )


def step_span(steps: tuple[int, ...]) -> str:
    """``(6, 7, 8)`` as "6-8"; steps that do not run on from each other, listed."""
    consecutive = list(steps) == list(range(steps[0], steps[0] + len(steps)))
    if consecutive and len(steps) > 1:
        return f"{steps[0]}-{steps[-1]}"
    return ", ".join(str(step) for step in steps)


def _played(values: list[float], steps: int) -> list[float]:
    """What WanGP applies at each of ``steps`` steps for one phase's list.

    Mirrors ``expand_one`` in WanGP's multiplier parser -- entry
    ``i * len // steps`` at step i -- so a list of any length is read the way
    it runs.
    """
    count = len(values)
    return [float(values[index * count // steps]) for index in range(steps)]


def derive_stage_2(stage_1: list[float], context: ScheduleContext) -> Stage2Fill:
    """Stage 2's values, from what stage 1 applies -- copied, never stretched.

    Where stage 1's noise levels are known, each refinement step takes what
    stage 1 applies at the matching level: a distilled stage 1's steps 6-8.
    Its first five steps, where layout and motion are set, have no counterpart
    in the refinement and are never read.  Otherwise there is no step to match,
    and stage 2 holds the strength stage 1 ends at: one number, no shape the
    run did not have.
    """
    played = _played(stage_1, context.phase_length(0))
    steps = tuple(step for step in context.stage_2_match if step <= len(played))
    if steps:
        return Stage2Fill(values=tuple(played[step - 1] for step in steps), steps=steps)
    return Stage2Fill(values=(played[-1],) * context.phase_length(1))


def stage_2_fill(
    info: codec.TokenInfo, phases: PhaseConfig, context: ScheduleContext | None
) -> Stage2Fill | None:
    """What "Fill from stage 1" offers this token, or ``None`` for nothing.

    Offered on a staged model in Two Phases, for a token whose stage 1 is scheduled and
    whose stage 2 has nothing drawn on it.  Whatever plain strength stage 2 has
    stays its schedule's base, so Clear phase puts it back.
    """
    if not (phases.staged and phases.effective > 1 and context and len(context.phase_steps) > 1):
        return None
    if info.kind != codec.SCHEDULED or info.schedule is None or info.schedule.shared:
        return None
    stage_1, stage_2 = info.schedule.values_for(0), info.schedule.values_for(1)
    if len(stage_1) < 2 or len(stage_2) > 1:
        return None
    return derive_stage_2(stage_1, context)


def stage_2_fill_payload(fill: Stage2Fill | None) -> dict[str, Any] | None:
    if fill is None:
        return None
    return {
        "values": [round(value, VALUE_DECIMALS) for value in fill.values],
        "steps": list(fill.steps),
        "empty": fill.empty,
        "note": fill.note,
    }


def entering_two_phases(stack: "Stack", phases: PhaseConfig) -> list[str]:
    """LoRAs whose stage-1 schedule has no stage 2 in Two Phases yet: the ones
    ``fit_phase_structure`` is about to give an empty one."""
    running = min(phases.effective, phases.capacity)
    if not (phases.staged and running > 1):
        return []
    entering = []
    for lora_id, token in zip(stack.ids, stack.tokens):
        parts = _phase_parts(token)
        if parts and len(parts) < running and _is_list(parts[-1]):
            entering.append(lora_id)
    return entering


def fill_stages_from_stage_1(
    stack: "Stack",
    lora_ids: list[str],
    phases: PhaseConfig,
    memory: dict[str, list[float]],
    schedules: dict[tuple[str, int], sch.PhaseSchedule],
    context: ScheduleContext | None = None,
) -> bool:
    """Fill each of ``lora_ids``' empty stage 2 from its stage 1, as the button does.

    What entering Two Phases does: the refinement carries on from where stage
    1 ends, matched by noise level.  A LoRA with nothing to carry keeps its
    empty stage 2.  A filled one is plain regions whose base is what stage 2
    was -- empty, for a LoRA entering Two Phases -- so Clear phase empties it
    again.
    """
    changed = False
    for lora_id in lora_ids:
        try:
            changed = schedule_fill_stage_2(stack, lora_id, phases, memory, schedules, context) or changed
        except sch.ScheduleError:
            # Nothing to carry, or no stage lengths to read it by.
            continue
    return changed


def schedule_fill_stage_2(
    stack: "Stack",
    lora_id: str,
    phases: PhaseConfig,
    memory: dict[str, list[float]],
    schedules: dict[tuple[str, int], sch.PhaseSchedule],
    context: ScheduleContext | None = None,
) -> bool:
    """Draw an empty stage 2 from stage 1, once.

    Regions like any others: nothing links them to stage 1 afterwards, so an
    edit to either stage stays in that stage.  Refused when there is nothing
    to carry -- an empty stage 2 already says that.
    """
    position = stack.index_of(lora_id)
    if position < 0:
        raise sch.ScheduleError("That LoRA is not active.")
    fill = stage_2_fill(codec.classify(stack.tokens[position], phases.capacity), phases, context)
    if fill is None:
        raise sch.ScheduleError(
            "Fill from stage 1 needs Two Phases on a staged model, a schedule on stage 1 and "
            "nothing drawn on stage 2."
        )
    if fill.empty:
        raise sch.ScheduleError(fill.note)

    target = _resolve(stack, lora_id, 1, phases, memory, schedules, context, create=True)
    schedule = target.schedule
    if schedule.regions:
        raise sch.ScheduleError("Stage 2 already has a schedule - clear it first.")
    drawn = sch.reconstruct_schedule(list(fill.values))
    schedule.slots = len(fill.values)
    schedule.regions = drawn.regions
    schedule.selected_region_id = drawn.selected_region_id
    sch.validate_regions(schedule.regions, schedule.slots)
    return _commit(stack, lora_id, target, phases, memory)


def stack_signature(selected, multipliers: str) -> str:
    """Canonical fingerprint of native state.

    The frontend compares this against what it last sent; when they match it
    knows the incoming payload is the echo of its own write and skips the
    resend that would otherwise loop.
    """
    stack = Stack.from_native(selected, multipliers)
    ids = [normalize_id(value) for value in (selected or [])]
    tokens = codec.align_tokens(codec.parse_multipliers(multipliers).tokens, len(ids))
    return _SIGNATURE_SEP.join(ids) + _SIGNATURE_SEP + codec.serialize(
        tokens, stack.separator_index, stack.disabled
    )
