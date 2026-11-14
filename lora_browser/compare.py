"""Two LoRA recipes side by side: what each one applies, LoRA by LoRA.

A recipe is what WanGP was given for one video -- its ``activated_loras`` and
``loras_multipliers``, and the steps and phases it ran -- as WanGP keeps it
next to every file in its gallery and writes it into the file itself.  The
compare view lines two of them up by the LoRA the user knows, whatever built
copy, group or bake carried it, and colours which one applies more, phase by
phase and step by step.

This module turns settings into those entries, and an edited recipe back into
native selections and multiplier text.  Pure: no WanGP, no Gradio.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from . import derived as dv
from . import multiplier_codec as codec
from . import stack_modes as sm
from . import ui_payloads as up
from .utils import display_name, format_number, is_finite_number, normalize_id

#: Strength WanGP accepts, either way (``ui_payloads.VALUE_MIN``/``MAX``).
VALUE_LIMIT = 10.0
#: More values than any run has steps: what a request may carry per phase.
MAX_VALUES = 1000
#: Settings read from a dropped file, as JSON text: generous for a recipe,
#: small next to anything that is not one.
MAX_SETTINGS_BYTES = 4 * 1024 * 1024


def local_id(reference: Any) -> str:
    """The LoRA a saved reference names, as the panel knows it.

    WanGP saves a LoRA it downloaded as its URL, with the folder it went to
    after a ``|``; everything else as its path in the LoRA folder.  The same
    file, however a recipe spelled it, must line up with itself.
    """
    text = normalize_id(reference)
    if text.startswith(("http:", "https:")):
        parts = text.split("|")
        name = parts[0].rstrip("/").rsplit("/", 1)[-1]
        folder = parts[1].strip("/ ") if len(parts) > 1 else ""
        return normalize_id(f"{folder}/{name}" if folder else name)
    return text


def _names(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if isinstance(item, str) and item.strip()]
    return []


def _multiplier_text(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return " ".join(str(item) for item in value)
    return "" if value is None else str(value)


# ------------------------------------------------------------------ phases


def _counts(total: int, parts: int) -> list[int]:
    base, extra = divmod(max(total, parts), parts)
    return [base + (1 if index < extra else 0) for index in range(parts)]


def _stretch(values: list[float], count: int) -> list[float]:
    """``values`` over ``count`` steps, the way WanGP spreads a list."""
    return [values[min(len(values) - 1, index * len(values) // count)] for index in range(count)]


def phase_lists(token: str, phases: up.PhaseConfig, steps=()) -> list[list[float]] | None:
    """What ``token`` applies in each phase this recipe ran: one value, or one
    per step.  ``None`` for syntax the panel keeps as it is (branches).

    A comma list with no ``;`` spans the whole run; with more than one phase
    running, each phase shows its own stretch of it -- by the steps each runs
    where that is known, evenly otherwise.
    """
    effective = max(1, min(int(phases.effective or 1), int(phases.capacity or 1)))
    info = codec.classify(token, max(1, int(phases.capacity or 1)))
    if info.kind == codec.ADVANCED:
        return None
    if info.kind == codec.SIMPLE:
        return [[float(value)] for value in (info.values or [1.0])[:effective]]
    schedule = info.schedule
    if not schedule.shared:
        lists = [[float(value) for value in values] for values in schedule.phase_values]
        while len(lists) < effective:
            lists.append(list(lists[-1]))
        return lists[:effective]
    values = [float(value) for value in schedule.phase_values[0]]
    if effective == 1:
        return [values]
    counts = [int(count) for count in steps][:effective]
    if len(counts) != effective or not all(count > 0 for count in counts):
        counts = _counts(len(values), effective)
    spread = _stretch(values, sum(counts))
    lists, start = [], 0
    for count in counts:
        lists.append(spread[start:start + count])
        start += count
    return lists


def _held(info: codec.TokenInfo, capacity: int) -> list[list[float]]:
    """Every phase a token holds, the ones this mode does not run included."""
    if info.kind == codec.SIMPLE:
        numbers = list(info.values or [1.0]) + list(info.overflow or [])
        while len(numbers) < capacity:
            numbers.append(numbers[-1])
        return [[float(number)] for number in numbers[:capacity]]
    if info.kind == codec.SCHEDULED and info.schedule is not None:
        lists = [[float(value) for value in values] for values in info.schedule.phase_values]
        while len(lists) < capacity:
            lists.append(list(lists[-1]))
        return lists[:capacity]
    return [[1.0] for _ in range(capacity)]


def _tidy(parts: list[list[float]]) -> str:
    """Phases as a native token, as short as it can be said."""
    if len(parts) == 1:
        return ",".join(format_number(value) for value in parts[0])
    if all(len(part) == 1 for part in parts):
        numbers = [part[0] for part in parts]
        if len(set(numbers)) == 1:
            # WanGP gives every phase a lone value.
            return format_number(numbers[0])
        return ";".join(format_number(number) for number in numbers)
    return codec.build_schedule(parts)


def clean_lists(value: Any) -> list[list[float]] | None:
    """Phase values from the browser, or ``None`` when they are not numbers
    WanGP takes."""
    if not isinstance(value, list) or not value:
        return None
    lists = []
    for part in value:
        if not isinstance(part, list) or not part or len(part) > MAX_VALUES:
            return None
        numbers = []
        for number in part:
            if isinstance(number, bool) or not is_finite_number(number):
                return None
            number = float(number)
            if abs(number) > VALUE_LIMIT:
                return None
            numbers.append(round(number, 4))
        lists.append(numbers)
    return lists


def token_for(original: str, lists: list[list[float]], phases: up.PhaseConfig) -> str:
    """``original`` with the phases this recipe runs set to ``lists``; a phase
    the mode does not run keeps what it had."""
    capacity = max(1, int(phases.capacity or 1))
    effective = max(1, min(int(phases.effective or 1), capacity))
    edited = [list(values) for values in (lists or [])[:effective] if values]
    if not edited:
        return original
    info = codec.classify(original, capacity)
    if info.kind == codec.SCHEDULED and info.schedule is not None and info.schedule.shared and effective == 1:
        # A list for the whole run stays one.
        return _tidy(edited[:1])
    held = _held(info, capacity)
    return _tidy([edited[index] if index < len(edited) else held[index] for index in range(capacity)])


# ------------------------------------------------------------------ entries


def _mode_label(stack: up.Stack, lora_id: str, source: str, mode: str) -> str:
    if mode == sm.VISUAL:
        sound_half = dv.split_id(source, sm.SOUND_HALF)
        return "Picture" if stack.has_entry(sound_half) else "Visual only"
    if mode == sm.SOUND_HALF:
        return "Sound"
    if mode == sm.AUDIO:
        return "Audio only"
    return ""


def entries(stack: up.Stack, phases: up.PhaseConfig, steps=(), known_ids=()) -> list[dict]:
    """One entry per LoRA and strength, in the stack's order: a LoRA with
    separate picture and sound strengths is two, a baked group one per
    member.  Keyed by the LoRA file the user knows."""
    groups = sm.groups(stack)
    found: dict[str, dict] = {}
    listed: list[dict] = []

    def add(entry: dict) -> None:
        key, count = entry["key"], 2
        while key in found:
            key = f"{entry['key']}#{count}"
            count += 1
        entry["key"] = key
        found[key] = entry
        listed.append(entry)

    system = [
        (lora_id, True, token)
        for position, (lora_id, token) in enumerate(zip(stack.ids, stack.tokens))
        if stack.is_system(position)
    ]
    system_ids = {entry[0] for entry in system}
    for lora_id, on, token in system + stack.user_entries():
        lists = phase_lists(token, phases, steps)
        group = next((each for each in groups.values() if each.holds(lora_id)), None)
        parsed = dv.parse(lora_id)
        baked = parsed is not None and parsed.kind == "bake"
        if baked and group is not None:
            covers = group.track_members(group.track_of(lora_id)) or group.members
        else:
            covers = [lora_id]
        for member in covers:
            source, mode = sm.source_of(member, known_ids)
            sound = sm.is_sound_entry(member) or (baked and sm.is_sound_entry(lora_id))
            add({
                "key": source + ("|sound" if sound else ""),
                "source": source,
                "name": display_name(source),
                "id": lora_id,
                "add_id": member if not baked else source,
                "mode": _mode_label(stack, member, source, mode),
                "group": group.letter if group is not None else "",
                "baked": baked,
                # Group members move as one: an edit to one is an edit to all.
                "link": f"{group.letter}|{group.track_of(member)}" if group is not None else "",
                "on": bool(on),
                "system": lora_id in system_ids,
                "raw": token,
                "phases": lists,
            })
    return listed


def recipe(
    selected, multipliers, model_def: dict | None, guidance, steps=None, sampler="", steps2=None,
    known_ids=(),
) -> dict:
    """A recipe as the compare view reads it."""
    ids = [local_id(value) for value in _names(selected)]
    text = _multiplier_text(multipliers)
    phases = up.resolve_phases(model_def or {}, guidance)
    context = up.ScheduleContext.from_native(steps, model_def or {}, phases, sampler, steps2)
    effective = max(1, min(phases.effective, phases.capacity))
    if context.phase_steps:
        counts = [int(count) for count in context.phase_steps[:effective]]
    else:
        counts = [int(context.steps)] if effective == 1 and context.steps else []
    stack = up.Stack.from_native(ids, text)
    return {
        "phases": effective,
        "capacity": phases.capacity,
        "staged": phases.staged,
        "steps": counts,
        "labels": [f"P{index + 1}" for index in range(effective)] if effective > 1 else [""],
        "entries": entries(stack, phases, counts, known_ids),
        "selected": ids,
        "multipliers": text,
        "guidance": guidance,
        "total_steps": int(context.steps or 0),
    }


def from_settings(settings: dict | None, model_def: dict | None, known_ids=()) -> dict | None:
    """The recipe a video was made with, from the settings WanGP saved with it."""
    if not isinstance(settings, dict):
        return None
    loras = settings.get("activated_loras")
    found = recipe(
        loras, settings.get("loras_multipliers", ""), model_def,
        settings.get("guidance_phases", 1), settings.get("num_inference_steps"),
        settings.get("sample_solver", ""), settings.get("num_inference_steps2"), known_ids,
    )
    found["model_type"] = str(settings.get("model_type") or settings.get("base_model_type") or "")
    found["listed"] = isinstance(loras, (list, tuple, str))
    return found


# ------------------------------------------------------------------- apply


def _set_token(stack: up.Stack, lora_id: str, token: str) -> None:
    if stack.set_token(lora_id, token):
        return
    for index, entry in enumerate(stack.disabled):
        if entry.id == normalize_id(lora_id):
            stack.disabled[index] = replace(entry, token=token)


def apply_edits(
    selected, multipliers: str, phases: up.PhaseConfig, changes: Any, added: Any,
    steps=(), known_ids=(),
) -> up.Stack:
    """The recipe ``selected``/``multipliers`` with the compare view's edits.

    ``changes`` maps an entry's key to its new ``phases`` (values per phase)
    and ``on``; ``added`` lists LoRAs to add, each ``{"id", "phases"}``.  What
    an entry cannot take -- WanGP's own LoRAs, a token in syntax the panel
    keeps as it is, values out of range -- is left as it was.
    """
    stack = up.Stack.from_native(list(selected or []), multipliers or "")
    current = {entry["key"]: entry for entry in entries(stack, phases, steps, known_ids)}
    written: dict[str, str] = {}
    switched: dict[str, bool] = {}
    for key, change in (changes.items() if isinstance(changes, dict) else []):
        entry = current.get(str(key))
        if entry is None or entry["system"] or not isinstance(change, dict):
            continue
        lists = clean_lists(change.get("phases"))
        if lists is not None and entry["phases"] is not None:
            written[entry["id"]] = token_for(entry["raw"], lists, phases)
        if isinstance(change.get("on"), bool):
            switched[entry["id"]] = change["on"]
    for lora_id, token in written.items():
        _set_token(stack, lora_id, token)
    for lora_id, on in switched.items():
        stack.set_enabled(lora_id, on)
    for item in (added if isinstance(added, list) else []):
        if not isinstance(item, dict):
            continue
        lora_id = local_id(item.get("id"))
        lists = clean_lists(item.get("phases"))
        if not lora_id or lists is None or stack.has_entry(lora_id):
            continue
        stack.add(lora_id, phases.capacity)
        _set_token(stack, lora_id, token_for(codec.default_token(phases.capacity), lists, phases))
    return stack
