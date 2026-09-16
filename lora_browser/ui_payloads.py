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
#: ``PHASE_RELATIVE`` - slots inside one phase, whose global step numbers depend
#:                      on runtime phase boundaries the plugin cannot see.
#: ``NORMALIZED_READONLY`` - preserved, but not offered for dragging.
COORD_GLOBAL_EXACT = "global_exact"
COORD_PHASE_RELATIVE = "phase_relative"
COORD_NORMALIZED_READONLY = "normalized_readonly"

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

    @property
    def hidden(self) -> int:
        return max(0, self.capacity - self.effective)


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

    return PhaseConfig(capacity=capacity, effective=max(1, min(capacity, guidance_count or 1)))


@dataclass(frozen=True)
class ScheduleContext:
    """What the editor knows about schedule time coordinates.

    ``steps`` is WanGP's ``num_inference_steps`` when the plugin could obtain
    the component, else 0.  ``boundaries_known`` stays False in v2: a
    phase-specific schedule is expanded inside a runtime phase interval that
    depends on model switching and guidance, and labelling slots with global
    step numbers the plugin cannot verify would be a lie.
    """

    steps: int = 0
    boundaries_known: bool = False

    @classmethod
    def from_native(cls, steps_value: Any) -> "ScheduleContext":
        try:
            steps = int(float(steps_value))
        except (TypeError, ValueError):
            steps = 0
        return cls(steps=max(0, steps))

    def target_slots(self) -> int:
        """One slot per inference step, or 0 when the step count is unknown.

        Clamped, so a step count beyond what a timeline can hold settles at the
        ceiling instead of never matching and asking to be refitted forever.
        """
        return sch.clamp_slots(self.steps) if self.steps else 0

    def default_slots(self, shared: bool = False) -> int:
        """Timeline length for a freshly opened schedule."""
        return self.target_slots() or sch.DEFAULT_SLOTS


def scheduling_allowed(phases: PhaseConfig) -> bool:
    """Whether a step schedule can say what it appears to say.

    WanGP's ``expand_slist`` stretches each phase's comma list to fill that
    phase's interval, so a list is step-exact exactly when its length equals the
    number of steps in the interval it covers.

    In One Phase mode the switch points sit at the end of the run, so phase 1
    covers every step: a list of ``num_inference_steps`` values runs one value
    per step, exactly as drawn.

    With two or more guidance phases, phase 1 covers only up to
    ``model_switch_step`` -- derived at generation time from the sampler's
    timesteps and the switch threshold, and not knowable while editing.  A
    four-slot schedule drawn against a four-step run would be squeezed into
    however many steps phase 1 turns out to be, so the timeline would be showing
    something WanGP is not going to do.  Rather than draw that, the editor does
    not offer scheduling at all there.
    """
    return phases.effective <= 1


SCHEDULING_DISABLED_REASON = (
    "Step schedules need One Phase guidance: with more phases WanGP squeezes "
    "each phase's values into that phase, so a timeline could not show what "
    "actually runs."
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
    """The selected LoRAs and their multiplier tokens, in native order."""

    ids: list[str] = field(default_factory=list)
    tokens: list[str] = field(default_factory=list)
    separator_index: int = -1

    @classmethod
    def from_native(cls, selected, multipliers: str) -> "Stack":
        ids = [normalize_id(value) for value in (selected or []) if normalize_id(value)]
        parsed = codec.parse_multipliers(multipliers)
        return cls(
            ids=ids,
            tokens=codec.align_tokens(parsed.tokens, len(ids)),
            separator_index=min(parsed.separator_index, len(ids)) if parsed.has_separator else -1,
        )

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
        previously occupied its index.
        """
        lora_id = normalize_id(lora_id)
        if not lora_id or self.index_of(lora_id) >= 0:
            return False
        self.ids.append(lora_id)
        self.tokens.append(codec.default_token(capacity))
        return True

    def remove(self, lora_id: str) -> bool:
        position = self.index_of(lora_id)
        if position < 0:
            return False
        self.ids.pop(position)
        self.tokens.pop(position)
        self.separator_index = codec.separator_after_removal(self.separator_index, [position])
        return True

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
        return values, codec.serialize(self.tokens, self.separator_index)


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
    for entry in inventory.entries:
        position = stack.index_of(entry.id)
        index = catalogue.get(entry.id)
        item: dict[str, Any] = {
            "id": entry.id,
            "name": entry.name,
            "label": entry.label,
            "parent": entry.parent,
            "active": position >= 0,
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
    """Editable rows, in native order, including LoRAs missing from disk."""
    memory = memory or {}
    catalogue = catalogue or {}
    schedules = schedules if schedules is not None else {}
    rows = []
    for position, lora_id in enumerate(stack.ids):
        entry = inventory.get(lora_id)
        index = catalogue.get(lora_id)
        fallback_name = lora_id.rsplit("/", 1)[-1]
        row: dict[str, Any] = {
            "id": lora_id,
            "name": entry.name if entry else fallback_name,
            "label": entry.label if entry else fallback_name,
            "civitai_name": getattr(index, "civitai_name", "") if index else "",
            "has_catalogue": bool(getattr(index, "has_sidecar", False)) if index else False,
            "system_managed": stack.is_system(position),
            "missing": entry is None,
        }
        row.update(
            multiplier_fields(
                stack.tokens[position], phases, lora_id, memory, schedules, context
            )
        )
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
        }

    if info.kind == codec.SCHEDULED and scheduling_allowed(phases):
        return _scheduled_fields(info, phases, lora_id, schedules or {}, context, with_schedules)

    if info.kind == codec.SCHEDULED:
        # More than one guidance phase: this token is on its way to being reset,
        # and until it is there is nothing here a strength control can mean.
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
            schedule_payload(stored, shared=shared, phases=phases, context=context)
            if stored is not None else None
        )
    return result


def schedule_payload(
    schedule: sch.PhaseSchedule,
    *,
    shared: bool,
    phases: PhaseConfig,
    context: ScheduleContext | None,
) -> dict[str, Any]:
    """One phase's schedule, as the frontend consumes it.

    Region ids travel with it; they are an editor concept and never appear in
    the native token.
    """
    return {
        "base": round(float(schedule.base), VALUE_DECIMALS),
        "slots": sch.held_slots(schedule.slots),
        "coordinate_mode": coordinate_mode(schedule, shared=shared, phases=phases, context=context),
        "steps": int(context.steps) if context else 0,
        "regions": [
            {
                "id": region.id,
                "start": int(region.start),
                "end": int(region.end),
                "strength": round(float(region.strength), VALUE_DECIMALS),
            }
            for region in sch.sort_regions(schedule.regions)
        ],
        "selected_region_id": schedule.selected_region_id,
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
) -> str:
    """Decide what slot numbers may claim to be.

    Exact global steps are only claimed for a schedule that really does span the
    whole run at one slot per step.  Everything else says so: guessing phase
    boundaries from equal percentages would mislabel every model whose switch
    point is not exactly halfway.
    """
    if not schedule.editable:
        return COORD_NORMALIZED_READONLY
    steps = int(context.steps) if context else 0
    if shared and steps and sch.held_slots(schedule.slots) == steps:
        return COORD_GLOBAL_EXACT
    if context and context.boundaries_known and steps:
        return COORD_GLOBAL_EXACT
    return COORD_PHASE_RELATIVE


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
    """
    if info.kind == codec.SCHEDULED and info.schedule is not None:
        return info.schedule.shared
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

    Kept when the stored regions still compile to exactly the values in the
    token -- that is what makes region ids stable across a round trip -- and
    rebuilt from the token otherwise.  Nothing here writes to the token: an
    imported schedule is rendered, never re-serialised.
    """
    live: set[tuple[str, int]] = set()
    if not scheduling_allowed(phases):
        # Nothing here could be edited truthfully; the tokens themselves are
        # reset by refit_schedules, which is the only thing allowed to write.
        schedules.clear()
        return

    for position, lora_id in enumerate(stack.ids):
        info = codec.classify(stack.tokens[position], phases.capacity)
        if info.kind == codec.ADVANCED:
            continue

        shared = token_is_shared(info, phases, lora_id, schedules)
        for phase in range(1 if shared else phases.capacity):
            key = schedule_key(lora_id, phase, shared)
            values = _phase_values(info, phases, lora_id, phase)
            stored = schedules.get(key)

            if stored is not None and _tracks(stored, values):
                live.add(key)
                if not stored.regions:
                    # Nothing has been drawn yet, so the timeline is free to
                    # follow the step counter the user is currently looking at.
                    stored.slots = (context or ScheduleContext()).default_slots()
                continue

            if len(values) > 1:
                # The token carries a schedule this editor state does not match:
                # WanGP is the truth, so derive the timeline from the token.
                live.add(key)
                schedules[key] = sch.reconstruct_schedule(values, raw=info.raw)

    for key in [key for key in schedules if key not in live]:
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
    """True when some timeline is not the length the step counter says."""
    if not scheduling_allowed(phases):
        # Any schedule still in a token has to go; see refit_schedules.
        return any(
            codec.classify(token, phases.capacity).kind == codec.SCHEDULED
            for token in stack.tokens
        )

    target = (context or ScheduleContext()).target_slots()
    if not target:
        return False
    return any(
        sch.clamp_slots(schedule.slots) != target
        for _, _, _, schedule in _drawn_schedules(stack, phases, schedules)
    )


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
      so it is **truncated or extended** at the end.  Steps that still exist keep
      exactly what they had and a new step arrives undefined.
    * a schedule that arrived from a preset, an .lset file or a hand edit was
      authored at its own resolution, and WanGP spreads it across the whole run.
      Truncating that would change what it renders, so it is **resampled** into
      step alignment once, and is step-aligned from then on.

    When the guidance mode has more than one phase, no schedule can be shown
    truthfully at all, so instead of refitting them this clears them -- see
    ``_clear_schedules_for_phase_mode``.

    ``dirty`` is what tells them apart: it is set the moment the panel edits a
    schedule, and clear for one that has only ever been read.
    """
    if not scheduling_allowed(phases):
        return _clear_schedules_for_phase_mode(stack, phases, schedules, memory)

    target = (context or ScheduleContext()).target_slots()
    if not target:
        return False

    changed = False
    for lora_id, phase, key, schedule in list(_drawn_schedules(stack, phases, schedules)):
        if sch.clamp_slots(schedule.slots) == target:
            continue
        schedules[key] = (
            sch.refit_schedule(schedule, target) if schedule.dirty
            else sch.normalize_schedule(schedule, target)
        )
        target_state = _resolve(stack, lora_id, phase, phases, memory, schedules, context, create=False)
        changed = _commit(stack, lora_id, target_state, phases, memory) or changed
    return changed


def _clear_schedules_for_phase_mode(
    stack: "Stack",
    phases: PhaseConfig,
    schedules: dict[tuple[str, int], sch.PhaseSchedule],
    memory: dict[str, list[float]],
) -> bool:
    """Drop every schedule when the guidance mode stops supporting them.

    A scheduled LoRA goes to 0 rather than to some strength picked on its
    behalf: the schedule said the multiplier varies over the run, and no single
    number carries that over, so the honest move is to switch the LoRA off and
    let the user set what they want.  Unscheduled LoRAs keep their strengths --
    those mean the same thing in any phase mode.
    """
    schedules.clear()
    changed = False
    for position, token in enumerate(stack.tokens):
        info = codec.classify(token, phases.capacity)
        if info.kind != codec.SCHEDULED:
            continue
        lora_id = stack.ids[position]
        zeroed = codec.build_token([0.0] * phases.capacity, phases.capacity)
        if stack.tokens[position] != zeroed:
            stack.tokens[position] = zeroed
            memory[normalize_id(lora_id)] = [0.0] * phases.capacity
            changed = True
    return changed


def _phase_values(info: codec.TokenInfo, phases: PhaseConfig, lora_id: str, phase: int) -> list[float]:
    """The native value list driving one phase, without inventing anything."""
    if info.kind == codec.SCHEDULED and info.schedule is not None:
        return info.schedule.values_for(phase)
    if info.kind == codec.SIMPLE and info.values:
        index = min(max(0, int(phase)), len(info.values) - 1)
        return [float(info.values[index])]
    return []


def _tracks(schedule: sch.PhaseSchedule, values: list[float]) -> bool:
    """True when ``schedule`` still says exactly what the token says.

    Compared against what the schedule would *emit*, so a timeline that is open
    but not yet doing anything keeps its slot count and regions instead of being
    rebuilt from the scalar it currently compiles to.
    """
    emitted = sch.native_values(schedule)
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
            slots=(context or ScheduleContext()).default_slots(shared),
            regions=[],
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
        strength=sanitize_value(schedule.base, 1.0),
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
    """
    target = _resolve(stack, lora_id, phase, phases, memory, schedules, context, create=False)
    schedule = target.schedule
    region = schedule.region(region_id)
    if region is None:
        raise sch.ScheduleError("That region no longer exists.")

    first, last = sch.clamp_region(
        start, end, schedule.slots, keep_width=keep_width, width=region.width
    )
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


def stack_signature(selected, multipliers: str) -> str:
    """Canonical fingerprint of native state.

    The frontend compares this against what it last sent; when they match it
    knows the incoming payload is the echo of its own write and skips the
    resend that would otherwise loop.
    """
    ids = [normalize_id(value) for value in (selected or [])]
    parsed = codec.parse_multipliers(multipliers)
    tokens = codec.align_tokens(parsed.tokens, len(ids))
    return _SIGNATURE_SEP.join(ids) + _SIGNATURE_SEP + codec.serialize(tokens, parsed.separator_index)
