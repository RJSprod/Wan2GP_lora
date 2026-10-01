"""Region editor model for one phase's step schedule.

WanGP expresses a time-varying multiplier as a comma separated list of values
inside one phase, e.g. ``1,0.8,0.4,0``.  That list is the native truth and
``multiplier_codec`` owns its grammar.  This module owns the *editor* view of
one such list: a set of non-overlapping regions, each holding a strength, over a
timeline whose uncovered slots are **zero**.

That last rule is the whole model.  A slot no region covers is a step the LoRA
is not applied on; it does not fall back to some resting strength hiding behind
the timeline.  Drawing regions is therefore the only way a scheduled phase says
anything, which is why a scheduled phase has no plain strength control at all.

``base`` is not a fill.  It is the value the phase is worth while nothing has
been drawn yet -- so opening a timeline changes nothing -- and the strength a
newly drawn region starts at.  The moment one region exists, every other slot
is zero.

Everything here is pure.  The timeline is where a mis-drag would silently
change what WanGP renders, so collision resolution and placement live in
testable functions rather than inside pointer event handlers.

Coordinates are 1-based inclusive slot numbers: a region ``start=2, end=5``
covers the 2nd through 5th value of the phase's list.  Slot numbers are not
necessarily inference steps -- see ``ui_payloads.coordinate_mode``.

A schedule drawn one slot per step is not resized by the step count; the step
count is a *window* onto it.  ``slots`` is how much of the schedule the run
reaches, and a region may lie wholly or partly past it: lowering the step count
hides what is past the new end rather than deleting it, and raising it again
brings that back exactly as it was.  So no sequence of step-count changes can
move or lose anything drawn -- see ``refit_schedule``.  Only what is inside the
window is ever compiled into the token or shown on the timeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from .utils import is_finite_number

#: A schedule always has at least one slot (that is just a scalar), and the
#: upper bound keeps an imported 500-value list from being offered for
#: slot-by-slot editing on a phone.  It bounds what may be *edited*, not what
#: may be held: a longer imported list is kept at its own length so that
#: preserving it, and re-gridding it later, stay lossless.
MIN_SLOTS = 1
MAX_SLOTS = 120

#: Ceiling on any schedule length at all, so a nonsense slot count cannot ask
#: for a list nothing would ever run.
HARD_MAX_SLOTS = 4096

#: Used when no better resolution is available: no ``num_inference_steps`` and
#: no phase boundaries to derive a length from.
DEFAULT_SLOTS = 20

#: The shortest window a drawn schedule is ever cut to.  One value is a scalar:
#: written to the token it would be indistinguishable from a plain strength
#: someone set, so the step count after it could not tell a schedule had been
#: cut short and bring the rest back.
MIN_WINDOW = 2

#: Versions of one phase's schedule remembered so that an event arriving with
#: an older token is recognised rather than rebuilt from; see ``remember``.
HISTORY_DEPTH = 16

#: "+ Region" aims for roughly a fifth of the timeline (spec: 20-25%).
REGION_WIDTH_FRACTION = 0.22

#: What a slot no region covers is worth. Not configurable: it is the model.
GAP_STRENGTH = 0.0

_ID_PREFIX = "r"


class ScheduleError(ValueError):
    """Raised when a region set could not be a valid WanGP schedule."""


@dataclass
class Region:
    """One override span. ``start``/``end`` are inclusive slot numbers."""

    id: str
    start: int
    end: int
    strength: float

    @property
    def width(self) -> int:
        return self.end - self.start + 1


@dataclass
class PhaseSchedule:
    """The editable state of one phase's schedule.

    ``source_values``/``source_raw`` exist so that an imported schedule can be
    rendered without being rewritten: until ``dirty`` is set, the plugin still
    knows the exact list WanGP was given and emits it byte-for-byte.
    """

    #: The strength this phase is worth while no region has been drawn, and the
    #: strength a new region starts at. Never painted into the gaps between
    #: regions -- those are ``GAP_STRENGTH``.
    base: float = 1.0
    #: How much of the schedule the run reaches: the length of the value list
    #: in the token, and of the timeline on screen.
    slots: int = DEFAULT_SLOTS
    #: May reach past ``slots``. What lies past the end of the run is kept, so
    #: that raising the step count again brings it back; see ``refit_schedule``.
    regions: list[Region] = field(default_factory=list)
    selected_region_id: str | None = None
    #: The values this schedule was imported from; ``None`` when it was created
    #: in the editor from a scalar.
    source_values: list[float] | None = None
    #: The native token the values came from, kept only for reference.
    source_raw: str | None = None
    #: ``False`` until a material edit; a clean schedule is never serialised.
    dirty: bool = False
    #: ``True`` when slot *i* is step *i* -- which is what a schedule drawn in
    #: the panel always is, and what an imported list is only by coincidence.
    #:
    #: It decides what a change to the step count means (see
    #: ``refit_schedule`` and ``normalize_schedule``), and it is separate from
    #: ``dirty`` because the two answer different questions. ``dirty`` asks
    #: "has this been edited since it was read", which is about serialising;
    #: it is false again after a reload, when the token is all that survived.
    #: Deciding alignment from it meant a schedule the user drew was treated
    #: as an import the first time they touched the step counter after a
    #: restart -- so 4 -> 5 stretched the regions and 5 -> 6 did not, with
    #: nothing on screen to say why.
    step_aligned: bool = False
    #: ``False`` when the list is too long to offer as a timeline.
    editable: bool = True
    #: ``True`` when editing requires an explicit resolution change first.
    normalization_required: bool = False
    #: Why the schedule is read-only, shown verbatim in the panel.
    reason: str = ""
    #: Recent versions of this phase's schedule as the panel wrote them, oldest
    #: first.  Every object that stands for the same phase shares the one list
    #: (``copy`` and ``replace`` pass it along), so whichever of them an edit
    #: lands on, the version is recorded where the next sync will look.  Memory
    #: only: it lasts as long as the process does, which is as long as anything
    #: past the end of the run needs remembering.
    history: list["PhaseSchedule"] = field(default_factory=list, repr=False, compare=False)

    def region(self, region_id: str) -> Region | None:
        for region in self.regions:
            if region.id == str(region_id):
                return region
        return None

    def copy(self) -> "PhaseSchedule":
        return replace(
            self,
            regions=[replace(region) for region in self.regions],
            source_values=list(self.source_values) if self.source_values is not None else None,
        )

    def snapshot(self) -> "PhaseSchedule":
        """A copy detached from ``history``, fit to be stored in it."""
        detached = self.copy()
        detached.history = []
        return detached


def clamp_slots(slots) -> int:
    """A slot count the timeline may be edited at."""
    try:
        number = int(slots)
    except (TypeError, ValueError):
        return DEFAULT_SLOTS
    return max(MIN_SLOTS, min(MAX_SLOTS, number))


def held_slots(slots) -> int:
    """A slot count a schedule may be *held* at, editable or not."""
    try:
        number = int(slots)
    except (TypeError, ValueError):
        return DEFAULT_SLOTS
    return max(MIN_SLOTS, min(HARD_MAX_SLOTS, number))


def preferred_width(slots: int) -> int:
    """Default width of a freshly added region: ~22% of the timeline."""
    return max(1, min(int(slots), int(round(int(slots) * REGION_WIDTH_FRACTION)) or 1))


def next_region_id(regions: list[Region]) -> str:
    """A fresh id that no existing region uses.

    Ids are stable for the life of a region -- the frontend addresses regions
    by id -- so they are never renumbered when a neighbour is deleted.
    """
    highest = 0
    for region in regions:
        text = str(region.id)
        if text.startswith(_ID_PREFIX) and text[len(_ID_PREFIX):].isdigit():
            highest = max(highest, int(text[len(_ID_PREFIX):]))
    return f"{_ID_PREFIX}{highest + 1}"


def sort_regions(regions: list[Region]) -> list[Region]:
    return sorted(regions, key=lambda region: (region.start, region.end))


def overlaps(left: Region, right: Region) -> bool:
    """Touching is not overlapping: 2-5 and 6-9 are adjacent, not in conflict."""
    return left.start <= right.end and right.start <= left.end


# ------------------------------------------------------------------ compile


def compile_schedule(schedule: PhaseSchedule) -> list[float]:
    """The value list this schedule means, one entry per slot.

    Slots no region covers are zero.  A timeline with nothing drawn on it is the
    exception: it is not a schedule of zeros, it is a phase that has not been
    scheduled, so it is worth its base everywhere.

    Regions are applied in slot order, so a caller that hands in overlapping
    regions still gets a deterministic result -- but committed region sets are
    validated to be disjoint, so that case only arises mid-gesture.
    """
    return compile_window(schedule, schedule.slots)


def compile_window(schedule: PhaseSchedule, slots) -> list[float]:
    """The value list this schedule means over its first ``slots`` steps.

    ``compile_schedule`` is this at the schedule's own window.  Any other
    window is the same schedule at another step count: a region past the end
    is simply not reached, and one crossing it contributes the part inside.
    """
    slots = held_slots(slots)
    if not schedule.regions:
        return [float(schedule.base)] * slots

    values = [GAP_STRENGTH] * slots
    for region in sort_regions(schedule.regions):
        start = max(1, int(region.start))
        end = min(slots, int(region.end))
        for slot in range(start, end + 1):
            values[slot - 1] = float(region.strength)
    return values


def visible_regions(schedule: PhaseSchedule) -> list[Region]:
    """The regions inside the window, cut at its end: what the timeline shows.

    A region the run does not reach is left out, and one crossing the end shows
    only the part the run reaches.  The cut is a view -- the regions themselves
    are untouched, which is what lets a longer run bring the rest back.
    """
    slots = held_slots(schedule.slots)
    shown = []
    for region in sort_regions(schedule.regions):
        if int(region.start) > slots:
            continue
        shown.append(replace(region, end=min(int(region.end), slots)))
    return shown


def last_drawn_step(schedule: PhaseSchedule) -> int:
    """The last step any region covers, wherever the window ends; 0 if none."""
    return max((int(region.end) for region in schedule.regions), default=0)


def native_values(schedule: PhaseSchedule) -> list[float]:
    """What to write into the native token for this phase.

    A timeline with nothing on it is emitted as the single scalar it is worth,
    which is what lets opening the scheduler leave the native token alone.  Once
    anything is drawn, the full list goes out: the zeros between regions are
    part of what the user drew.
    """
    if not schedule.regions:
        return [float(schedule.base)]
    return compile_schedule(schedule)


def is_materialized(schedule: PhaseSchedule) -> bool:
    """True when something has actually been scheduled."""
    return bool(schedule.regions)


# -------------------------------------------------------------- reconstruct


def reconstruct_schedule(
    values: list[float],
    *,
    raw: str | None = None,
    slots: int | None = None,
) -> PhaseSchedule:
    """Turn an imported value list back into regions over zero.

    Every contiguous run of non-zero values becomes one region and the zeros
    between them stay gaps, which makes the decomposition deterministic and
    exactly reversible: ``compile_schedule(reconstruct_schedule(v)) == v``.
    """
    numbers = [float(value) for value in values] or [1.0]
    total = len(numbers) if slots is None else clamp_slots(slots)

    regions: list[Region] = []
    index = 0
    while index < len(numbers):
        value = numbers[index]
        if _same(value, GAP_STRENGTH):
            index += 1
            continue
        run_end = index
        while run_end + 1 < len(numbers) and _same(numbers[run_end + 1], value):
            run_end += 1
        regions.append(
            Region(id=f"{_ID_PREFIX}{len(regions) + 1}", start=index + 1, end=run_end + 1, strength=value)
        )
        index = run_end + 1

    # The base only seeds new regions and survives "Clear phase"; the first real
    # strength in the list is the best guess at what the phase is worth. An
    # all-zero list is the one case where the timeline is genuinely empty, and
    # zero is then what it is worth.
    base = next((value for value in numbers if not _same(value, GAP_STRENGTH)), GAP_STRENGTH)

    schedule = PhaseSchedule(
        base=base,
        slots=held_slots(len(numbers)) if slots is None else total,
        regions=regions,
        selected_region_id=regions[0].id if regions else None,
        source_values=list(numbers),
        source_raw=raw,
        dirty=False,
    )
    if len(numbers) > MAX_SLOTS:
        # Honest refusal: a 500-slot timeline cannot be dragged usefully, and
        # regridding it on load would change what WanGP renders.
        schedule.editable = False
        schedule.normalization_required = True
        schedule.reason = (
            f"This schedule has {len(numbers)} values; normalise it to at most "
            f"{MAX_SLOTS} slots to edit it visually."
        )
    return schedule


def _same(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-9)


# ---------------------------------------------------------------- placement


def free_gaps(regions: list[Region], slots: int) -> list[tuple[int, int]]:
    """Inclusive free intervals, left to right."""
    slots = clamp_slots(slots)
    gaps: list[tuple[int, int]] = []
    cursor = 1
    for region in sort_regions(regions):
        if cursor < region.start:
            gaps.append((cursor, min(region.start - 1, slots)))
        cursor = max(cursor, int(region.end) + 1)
    if cursor <= slots:
        gaps.append((cursor, slots))
    return [gap for gap in gaps if gap[0] <= gap[1]]


def gap_containing(regions: list[Region], slots: int, slot: int) -> tuple[int, int] | None:
    """The free interval ``slot`` falls in, or ``None`` if it is occupied."""
    for gap in free_gaps(regions, slots):
        if gap[0] <= int(slot) <= gap[1]:
            return gap
    return None


def fit_region(
    regions: list[Region],
    slots: int,
    start,
    end=None,
    width: int | None = None,
) -> tuple[int, int] | None:
    """Fit a span the user drew into the free run that holds its start.

    Drawing stops at the neighbouring region rather than overrunning it: a
    gesture that begins in empty space should not delete what is next to it,
    which is the difference between drawing a region and dragging one.
    """
    slots = clamp_slots(slots)
    try:
        first = int(round(float(start)))
    except (TypeError, ValueError):
        return None
    first = max(1, min(first, slots))

    gap = gap_containing(regions, slots, first)
    if gap is None:
        return None

    if end is None:
        last = first + max(1, int(width) if width else preferred_width(slots)) - 1
    else:
        try:
            last = int(round(float(end)))
        except (TypeError, ValueError):
            return None
    if last < first:
        first, last = last, first

    first = max(gap[0], min(first, gap[1]))
    last = max(first, min(last, gap[1]))
    return (first, last)


def find_region_slot(regions: list[Region], slots: int, width: int | None = None) -> tuple[int, int] | None:
    """Pick genuine free space for a new region.

    Order: the free tail after the last region, then the first hole that fits,
    then the largest hole.  ``None`` means the timeline is full -- the caller
    refuses rather than creating an overlap and relying on collision
    resolution to sort it out.
    """
    slots = clamp_slots(slots)
    target = max(1, min(slots, int(width) if width else preferred_width(slots)))
    gaps = free_gaps(regions, slots)
    if not gaps:
        return None

    tail = gaps[-1]
    if tail[1] == slots and tail[1] - tail[0] + 1 >= target:
        return (tail[0], tail[0] + target - 1)

    for gap in gaps:
        if gap[1] - gap[0] + 1 >= target:
            return (gap[0], gap[0] + target - 1)

    largest = max(gaps, key=lambda gap: (gap[1] - gap[0], -gap[0]))
    return (largest[0], largest[1])


# --------------------------------------------------------------- collisions


def resolve_collision(regions: list[Region], winner: Region, new_id: str | None = None) -> list[Region]:
    """Settle a dropped region against the committed set.  The winner wins.

    * partial overlap -> the loser shrinks away from the winner,
    * winner strictly inside a loser -> the loser splits in two,
    * winner covers a loser -> the loser is deleted.

    The left fragment of a split keeps the loser's id so its identity follows
    the part that did not move; the right fragment gets a fresh one.
    """
    kept: list[Region] = []
    used_ids = {str(region.id) for region in regions} | {str(winner.id)}
    fresh = str(new_id) if new_id else None

    def mint() -> str:
        nonlocal fresh
        if fresh and fresh not in used_ids:
            candidate = fresh
        else:
            candidate = next_region_id([Region(id=value, start=1, end=1, strength=0.0) for value in used_ids])
        fresh = None
        used_ids.add(candidate)
        return candidate

    for original in regions:
        if str(original.id) == str(winner.id):
            continue
        loser = replace(original)
        if not overlaps(winner, loser):
            kept.append(loser)
            continue

        if winner.start <= loser.start and winner.end >= loser.end:
            # Fully covered, including the equal-bounds case: the loser is gone.
            continue

        if winner.start > loser.start and winner.end < loser.end:
            kept.append(replace(loser, end=winner.start - 1))
            kept.append(
                Region(id=mint(), start=winner.end + 1, end=loser.end, strength=loser.strength)
            )
            continue

        if winner.start <= loser.start:
            trimmed = replace(loser, start=winner.end + 1)
        else:
            trimmed = replace(loser, end=winner.start - 1)
        if trimmed.start <= trimmed.end:
            kept.append(trimmed)

    kept.append(replace(winner))
    return sort_regions(kept)


# --------------------------------------------------------------- validation


def clamp_region(
    start,
    end,
    slots: int,
    keep_width: bool = False,
    width: int | None = None,
) -> tuple[int, int]:
    """Force a dragged span onto whole slots inside the timeline.

    ``keep_width`` is for a body drag, which must preserve duration.  The
    duration it preserves is ``width`` -- the region's committed length -- not
    whatever length the reported bounds happen to imply: a drag that ran off the
    end of the timeline reports a clipped span, and growing the region to fill
    the gap would silently swallow its neighbours.

    A resize clamps the moved edge instead, and never lets start pass end.
    """
    slots = clamp_slots(slots)
    try:
        first, last = int(round(float(start))), int(round(float(end)))
    except (TypeError, ValueError):
        return (1, 1)
    if last < first:
        first, last = last, first

    if keep_width:
        span = int(width) if width else last - first + 1
        span = max(1, min(span, slots))
        first = max(1, min(first, slots - span + 1))
        return (first, first + span - 1)

    first = max(1, min(first, slots))
    last = max(first, min(last, slots))
    return (first, last)


def validate_regions(regions: list[Region], slots: int) -> None:
    """Raise unless ``regions`` could be committed as-is.

    A region may run past ``slots``: that is a schedule kept past the end of a
    shorter run (see ``refit_schedule``), not an error.  No region can reach
    past the longest timeline there is, though, because none could have been
    drawn there.
    """
    limit = max(held_slots(slots), MAX_SLOTS)
    previous: Region | None = None
    for region in sort_regions(regions):
        if int(region.start) < 1 or int(region.end) > limit:
            raise ScheduleError(f"Region {region.id} lies outside slots 1-{limit}.")
        if int(region.start) > int(region.end):
            raise ScheduleError(f"Region {region.id} has no length.")
        if not is_finite_number(region.strength):
            raise ScheduleError(f"Region {region.id} has a non-numeric strength.")
        if previous is not None and overlaps(previous, region):
            raise ScheduleError(f"Regions {previous.id} and {region.id} overlap.")
        previous = region


# ------------------------------------------------------------ normalisation


def normalize_values(values: list[float], target_slots: int) -> list[float]:
    """Resample a value list onto a different number of slots.

    Nearest-neighbour on purpose: interpolation would invent multipliers the
    user never chose.  Only ever called from an explicit user action.
    """
    numbers = [float(value) for value in values] or [1.0]
    target = clamp_slots(target_slots)
    if target == len(numbers):
        return list(numbers)
    result = []
    for index in range(target):
        source = int(index * len(numbers) / target)
        result.append(numbers[min(source, len(numbers) - 1)])
    return result


def refit_schedule(schedule: PhaseSchedule, target_slots: int) -> PhaseSchedule:
    """Move the end of the timeline to ``target_slots``.  Nothing drawn moves.

    This is what a change to the inference step count means: slot *i* is step
    *i*, so the step count only decides how much of the schedule the run
    reaches.  More steps arrive at the end, empty unless something was drawn
    there before; fewer steps cut the run short, so the token stops at the new
    end and the timeline shows a region crossing it only up to there.

    The cut is not a deletion.  Every region is kept as drawn, so raising the
    step count again brings back exactly what the lower one hid -- and a step
    count that is only passed through on the way to another, as typing ``12``
    passes through ``1``, costs nothing.

    Deliberately not ``normalize_schedule``: resampling would slide every region
    along the timeline to keep its *proportion* of the run, which is right for a
    schedule spread across the whole run and wrong for one drawn step by step.
    """
    target = clamp_slots(target_slots)
    rebuilt = schedule.copy()
    rebuilt.slots = target
    shown = visible_regions(rebuilt)
    if not any(region.id == rebuilt.selected_region_id for region in shown):
        # The selection is a region the panel can show, so a hidden one hands
        # it to whatever is still in view.
        rebuilt.selected_region_id = shown[0].id if shown else None
    rebuilt.dirty = True
    rebuilt.step_aligned = True
    return rebuilt


def normalize_schedule(schedule: PhaseSchedule, target_slots: int) -> PhaseSchedule:
    """Re-grid a schedule to ``target_slots``, keeping what it currently means.

    Resampled from the whole schedule, including the part beyond what a
    timeline would show: an imported list too long to edit must not lose its
    tail by being re-gridded.  What is kept past the end of the run is a
    different thing -- the run does not reach it, so there is nothing of it in
    the list to resample, and a re-grid lets it go.
    """
    target = clamp_slots(target_slots)
    values = normalize_values(compile_schedule(schedule), target)
    rebuilt = reconstruct_schedule(values, raw=schedule.source_raw, slots=target)
    rebuilt.dirty = True
    # Resampling is what turns a list authored at its own resolution into one
    # slot per step. That is the whole point of it, and it happens once.
    rebuilt.step_aligned = True
    rebuilt.source_values = schedule.source_values
    rebuilt.editable = True
    rebuilt.normalization_required = False
    rebuilt.reason = ""
    rebuilt.history = schedule.history
    return rebuilt


def remember(schedule: PhaseSchedule) -> None:
    """Record ``schedule`` as a version the panel has written to the token.

    WanGP answers every write with change events, and each event carries the
    multiplier text as it stood when the event was *sent* -- which, while the
    step slider is being dragged or a strength is being tapped, is routinely a
    write or two behind.  A token the panel wrote is recognised from here when
    it comes back late, so the schedule it came from is put back whole,
    including anything past the end of the run, instead of being rebuilt from
    the token, which keeps only what the run reaches.
    """
    schedule.history.append(schedule.snapshot())
    del schedule.history[:-HISTORY_DEPTH]
