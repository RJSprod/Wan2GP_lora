"""Visual-only and audio-only LoRAs, separate picture and sound strengths,
groups and bakes, over the native stack.

Everything here edits a ``ui_payloads.Stack`` and says whether it changed,
like the rest of the state model, so all of it is testable without WanGP.
What each feature leaves in WanGP's own state -- which is what presets,
``.lset`` files, the queue and settings saved with media keep:

* A LoRA in visual-only or audio-only mode is selected as its built copy,
  ``_lora_browser/<name>.visual.safetensors`` (``derived``).  Normal selects
  the original again.
* A LoRA whose picture and sound have strengths of their own is selected as
  two halves, ``<name>.visual`` and ``<name>.sound``, each with its own
  multiplier.  The sound half holds the sound side alone; the layers both
  sides share stay in the visual half, so they are applied once, at the
  picture's strength.  Nothing else marks it: a sound half beside its visual
  half is a LoRA with two strengths.
* A LoRA its mode leaves nothing to apply -- one that only changes the
  picture, set to audio only -- is off, as a ``#lora_skip`` line.  A mode
  that gives it something back switches it on again; a LoRA switched off by
  hand stays off.
* A group is a set of ``#lora_group`` lines.  Unbaked, its members stay
  selected one by one, side by side, each carrying the group's multiplier.
  Baked, the baked file is selected in their place and the lines still name
  the members, each as the id its mode gives it.
* A group with separate strengths holds its members as halves -- a Normal
  member as both, a visual-only member as its visual half, an audio-only one
  as its sound half -- and carries two multipliers, one per side: one
  ``track`` each.  Baked, each side with two or more members is baked into a
  file of its own (the sound side's id ends ``.sound``); a side with one
  member keeps it selected.
* A shrunk group (``shrink``) also says, in ``#lora_weight`` lines, what each
  member brings to its baked file: the strength that member ran at, for a
  shrink of single strengths, or 1 for LoRAs that shared a schedule.  It is
  always baked, as one file whatever its members' sides -- their strengths
  are in it -- and runs at the group's multiplier on top.  Unbaking it gives
  each member back as a LoRA of its own, at its weight times that multiplier.
* A shrunk group whose LoRAs run different strengths per phase is baked a
  file per phase (``phased``): its weights are one per phase (``0.8;0.5``),
  and its files -- ``.p1``, ``.p2`` -- each hold every member at that phase's
  weight and run in that phase only, 0 in every other (``pin_tracks``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from . import derived as dv
from . import multiplier_codec as codec
from .derived import PHASE_TRACKS
from .utils import format_number, normalize_id

NORMAL = "normal"
VISUAL = "visual"
AUDIO = "audio"
#: The modes a LoRA has on its own or as a member of a group.
MODES = (NORMAL, VISUAL, AUDIO)
#: A LoRA whose picture and sound each have a strength of their own.  A mode
#: of a single LoRA; a group has separate strengths as a whole.
SEPARATE = "separate"
SINGLE_MODES = MODES + (SEPARATE,)
#: The sound half's copy: the sound side only.
SOUND_HALF = "sound"

#: Which multiplier an entry takes: a group's one (``ALL``), or -- where
#: picture and sound have strengths of their own -- the picture's or the
#: sound's.
ALL = "all"
PICTURE = "picture"
SOUND = "sound"

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DEFAULT_TOKEN = "1"

ON, OFF, SKIP = "on", "off", "skip"


def _is_bake(lora_id: str) -> bool:
    parsed = dv.parse(lora_id)
    return parsed is not None and parsed.kind == "bake"


def is_sound_entry(lora_id: str) -> bool:
    """A sound half or a sound-side bake: entries that only exist where
    picture and sound have strengths of their own."""
    parsed = dv.parse(lora_id)
    if parsed is None:
        return False
    if parsed.kind == "split":
        return parsed.mode == SOUND_HALF
    return parsed.track == dv.SOUND_TRACK


def entry_track(lora_id: str) -> str:
    """The strength an entry takes where picture and sound have their own:
    the sound's for sound halves, audio-only copies and sound-side bakes."""
    parsed = dv.parse(lora_id)
    if parsed is None:
        return PICTURE
    if parsed.kind == "split":
        return SOUND if parsed.mode in (SOUND_HALF, AUDIO) else PICTURE
    return SOUND if parsed.track == dv.SOUND_TRACK else PICTURE


def source_of(lora_id: str, known_ids=()) -> tuple[str, str]:
    """``(original id, mode)`` for any id the stack can hold; a sound half's
    mode is ``sound``."""
    parsed = dv.parse(lora_id)
    if parsed is not None and parsed.kind == "split":
        return dv.original_id(parsed, known_ids), parsed.mode
    return normalize_id(lora_id), NORMAL


def member_mode(entries: list[str], known_ids=()) -> str:
    """The mode of one LoRA selected as ``entries``: both halves are Normal,
    a sound half alone is audio only."""
    modes = {source_of(entry, known_ids)[1] for entry in entries}
    if VISUAL in modes and (SOUND_HALF in modes or AUDIO in modes):
        return NORMAL
    if len(modes) == 1:
        mode = modes.pop()
        return AUDIO if mode == SOUND_HALF else mode
    return NORMAL


def member_ids(source: str, mode: str, separate: bool) -> list[tuple[str, str]]:
    """The entries a LoRA in ``mode`` is selected as, each with the strength
    it takes: one entry, or -- with separate strengths -- its halves."""
    if not separate:
        return [(dv.split_id(source, mode), ALL)]
    if mode == VISUAL:
        return [(dv.split_id(source, VISUAL), PICTURE)]
    if mode == AUDIO:
        return [(dv.split_id(source, SOUND_HALF), SOUND)]
    return [(dv.split_id(source, VISUAL), PICTURE), (dv.split_id(source, SOUND_HALF), SOUND)]


@dataclass
class Group:
    letter: str
    #: Native ids, in order; each says its member's mode.  A member with
    #: separate strengths is listed as both its halves.
    members: list[str] = field(default_factory=list)
    #: Baked files by the strength they carry: ``ALL``, or ``PICTURE`` and
    #: ``SOUND`` for a group with separate strengths.
    bakes: dict[str, str] = field(default_factory=dict)
    #: A shrunk group's members, each with what it brings to the baked file:
    #: one weight, or one per phase where it is baked a file per phase.
    weights: dict[str, list[float]] = field(default_factory=dict)

    @property
    def weighted(self) -> bool:
        """A shrunk group: its members' strengths are in its baked file."""
        return bool(self.weights)

    def weight(self, lora_id: str, track: str = ALL) -> float:
        """What ``lora_id`` brings to the file of ``track``: its one weight,
        or its weight in that file's phase."""
        values = self.weights.get(lora_id) or [1.0]
        phase = dv.phase_of(track)
        if phase <= 0:
            return float(values[0])
        return float(values[min(phase, len(values)) - 1])

    @property
    def phased(self) -> bool:
        """A shrunk group baked a file per phase: its LoRAs bring different
        strengths to different phases."""
        if not self.weights:
            return False
        if self.bakes:
            return any(track in PHASE_TRACKS for track in self.bakes)
        return any(len({round(value, 6) for value in values}) > 1 for values in self.weights.values())

    @property
    def phase_count(self) -> int:
        """How many phases a phased group's weights and files speak for."""
        widest = max((len(values) for values in self.weights.values()), default=1)
        return max([widest] + [dv.phase_of(track) for track in self.bakes])

    @property
    def separate(self) -> bool:
        """Picture and sound have strengths of their own -- never in a shrunk
        group, whose one file holds each side at its own strength."""
        if self.weights:
            return False
        return any(is_sound_entry(value) for value in self.members) or SOUND in self.bakes

    @property
    def tracks(self) -> list[str]:
        """The strengths the group carries, picture first -- a phased group's
        files, phase by phase: those its bakes have, or before it is baked,
        each phase some member brings something to."""
        if self.phased:
            present = [track for track in PHASE_TRACKS if track in self.bakes]
            if present:
                return present
            count = min(self.phase_count, len(PHASE_TRACKS))
            return [PHASE_TRACKS[index] for index in range(count)
                    if any(self.weight(member, PHASE_TRACKS[index]) != 0 for member in self.members)] \
                or [PHASE_TRACKS[0]]
        if not self.separate:
            return [ALL]
        present = {entry_track(member) for member in self.members} | set(self.bakes)
        return [track for track in (PICTURE, SOUND) if track in present]

    def track_of(self, lora_id: str) -> str:
        if self.phased:
            return self.tracks[0]
        return entry_track(lora_id) if self.separate else ALL

    def track_members(self, track: str) -> list[str]:
        if self.phased and track in PHASE_TRACKS:
            return list(self.members)          # every LoRA is in every phase's file
        return [member for member in self.members if self.track_of(member) == track]

    @property
    def baked(self) -> bool:
        return bool(self.bakes)

    @property
    def bake(self) -> str:
        """The baked file -- the picture's, where each side has one; phase
        1's, where each phase has one."""
        return next((self.bakes[track] for track in (ALL, PICTURE, SOUND) + PHASE_TRACKS if track in self.bakes), "")

    @property
    def files(self) -> list[str]:
        """Every baked file, picture first, phase 1 first."""
        return [self.bakes[track] for track in (ALL, PICTURE, SOUND) + PHASE_TRACKS if track in self.bakes]

    def inside(self, lora_id: str) -> bool:
        """``lora_id`` is a member baked into the group's file: named by the
        group's lines, never an entry of its own.  An entry with that id is
        the same LoRA added beside the file."""
        return lora_id in self.members and self.track_of(lora_id) in self.bakes

    def holds(self, lora_id: str) -> bool:
        """An entry of the group's: one of its files, or a member on a side
        that is not baked."""
        return lora_id in self.bakes.values() or (lora_id in self.members and not self.inside(lora_id))

    @property
    def entries(self) -> list[str]:
        """What the group is in the stack: its files and unbaked members."""
        return list(self.files) + [member for member in self.members if not self.inside(member)]


def groups(stack) -> dict[str, Group]:
    """The live groups, by letter (see ``Stack.live_groups``)."""
    found = {}
    weights_by_letter = getattr(stack, "weights", {}) or {}
    for letter, values in stack.live_groups().items():
        members = [value for value in values if not _is_bake(value)]
        weights = {
            member: [float(item) for item in (value if isinstance(value, (list, tuple)) else [value])]
            for member, value in (weights_by_letter.get(letter) or {}).items() if member in members
        }
        separate = not weights and any(is_sound_entry(value) for value in values)
        bakes: dict[str, str] = {}
        for value in values:
            if _is_bake(value):
                parsed = dv.parse(value)
                if weights and parsed.track in PHASE_TRACKS:
                    track = parsed.track
                elif is_sound_entry(value):
                    track = SOUND
                else:
                    track = PICTURE if separate else ALL
                bakes.setdefault(track, value)
        found[letter] = Group(letter, members, bakes, weights)
    return found


def group_of(stack, lora_id: str) -> Group | None:
    target = normalize_id(lora_id)
    for group in groups(stack).values():
        if group.holds(target):
            return group
    return None


def file_mode(group: Group, track: str) -> str:
    """How a baked file of ``group`` is used: ``normal``, or a copy of it
    visual or audio only."""
    parsed = dv.parse(group.bakes.get(track, ""))
    return parsed.mode if parsed is not None and parsed.mode in dv.FILE_MODES else NORMAL


def file_modes(group: Group) -> dict[str, str]:
    """``track -> mode`` for each of a group's files used visual or audio only."""
    return {track: file_mode(group, track) for track in group.bakes if file_mode(group, track) != NORMAL}


def set_file_mode(stack, letter: str, track: str, mode: str) -> str:
    """Use a shrunk group's file for ``track`` visual or audio only -- a copy
    made from the built file, nothing baked again -- or as it is.  Returns
    the id it is selected by now, or ``""`` when nothing changed."""
    group = groups(stack).get(letter)
    if group is None or track not in group.bakes:
        return ""
    old = group.bakes[track]
    new = dv.bake_mode_id(old, mode)
    if new == old or _named(stack, new):
        return ""
    rename(stack, old, new)
    return new


def member_group(stack, lora_id: str) -> Group | None:
    """The group ``lora_id`` stands for: the one whose entry it is, or --
    with no entry of that id beside it -- the one whose file it is baked
    into."""
    target = normalize_id(lora_id)
    found = group_of(stack, target)
    if found is not None or stack.has_entry(target):
        return found
    return next((group for group in groups(stack).values() if group.inside(target)), None)


def next_letter(stack) -> str:
    used = set(stack.groups)
    return next((letter for letter in LETTERS if letter not in used), "")


def lead_of(stack, group: Group, track: str = "") -> str:
    """The entry that holds one of a group's multipliers and is edited for it.

    A baked side's file; otherwise its first member that is on, because a
    member that is off carries no schedule state to edit -- and every member
    on that side has the same multiplier, so which one leads changes nothing
    else.  ``track`` defaults to the group's first.
    """
    track = track or group.tracks[0]
    if track in group.bakes:
        return group.bakes[track]
    members = group.track_members(track) or group.members
    selected = set(stack.ids)
    return next((member for member in members if member in selected), members[0])


def group_on(stack, group: Group) -> bool:
    selected = set(stack.ids)
    return any(value in selected for value in group.entries)


# -------------------------------------------------------------- entries


def _states(stack) -> tuple[list[tuple[str, bool, str]], set[str]]:
    return stack.user_entries(), stack.skipped()


def _token(stack, lora_id: str) -> str | None:
    position = stack.index_of(lora_id)
    if position >= 0:
        return stack.tokens[position]
    entry = stack.disabled_entry(lora_id)
    return entry.token if entry is not None else None


def _named(stack, lora_id: str) -> bool:
    """In the stack, or named by a group's lines as one of its entries -- a
    member baked into a file leaves its id free for the same LoRA beside it."""
    if stack.has_entry(lora_id):
        return True
    found = groups(stack)
    return any(lora_id in members and (letter not in found or not found[letter].inside(lora_id))
               for letter, members in stack.groups.items())


def _free(stack, new_ids, old_ids) -> bool:
    """None of ``new_ids`` is taken by something other than ``old_ids``."""
    old = set(old_ids)
    return not any(value not in old and _named(stack, value) for value in new_ids)


def set_states(stack, wanted: dict[str, str]) -> bool:
    """Switch user entries on, off, or off-for-nothing-to-apply, in place."""
    entries, skipped = _states(stack)
    changed = False
    updated = []
    for lora_id, on, token in entries:
        state = wanted.get(lora_id)
        if state is None:
            updated.append((lora_id, on, token))
            continue
        now = ON if on else (SKIP if lora_id in skipped else OFF)
        if now != state:
            changed = True
        skipped.discard(lora_id)
        if state == SKIP:
            skipped.add(lora_id)
        updated.append((lora_id, state == ON, token))
    if changed:
        stack.set_user_entries(updated, skipped)
    return changed


def rename(stack, old: str, new: str) -> bool:
    """Give one LoRA a new id where it stands -- selected, off, or only named
    by its group's lines -- keeping its place and multiplier."""
    old, new = normalize_id(old), normalize_id(new)
    if old == new or not new:
        return False
    changed = False
    position = stack.index_of(old)
    if position >= 0:
        stack.ids[position] = new
        changed = True
    else:
        entry = stack.disabled_entry(old)
        if entry is not None:
            entry.id = new
            changed = True
    found_groups = groups(stack)
    for letter, members in stack.groups.items():
        if letter in found_groups and found_groups[letter].inside(old):
            continue                     # the file's member, not this entry
        for index, value in enumerate(members):
            if value == old:
                members[index] = new
                changed = True
        if old in getattr(stack, "weights", {}).get(letter, {}):
            stack.weights[letter][new] = stack.weights[letter].pop(old)
    return changed


def replace(stack, old: list[str], new: list[tuple[str, str]], letter: str = "") -> bool:
    """Put ``new`` -- ``(id, token)`` pairs -- where the first of ``old``
    stands, in place of all of ``old``: in the stack, and in any group's
    lines.  An entry that stays keeps its state; a new one is on if any of
    ``old`` was, off for nothing to apply if every one of ``old`` was.

    ``letter`` names the group whose members ``old`` are: only its lines
    change, and an entry only where that member is one -- a member baked into
    its file shares its id with no entry, so one with it is the same LoRA
    beside the file, and stays as it is.  Without it, a group's lines change
    only where they name ``old`` as one of its entries."""
    old = [normalize_id(value) for value in old]
    new = [(normalize_id(value), token or DEFAULT_TOKEN) for value, token in new]
    found_groups = groups(stack)
    owner = found_groups.get(letter) if letter else None
    gone = {value for value in old if owner is None or not owner.inside(value)}
    lines_gone = set(old)
    entries, skipped = _states(stack)
    hits = [index for index, entry in enumerate(entries) if entry[0] in gone]
    changed = False
    if hits:
        before = {entry[0]: entry[1] for entry in entries}
        on = any(entries[index][1] for index in hits)
        every_skip = all(entries[index][0] in skipped for index in hits)
        rest = [entry for entry in entries if entry[0] not in gone]
        placed = []
        for value, token in new:
            if value in before:
                state = before[value]
            else:
                state = on
                if not on and every_skip:
                    skipped.add(value)
            placed.append((value, state, token))
        kept = {value for value, _ in new}
        skipped -= gone - kept
        rest[hits[0]:hits[0]] = placed
        stack.set_user_entries(rest, skipped)
        changed = True
    for name, members in stack.groups.items():
        if letter and name != letter:
            continue
        group = found_groups.get(name)
        here = {value for value in lines_gone if letter or group is None or not group.inside(value)}
        found = [index for index, value in enumerate(members) if value in here]
        if not found:
            continue
        rest = [value for value in members if value not in here]
        at = sum(1 for value in members[:found[0]] if value not in here)
        rest[at:at] = [value for value, _ in new if value not in rest]
        stack.groups[name] = rest
        changed = True
    return changed


# ------------------------------------------------------- single LoRAs


def partner(stack, lora_id: str, known_ids=()) -> str:
    """The other half of a LoRA with separate picture and sound strengths
    that is not in a group, or ``""``."""
    lora_id = normalize_id(lora_id)
    parsed = dv.parse(lora_id)
    if parsed is None or parsed.kind != "split" or parsed.mode not in (VISUAL, SOUND_HALF):
        return ""
    if group_of(stack, lora_id) is not None:
        return ""
    source = dv.original_id(parsed, known_ids)
    other = dv.split_id(source, SOUND_HALF if parsed.mode == VISUAL else VISUAL)
    if stack.has_entry(other) and group_of(stack, other) is None:
        return other
    return ""


def unit_of(stack, lora_id: str, known_ids=()) -> list[str]:
    """Every entry one of the user's LoRAs is selected as: its two halves,
    the picture's first, or just itself."""
    lora_id = normalize_id(lora_id)
    other = partner(stack, lora_id, known_ids)
    if not other:
        return [lora_id]
    return [lora_id, other] if entry_track(lora_id) == PICTURE else [other, lora_id]


def mode_of(stack, lora_id: str, known_ids=()) -> str:
    """The mode of a LoRA not in a group, ``separate`` included."""
    if partner(stack, lora_id, known_ids):
        return SEPARATE
    mode = source_of(lora_id, known_ids)[1]
    return AUDIO if mode == SOUND_HALF else mode


# ---------------------------------------------------------------- modes


def set_mode(stack, lora_id: str, mode: str, known_ids=()) -> str:
    """Point a LoRA at its copy in ``mode``, or back at its original.

    Returns its new id -- the picture half's, for ``separate`` -- or ``""``
    when nothing changed: the mode it already has, or a copy the stack
    already holds as a LoRA of its own.  Settle the stack afterwards
    (:func:`settle`): the new mode may leave it nothing.
    """
    made = set_mode_entries(stack, lora_id, mode, known_ids)
    return made[0] if made else ""


def set_mode_entries(stack, lora_id: str, mode: str, known_ids=(), letter: str = "") -> list[str]:
    """:func:`set_mode`, returning every entry the LoRA is now selected as.

    A single LoRA takes any of :data:`SINGLE_MODES`.  ``separate`` selects it
    as its two halves, both at its strength until one is changed; leaving it,
    Normal and visual only keep the picture's strength, audio only the
    sound's.  A member of a group takes Normal, visual or audio only, as the
    halves its group's strengths need.
    """
    lora_id = normalize_id(lora_id)
    named = groups(stack).get(letter) if letter else None
    # ``letter`` names the group whose member it is: one baked into its file
    # shares its id with any copy of the same LoRA beside the file.
    group = named if named is not None and lora_id in named.members else member_group(stack, lora_id)
    if group is not None:
        return _set_member_mode(stack, group, lora_id, mode, known_ids) if mode in MODES else []
    if mode not in SINGLE_MODES:
        return []
    entries = unit_of(stack, lora_id, known_ids)
    current = mode_of(stack, entries[0], known_ids)
    if mode == current:
        return []
    source = source_of(entries[0], known_ids)[0]
    picture = _token(stack, entries[0]) or DEFAULT_TOKEN
    sound = _token(stack, entries[-1]) or picture
    if mode == SEPARATE:
        new = [(dv.split_id(source, VISUAL), picture), (dv.split_id(source, SOUND_HALF), sound)]
    elif mode == AUDIO:
        new = [(dv.split_id(source, AUDIO), sound)]
    else:
        new = [(dv.split_id(source, mode), picture)]
    if not _free(stack, [value for value, _ in new], entries):
        return []
    replace(stack, entries, new)
    return [value for value, _ in new]


def _track_tokens(stack, group: Group) -> dict[str, str]:
    """Each of a group's multipliers, with the first standing in for a side
    the group does not have yet."""
    tokens = {track: _token(stack, lead_of(stack, group, track)) for track in group.tracks}
    first = next((token for token in tokens.values() if token), DEFAULT_TOKEN)
    return {track: tokens.get(track) or first for track in (ALL, PICTURE, SOUND) + PHASE_TRACKS}


def group_token(stack, group: Group) -> str:
    """A shrunk group's multiplier as one token: its file's, or -- baked a
    file per phase -- each phase's from that phase's own file."""
    tokens = _track_tokens(stack, group)
    if not group.phased:
        return tokens[ALL]
    parts = []
    for phase in range(1, group.phase_count + 1):
        track = PHASE_TRACKS[phase - 1] if phase <= len(PHASE_TRACKS) else ""
        values = codec.token_parts(tokens[track], phase) if track in group.tracks else None
        parts.append(values[phase - 1] if values else [0.0])
    return codec.compose_token(parts)


def member_token(stack, group: Group, member: str) -> str:
    """What one LoRA of a shrunk group runs at: its weight times the group's
    multiplier -- phase by phase, every phase written, where it is baked a
    file per phase."""
    if not group.phased:
        return _weighted_token(group_token(stack, group), group.weight(member))
    tokens = _track_tokens(stack, group)
    parts = []
    for phase in range(1, group.phase_count + 1):
        track = PHASE_TRACKS[phase - 1] if phase <= len(PHASE_TRACKS) else ""
        values = codec.token_parts(tokens[track], phase) if track in group.tracks else None
        weight = group.weight(member, track) if track in group.tracks else 0.0
        parts.append([value * weight for value in values[phase - 1]] if values else [0.0])
    return codec.compose_token(parts)


def pin_tracks(stack, capacity: int) -> bool:
    """Keep every file of a group baked a file per phase in its own phase:
    0 in every other phase of this ``capacity``-phase run, whatever was set,
    typed or pasted -- a phase 1 file never runs in phase 2."""
    changed = False
    for group in groups(stack).values():
        if not group.weighted:
            continue
        for track, value in group.bakes.items():
            phase = dv.phase_of(track)
            if not phase:
                continue
            position = stack.index_of(value)
            entry = stack.disabled_entry(value) if position < 0 else None
            current = stack.tokens[position] if position >= 0 else (entry.token if entry is not None else None)
            if current is None:
                continue
            pinned = codec.pin_token(current, phase, capacity)
            if pinned == current:
                continue
            if position >= 0:
                stack.tokens[position] = pinned
            else:
                entry.token = pinned
            changed = True
    return changed


def _set_member_mode(stack, group: Group, lora_id: str, mode: str, known_ids=()) -> list[str]:
    if group.weighted:
        return []                   # baked in that mode: unshrink to change it
    source = source_of(lora_id, known_ids)[0]
    old = [member for member in group.members if source_of(member, known_ids)[0] == source]
    if not old or member_mode(old, known_ids) == mode:
        return []
    tokens = _track_tokens(stack, group)
    new = [(value, tokens[track]) for value, track in member_ids(source, mode, group.separate)]
    if not _free(stack, [value for value, _ in new], old):
        return []
    replace(stack, old, new, letter=group.letter)
    return [value for value, _ in new]


def settle(stack, empty: Callable[[str], bool]) -> dict[str, list[str]]:
    """Switch off what its mode leaves nothing to apply, and back on what a
    mode gives something again.

    ``empty(id)`` says whether a LoRA in the stack would apply nothing.  A
    LoRA, or a whole group, switched off by hand is never switched on here.
    Returns the ids it switched ``off`` and ``on``.
    """
    entries, skipped = _states(stack)
    on_now = {lora_id for lora_id, on, _ in entries if on}
    present = {lora_id for lora_id, _, _ in entries}
    found = groups(stack)
    member_group = {member: group for group in found.values() for member in group.members
                    if not group.inside(member)}
    bake_track = {value: (group, track) for group in found.values() for track, value in group.bakes.items()}
    wanted: dict[str, str] = {}
    moved: dict[str, list[str]] = {"off": [], "on": []}

    def want(lora_id: str, state: str) -> None:
        wanted[lora_id] = state
        moved["off" if state == SKIP else "on"].append(lora_id)

    for lora_id, on, _ in entries:
        skip = lora_id in skipped
        if lora_id in bake_track:
            group, track = bake_track[lora_id]
            nothing = all(empty(member) for member in group.track_members(track))
        elif lora_id in member_group:
            group = member_group[lora_id]
            nothing = empty(lora_id)
            if not nothing and skip:
                # A member comes back on unless its group was switched off by
                # hand: nothing on, and something off that is not a skip.
                mine = [value for value in group.members + group.files if value in present]
                by_hand = not any(value in on_now for value in mine) and any(
                    value not in skipped and value not in on_now for value in mine
                )
                if by_hand:
                    continue
        else:
            nothing = empty(lora_id)
        if nothing and on:
            want(lora_id, SKIP)
        elif not nothing and skip:
            want(lora_id, ON)
    set_states(stack, wanted)
    return moved


# --------------------------------------------------------------- groups


def _units(stack, ids, known_ids=()) -> list[list[str]]:
    """The user's LoRAs ``ids`` name, each as every entry it is selected as."""
    units, seen = [], set()
    for value in ids:
        unit = [entry for entry in unit_of(stack, value, known_ids) if stack.has_entry(entry)]
        if not unit or unit[0] in seen:
            continue
        seen.update(unit)
        units.append(unit)
    return units


def create_group(stack, ids, lead: str = "", known_ids=()) -> str:
    """Group two or more of the user's LoRAs; returns the new letter.

    They move to sit together where the first of them was, take ``lead``'s
    multiplier, and are on together if any of them was on.  A lead with
    separate picture and sound strengths gives the group both: every member
    is selected as the halves its mode needs.
    """
    units = _units(stack, ids, known_ids)
    if len(units) < 2 or any(group_of(stack, entry) for unit in units for entry in unit):
        return ""
    letter = next_letter(stack)
    if not letter:
        return ""
    target = normalize_id(lead)
    lead_unit = next((unit for unit in units if target in unit), units[0])
    separate = len(lead_unit) == 2
    picture = _token(stack, lead_unit[0]) or DEFAULT_TOKEN
    tokens = {ALL: picture, PICTURE: picture, SOUND: _token(stack, lead_unit[-1]) or picture}
    placed = []
    for unit in [lead_unit] + [unit for unit in units if unit is not lead_unit]:
        source = source_of(unit[0], known_ids)[0]
        mode = NORMAL if len(unit) == 2 else mode_of(stack, unit[0], known_ids)
        placed += [(value, tokens[track]) for value, track in member_ids(source, mode, separate)]
    old = [entry for unit in units for entry in unit]
    if not _free(stack, [value for value, _ in placed], old):
        return ""

    entries, skipped = _states(stack)
    gone = set(old)
    before = {entry[0]: entry[1] for entry in entries}
    on = any(before.get(entry, False) for entry in old)
    position = min(index for index, entry in enumerate(entries) if entry[0] in gone)
    rest = [entry for entry in entries if entry[0] not in gone]
    rows = []
    for value, token in placed:
        if value in before:
            state = (value not in skipped) if on else before[value]
        else:
            state = on
        rows.append((value, state, token))
    skipped -= gone - {value for value, _ in placed}
    rest[position:position] = rows
    stack.set_user_entries(rest, skipped)
    stack.groups[letter] = [value for value, _ in placed]
    return letter


def add_to_group(stack, letter: str, lora_id: str, known_ids=()) -> bool:
    """Bring one more LoRA into a group, with the group's multiplier -- or
    multipliers, selected as the halves the group's strengths need.

    In a baked group it leaves the stack and becomes one of the members the
    lines name: the caller re-bakes (:func:`set_bakes`).
    """
    group = groups(stack).get(letter)
    units = _units(stack, [lora_id], known_ids)
    if group is None or group.weighted or not units or any(group_of(stack, entry) for entry in units[0]):
        return False
    unit = units[0]
    source = source_of(unit[0], known_ids)[0]
    mode = NORMAL if len(unit) == 2 else mode_of(stack, unit[0], known_ids)
    tokens = _track_tokens(stack, group)
    new = [(value, tokens[track]) for value, track in member_ids(source, mode, group.separate)]
    if not _free(stack, [value for value, _ in new], unit):
        return False
    entries, skipped = _states(stack)
    on = group_on(stack, group)
    rest = [entry for entry in entries if entry[0] not in unit]
    skipped -= set(unit)
    if group.baked:
        stack.set_user_entries(rest, skipped)
        stack.groups[letter] = group.members + [value for value, _ in new] + group.files
        return True
    last = max(index for index, entry in enumerate(rest) if entry[0] in group.members)
    rest[last + 1:last + 1] = [(value, on, token) for value, token in new]
    stack.set_user_entries(rest, skipped)
    stack.groups[letter] = group.members + [value for value, _ in new]
    return True


def _single_out(stack, entries: list[str], known_ids=()) -> None:
    """A LoRA no longer in a group: a sound half on its own is the LoRA's
    audio-only copy (its pair, if any, is a LoRA with two strengths)."""
    for value in entries:
        parsed = dv.parse(value)
        if parsed is None or parsed.kind != "split" or parsed.mode != SOUND_HALF:
            continue
        if stack.has_entry(value) and not partner(stack, value, known_ids):
            target = dv.split_id(dv.original_id(parsed, known_ids), AUDIO)
            if not _named(stack, target):
                rename(stack, value, target)


def to_doubles(stack, members) -> list[str]:
    """Entries of the same LoRAs as ``members`` -- added on their own beside
    the baked file that holds them -- become doubles as those members come
    back out of it: WanGP cannot apply one LoRA twice, so the copy that was
    added beside it stays off, and cannot be switched on, until the other is
    gone.  Returns the ids made doubles."""
    stems = {dv.stem_of(normalize_id(value)) for value in members}
    if not stems:
        return []
    entries, skipped = _states(stack)
    moving = [(index, entry) for index, entry in enumerate(entries) if dv.stem_of(entry[0]) in stems]
    if not moving:
        return []
    ids = {entry[0] for _, entry in moving}
    stack.set_user_entries([entry for entry in entries if entry[0] not in ids], skipped - ids)
    stack.doubles = [double for double in getattr(stack, "doubles", []) if double.id not in ids] + [
        codec.DisabledLora(id=lora_id, token=token, at=index, double=True)
        for index, (lora_id, _, token) in moving
    ]
    return [entry[0] for _, entry in moving]


def seat_doubles(stack, ids) -> None:
    """Each of these doubles right after the copy of its LoRA that stays."""
    wanted = set(ids)
    if not wanted:
        return
    entries = stack.user_entries()
    for double in getattr(stack, "doubles", []):
        if double.id not in wanted:
            continue
        stem = dv.stem_of(double.id)
        twins = [index for index, entry in enumerate(entries) if dv.stem_of(entry[0]) == stem]
        if twins:
            double.at = twins[-1] + 1


def release_doubles(stack) -> bool:
    """A double whose other copy is no entry any more -- removed, or baked
    into a file again -- is a LoRA of its own again: off, where it was, and
    free to be switched on (beside the file, where it is in one)."""
    doubles = getattr(stack, "doubles", [])
    if not doubles:
        return False
    present = {dv.stem_of(lora_id) for lora_id, _, _ in stack.user_entries()}
    free = [double for double in doubles if dv.stem_of(double.id) not in present]
    if not free:
        return False
    stack.doubles = [double for double in doubles if double not in free]
    entries, skipped = _states(stack)
    for double in sorted(free, key=lambda item: item.at):
        entries.insert(max(0, min(int(double.at), len(entries))), (double.id, False, double.token))
    stack.set_user_entries(entries, skipped)
    return True


def leave_group(stack, letter: str, lora_id: str, known_ids=()) -> bool:
    """Take one LoRA out of a group; it keeps the group's multiplier -- or,
    with separate strengths, the picture's and the sound's.

    A group left with one LoRA is no group, so it is dissolved.  Out of a
    baked group the LoRA is selected again, beside the baked file, which the
    caller re-bakes.
    """
    group = groups(stack).get(letter)
    if group is None:
        return False
    source = source_of(lora_id, known_ids)[0]
    mine = [member for member in group.members if source_of(member, known_ids)[0] == source]
    if not mine:
        return False
    if len({source_of(member, known_ids)[0] for member in group.members}) <= 2:
        return ungroup(stack, letter, known_ids)
    members = [member for member in group.members if member not in mine]
    if group.baked:
        tokens = _track_tokens(stack, group)
        coming = [(member, member_token(stack, group, member) if group.weighted else tokens[group.track_of(member)])
                  for member in mine if group.inside(member)]
        made = to_doubles(stack, [member for member, _ in coming])
        entries, skipped = _states(stack)
        values = set(group.entries)
        spots = [index for index, entry in enumerate(entries) if entry[0] in values]
        on = any(entries[index][1] for index in spots)
        at = spots[-1] + 1 if spots else len(entries)
        entries[at:at] = [(member, on, token) for member, token in coming if not stack.has_entry(member)]
        stack.set_user_entries(entries, skipped)
        seat_doubles(stack, made)
        stack.groups[letter] = members + group.files
        if group.weighted:
            stack.weights[letter] = {member: list(group.weights.get(member, [1.0])) for member in members}
    else:
        stack.groups[letter] = members
    _single_out(stack, mine, known_ids)
    return True


def ungroup(stack, letter: str, known_ids=()) -> bool:
    group = groups(stack).get(letter)
    if group is None:
        stack.groups.pop(letter, None)
        getattr(stack, "weights", {}).pop(letter, None)
        return False
    if group.weighted:
        return unshrink(stack, letter)
    if group.baked:
        unbake(stack, letter)
        group = groups(stack).get(letter) or group
    stack.groups.pop(letter, None)
    _single_out(stack, group.members, known_ids)
    return True


def remove_group(stack, letter: str) -> bool:
    """Take a whole group out of the stack."""
    group = groups(stack).get(letter)
    if group is None:
        return False
    entries, skipped = _states(stack)
    gone = set(group.entries)
    stack.set_user_entries([entry for entry in entries if entry[0] not in gone], skipped)
    stack.groups.pop(letter, None)
    getattr(stack, "weights", {}).pop(letter, None)
    return True


def set_group_enabled(stack, letter: str, on: bool) -> bool:
    """Switch a group on or off as one.  Settle afterwards: a member its mode
    leaves nothing goes back to being skipped."""
    group = groups(stack).get(letter)
    if group is None:
        return False
    targets = [value for value in group.files + group.members if stack.has_entry(value)]
    return set_states(stack, {lora_id: ON if on else OFF for lora_id in targets})


def set_separate(stack, letter: str, separate: bool, known_ids=()) -> bool:
    """Give a group separate picture and sound strengths, or one again.

    Separating, both start at the group's strength; joining, the group keeps
    the picture's.  Every member is selected as the halves -- or the one
    entry -- its mode needs.  An unbaked group only: unbake first, and bake
    again after.
    """
    group = groups(stack).get(letter)
    if group is None or group.baked or group.weighted or group.separate == bool(separate):
        return False
    tokens = _track_tokens(stack, group)
    token = tokens[PICTURE] if group.separate else tokens[ALL]
    by_source: dict[str, list[str]] = {}
    for member in group.members:
        by_source.setdefault(source_of(member, known_ids)[0], []).append(member)
    plan = []
    for source, entries in by_source.items():
        mode = member_mode(entries, known_ids)
        if separate:
            new = [(value, tokens[ALL]) for value, _ in member_ids(source, mode, True)]
        else:
            new = [(value, token) for value, _ in member_ids(source, mode, False)]
        plan.append((entries, new))
    old = [value for entries, _ in plan for value in entries]
    if not _free(stack, [value for _, new in plan for value, _ in new], old):
        return False
    for entries, new in plan:
        replace(stack, entries, new)
    return True


def set_bakes(stack, letter: str, wanted: dict[str, str]) -> bool:
    """Select each side of a group as ``wanted`` says: as its baked file
    (``track -> id``), or as its members.  Only a side with two or more
    members can be baked; the group keeps its place, its multipliers and its
    on/off state."""
    group = groups(stack).get(letter)
    if group is None:
        return False
    wanted = {
        track: normalize_id(value) for track, value in (wanted or {}).items()
        if value and track in group.tracks and len(group.track_members(track)) >= 2
    }
    if wanted == group.bakes:
        return False
    if any(value not in group.bakes.values() and _named(stack, value) for value in wanted.values()):
        return False
    tokens = _track_tokens(stack, group)
    made = to_doubles(stack, [member for member in group.members
                              if group.inside(member) and group.track_of(member) not in wanted])
    entries, skipped = _states(stack)
    mine = set(group.entries)
    spots = [index for index, entry in enumerate(entries) if entry[0] in mine]
    if not spots:
        return False
    on = any(entries[index][1] for index in spots)
    # Off only because nothing in it applied: what replaces it is off that way too.
    every_skip = not on and all(entries[index][0] in skipped for index in spots)
    rebuilt = [(wanted[track], tokens[track]) for track in group.tracks if track in wanted]
    rebuilt += [
        (member, tokens[group.track_of(member)])
        for member in group.members if group.track_of(member) not in wanted
    ]
    kept = {value for value, _ in rebuilt}
    placed = [(value, on and value not in skipped, token) for value, token in rebuilt]
    skipped -= mine - kept
    if every_skip:
        skipped |= kept
    rest = [entry for entry in entries if entry[0] not in mine]
    rest[spots[0]:spots[0]] = placed
    stack.set_user_entries(rest, skipped)
    stack.groups[letter] = list(group.members) + [wanted[track] for track in group.tracks if track in wanted]
    seat_doubles(stack, made)
    return True


def bake(stack, letter: str, bake_id) -> bool:
    """Select the group's baked file -- or one per side, ``track -> id`` --
    in place of its members.

    The file takes the group's place, multiplier and on/off state; the
    members stay named in the group's lines.
    """
    group = groups(stack).get(letter)
    if group is None or group.baked:
        return False
    wanted = bake_id if isinstance(bake_id, dict) else {group.tracks[0]: bake_id}
    return set_bakes(stack, letter, wanted)


def unbake(stack, letter: str) -> bool:
    """Select the members again in place of the baked files.  Settle
    afterwards: a member whose mode leaves it nothing is skipped.  A shrunk
    group is unshrunk: its members go back to being LoRAs of their own."""
    group = groups(stack).get(letter)
    if group is None or not group.baked:
        return False
    if group.weighted:
        return unshrink(stack, letter)
    return set_bakes(stack, letter, {})


def _weighted_token(token: str, weight: float) -> str:
    """What a member ran at inside a shrunk group: its weight times the
    group's multiplier, in that multiplier's phases and steps."""
    scaled = codec.scale_token(token or DEFAULT_TOKEN, weight)
    return scaled if scaled is not None else format_number(weight)


def unshrink(stack, letter: str) -> bool:
    """A shrunk group's LoRAs as LoRAs of their own again, where the group
    stood, each at its weight times the group's multiplier and on if the
    group was."""
    group = groups(stack).get(letter)
    if group is None or not group.weighted:
        return False
    restored = [(member, member_token(stack, group, member)) for member in group.members]
    # Coming out of its file: a copy beside the file becomes a double.  A
    # group that lost its file holds its members as entries already.
    made = to_doubles(stack, [member for member in group.members if group.inside(member)])
    entries, skipped = _states(stack)
    mine = set(group.entries)
    spots = [index for index, entry in enumerate(entries) if entry[0] in mine]
    on = any(entries[index][1] for index in spots) if spots else True
    rest = [entry for entry in entries if entry[0] not in mine]
    at = sum(1 for index, entry in enumerate(entries) if entry[0] not in mine and index < spots[0]) if spots else len(rest)
    rest[at:at] = [(member, on, token) for member, token in restored]
    stack.set_user_entries(rest, skipped - mine)
    stack.groups.pop(letter, None)
    stack.weights.pop(letter, None)
    seat_doubles(stack, made)
    return True


def rebake(stack, letter: str, bake_id) -> bool:
    """Point a baked group at other baked files -- its members changed."""
    group = groups(stack).get(letter)
    if group is None or not group.baked:
        return False
    if isinstance(bake_id, dict):
        return set_bakes(stack, letter, bake_id)
    return rename(stack, group.bake, bake_id)


def sync_tokens(stack) -> bool:
    """Give every member of an unbaked side its lead's multiplier."""
    changed = False
    for group in groups(stack).values():
        if group.weighted:
            continue                    # each member brings its own strength
        for track in group.tracks:
            if track in group.bakes:
                continue
            token = _token(stack, lead_of(stack, group, track))
            if token is None:
                continue
            for member in group.track_members(track):
                position = stack.index_of(member)
                if position >= 0:
                    if stack.tokens[position] != token:
                        stack.tokens[position] = token
                        changed = True
                    continue
                entry = stack.disabled_entry(member)
                if entry is not None and entry.token != token:
                    entry.token = token
                    changed = True
    return changed


def repair(stack, known_ids=()) -> bool:
    """Bring back the shapes the rest of the model relies on, after a hand
    edit or a preset took one apart: a sound half on its own is that LoRA's
    audio-only copy, and a group with separate strengths holds a Normal
    member as both halves."""
    before = (list(stack.ids), [(entry.id, entry.at) for entry in stack.disabled], repr(stack.groups))
    # A shrunk group is only ever its baked file: one that lost it -- a hand
    # edit, a preset -- is its LoRAs again, at their own strengths.
    for letter, group in list(groups(stack).items()):
        if group.weighted and not group.baked:
            unshrink(stack, letter)
    doubles_before = [(double.id, double.at) for double in getattr(stack, "doubles", [])]
    release_doubles(stack)
    _single_out(stack, [lora_id for lora_id, _, _ in stack.user_entries()
                        if group_of(stack, lora_id) is None], known_ids)
    for group in groups(stack).values():
        if not group.separate:
            continue
        tokens = _track_tokens(stack, group)
        for member in list(group.members):
            source, mode = source_of(member, known_ids)
            if mode != NORMAL:
                continue
            new = [(value, tokens[track]) for value, track in member_ids(source, NORMAL, True)]
            if _free(stack, [value for value, _ in new], [member]):
                replace(stack, [member], new)
    after = (list(stack.ids), [(entry.id, entry.at) for entry in stack.disabled], repr(stack.groups))
    return before != after or doubles_before != [(double.id, double.at) for double in getattr(stack, "doubles", [])]


# ------------------------------------------------------------------ rows


def fold_rows(rows: list[dict], stack, describe: Callable[[str], dict], known_ids=()) -> list[dict]:
    """One row per LoRA and per group, however many entries each is.

    A LoRA with separate picture and sound strengths is its picture half's
    row, with ``mode: "separate"`` and the sound half's row as its
    ``sound_track``.  A group's row is its lead's -- strengths, schedules and
    all, so the editor edits the group exactly as it edits a LoRA -- with
    ``kind: "group"``, its letter, and its ``members``, one per LoRA (from
    their own rows, or from ``describe(id)`` for members a baked side holds),
    each listing its ``entries``.  With separate strengths, the picture
    side's lead is the row and the sound side's its ``sound_track``.
    """
    found = groups(stack)
    by_id = {row["id"]: row for row in rows}
    owner = {}
    for group in found.values():
        for value in group.entries:
            owner[value] = group
    folded, done = [], set()
    for row in rows:
        group = owner.get(row["id"])
        if group is None:
            other = partner(stack, row["id"], known_ids)
            if not other:
                folded.append(row)
                continue
            if entry_track(row["id"]) == SOUND and other in by_id:
                continue                     # shown as its picture half's sound track
            merged = dict(row)
            merged.update(mode=SEPARATE, track=PICTURE)
            if other in by_id:
                merged["sound_track"] = dict(by_id[other], track=SOUND)
            folded.append(merged)
            continue
        if group.letter in done:
            continue
        done.add(group.letter)
        tracks = group.tracks
        lead = by_id.get(lead_of(stack, group, tracks[0]), row)
        merged = dict(lead)
        merged.update(
            kind="group",
            group=group.letter,
            baked=group.baked,
            bake_id=group.bake,
            bake_ids=group.files,
            separate=group.separate,
            shrink=group.weighted,
            track=tracks[0],
            name=f"Shrunk {group.letter}" if group.weighted else f"Group {group.letter}",
            label=f"Shrunk {group.letter}" if group.weighted else f"Group {group.letter}",
            civitai_name="",
            has_catalogue=False,
            missing=False,
            enabled=group_on(stack, group),
            members=_members(group, stack, by_id, describe, known_ids),
        )
        if group.phased:
            # A file per phase, each edited in its own phase only.
            merged["phase_tracks"] = [
                dict(by_id[group.bakes[track]], track=track, pinned_phase=dv.phase_of(track))
                for track in tracks if group.bakes.get(track) in by_id
            ]
            merged["group_token"] = group_token(stack, group)
        elif len(tracks) > 1:
            sound = by_id.get(lead_of(stack, group, tracks[1]))
            if sound is not None:
                merged["sound_track"] = dict(sound, track=tracks[1])
        folded.append(merged)
    return folded


def _members(group: Group, stack, by_id: dict, describe, known_ids=()) -> list[dict]:
    out: list[dict] = []
    by_source: dict[str, dict] = {}
    for value in group.members:
        source = source_of(value, known_ids)[0]
        if source in by_source:
            by_source[source]["entries"].append(value)
            continue
        member = _member(by_id.get(value), value, stack, describe)
        member["entries"] = [value]
        by_source[source] = member
        out.append(member)
    for member in out:
        member["member_mode"] = member_mode(member["entries"], known_ids)
        if group.weighted:
            # What each of its entries brings to the baked file: one weight,
            # or the picture's and the sound's -- each, where the group has a
            # file per phase, one per phase.
            if group.phased:
                member["weights"] = [[group.weight(entry, track) for track in group.tracks]
                                     for entry in member["entries"]]
                member["phases"] = [dv.phase_of(track) for track in group.tracks]
            else:
                member["weights"] = [group.weight(entry) for entry in member["entries"]]
    return out


def _member(row: dict | None, lora_id: str, stack, describe) -> dict:
    if row is None:
        member = dict(describe(lora_id))
        member.setdefault("id", lora_id)
        member["enabled"] = True
        member["skipped"] = False
        return member
    return {
        "id": row["id"],
        "name": row.get("name", ""),
        "label": row.get("label", ""),
        "civitai_name": row.get("civitai_name", ""),
        "has_catalogue": row.get("has_catalogue", False),
        "missing": row.get("missing", False),
        "enabled": row.get("enabled", True),
        "skipped": lora_id in stack.skipped(),
    }
