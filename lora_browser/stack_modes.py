"""Visual-only and audio-only LoRAs, groups and bakes, over the native stack.

Everything here edits a ``ui_payloads.Stack`` and says whether it changed,
like the rest of the state model, so all of it is testable without WanGP.
What each feature leaves in WanGP's own state -- which is what presets,
``.lset`` files, the queue and settings saved with media keep:

* A LoRA in visual-only or audio-only mode is selected as its built copy,
  ``_lora_browser/<name>.visual.safetensors`` (``derived``).  Normal selects
  the original again.
* A LoRA its mode leaves nothing to apply -- one that only changes the
  picture, set to audio only -- is off, as a ``#lora_skip`` line.  A mode
  that gives it something back switches it on again; a LoRA switched off by
  hand stays off.
* A group is a set of ``#lora_group`` lines.  Unbaked, its members stay
  selected one by one, side by side, each carrying the group's multiplier.
  Baked, the baked file is selected in their place and the lines still name
  the members, each as the id its mode gives it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from . import derived as dv
from .utils import normalize_id

NORMAL = "normal"
VISUAL = "visual"
AUDIO = "audio"
MODES = (NORMAL, VISUAL, AUDIO)
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

ON, OFF, SKIP = "on", "off", "skip"


def _is_bake(lora_id: str) -> bool:
    parsed = dv.parse(lora_id)
    return parsed is not None and parsed.kind == "bake"


def source_of(lora_id: str, known_ids=()) -> tuple[str, str]:
    """``(original id, mode)`` for any id the stack can hold."""
    parsed = dv.parse(lora_id)
    if parsed is not None and parsed.kind == "split":
        return dv.original_id(parsed, known_ids), parsed.mode
    return normalize_id(lora_id), NORMAL


@dataclass
class Group:
    letter: str
    #: Native ids, in order; each says its member's mode.
    members: list[str] = field(default_factory=list)
    #: The baked file's id, when the group is baked.
    bake: str = ""

    @property
    def baked(self) -> bool:
        return bool(self.bake)


def groups(stack) -> dict[str, Group]:
    """The live groups, by letter (see ``Stack.live_groups``)."""
    found = {}
    for letter, values in stack.live_groups().items():
        bakes = [value for value in values if _is_bake(value)]
        members = [value for value in values if not _is_bake(value)]
        found[letter] = Group(letter, members, bakes[0] if bakes else "")
    return found


def group_of(stack, lora_id: str) -> Group | None:
    target = normalize_id(lora_id)
    for group in groups(stack).values():
        if target in group.members or target == group.bake:
            return group
    return None


def next_letter(stack) -> str:
    used = set(stack.groups)
    return next((letter for letter in LETTERS if letter not in used), "")


def lead_of(stack, group: Group) -> str:
    """The entry that holds a group's multiplier and is edited for it.

    A baked group's file; otherwise its first member that is on, because a
    member that is off carries no schedule state to edit -- and every member
    has the same multiplier, so which one leads changes nothing else.
    """
    if group.baked:
        return group.bake
    selected = set(stack.ids)
    return next((member for member in group.members if member in selected), group.members[0])


def group_on(stack, group: Group) -> bool:
    if group.baked:
        return stack.index_of(group.bake) >= 0
    selected = set(stack.ids)
    return any(member in selected for member in group.members)


# -------------------------------------------------------------- entries


def _states(stack) -> tuple[list[tuple[str, bool, str]], set[str]]:
    return stack.user_entries(), stack.skipped()


def _token(stack, lora_id: str) -> str | None:
    position = stack.index_of(lora_id)
    if position >= 0:
        return stack.tokens[position]
    entry = stack.disabled_entry(lora_id)
    return entry.token if entry is not None else None


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
    for members in stack.groups.values():
        for index, value in enumerate(members):
            if value == old:
                members[index] = new
                changed = True
    return changed


# ---------------------------------------------------------------- modes


def set_mode(stack, lora_id: str, mode: str, known_ids=()) -> str:
    """Point a LoRA at its copy in ``mode``, or back at its original.

    Returns its new id, or ``""`` when nothing changed -- the mode it already
    has, or a copy the stack already holds as a LoRA of its own.  Settle the
    stack afterwards (:func:`settle`): the new mode may leave it nothing.
    """
    if mode not in MODES:
        return ""
    source, current = source_of(lora_id, known_ids)
    if mode == current:
        return ""
    new_id = dv.split_id(source, mode)
    if stack.has_entry(new_id) or any(new_id in members for members in stack.groups.values()):
        return ""
    return new_id if rename(stack, lora_id, new_id) else ""


def settle(stack, empty: Callable[[str], bool]) -> dict[str, list[str]]:
    """Switch off what its mode leaves nothing to apply, and back on what a
    mode gives something again.

    ``empty(id)`` says whether a LoRA in the stack would apply nothing.  A
    LoRA, or a whole group, switched off by hand is never switched on here.
    Returns the ids it switched ``off`` and ``on``.
    """
    entries, skipped = _states(stack)
    on_now = {lora_id for lora_id, on, _ in entries if on}
    found = groups(stack)
    member_group = {member: group for group in found.values() if not group.baked for member in group.members}
    bake_group = {group.bake: group for group in found.values() if group.baked}
    wanted: dict[str, str] = {}
    moved: dict[str, list[str]] = {"off": [], "on": []}

    def want(lora_id: str, state: str) -> None:
        wanted[lora_id] = state
        moved["off" if state == SKIP else "on"].append(lora_id)

    for lora_id, on, _ in entries:
        skip = lora_id in skipped
        if lora_id in bake_group:
            nothing = all(empty(member) for member in bake_group[lora_id].members)
        elif lora_id in member_group:
            group = member_group[lora_id]
            nothing = empty(lora_id)
            if not nothing and skip:
                # A member comes back on unless its group was switched off by
                # hand: nothing on, and something off that is not a skip.
                by_hand = not any(member in on_now for member in group.members) and any(
                    member not in skipped and member not in on_now for member in group.members
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


def create_group(stack, ids, lead: str = "") -> str:
    """Group two or more of the user's LoRAs; returns the new letter.

    They move to sit together where the first of them was, take ``lead``'s
    multiplier, and are on together if any of them was on.
    """
    targets = [normalize_id(value) for value in ids if normalize_id(value)]
    entries, skipped = _states(stack)
    members = [entry for entry in entries if entry[0] in targets]
    if len(members) < 2 or any(group_of(stack, entry[0]) for entry in members):
        return ""
    letter = next_letter(stack)
    if not letter:
        return ""
    lead = normalize_id(lead) if normalize_id(lead) in targets else members[0][0]
    members.sort(key=lambda entry: entry[0] != lead)              # the lead first
    token = next(token for lora_id, _, token in members if lora_id == lead)
    on = any(entry_on for _, entry_on, _ in members)
    position = min(index for index, entry in enumerate(entries) if entry[0] in targets)
    rest = [entry for entry in entries if entry[0] not in targets]
    placed = [
        (lora_id, (lora_id not in skipped) if on else entry_on, token)
        for lora_id, entry_on, _ in members
    ]
    stack.set_user_entries(rest[:position] + placed + rest[position:], skipped)
    stack.groups[letter] = [lora_id for lora_id, _, _ in placed]
    return letter


def add_to_group(stack, letter: str, lora_id: str) -> bool:
    """Bring one more LoRA into a group, with the group's multiplier.

    In a baked group it leaves the stack and becomes one of the members the
    lines name: the caller re-bakes (:func:`rebake`).
    """
    group = groups(stack).get(letter)
    target = normalize_id(lora_id)
    if group is None or not stack.has_entry(target) or group_of(stack, target):
        return False
    entries, skipped = _states(stack)
    lead = lead_of(stack, group)
    token = _token(stack, lead) or "1"
    on = group_on(stack, group)
    rest = [entry for entry in entries if entry[0] != target]
    if group.baked:
        stack.set_user_entries(rest, skipped)
        stack.groups[letter] = group.members + [target, group.bake]
        return True
    last = max(index for index, entry in enumerate(rest) if entry[0] in group.members)
    rest.insert(last + 1, (target, on and target not in skipped, token))
    stack.set_user_entries(rest, skipped)
    stack.groups[letter] = group.members + [target]
    return True


def leave_group(stack, letter: str, lora_id: str) -> bool:
    """Take one LoRA out of a group; it keeps the group's multiplier.

    A group left with one LoRA is no group, so it is dissolved.  Out of a
    baked group the LoRA is selected again, beside the baked file, which the
    caller re-bakes.
    """
    group = groups(stack).get(letter)
    target = normalize_id(lora_id)
    if group is None or target not in group.members:
        return False
    if len(group.members) <= 2:
        return ungroup(stack, letter)
    members = [member for member in group.members if member != target]
    if group.baked:
        entries, skipped = _states(stack)
        position = next(index for index, entry in enumerate(entries) if entry[0] == group.bake)
        _, on, token = entries[position]
        entries.insert(position + 1, (target, on, token))
        stack.set_user_entries(entries, skipped)
        stack.groups[letter] = members + [group.bake]
    else:
        stack.groups[letter] = members
    return True


def ungroup(stack, letter: str) -> bool:
    group = groups(stack).get(letter)
    if group is None:
        stack.groups.pop(letter, None)
        return False
    if group.baked:
        unbake(stack, letter)
    stack.groups.pop(letter, None)
    return True


def remove_group(stack, letter: str) -> bool:
    """Take a whole group out of the stack."""
    group = groups(stack).get(letter)
    if group is None:
        return False
    entries, skipped = _states(stack)
    gone = set(group.members) | ({group.bake} if group.bake else set())
    stack.set_user_entries([entry for entry in entries if entry[0] not in gone], skipped)
    stack.groups.pop(letter, None)
    return True


def set_group_enabled(stack, letter: str, on: bool) -> bool:
    """Switch a group on or off as one.  Settle afterwards: a member its mode
    leaves nothing goes back to being skipped."""
    group = groups(stack).get(letter)
    if group is None:
        return False
    targets = [group.bake] if group.baked else group.members
    return set_states(stack, {lora_id: ON if on else OFF for lora_id in targets})


def bake(stack, letter: str, bake_id: str) -> bool:
    """Select the group's baked file in place of its members.

    The file takes the group's place, multiplier and on/off state; the
    members stay named in the group's lines.
    """
    group = groups(stack).get(letter)
    bake_id = normalize_id(bake_id)
    if group is None or group.baked or stack.has_entry(bake_id):
        return False
    entries, skipped = _states(stack)
    lead = lead_of(stack, group)
    token = _token(stack, lead) or "1"
    on = group_on(stack, group)
    every_skip = all(member in skipped for member in group.members)
    position = min(index for index, entry in enumerate(entries) if entry[0] in group.members)
    rest = [entry for entry in entries if entry[0] not in group.members]
    rest.insert(position, (bake_id, on, token))
    if every_skip:
        skipped.add(bake_id)
    stack.set_user_entries(rest, skipped)
    stack.groups[letter] = group.members + [bake_id]
    return True


def unbake(stack, letter: str) -> bool:
    """Select the members again in place of the baked file.  Settle
    afterwards: a member whose mode leaves it nothing is skipped."""
    group = groups(stack).get(letter)
    if group is None or not group.baked:
        return False
    entries, skipped = _states(stack)
    position = next(index for index, entry in enumerate(entries) if entry[0] == group.bake)
    _, on, token = entries.pop(position)
    for offset, member in enumerate(group.members):
        entries.insert(position + offset, (member, on, token))
    skipped.discard(group.bake)
    stack.set_user_entries(entries, skipped)
    stack.groups[letter] = list(group.members)
    return True


def rebake(stack, letter: str, bake_id: str) -> bool:
    """Point a baked group at another baked file -- its members changed."""
    group = groups(stack).get(letter)
    if group is None or not group.baked:
        return False
    return rename(stack, group.bake, bake_id)


def sync_tokens(stack) -> bool:
    """Give every member of an unbaked group its lead's multiplier."""
    changed = False
    for group in groups(stack).values():
        if group.baked:
            continue
        token = _token(stack, lead_of(stack, group))
        if token is None:
            continue
        for member in group.members:
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


# ------------------------------------------------------------------ rows


def fold_rows(rows: list[dict], stack, describe: Callable[[str], dict]) -> list[dict]:
    """One row per group in place of its members' rows.

    The group's row is its lead's -- strengths, schedules and all, so the
    editor edits the group exactly as it edits a LoRA -- with ``kind:
    "group"``, its letter, and its ``members`` (from their own rows, or from
    ``describe(id)`` for a baked group's members, which have none).
    """
    found = groups(stack)
    if not found:
        return rows
    by_id = {row["id"]: row for row in rows}
    owner = {}
    for group in found.values():
        for member in group.members:
            owner[member] = group
        if group.baked:
            owner[group.bake] = group
    folded, done = [], set()
    for row in rows:
        group = owner.get(row["id"])
        if group is None:
            folded.append(row)
            continue
        if group.letter in done:
            continue
        done.add(group.letter)
        lead = by_id.get(lead_of(stack, group), row)
        merged = dict(lead)
        merged.update(
            kind="group",
            group=group.letter,
            baked=group.baked,
            bake_id=group.bake,
            name=f"Group {group.letter}",
            label=f"Group {group.letter}",
            civitai_name="",
            has_catalogue=False,
            missing=False,
            enabled=group_on(stack, group),
            members=[_member(by_id.get(member), member, stack, describe) for member in group.members],
        )
        folded.append(merged)
    return folded


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
