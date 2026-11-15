"""The built files a stack uses: what they are, whether they are ready, and
building the ones it lacks.

Every payload asks this, per row, which original a LoRA is, which mode it
runs in, what each of its streams holds, and whether its copy -- or its
group's bake -- is on disk.  And every payload queues whatever the stack
selects but does not have yet: a copy is built the moment a setting needs it,
whether that setting came from a toggle, a recalled profile, a pasted stack
or a preset.  A file already built is used at once; one whose original
changed is built again.

If WanGP is asked to generate while a file is still building, the file is not
there yet and WanGP reports it missing: the panel warns about exactly that
while anything is building.
"""

from __future__ import annotations

import os

from . import bake as bk
from . import builder as bd
from . import derived as dv
from . import lora_files as lf
from . import stack_modes as sm
from .utils import display_name, normalize_id


class StackFiles:
    def __init__(self, store: dv.Store | None, builder: bd.Builder, summaries: lf.SummaryCache,
                 inventory, sided: bool) -> None:
        self.store = store
        self.builder = builder
        self.summaries = summaries
        self.inventory = inventory
        #: LTX-2: LoRAs have a picture side and a sound side to choose between.
        self.sided = bool(sided)
        self.known = list(inventory.ids)

    # ---------------------------------------------------------- originals

    def source(self, native_id: str) -> tuple[str, str]:
        return sm.source_of(native_id, self.known)

    def path_of(self, source_id: str) -> str | None:
        entry = self.inventory.get(source_id)
        if entry is not None and entry.path:
            return entry.path
        lora_dir = getattr(self.inventory, "lora_dir", "")
        if lora_dir and not os.path.isabs(source_id) and "://" not in source_id:
            return os.path.normpath(os.path.join(lora_dir, normalize_id(source_id)))
        return None

    def summary(self, source_id: str) -> lf.StreamSummary | None:
        return self.summaries.get(self.path_of(source_id))

    def empty(self, native_id: str) -> bool:
        """The LoRA's mode leaves it nothing to apply."""
        source, mode = self.source(native_id)
        if mode == sm.NORMAL or not self.sided:
            return False
        summary = self.summary(source)
        return summary is not None and not summary.has(mode)

    def sides(self, native_id: str) -> str:
        """What a member brings to a bake: ``p``icture, ``s``ound, both, or
        ``-``.  A mode that changes nothing -- visual only on a LoRA that only
        changes the picture -- brings the same as Normal.  A sound half brings
        less than an audio-only copy when the LoRA has shared layers, and says
        so."""
        source, mode = self.source(native_id)
        if not self.sided:
            return "all"
        summary = self.summary(source)
        if summary is None:
            return mode
        picture = summary.picture.tensors > 0 and mode in (sm.NORMAL, sm.VISUAL)
        sound = summary.sound.tensors > 0 and mode in (sm.NORMAL, sm.AUDIO, sm.SOUND_HALF)
        text = ("p" if picture else "") + ("s" if sound else "") or "-"
        if mode == sm.SOUND_HALF and summary.shared.tensors:
            text += "/no-shared"
        return text

    def brings(self, native_id: str) -> tuple[bool, bool]:
        """Whether this entry applies anything to the picture, and to the
        sound, in the mode it is in."""
        source, mode = self.source(native_id)
        summary = self.summary(source)
        if summary is None:
            return mode != sm.AUDIO and mode != sm.SOUND_HALF, mode != sm.VISUAL
        return (
            summary.picture.tensors > 0 and mode in (sm.NORMAL, sm.VISUAL),
            summary.sound.tensors > 0 and mode in (sm.NORMAL, sm.AUDIO, sm.SOUND_HALF),
        )

    def has_both(self, native_id: str) -> bool:
        """The LoRA changes the picture and the sound, so each can have a
        strength of its own."""
        summary = self.summary(self.source(native_id)[0])
        return summary is not None and summary.both

    # --------------------------------------------------------------- bakes

    def bake_ids(self, group: sm.Group, rank: int) -> dict[str, str]:
        """The file each side of ``group`` bakes into at ``rank``: the whole
        group's, or -- with separate strengths -- the picture's and the
        sound's, for each side with two or more members.  A shrunk group's
        members are named with what each brings."""
        found = {}
        for track in group.tracks:
            members = self.bakeable(group, track)
            if len(members) >= 2:
                suffix = dv.SOUND_TRACK if track == sm.SOUND else (track if track in dv.PHASE_TRACKS else "")
                found[track] = dv.bake_id(
                    [(self.source(member)[0], self.sides(member), group.weight(member, track)) for member in members],
                    rank, track=suffix,
                )
        return found

    def bake_id(self, group: sm.Group, rank: int) -> str:
        """The group's baked file -- the picture's, where each side has one."""
        found = self.bake_ids(group, rank)
        return next((found[track] for track in (sm.ALL, sm.PICTURE, sm.SOUND) if track in found), "")

    def default_rank(self, group: sm.Group) -> int:
        """The rank of the highest-rank LoRA in the group that brings anything."""
        ranks = []
        for member in group.members:
            if self.empty(member):
                continue
            summary = self.summary(self.source(member)[0])
            if summary is not None and summary.rank:
                ranks.append(summary.rank)
        return max(ranks, default=0)

    def recognised(self, group: sm.Group) -> str:
        """Why the group cannot be baked because WanGP recognises one of its
        LoRAs by its file name, or ""."""
        for member in group.members:
            source = self.source(member)[0]
            signature = dv.recognised_by_name(source)
            if signature:
                return (f"WanGP recognises {display_name(source)} by its file name ('{signature}'); "
                        "baked into a file of its own, it would not. Take it out of the group to bake the rest.")
        return ""

    def bakeable(self, group: sm.Group, track: str) -> list[str]:
        """A side's members that bring it something: a picture-only LoRA's
        sound half, in a group with separate strengths, brings nothing."""
        return [member for member in group.track_members(track) if not self.empty(member)]

    def _members(self, group: sm.Group, track: str = "") -> list[tuple[str, str, str, str, float]] | None:
        found = []
        for member in (self.bakeable(group, track) if track else group.members):
            source, mode = self.source(member)
            path = self.path_of(source)
            if not path or not os.path.isfile(path):
                return None
            found.append((source, path, mode, display_name(source), group.weight(member, track or sm.ALL)))
        return found

    def bake_options(self, stack, letter: str) -> dict:
        """What the bake dialog offers: sizes per rank, read from headers only."""
        group = sm.groups(stack).get(letter)
        if group is None:
            return {"error": "That group is gone."}
        recognised = self.recognised(group)
        if recognised:
            return {"error": recognised}
        plans = []
        for track in group.tracks:
            members = self._members(group, track)
            if members is None:
                return {"error": "A LoRA in this group is missing locally."}
            if len(members) < 2:
                continue                    # a side of one LoRA stays as it is
            try:
                plans.append(bk.plan([bk.Member(path, mode, name, weight) for _, path, mode, name, weight in members],
                                     sided=self.sided))
            except bk.BakeError as error:
                return {"error": str(error)}
        if not plans:
            return {"error": "No side of this group has two LoRAs to bake together."}
        combined = max(plan.combined_rank for plan in plans)
        default = min(max(plan.default_rank for plan in plans) or combined, combined)
        ranks = sorted({combined, default, max(1, default // 2)}, reverse=True)
        return {
            "group": letter,
            "combined_rank": combined,
            "default_rank": default,
            "current_bytes": sum(plan.size_for(plan.combined_rank) for plan in plans),
            "options": [
                {"rank": rank, "bytes": sum(plan.size_for(rank) for plan in plans),
                 "exact": all(rank >= plan.combined_rank for plan in plans)}
                for rank in ranks
            ],
            "baked_rank": dv.parse(group.bake).rank if group.baked else 0,
            # A group with separate strengths bakes a file per side.
            "files": len(plans),
        }

    # --------------------------------------------------------------- state

    def state(self, native_id: str) -> dict:
        """Whether a built file is ready, and how far along it is if not."""
        parsed = dv.parse(native_id)
        if parsed is None:
            return {"state": "original"}
        if self.store is None:
            return {"state": "unavailable"}
        busy = self.builder.busy(native_id)
        if busy == "building":
            return {"state": "building", "pct": round(self.builder.progress(native_id) * 100)}
        if busy == "queued":
            return {"state": "queued"}
        if busy == "failed":
            return {"state": "failed", "error": self.builder.failure(native_id)}
        if self.store.ready(native_id):
            entry = self.store.entry(native_id) or {}
            state = {"state": "built"}
            try:
                state["bytes"] = os.path.getsize(self.store.path(native_id))
            except OSError:
                pass
            for key in ("kept", "exact", "rank"):
                if key in entry:
                    state[key] = entry[key]
            return state
        if parsed.kind == "split":
            source = dv.original_id(parsed, self.known)
            path = self.path_of(source)
            if not path or not os.path.isfile(path):
                return {"state": "missing", "error": "The original is not here."}
        return {"state": "pending"}

    # ------------------------------------------------------------ building

    def job(self, stack, native_id: str) -> bd.Job | None:
        parsed = dv.parse(native_id)
        if parsed is None or self.store is None:
            return None
        if parsed.kind == "split":
            if not self.sided:
                return None
            source = dv.original_id(parsed, self.known)
            path = self.path_of(source)
            if not path or not os.path.isfile(path):
                return None
            summary = self.summary(source)
            if summary is not None and not summary.has(parsed.mode):
                return None
            return bd.split_job(self.store, native_id, source, path, parsed.mode, display_name(source))
        group = sm.group_of(stack, native_id)
        track = next((key for key, value in group.bakes.items() if value == native_id), None) if group else None
        if track is None:
            # Nothing in the stack says what it holds: its recipe does.
            return self.recipe_job(native_id, parsed.rank)
        members = self._members(group, track)
        if members is None or len(members) < 2:
            return None
        name = "Shrunk" if group.weighted else "Group"
        phase = dv.phase_of(track)
        label = f"{name} {group.letter}" + (f", phase {phase}" if phase else
                                            {sm.PICTURE: ", picture", sm.SOUND: ", sound"}.get(track, ""))
        return bd.bake_job(self.store, native_id, members, parsed.rank, self.sided, label,
                           shrunk=group.weighted, track=track if phase else "")

    def _recipe_members(self, recipe: dict) -> list[tuple[str, str, str, str, float]] | None:
        found = []
        for item in recipe.get("members") or []:
            if not isinstance(item, dict):
                continue
            source = normalize_id(item.get("id"))
            path = self.path_of(source) if source else None
            if not path or not os.path.isfile(path):
                return None
            try:
                weight = float(item.get("weight", 1.0))
            except (TypeError, ValueError):
                return None
            found.append((source, path, str(item.get("mode") or sm.NORMAL),
                          str(item.get("name") or display_name(source)), weight))
        return found

    def recipe_job(self, native_id: str, rank: int) -> bd.Job | None:
        """A bake built again from the recipe kept when it was first built."""
        if self.store is None:
            return None
        try:
            recipe = self.store.recipe(native_id)
        except OSError:
            return None
        if not recipe or recipe.get("kind") != "bake":
            return None
        members = self._recipe_members(recipe)
        if members is None or len(members) < 2:
            return None
        return bd.bake_job(self.store, native_id, members, rank, self.sided,
                           str(recipe.get("label") or "Baked file"), shrunk=bool(recipe.get("shrink")),
                           track=str(recipe.get("track") or ""))

    def restore_groups(self, stack) -> bool:
        """Give a baked file the stack names without the lines that say what
        it holds -- a setting from somewhere that dropped them -- its group
        back, from the recipe kept when it was built."""
        if self.store is None:
            return False
        held = {value for group in sm.groups(stack).values() for value in group.files}
        named = {value for members in stack.groups.values() for value in members}
        # Each orphan bake, as its recipe says: members, weights, phase.  A
        # group baked a file per phase comes back as one group with them all.
        found: list[tuple[str, list[str], dict[str, float], bool, int]] = []
        for lora_id, _, _ in stack.user_entries():
            parsed = dv.parse(lora_id)
            if parsed is None or parsed.kind != "bake" or lora_id in held:
                continue
            try:
                recipe = self.store.recipe(lora_id)
            except OSError:
                return False
            if not recipe or recipe.get("kind") != "bake":
                continue
            members, weights = [], {}
            for item in recipe.get("members") or []:
                if not isinstance(item, dict) or not item.get("id"):
                    continue
                source, mode = normalize_id(item.get("id")), str(item.get("mode") or sm.NORMAL)
                native = dv.split_id(source, mode) if mode in dv.MODES else source
                members.append(native)
                try:
                    weights[native] = float(item.get("weight", 1.0))
                except (TypeError, ValueError):
                    weights[native] = 1.0
            if len(members) >= 2:
                found.append((lora_id, members, weights, bool(recipe.get("shrink")), dv.phase_of(parsed.track)))
        changed = False
        done: set[str] = set()
        for lora_id, members, weights, shrunk, phase in found:
            if lora_id in done:
                continue
            files, per_phase = [lora_id], {phase: weights}
            if phase:
                # Its other phases' files: the same LoRAs, a phase of their own.
                for other, others, other_weights, _, other_phase in found:
                    if other not in done and other != lora_id and other_phase and other_phase not in per_phase \
                            and others == members:
                        files.append(other)
                        per_phase[other_phase] = other_weights
            done.update(files)
            if any(stack.has_entry(member) or member in named for member in members):
                continue
            letter = sm.next_letter(stack)
            if not letter:
                break
            stack.groups[letter] = members + files
            if phase:
                count = max(per_phase)
                stack.weights[letter] = {
                    member: [per_phase.get(index, {}).get(member, 0.0) for index in range(1, count + 1)]
                    for member in members
                }
            elif shrunk or any(abs(weight - 1.0) > 1e-9 for weight in weights.values()):
                stack.weights[letter] = {member: [weight] for member, weight in weights.items()}
            named.update(members)
            changed = True
        return changed

    def ensure(self, stack) -> None:
        """Queue every file the stack selects and does not have ready."""
        if self.store is None:
            return
        for native_id in list(stack.ids):
            if dv.parse(native_id) is None or self.builder.busy(native_id):
                continue
            try:
                if self.store.ready(native_id):
                    continue
            except OSError:
                return
            job = self.job(stack, native_id)
            if job is not None:
                self.builder.submit(job)

    def waiting(self, stack) -> list[str]:
        """Selected built files that are not on disk yet."""
        if self.store is None:
            return []
        return [
            native_id for native_id in stack.ids
            if dv.parse(native_id) is not None and not self.store.ready(native_id)
        ]

    # ---------------------------------------------------------------- rows

    def describe(self, native_id: str, catalogue: dict | None = None) -> dict:
        """Name and catalogue fields for a LoRA that has no row of its own --
        a member of a baked group."""
        source, _ = self.source(native_id)
        entry = self.inventory.get(source)
        index = (catalogue or {}).get(source)
        fallback = source.rsplit("/", 1)[-1]
        return {
            "id": native_id,
            "name": entry.name if entry else fallback,
            "label": entry.label if entry else fallback,
            "civitai_name": getattr(index, "civitai_name", "") if index else "",
            "has_catalogue": bool(getattr(index, "has_sidecar", False)) if index else False,
            "missing": entry is None,
        }

    def _decorate(self, payload: dict, native_id: str, catalogue: dict | None, skipped: set[str]) -> None:
        source, mode = self.source(native_id)
        payload["source_id"] = source
        payload["mode"] = mode
        payload["skipped"] = native_id in skipped
        if dv.parse(native_id) is not None and dv.parse(native_id).kind == "split":
            # The copy shows as its original: name, picture, catalogue.
            payload.update({key: value for key, value in self.describe(native_id, catalogue).items() if key != "id"})
        if self.sided:
            summary = self.summary(source)
            payload["streams"] = summary.to_dict() if summary is not None else None
        payload["empty"] = self.empty(native_id)
        payload["build"] = self.state(native_id)

    def decorate(self, rows: list[dict], stack, catalogue: dict | None = None) -> list[dict]:
        skipped = stack.skipped()
        found = sm.groups(stack)
        for row in rows:
            if row.get("kind") != "group":
                self._decorate(row, row["id"], catalogue, skipped)
                sound = row.get("sound_track")
                if sound:
                    # Two halves, one LoRA: built when both are.
                    self._decorate(sound, sound["id"], catalogue, skipped)
                    row["mode"] = sm.SEPARATE
                    row["build"] = combine([dict(row["build"], id=row["id"], track=sm.PICTURE),
                                            dict(sound["build"], id=sound["id"], track=sm.SOUND)])
                    row["empty"] = row["empty"] and sound["empty"]
                continue
            for member in row.get("members", []):
                self._decorate(member, member["id"], catalogue, skipped)
                entries = member.get("entries") or [member["id"]]
                member["mode"] = member.get("member_mode") or member["mode"]
                if len(entries) > 1:
                    member["build"] = combine([dict(self.state(value), id=value) for value in entries])
                    member["empty"] = all(self.empty(value) for value in entries)
            group = found.get(row["group"])
            if group is None:
                continue
            info = {"default_rank": self.default_rank(group)}
            if group.baked:
                parsed = dv.parse(group.bake)
                info.update(combine([dict(self.state(value), id=value, track=track)
                                     for track, value in group.bakes.items()]))
                info["rank"] = parsed.rank if parsed else 0
            row["bake"] = info
            row["source_id"] = ""
            row["mode"] = sm.NORMAL
        return rows

    def aliases(self, stack) -> dict[str, tuple[str, bool]]:
        """Original id -> ``(the id the stack holds it by, whether it is on)``,
        for the tile grid: a LoRA in visual-only mode, or baked into a group,
        is still that LoRA's tile."""
        found: dict[str, tuple[str, bool]] = {}
        for lora_id, on, _ in stack.user_entries():
            source, _ = self.source(lora_id)
            if dv.parse(lora_id) is None or dv.parse(lora_id).kind == "split":
                found.setdefault(source, (lora_id, on))
        for group in sm.groups(stack).values():
            if group.baked:
                on = any(stack.index_of(value) >= 0 for value in group.files)
                for member in group.members:
                    found.setdefault(self.source(member)[0], (member, on))
        return found


#: Which state a set of files is in, worst first: what a row shows for two
#: halves, or for a group baked into a file per side.
_ORDER = ("failed", "missing", "unavailable", "building", "queued", "pending", "built", "original")


def combine(states: list[dict]) -> dict:
    """One state for several files: the furthest from ready, with the sizes
    of all of them and the share the least faithful one kept."""
    if not states:
        return {"state": "original"}
    worst = min(states, key=lambda state: _ORDER.index(state.get("state", "original"))
                if state.get("state", "original") in _ORDER else 0)
    merged = dict(worst)
    sizes = [state["bytes"] for state in states if "bytes" in state]
    if sizes:
        merged["bytes"] = sum(sizes)
    kept = [state["kept"] for state in states if state.get("kept") is not None]
    if kept:
        merged["kept"] = min(kept)
    exact = [state["exact"] for state in states if "exact" in state]
    if exact:
        merged["exact"] = all(exact)
    merged["files"] = states
    return merged
