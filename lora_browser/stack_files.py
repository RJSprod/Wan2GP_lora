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
        changes the picture -- brings the same as Normal."""
        source, mode = self.source(native_id)
        if not self.sided:
            return "all"
        summary = self.summary(source)
        if summary is None:
            return mode
        picture = summary.picture.tensors > 0 and mode != sm.AUDIO
        sound = summary.sound.tensors > 0 and mode != sm.VISUAL
        return ("p" if picture else "") + ("s" if sound else "") or "-"

    # --------------------------------------------------------------- bakes

    def bake_id(self, group: sm.Group, rank: int) -> str:
        return dv.bake_id([(self.source(member)[0], self.sides(member)) for member in group.members], rank)

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

    def _members(self, group: sm.Group) -> list[tuple[str, str, str, str]] | None:
        found = []
        for member in group.members:
            source, mode = self.source(member)
            path = self.path_of(source)
            if not path or not os.path.isfile(path):
                return None
            found.append((source, path, mode, display_name(source)))
        return found

    def bake_options(self, stack, letter: str) -> dict:
        """What the bake dialog offers: sizes per rank, read from headers only."""
        group = sm.groups(stack).get(letter)
        if group is None:
            return {"error": "That group is gone."}
        members = self._members(group)
        if members is None:
            return {"error": "A LoRA in this group is missing locally."}
        try:
            plan = bk.plan([bk.Member(path, mode, name) for _, path, mode, name in members], sided=self.sided)
        except bk.BakeError as error:
            return {"error": str(error)}
        combined = plan.combined_rank
        default = min(plan.default_rank or combined, combined)
        ranks = sorted({combined, default, max(1, default // 2)}, reverse=True)
        return {
            "group": letter,
            "combined_rank": combined,
            "default_rank": default,
            "current_bytes": plan.size_for(combined),
            "options": [{"rank": rank, "bytes": plan.size_for(rank), "exact": rank >= combined} for rank in ranks],
            "baked_rank": dv.parse(group.bake).rank if group.baked else 0,
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
        if group is None or group.bake != native_id:
            return None
        members = self._members(group)
        if members is None:
            return None
        return bd.bake_job(self.store, native_id, members, parsed.rank, self.sided, f"Group {group.letter}")

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
                continue
            for member in row.get("members", []):
                self._decorate(member, member["id"], catalogue, skipped)
            group = found.get(row["group"])
            if group is None:
                continue
            info = {"default_rank": self.default_rank(group)}
            if group.baked:
                parsed = dv.parse(group.bake)
                info.update(self.state(group.bake), rank=parsed.rank if parsed else 0)
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
                on = stack.index_of(group.bake) >= 0
                for member in group.members:
                    found.setdefault(self.source(member)[0], (member, on))
        return found
