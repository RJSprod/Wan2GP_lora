"""Shrink: bake a stack down into as few LoRA files as run the same.

Every LoRA that runs at one strength -- the same value in every phase and at
every step -- goes into one file, each baked in at that strength, and the
file runs at 1: a LoRA at 0.7 still brings 0.7.  LoRAs that share exactly
the same schedule (or the same strength per phase) go into a file of their
own per schedule, each at 1, and that file runs the schedule: the sum of
LoRAs that each follow a schedule is that sum following it.  A LoRA with no
other on its schedule stays as it is.

Each file is fitted to the rank of the highest-rank LoRA in it.  A LoRA in a
mode is baked as that mode runs it -- an audio-only LoRA brings its sound
side, a LoRA with separate picture and sound strengths its picture half at
the one and its sound half at the other -- and its two halves go into the
shrink together or not at all, so a sound half never runs on its own.

What goes in is what the stack runs: LoRAs that are on, on the user's side
of the bar.  A group's members come with the group's strength; a group
already baked, or shrunk before, comes as its members, so shrinking again
re-plans everything from the originals.  Left as they are, and said why:
LoRAs switched off, WanGP's own, multipliers in syntax that cannot be scaled,
LoRAs not on this machine, files a bake cannot combine (DoRA, LoKr...), and
LoRAs WanGP recognises by their file name (``derived.recognised_by_name``):
in a file with a name of its own, WanGP would no longer treat them so.

The result is a shrunk group per file (``stack_modes``): ``#lora_group``
lines naming its LoRAs, ``#lora_weight`` lines saying what each brings, and
the baked file in their place -- everything needed to build it again, kept
in the multiplier text that presets, the queue and saved settings keep.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field

from . import bake as bk
from . import derived as dv
from . import multiplier_codec as codec
from . import stack_modes as sm
from .utils import display_name, format_number

FLAT = "flat"
SHAPED = "shaped"


@dataclass
class Piece:
    """One thing the stack runs, as a shrink sees it."""

    #: The id the stack holds it by: an original, a mode's copy, or a half.
    entry: str
    source: str
    mode: str
    #: What it runs at now -- inside a shrunk group, its weight times the
    #: group's multiplier.
    token: str
    #: The stack entry it rides on: itself, or its group's baked file.
    holder: str
    #: The group it comes from, "" for a LoRA of its own.
    origin: str = ""
    name: str = ""
    rank: int = 0
    #: Its one strength, for a LoRA that runs at one; None otherwise.
    weight: float | None = None
    #: The schedule it shares, for a LoRA that does not run at one strength.
    shape: str = ""
    #: Why it stays as it is; "" while it can go in.
    reason: str = ""

    @property
    def unit(self) -> str:
        """One key per LoRA: its two halves share it."""
        return self.source

    def to_dict(self) -> dict:
        found = {"id": self.entry, "source": self.source, "mode": self.mode, "name": self.name,
                 "token": self.token, "rank": self.rank}
        if self.weight is not None:
            found["weight"] = self.weight
        if self.reason:
            found["reason"] = self.reason
        return found


@dataclass
class Bundle:
    """The LoRAs one shrunk file holds."""

    kind: str
    #: What the baked file runs at: 1, or the schedule its LoRAs share.
    token: str
    pieces: list[Piece] = field(default_factory=list)
    rank: int = 0
    #: Bytes the LoRAs' factors take now, and the baked file's.
    bytes_before: int = 0
    bytes_after: int = 0

    @property
    def loras(self) -> int:
        return len({piece.unit for piece in self.pieces})

    @property
    def schedule(self) -> bool:
        """Its LoRAs share a schedule over the steps, not just a strength
        per phase."""
        return "," in self.token

    def weights(self) -> dict[str, float]:
        return {piece.entry: (piece.weight if self.kind == FLAT else 1.0) for piece in self.pieces}

    def to_dict(self) -> dict:
        return {
            "kind": self.kind, "token": self.token, "rank": self.rank, "loras": self.loras,
            "schedule": self.schedule,
            "bytes_before": self.bytes_before, "bytes_after": self.bytes_after,
            "members": [piece.to_dict() for piece in self.pieces],
        }


@dataclass
class Plan:
    bundles: list[Bundle] = field(default_factory=list)
    #: LoRAs that are on and stay as they are, each with why.
    kept: list[Piece] = field(default_factory=list)
    #: Files WanGP loads for the user's LoRAs now, and after the shrink.
    files_before: int = 0
    files_after: int = 0

    def to_dict(self) -> dict:
        return {
            "bundles": [bundle.to_dict() for bundle in self.bundles],
            "kept": [piece.to_dict() for piece in self.kept],
            "files_before": self.files_before, "files_after": self.files_after,
        }


# ------------------------------------------------------------------ pieces


def _values_key(values: list[list[float]]) -> str:
    return ";".join(",".join(format_number(value) for value in phase) for phase in values)


def _classify(piece: Piece) -> None:
    values = codec.token_values(piece.token)
    if values is None:
        piece.reason = "Its multiplier uses syntax a shrink cannot scale."
        return
    every = [round(value, 4) for phase in values for value in phase]
    if every and all(value == every[0] for value in every):
        piece.weight = every[0]
    else:
        piece.shape = _values_key(values)


def collect(stack, files) -> tuple[list[Piece], int]:
    """Every piece the stack runs on the user's side, and how many files
    WanGP loads for it now."""
    found = sm.groups(stack)
    by_member = {member: group for group in found.values() for member in group.members}
    by_bake = {value: (group, track) for group in found.values() for track, value in group.bakes.items()}
    pieces: list[Piece] = []
    loaded = 0
    for lora_id, on, token in stack.user_entries():
        if not on:
            continue
        loaded += 1
        if lora_id in by_bake:
            group, track = by_bake[lora_id]
            for member in files.bakeable(group, track):
                ran = sm._weighted_token(token, group.weight(member)) if group.weighted else token
                pieces.append(_piece(files, member, ran, lora_id, group.letter))
            continue
        group = by_member.get(lora_id)
        pieces.append(_piece(files, lora_id, token, lora_id, group.letter if group else ""))
    return pieces, loaded


def _piece(files, entry: str, token: str, holder: str, origin: str) -> Piece:
    source, mode = files.source(entry)
    summary = files.summary(source)
    piece = Piece(entry=entry, source=source, mode=mode, token=token, holder=holder, origin=origin,
                  name=display_name(source), rank=int(getattr(summary, "rank", 0) or 0))
    path = files.path_of(source)
    recognised = dv.recognised_by_name(source)
    if files.empty(entry):
        piece.reason = "Its mode leaves it nothing to apply."
    elif recognised:
        piece.reason = (f"WanGP recognises it by its file name ('{recognised}'); "
                        "inside a shrunk file it would not.")
    elif not path or not os.path.isfile(path):
        piece.reason = "Its file is not on this machine."
    else:
        problem = bk.check(bk.Member(path, mode, piece.name), sided=files.sided)
        if problem:
            piece.reason = problem
    if not piece.reason:
        _classify(piece)
    return piece


# -------------------------------------------------------------------- plan


def plan(stack, files) -> Plan:
    """What a shrink would make of ``stack``: its files, and what stays."""
    pieces, loaded = collect(stack, files)
    result = Plan(files_before=loaded)
    good = [piece for piece in pieces if not piece.reason]
    bundles = [Bundle(FLAT, "1", [piece for piece in good if piece.weight is not None])]
    shaped: dict[str, Bundle] = {}
    for piece in good:
        if piece.weight is None:
            bundle = shaped.get(piece.shape)
            if bundle is None:
                bundle = shaped[piece.shape] = Bundle(SHAPED, piece.token)
            bundle.pieces.append(piece)
    bundles += list(shaped.values())

    # A file needs two LoRAs; a LoRA goes in whole -- both halves -- or not.
    broken = {piece.unit for piece in pieces if piece.reason}
    while True:
        alone = [bundle for bundle in bundles if bundle.loras < 2]
        for bundle in alone:
            for piece in bundle.pieces:
                broken.add(piece.unit)
                if not piece.reason:
                    piece.reason = ("No other LoRA runs at one strength." if bundle.kind == FLAT
                                    else "No other LoRA runs this schedule." if bundle.schedule
                                    else "No other LoRA has these strengths per phase.")
        moved = False
        for bundle in bundles:
            keep = [piece for piece in bundle.pieces if piece.unit not in broken]
            for piece in bundle.pieces:
                if piece.unit in broken and not piece.reason:
                    piece.reason = "Its other side stays as it is."
            if len(keep) != len(bundle.pieces):
                bundle.pieces = keep
                moved = True
        bundles = [bundle for bundle in bundles if bundle.pieces]
        if not moved and not any(bundle.loras < 2 for bundle in bundles):
            break

    for bundle in bundles:
        bundle.rank = max((piece.rank for piece in bundle.pieces), default=0)
        members = [bk.Member(files.path_of(piece.source), piece.mode, piece.name,
                             piece.weight if bundle.kind == FLAT else 1.0) for piece in bundle.pieces]
        try:
            baked = bk.plan(members, sided=files.sided)
        except bk.BakeError as error:
            for piece in bundle.pieces:
                piece.reason = str(error)
            bundle.pieces = []
            continue
        if bundle.rank < 1:
            bundle.rank = baked.default_rank or baked.combined_rank
        bundle.bytes_before = baked.size_for(baked.combined_rank)
        bundle.bytes_after = baked.size_for(bundle.rank)
    result.bundles = [bundle for bundle in bundles if bundle.pieces][:len(sm.LETTERS)]
    taken = {id(piece) for bundle in result.bundles for piece in bundle.pieces}
    result.kept = [piece for piece in pieces if id(piece) not in taken]
    for piece in result.kept:
        if not piece.reason:
            piece.reason = "Its other side stays as it is."
    # What WanGP would load after: counted on a copy, shrunk.
    trial = copy.deepcopy(stack)
    apply(trial, result, files)
    result.files_after = sum(1 for _, on, _ in trial.user_entries() if on)
    return result


# ------------------------------------------------------------------- apply


def apply(stack, result: Plan, files) -> dict[str, str]:
    """Rewrite ``stack`` as ``result`` plans it: each bundle a shrunk group
    whose baked file stands where its first LoRA stood.  A group that gave a
    LoRA to a bundle is taken apart; what it does not give stays as LoRAs of
    their own, at what they ran at.  Returns ``letter -> baked file id``."""
    if not result.bundles:
        return {}
    found = sm.groups(stack)

    # Each bundle a letter and a file: the letters of groups that stay are
    # theirs, and those of groups a bundle takes apart are free again.
    giving = {piece.origin for bundle in result.bundles for piece in bundle.pieces if piece.origin}
    letters = [letter for letter in sm.LETTERS if letter not in stack.groups or letter in giving]
    made: dict[str, str] = {}
    shrunk: dict[int, tuple[str, str, sm.Group]] = {}
    for index, bundle in enumerate(result.bundles):
        if not letters:
            break
        group = sm.Group(letters[0], [piece.entry for piece in bundle.pieces], {}, bundle.weights())
        baked = files.bake_ids(group, bundle.rank).get(sm.ALL, "")
        if baked:
            letter = letters.pop(0)
            shrunk[index] = (letter, baked, group)
            made[letter] = baked
    taken = {piece.entry: index for index in shrunk for piece in result.bundles[index].pieces}
    apart = {piece.origin for index in shrunk for piece in result.bundles[index].pieces if piece.origin}
    # What a group taken apart does not give stays where its baked file was.
    staying: dict[str, list[Piece]] = {}
    for index, bundle in enumerate(result.bundles):
        if index not in shrunk:
            for piece in bundle.pieces:
                staying.setdefault(piece.holder, []).append(piece)
    for piece in result.kept:
        staying.setdefault(piece.holder, []).append(piece)

    entries, skipped = stack.user_entries(), stack.skipped()
    bakes = {value: group for group in found.values() for value in group.files}
    placed: set[int] = set()
    rows = []

    def place(index: int, on: bool) -> None:
        if index not in placed:
            placed.add(index)
            rows.append((shrunk[index][1], on, result.bundles[index].token))

    for lora_id, on, token in entries:
        owner = bakes.get(lora_id)
        if owner is not None and owner.letter in apart:
            for member in owner.members:
                if member in taken:
                    place(taken[member], on)
            for piece in staying.get(lora_id, []):
                rows.append((piece.entry, on, piece.token))
            continue
        if on and lora_id in taken:
            place(taken[lora_id], True)
            continue
        rows.append((lora_id, on, token))
    stack.set_user_entries(rows, skipped)
    groups_now = {letter: list(members) for letter, members in stack.groups.items() if letter not in apart}
    weights_now = {letter: dict(found_weights) for letter, found_weights in stack.weights.items()
                   if letter in groups_now}
    for letter, baked, group in shrunk.values():
        groups_now[letter] = list(group.members) + [baked]
        weights_now[letter] = dict(group.weights)
    stack.groups = groups_now
    stack.weights = weights_now
    return made
