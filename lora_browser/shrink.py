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
from .utils import display_name

FLAT = "flat"
SHAPED = "shaped"
#: A phase where a LoRA runs one strength.
ONE = "c"


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
    #: The stack entry it rides on: itself, or its group's (first) baked file.
    holder: str
    #: The group it comes from, "" for a LoRA of its own.
    origin: str = ""
    name: str = ""
    rank: int = 0
    #: Which LoRAs it can share files with: the shape of each phase -- one
    #: strength (``ONE``) or its exact steps -- or one schedule over the run.
    key: tuple = ()
    #: What it brings to its file in each phase: that phase's strength where
    #: it runs one, 1 where it runs steps.
    weights: list[float] = field(default_factory=list)
    #: Why it stays as it is; "" while it can go in.
    reason: str = ""

    @property
    def unit(self) -> str:
        """One key per LoRA: its two halves share it."""
        return self.source

    def to_dict(self) -> dict:
        found = {"id": self.entry, "source": self.source, "mode": self.mode, "name": self.name,
                 "token": self.token, "rank": self.rank}
        if self.weights:
            found["weights"] = list(self.weights)
            if len({round(value, 6) for value in self.weights}) == 1:
                found["weight"] = self.weights[0]
        if self.reason:
            found["reason"] = self.reason
        return found


@dataclass
class Bundle:
    """The LoRAs shrunk into one group: one file, or one file per phase."""

    key: tuple
    pieces: list[Piece] = field(default_factory=list)
    rank: int = 0
    #: Bytes the LoRAs' factors take now, and the baked files'.
    bytes_before: int = 0
    bytes_after: int = 0
    #: How many phases a token speaks for here.
    count: int = 1

    @property
    def loras(self) -> int:
        return len({piece.unit for piece in self.pieces})

    @property
    def run(self) -> bool:
        """One schedule over the whole run, not phase by phase."""
        return bool(self.key) and self.key[0] == "run"

    @property
    def kind(self) -> str:
        return FLAT if not self.run and all(shape == ONE for shape in self.key[1]) else SHAPED

    @property
    def schedule(self) -> bool:
        """Its LoRAs share steps, not only strengths."""
        return self.kind == SHAPED

    @property
    def phased(self) -> bool:
        """A file per phase: what its LoRAs bring differs from phase to phase."""
        return any(len({round(value, 6) for value in piece.weights}) > 1 for piece in self.pieces)

    def parts(self) -> list[list[float]]:
        """The group's multiplier, phase by phase: 1 where its LoRAs run one
        strength (their weights hold it), their steps where they share steps."""
        if self.run:
            return [list(self.key[1])]
        return [[1.0] if shape == ONE else list(shape) for shape in self.key[1]]

    @property
    def token(self) -> str:
        """What the group runs at: its one file's multiplier, or each phase's
        from that phase's file."""
        return codec.compose_token(self.parts())

    def tracks(self) -> list[tuple[str, str]]:
        """``(track, multiplier)`` per file: the one file, or -- phase by
        phase, for each phase some LoRA brings something to -- a file that
        runs in its phase only."""
        if not self.phased:
            return [(sm.ALL, self.token)]
        found = []
        for index in range(min(self.count, len(dv.PHASE_TRACKS))):
            if any(piece.weights[min(index, len(piece.weights) - 1)] != 0 for piece in self.pieces):
                found.append((dv.PHASE_TRACKS[index], codec.pin_token(self.token, index + 1, self.count)))
        return found or [(dv.PHASE_TRACKS[0], codec.pin_token(self.token, 1, self.count))]

    @property
    def worth(self) -> bool:
        """It leaves fewer files than the LoRAs it holds: two LoRAs at least,
        and more LoRAs than it has files -- a file per phase for two LoRAs
        would be as many files as before."""
        return self.loras >= 2 and self.loras > len(self.tracks())

    def weights(self) -> dict[str, list[float]]:
        """What each LoRA brings: one weight, or one per phase."""
        if self.phased:
            return {piece.entry: list(piece.weights) for piece in self.pieces}
        return {piece.entry: [piece.weights[0] if piece.weights else 1.0] for piece in self.pieces}

    def to_dict(self) -> dict:
        return {
            "kind": self.kind, "token": self.token, "rank": self.rank, "loras": self.loras,
            "schedule": self.schedule, "phased": self.phased,
            "files": [{"phase": dv.phase_of(track), "token": token} for track, token in self.tracks()],
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


def _classify(piece: Piece, capacity: int) -> None:
    """A piece's shape, phase by phase, and what it brings to each phase."""
    values = codec.token_values(piece.token)
    if values is None:
        piece.reason = "Its multiplier uses syntax a shrink cannot scale."
        return
    every = [round(value, 4) for phase in values for value in phase]
    constant = all(value == every[0] for value in every)
    if capacity > 1 and len(values) == 1 and len(values[0]) > 1 and not constant:
        # A comma list with no phases: WanGP spreads it over the whole run.
        piece.key = ("run", tuple(round(value, 4) for value in values[0]))
        piece.weights = [1.0]
        return
    count = max(capacity, len(values), 1)
    shapes, weights = [], []
    for part in codec.token_parts(piece.token, count):
        rounded = [round(value, 4) for value in part]
        if all(value == rounded[0] for value in rounded):
            shapes.append(ONE)
            weights.append(rounded[0])
        else:
            shapes.append(tuple(rounded))
            weights.append(1.0)
    piece.key = ("phases", tuple(shapes))
    piece.weights = weights


def collect(stack, files) -> tuple[list[Piece], int]:
    """Every piece the stack runs on the user's side, and how many files
    WanGP loads for it now."""
    found = sm.groups(stack)
    by_member = {member: group for group in found.values() for member in group.members}
    by_bake = {value: (group, track) for group in found.values() for track, value in group.bakes.items()}
    pieces: list[Piece] = []
    seen: set[str] = set()
    loaded = 0
    for lora_id, on, token in stack.user_entries():
        if not on:
            continue
        loaded += 1
        if lora_id in by_bake:
            group, track = by_bake[lora_id]
            if group.weighted:
                # A shrunk group is its LoRAs, each at what it runs at --
                # once, however many files it has.
                if group.letter in seen:
                    continue
                seen.add(group.letter)
                for member in files.bakeable(group, track):
                    pieces.append(_piece(files, member, sm.member_token(stack, group, member), lora_id, group.letter))
                continue
            for member in files.bakeable(group, track):
                pieces.append(_piece(files, member, token, lora_id, group.letter))
            continue
        group = by_member.get(lora_id)
        pieces.append(_piece(files, lora_id, token, lora_id, group.letter if group else ""))
    return pieces, loaded


def _piece(files, entry: str, token: str, holder: str, origin: str) -> Piece:
    source, mode = files.source(entry)
    summary = files.summary(source)
    return Piece(entry=entry, source=source, mode=mode, token=token, holder=holder, origin=origin,
                 name=display_name(source), rank=int(getattr(summary, "rank", 0) or 0))


def _check(piece: Piece, files, capacity: int) -> None:
    path = files.path_of(piece.source)
    recognised = dv.recognised_by_name(piece.source)
    if files.empty(piece.entry):
        piece.reason = "Its mode leaves it nothing to apply."
    elif recognised:
        piece.reason = (f"WanGP recognises it by its file name ('{recognised}'); "
                        "inside a shrunk file it would not.")
    elif not path or not os.path.isfile(path):
        piece.reason = "Its file is not on this machine."
    else:
        problem = bk.check(bk.Member(path, piece.mode, piece.name), sided=files.sided)
        if problem:
            piece.reason = problem
    if not piece.reason:
        _classify(piece, capacity)


# -------------------------------------------------------------------- plan


def plan(stack, files, capacity: int = 1) -> Plan:
    """What a shrink would make of ``stack`` in a ``capacity``-phase run:
    its files, and what stays."""
    pieces, loaded = collect(stack, files)
    for piece in pieces:
        _check(piece, files, capacity)
    result = Plan(files_before=loaded)
    bundles: dict[tuple, Bundle] = {}
    for piece in pieces:
        if not piece.reason:
            bundle = bundles.get(piece.key)
            if bundle is None:
                bundle = bundles[piece.key] = Bundle(piece.key, count=max(capacity, len(piece.weights), 1))
            bundle.pieces.append(piece)
    listed = list(bundles.values())

    # A file needs two LoRAs; a LoRA goes in whole -- both halves -- or not.
    broken = {piece.unit for piece in pieces if piece.reason}
    while True:
        for bundle in [bundle for bundle in listed if not bundle.worth]:
            for piece in bundle.pieces:
                broken.add(piece.unit)
                if piece.reason:
                    continue
                if bundle.loras >= 2:
                    piece.reason = (f"A file per phase would be {len(bundle.tracks())} files "
                                    f"for {bundle.loras} LoRAs: no fewer than now.")
                else:
                    piece.reason = ("No other LoRA runs at one strength." if bundle.kind == FLAT
                                    else "No other LoRA runs this schedule.")
        moved = False
        for bundle in listed:
            keep = [piece for piece in bundle.pieces if piece.unit not in broken]
            for piece in bundle.pieces:
                if piece.unit in broken and not piece.reason:
                    piece.reason = "Its other side stays as it is."
            if len(keep) != len(bundle.pieces):
                bundle.pieces = keep
                moved = True
        listed = [bundle for bundle in listed if bundle.pieces]
        if not moved and all(bundle.worth for bundle in listed):
            break

    for bundle in listed:
        bundle.rank = max((piece.rank for piece in bundle.pieces), default=0)
        members = [bk.Member(files.path_of(piece.source), piece.mode, piece.name) for piece in bundle.pieces]
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
        bundle.bytes_after = baked.size_for(bundle.rank) * len(bundle.tracks())
    result.bundles = [bundle for bundle in listed if bundle.pieces][:len(sm.LETTERS)]
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


def apply(stack, result: Plan, files) -> dict[str, list[str]]:
    """Rewrite ``stack`` as ``result`` plans it: each bundle a shrunk group
    whose baked files -- one, or one per phase -- stand where its first LoRA
    stood.  A group that gave a LoRA to a bundle is taken apart; what it does
    not give stays as LoRAs of their own, at what they ran at.  Returns
    ``letter -> its baked files``."""
    if not result.bundles:
        return {}
    found = sm.groups(stack)

    # Each bundle a letter and its files: the letters of groups that stay are
    # theirs, and those of groups a bundle takes apart are free again.
    giving = {piece.origin for bundle in result.bundles for piece in bundle.pieces if piece.origin}
    letters = [letter for letter in sm.LETTERS if letter not in stack.groups or letter in giving]
    made: dict[str, list[str]] = {}
    shrunk: dict[int, tuple[str, list[tuple[str, str]], sm.Group]] = {}
    for index, bundle in enumerate(result.bundles):
        if not letters:
            break
        group = sm.Group(letters[0], [piece.entry for piece in bundle.pieces], {}, bundle.weights())
        ids = files.bake_ids(group, bundle.rank)
        placed_files = [(ids[track], token) for track, token in bundle.tracks() if ids.get(track)]
        if placed_files:
            letter = letters.pop(0)
            shrunk[index] = (letter, placed_files, group)
            made[letter] = [value for value, _ in placed_files]
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
    done: set[str] = set()
    rows = []

    def place(index: int, on: bool) -> None:
        if index not in placed:
            placed.add(index)
            rows.extend((value, on, token) for value, token in shrunk[index][1])

    for lora_id, on, token in entries:
        owner = bakes.get(lora_id)
        if owner is not None and owner.letter in apart:
            if owner.letter in done:
                continue                     # a file per phase: taken apart once
            done.add(owner.letter)
            for member in owner.members:
                if member in taken:
                    place(taken[member], on)
            for value in owner.files:
                for piece in staying.get(value, []):
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
    for letter, placed_files, group in shrunk.values():
        groups_now[letter] = list(group.members) + [value for value, _ in placed_files]
        weights_now[letter] = {member: list(values) for member, values in group.weights.items()}
    stack.groups = groups_now
    stack.weights = weights_now
    return made
