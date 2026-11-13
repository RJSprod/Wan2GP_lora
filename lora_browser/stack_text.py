"""A LoRA stack as plain text: one ``name: x<multiplier>`` line per LoRA.

    Vocal_Enhance_1: x0,0,0,0,0.325,0.325,0,0;0
    Body_Structure_1: x1;0.25

The multiplier is a WanGP token exactly as ``loras_multipliers`` holds it --
``;`` between phases, ``,`` between steps, ``:`` between branches -- so a
stack pasted here means what the same tokens typed into WanGP would mean.
The ``x`` is optional, and so is the multiplier: a bare name loads at 1.
Blank lines are skipped, and so is a line starting with ``#`` or ``//``.

Names are matched against what WanGP offers the current model, most exact
first: the LoRA's own path with or without the extension, its filename with or
without the extension, its Civitai name, and last the same names with case,
spaces and punctuation ignored.  A name that two LoRAs answer to at the same
level is refused rather than guessed at, and so is a name nothing answers to.

``format_stack`` writes the same text back out, naming each LoRA by the
shortest name that reads back to it.

Tags in brackets after the name carry what the panel adds to a LoRA:

    Vocal_Enhance_1 [audio]: x0,0,0,0,0.325,0.325,0,0;0
    Body_Structure_1 [group A, baked r32]: x1;0.25
    Skin_Texture_v3 [visual, group A, baked r32]: x1;0.25

``visual``/``audio`` (or ``normal``) is the mode, ``group A`` the group it
moves with, and ``baked r32`` that the group is baked at rank 32.  Brackets
holding anything else are part of the name, as they always were.

A LoRA whose picture and sound have strengths of their own is two lines, one
per side:

    Rain_Ambience [picture]: x1;1
    Rain_Ambience [sound]: x0.4;0.2
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from .utils import is_finite_number, normalize_id

#: Name, separator, optional multiplication mark, token.  The name is lazy, so
#: the first ``:`` whose remainder is a whole token is the separator -- a
#: Civitai name with a colon of its own ("Detail Tweaker: XL: x0.8") still
#: splits where the multiplier starts, and so does a branch token ("0.5:0.9").
_LINE = re.compile(
    r"^(?P<name>.*?\S)\s*:\s*(?:[xX×*]\s*)?(?P<token>[-+.\d][-+.\deE,;:\s]*)$"
)
#: A name with nothing after it, or with a separator and no number yet.
_BARE = re.compile(r"^(?P<name>.*?\S)\s*(?::\s*[xX×*]?\s*)?$")
#: A list bullet in front of a line copied out of notes or a chat message.
_BULLET = re.compile(r"^[-*•]\s+")
_SPACE = re.compile(r"\s+")
_ATOMS = re.compile(r"[;,:]")
_LOOSE = re.compile(r"[\W_]+")
#: A name ending in a bracketed list, and the tags the list may hold.
_TAGGED = re.compile(r"^(?P<name>.*?\S)\s*\[(?P<tags>[^\[\]]*)\]$")
_GROUP_TAG = re.compile(r"^group\s+([a-z])$")
_BAKED_TAG = re.compile(r"^baked\s+r?([1-9]\d{0,4})$")
#: ``picture`` and ``sound`` are the two sides of a LoRA with separate
#: strengths; ``picture`` reads the same as ``visual``.
TAG_MODES = ("normal", "visual", "audio", "picture", "sound")


@dataclass(frozen=True)
class StackLine:
    """One LoRA line of a pasted stack."""

    #: 1-based line number in the pasted text.
    number: int
    #: The line as pasted, trimmed.
    text: str
    name: str
    #: The WanGP token with any spaces taken out, or "" for no multiplier.
    token: str
    #: From the tags: the mode ("" when not given), the group letter, and the
    #: rank the group is baked at (0 for not baked).
    mode: str = ""
    group: str = ""
    baked: int = 0


@dataclass(frozen=True)
class LineProblem:
    """Why one line of a pasted stack was not loaded."""

    number: int
    text: str
    reason: str

    def to_dict(self) -> dict:
        return {"line": self.number, "text": self.text, "reason": self.reason}


def parse(text: str) -> list[StackLine]:
    """Every LoRA line in ``text``, in order.

    Nothing here knows what LoRAs exist, so every line that is not blank, a
    comment or bare punctuation (a code fence, a rule) comes back as a line;
    resolving its name is what decides whether it loads.
    """
    lines = []
    for number, raw in enumerate(str(text or "").replace("\r", "").split("\n"), start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith(("#", "//")) or not _loose(stripped):
            continue
        body = _BULLET.sub("", stripped)
        match = _LINE.match(body)
        if match:
            name, token = match.group("name"), _SPACE.sub("", match.group("token"))
        else:
            bare = _BARE.match(body)
            name, token = (bare.group("name") if bare else body), ""
        name, mode, group, baked = split_tags(name.strip())
        lines.append(StackLine(number=number, text=stripped, name=name, token=token,
                               mode=mode, group=group, baked=baked))
    return lines


def split_tags(name: str) -> tuple[str, str, str, int]:
    """``(name, mode, group, baked rank)``: the tags taken off the end of a
    name, or the name untouched when its brackets hold anything else."""
    match = _TAGGED.match(str(name or ""))
    if not match:
        return name, "", "", 0
    mode, group, baked = "", "", 0
    for part in (piece.strip().lower() for piece in match.group("tags").split(",")):
        if not part:
            continue
        if part in TAG_MODES:
            mode = part
            continue
        found = _GROUP_TAG.match(part)
        if found:
            group = found.group(1).upper()
            continue
        found = _BAKED_TAG.match(part)
        if found:
            baked = int(found.group(1))
            continue
        return name, "", "", 0
    return match.group("name").strip(), mode, group, baked


def tags_text(mode: str = "", group: str = "", baked: int = 0) -> str:
    """`` [visual, group A, baked r32]``, or "" for a plain LoRA."""
    tags = []
    if mode and mode != "normal":
        tags.append(mode)
    if group:
        tags.append(f"group {group}")
        if baked:
            tags.append(f"baked r{int(baked)}")
    return f" [{', '.join(tags)}]" if tags else ""


def token_problem(token: str) -> str:
    """Why WanGP could not read ``token`` as a multiplier, or "" when it can.

    Only the syntax: every value between the separators must be a number.  How
    many phases a token may name depends on the model and its phase mode,
    which the caller knows.
    """
    if not token:
        return ""
    if not all(is_finite_number(atom) for atom in _ATOMS.split(token)):
        return f"'{token}' is not a multiplier WanGP can read"
    return ""


def phase_count(token: str) -> int:
    """How many phases a token names: one more than it has ``;``."""
    return token.count(";") + 1 if token else 1


def _loose(text: str) -> str:
    """``text`` with case, spaces and punctuation ignored."""
    return _LOOSE.sub("", str(text or "").casefold())


class NameIndex:
    """The LoRAs one model offers, by every name a pasted line might use."""

    #: Most exact first: the first level that knows a name decides it.  A path
    #: without its extension ("styles/c") is still a path, so it is matched
    #: before a filename shared by every folder that has one.
    LEVELS = ("path", "path_ci", "path_stem", "file", "stem", "civitai", "loose")

    def __init__(self, inventory, civitai_names: dict[str, str] | None = None) -> None:
        names = civitai_names or {}
        self._levels: dict[str, dict[str, dict[str, None]]] = {level: {} for level in self.LEVELS}
        self._labels: dict[str, str] = {}
        for entry in inventory.entries:
            filename = entry.name
            stem = os.path.splitext(filename)[0] or filename
            civitai = str(names.get(entry.id, "") or "").strip()
            self._labels[entry.id] = filename
            self._add("path", entry.id, entry.id)
            self._add("path_ci", entry.id.casefold(), entry.id)
            self._add("path_stem", os.path.splitext(entry.id)[0].casefold(), entry.id)
            self._add("file", filename.casefold(), entry.id)
            self._add("stem", stem.casefold(), entry.id)
            if civitai:
                self._add("civitai", civitai.casefold(), entry.id)
            for name in (stem, filename, civitai):
                if _loose(name):
                    self._add("loose", _loose(name), entry.id)

    def _add(self, level: str, key: str, lora_id: str) -> None:
        self._levels[level].setdefault(key, {})[lora_id] = None

    def label(self, lora_id: str) -> str:
        """The filename a pasted name resolved to, for messages."""
        return self._labels.get(lora_id, normalize_id(lora_id).rsplit("/", 1)[-1])

    def resolve(self, name: str) -> tuple[str, list[str]]:
        """``(lora id, [])`` for a match, ``("", candidates)`` for a name more
        than one LoRA answers to, and ``("", [])`` for a name none does."""
        text = str(name or "").strip()
        path = normalize_id(text)
        filename = path.rsplit("/", 1)[-1]
        probes = (
            ("path", path),
            ("path_ci", path.casefold()),
            ("path_stem", path.casefold()),
            ("file", filename.casefold()),
            ("stem", filename.casefold()),
            ("civitai", text.casefold()),
            ("loose", _loose(text)),
            ("loose", _loose(filename)),
        )
        for level, key in probes:
            if not key:
                continue
            found = list(self._levels[level].get(key, {}))
            if len(found) == 1:
                return found[0], []
            if found:
                return "", found
        return "", []

    def name_for(self, lora_id: str) -> str:
        """The shortest name that reads back as ``lora_id``: its filename
        without the extension when no other LoRA answers to that, else its
        filename, else its path."""
        lora_id = normalize_id(lora_id)
        filename = lora_id.rsplit("/", 1)[-1]
        for candidate in (os.path.splitext(filename)[0], filename):
            if candidate and not candidate.startswith(("#", "//")) and self.resolve(candidate)[0] == lora_id:
                return candidate
        return lora_id


def format_stack(entries, index: NameIndex) -> str:
    """``entries`` -- ``(lora id, token)`` pairs, or ``(lora id, token, mode,
    group, baked)`` -- as text ``parse`` reads back."""
    lines = []
    for entry in entries:
        lora_id, token = entry[0], entry[1]
        tags = tags_text(*entry[2:5]) if len(entry) > 2 else ""
        lines.append(f"{index.name_for(lora_id)}{tags}: x{token}")
    return "\n".join(lines)
