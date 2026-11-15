"""Where built LoRA files live, how WanGP finds them, and what they came from.

Built files -- visual-only and audio-only copies, and baked groups -- are
kept in the extension's own folder, one store per LoRA folder:

    <plugin>/derived/<LoRA folder name>-<hash of its path>/

WanGP only offers LoRAs found under the model's LoRA folder (``setup_loras``
globs ``<lora dir>/**``, following links), so the store is linked into that
folder as ``_lora_browser``: a symlink, or on Windows a directory junction,
which needs no administrator rights.  Where neither can be made -- a FAT or
network drive -- ``_lora_browser`` is created as a real folder and the files
are written straight into it.

A built file is addressed the way WanGP addresses every LoRA, by its path
relative to the LoRA folder, so a stack that uses one -- in a preset, the
queue, a generated video's settings -- names it like any other:

    _lora_browser/<original without its extension>.<visual|audio|sound>.safetensors
    _lora_browser/_bakes/<hash of the members>.<r<rank>|q<per mille>>[.sound|.p1...].safetensors

A bake's size is ``r32`` -- every layer re-fitted to rank 32 -- or ``q990``
-- a compression level: the smallest file keeping 99.0% of the combined
change, each layer at a rank of its own.  A shrunk file may also be used
visual or audio only: ``<bake>.visual.safetensors`` is a copy of the built
file keeping its picture side, made from the file itself -- nothing is
baked again -- and its recipe names the file it came from.

``visual`` and ``audio`` are the visual-only and audio-only copies; a
shrunk group whose LoRAs run different strengths per phase is baked a file
per phase, ``.p1``, ``.p2``, each running in its phase only.  A LoRA
whose picture and sound have strengths of their own is selected as two
halves, its ``visual`` copy and its ``sound`` copy -- the sound side alone,
since the visual copy already holds the layers both sides share.  A group
like that bakes each side into a file of its own; the sound one says so.

A copy keeps its original's file name, with the mode added, because WanGP
recognises some LoRAs by name (LTX-2's IC-LoRAs).

``manifest.json`` in the store records what each file was built from -- the
originals' size and modification time, and the SHA-256 of a split's source --
so a copy whose original has been replaced is rebuilt rather than used.

``recipes.json`` beside it keeps every built file's recipe -- a copy's
original and mode, a bake's members, what each brings, and its rank -- and
keeps it when the file is deleted: a stack, a profile or a video's settings
that names the file again gets it built again from its originals, even where
nothing else in them says what it held.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass

from .utils import format_number, is_within, normalize_id

LINK_NAME = "_lora_browser"
BAKES = "_bakes"
MANIFEST = "manifest.json"
RECIPES = "recipes.json"
MODES = ("visual", "audio", "sound")
#: The track a group's sound-side bake carries.
SOUND_TRACK = "sound"
#: The tracks of a shrunk group baked a file per phase: each file runs in
#: its phase only.
PHASE_TRACKS = ("p1", "p2", "p3")

_SPLIT = re.compile(r"^_lora_browser/(?!_bakes/)(?P<stem>.+)\.(?P<mode>visual|audio|sound)\.safetensors$")
_BAKE = re.compile(
    r"^_lora_browser/_bakes/(?P<hash>[0-9a-f]{6,64})\.(?:r(?P<rank>[1-9]\d{0,4})|q(?P<quality>[1-9]\d{2}))"
    r"(?:\.(?P<track>sound|p[1-3]))?(?:\.(?P<mode>visual|audio))?\.safetensors$"
)
#: The modes a baked file can be copied in.
FILE_MODES = ("visual", "audio")
_SIZE = re.compile(r"^(?:r?(?P<rank>[1-9]\d{0,4})|q(?P<quality>[1-9]\d{2}))$")


def size_tag(value) -> str:
    """A bake's size as its id spells it: ``r32``, one rank for every layer,
    or ``q990``, a share of the combined change kept -- per mille -- each
    layer at a rank of its own.  Takes a rank as a number, or a tag; "" for
    anything else."""
    if isinstance(value, bool):
        return ""
    if isinstance(value, int):
        return f"r{value}" if 0 < value < 100000 else ""
    match = _SIZE.match(str(value or "").strip().lower())
    if not match:
        return ""
    return f"q{match.group('quality')}" if match.group("quality") else f"r{int(match.group('rank'))}"


def size_parts(value) -> tuple[int, int]:
    """``(rank, quality)`` of a size: one of them, the other 0."""
    tag = size_tag(value)
    if not tag:
        return 0, 0
    return (0, int(tag[1:])) if tag.startswith("q") else (int(tag[1:]), 0)


def phase_of(track: str) -> int:
    """The phase a phase track's file runs in (``p2`` -> 2), else 0."""
    return PHASE_TRACKS.index(track) + 1 if track in PHASE_TRACKS else 0


_ORIGINAL_EXTENSIONS = (".safetensors", ".sft")


@dataclass(frozen=True)
class Derived:
    """A built file's id, read back."""

    kind: str                 # "split" or "bake"
    id: str
    #: Split: the original's id without its extension, and the mode.
    stem: str = ""
    mode: str = ""
    #: Bake: the members' hash and its size -- a rank for every layer, or a
    #: share of the combined change kept, per mille -- and ``sound`` for the
    #: sound side of a group whose picture and sound have strengths of their
    #: own, ``p1``... for a shrunk group's file for one phase.
    hash: str = ""
    rank: int = 0
    quality: int = 0
    track: str = ""

    @property
    def size(self) -> str:
        """``r32`` or ``q990``, as the id spells it."""
        return f"q{self.quality}" if self.quality else (f"r{self.rank}" if self.rank else "")

    @property
    def base(self) -> str:
        """A bake's own file: its id without a visual or audio mode."""
        if self.kind == "bake" and self.mode:
            return self.id[: -len(f".{self.mode}.safetensors")] + ".safetensors"
        return self.id


#: What WanGP looks for in a LoRA's file name to treat it as more than its
#: weights: LTX-2's IC-LoRAs ("ic-lora", their reference scale read from
#: "-ref0.5") and the LoRAs it adds itself unless one of yours is named like
#: them (distilled, union control, in/outpainting, ID), and Wan 2.2's
#: Lightning I2V accelerators, which pick the sampler.  Baked into a file
#: with a name of its own, such a LoRA would lose that -- WanGP would even add
#: its own default beside it -- so a bake or a shrink leaves it as it is.
_RECOGNISED = ("ic-lora", "distilled-lora", "union-control", "in-outpainting", "id-lora-celebvhq")


def recognised_by_name(lora_id: str) -> str:
    """The part of ``lora_id``'s file name WanGP recognises it by, or ""."""
    name = os.path.basename(normalize_id(lora_id)).lower()
    found = next((signature for signature in _RECOGNISED if signature in name), "")
    if not found and "i2v" in name and ("lightx2v" in name or "seko" in name):
        found = "lightx2v" if "lightx2v" in name else "seko"
    return found


def is_derived(lora_id: str) -> bool:
    return normalize_id(lora_id).startswith(LINK_NAME + "/")


def parse(lora_id: str) -> Derived | None:
    value = normalize_id(lora_id)
    match = _SPLIT.match(value)
    if match:
        return Derived("split", value, stem=match.group("stem"), mode=match.group("mode"))
    match = _BAKE.match(value)
    if match:
        return Derived(
            "bake", value, hash=match.group("hash"), rank=int(match.group("rank") or 0),
            quality=int(match.group("quality") or 0), track=match.group("track") or "",
            mode=match.group("mode") or "",
        )
    return None


def stem_of(lora_id: str) -> str:
    """One key per original LoRA, whichever copy of it ``lora_id`` names: a
    LoRA and its visual, audio and sound copies share it.  A bake is its own."""
    value = normalize_id(lora_id)
    parsed = parse(value)
    if parsed is not None:
        return parsed.stem if parsed.kind == "split" else parsed.base
    stem, extension = os.path.splitext(value)
    return stem if extension.lower() in _ORIGINAL_EXTENSIONS else value


def split_id(original_id: str, mode: str) -> str:
    """The id of ``original_id``'s copy in ``mode``; the original itself for
    ``normal``."""
    original_id = normalize_id(original_id)
    if mode not in MODES:
        return original_id
    stem, _ = os.path.splitext(original_id)
    return f"{LINK_NAME}/{stem}.{mode}.safetensors"


def original_id(derived: Derived, known_ids) -> str:
    """The original a split copy was made from, as the inventory names it."""
    known = set(known_ids or [])
    for extension in _ORIGINAL_EXTENSIONS:
        candidate = derived.stem + extension
        if candidate in known:
            return candidate
    return derived.stem + _ORIGINAL_EXTENSIONS[0]


def bake_hash(members) -> str:
    """One name per set of members and what each brings.

    ``members`` is ``(original id, sides)`` pairs, ``sides`` saying which of
    its streams a member contributes -- or ``(original id, sides, weight)``,
    a shrunk group's member scaled by what it brings.  Two groups that would
    bake the same thing share a file; changing what any member brings changes
    the name, so a bake is never mistaken for one built from something else.
    A weight of 1 is left out, so every name made before weights is the same.
    """
    lines = []
    for member in members:
        lora_id, sides = member[0], member[1]
        line = f"{normalize_id(lora_id)}\t{sides}"
        weight = float(member[2]) if len(member) > 2 else 1.0
        if abs(weight - 1.0) > 1e-9:
            line += f"\t{format_number(weight)}"
        lines.append(line)
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()[:12]


def bake_id(members, size, track: str = "") -> str:
    """A bake's id at ``size`` (a rank, or a :func:`size_tag`): ``.sound``
    for a group's sound side, ``.p1``... for a shrunk group's file for one
    phase."""
    tag = size_tag(size)
    if not tag:
        raise ValueError(f"not a bake size: {size!r}")
    suffix = f".{track}" if track == SOUND_TRACK or track in PHASE_TRACKS else ""
    return f"{LINK_NAME}/{BAKES}/{bake_hash(members)}.{tag}{suffix}.safetensors"


def bake_mode_id(bake_id: str, mode: str) -> str:
    """A baked file's copy in ``mode`` -- visual or audio only -- or, for
    ``normal``, the file itself."""
    parsed = parse(bake_id)
    if parsed is None or parsed.kind != "bake":
        return normalize_id(bake_id)
    base = parsed.base
    if mode not in FILE_MODES:
        return base
    return base[: -len(".safetensors")] + f".{mode}.safetensors"


def bake_spec(size, modes: dict | None = None) -> str:
    """How a group is baked, as one value: its size, and any of its files
    used visual or audio only -- ``q999+p1.audio``."""
    tag = size_tag(size)
    extra = "".join(f"+{track}.{mode}" for track, mode in sorted((modes or {}).items()) if mode in FILE_MODES)
    return tag + extra if tag else ""


def parse_bake_spec(value) -> tuple[str, dict[str, str]]:
    """``(size, {track: mode})`` of a :func:`bake_spec`; a plain rank or
    size reads as itself."""
    if isinstance(value, int) and not isinstance(value, bool):
        return size_tag(value), {}
    head, *rest = str(value or "").strip().split("+")
    modes = {}
    for part in rest:
        track, _, mode = part.partition(".")
        if track in ("all", "picture") + PHASE_TRACKS and mode in FILE_MODES:
            modes[track] = mode
    return size_tag(head), modes


def folder_key(lora_dir: str) -> str:
    """A store name for one LoRA folder: readable, and unique per path."""
    real = os.path.realpath(lora_dir or ".")
    base = re.sub(r"[^A-Za-z0-9_.-]+", "_", os.path.basename(real.rstrip("/\\")) or "loras")
    digest = hashlib.sha256(real.encode("utf-8")).hexdigest()[:8]
    return f"{base}-{digest}"


def _make_link(target: str, link: str) -> bool:
    try:
        os.symlink(target, link, target_is_directory=True)
        return True
    except (OSError, NotImplementedError, AttributeError):
        pass
    if os.name == "nt":
        try:
            import _winapi  # noqa: PLC0415 - Windows only

            _winapi.CreateJunction(target, link)
            return True
        except (ImportError, OSError, AttributeError):
            pass
    return False


class Store:
    """The built files for one LoRA folder."""

    def __init__(self, plugin_dir: str, lora_dir: str) -> None:
        self.lora_dir = lora_dir
        self.root = os.path.join(plugin_dir, "derived", folder_key(lora_dir))
        self.link = os.path.join(lora_dir, LINK_NAME)
        self._lock = threading.RLock()
        self._base: str | None = None

    # ------------------------------------------------------------ the link

    def base(self) -> str:
        """The folder files are written into, made visible to WanGP first.

        The store is linked into the LoRA folder; a link that already points
        there is reused, a ``_lora_browser`` that is a real folder is written
        into as it is, and a dangling link (the plugin moved) is replaced.
        """
        with self._lock:
            if self._base and os.path.isdir(self._base) and os.path.isdir(self.link):
                return self._base
            os.makedirs(self.root, exist_ok=True)
            if os.path.islink(self.link) or _is_junction(self.link):
                if os.path.isdir(self.link):
                    self._base = os.path.realpath(self.link)
                    return self._base
                _remove_link(self.link)
            elif os.path.isdir(self.link):
                self._base = self.link
                return self._base
            elif os.path.exists(self.link):
                raise OSError(f"'{self.link}' exists and is not a folder")
            if _make_link(self.root, self.link) and os.path.isdir(self.link):
                self._base = self.root
            else:
                os.makedirs(self.link, exist_ok=True)
                self._base = self.link
            return self._base

    @property
    def linked(self) -> bool:
        """True when the files live in the extension's folder, False when
        they had to be written into the LoRA folder itself."""
        return os.path.realpath(self.base()) == os.path.realpath(self.root)

    # ------------------------------------------------------------- files

    def path(self, derived_id: str) -> str:
        """Where WanGP opens ``derived_id``: inside the LoRA folder."""
        value = normalize_id(derived_id)
        if not is_derived(value):
            raise ValueError(f"'{derived_id}' is not a built LoRA")
        path = os.path.normpath(os.path.join(self.lora_dir, value))
        if not is_within(self.link, path) and not is_within(self.lora_dir, path):
            raise ValueError(f"'{derived_id}' points outside the LoRA folder")
        return path

    def write_path(self, derived_id: str) -> str:
        """Where to write ``derived_id``, inside the store itself."""
        relative = normalize_id(derived_id)[len(LINK_NAME) + 1:]
        base = self.base()
        path = os.path.normpath(os.path.join(base, relative))
        if not is_within(base, path):
            raise ValueError(f"'{derived_id}' points outside the store")
        return path

    def exists(self, derived_id: str) -> bool:
        try:
            return os.path.isfile(self.path(derived_id))
        except ValueError:
            return False

    # ---------------------------------------------------------- manifest

    def _manifest_path(self) -> str:
        return os.path.join(self.base(), MANIFEST)

    def manifest(self) -> dict:
        with self._lock:
            try:
                with open(self._manifest_path(), "r", encoding="utf-8") as handle:
                    data = json.load(handle)
            except (OSError, ValueError):
                return {}
            return data if isinstance(data, dict) else {}

    def _write_manifest(self, data: dict) -> None:
        self._write_json(self._manifest_path(), data)

    def _write_json(self, path: str, data: dict) -> None:
        handle, temp = tempfile.mkstemp(prefix=".manifest-", suffix=".json", dir=os.path.dirname(path))
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as writer:
                json.dump(data, writer, indent=1, sort_keys=True)
            os.replace(temp, path)
        except BaseException:
            try:
                os.remove(temp)
            except OSError:
                pass
            raise

    def record(self, derived_id: str, entry: dict) -> None:
        with self._lock:
            data = self.manifest()
            data[normalize_id(derived_id)] = dict(entry, created=entry.get("created") or time.time())
            self._write_manifest(data)
            self.remember(derived_id, recipe_of(entry))

    # ----------------------------------------------------------- recipes

    def _recipes_path(self) -> str:
        return os.path.join(self.base(), RECIPES)

    def _existing_base(self) -> str | None:
        """The folder files are in, if there is one yet -- making nothing."""
        if self._base and os.path.isdir(self._base):
            return self._base
        if os.path.isdir(self.link):
            return os.path.realpath(self.link)
        return None

    def recipes(self) -> dict:
        """Every recipe kept, built file on disk or not.  Reading makes no
        folder and no link: a store nothing was ever built in has none."""
        base = self._existing_base()
        if base is None:
            return {}
        with self._lock:
            try:
                with open(os.path.join(base, RECIPES), "r", encoding="utf-8") as handle:
                    data = json.load(handle)
            except (OSError, ValueError):
                return {}
            return data if isinstance(data, dict) else {}

    def recipe(self, derived_id: str) -> dict | None:
        value = self.recipes().get(normalize_id(derived_id))
        return value if isinstance(value, dict) else None

    def remember(self, derived_id: str, recipe: dict) -> None:
        """Keep what ``derived_id`` is made of -- for as long as the store is
        there, the file or not."""
        if not recipe:
            return
        with self._lock:
            data = self.recipes()
            data[normalize_id(derived_id)] = dict(recipe, remembered=time.time())
            self._write_json(self._recipes_path(), data)

    def entry(self, derived_id: str) -> dict | None:
        value = self.manifest().get(normalize_id(derived_id))
        return value if isinstance(value, dict) else None

    def source_stamp(self, lora_id: str) -> dict:
        """What the manifest remembers about an original: size and mtime."""
        path = os.path.join(self.lora_dir, normalize_id(lora_id))
        try:
            stat = os.stat(path)
        except OSError:
            return {"id": normalize_id(lora_id)}
        return {"id": normalize_id(lora_id), "size": stat.st_size, "mtime": stat.st_mtime}

    def stale(self, derived_id: str) -> bool:
        """A built file whose originals have changed since it was built.

        Only a file the manifest knows can be called stale; one it does not
        (a store copied in from elsewhere) is trusted as it is.
        """
        entry = self.entry(derived_id)
        if not entry:
            return False
        for source in entry.get("sources", []) or []:
            if not isinstance(source, dict) or "size" not in source:
                continue
            now = self.source_stamp(source.get("id", ""))
            if now.get("size") != source.get("size") or abs(float(now.get("mtime", 0)) - float(source.get("mtime", 0))) > 1e-3:
                return True
        return False

    def ready(self, derived_id: str) -> bool:
        return self.exists(derived_id) and not self.stale(derived_id)

    # ------------------------------------------------------------ listing

    def files(self) -> list[dict]:
        """Every built file on disk, with what the manifest says about it."""
        try:
            base = self.base()
        except OSError:
            return []
        manifest = self.manifest()
        found = []
        for folder, _, names in os.walk(base):
            for name in names:
                if not name.endswith(".safetensors"):
                    continue
                path = os.path.join(folder, name)
                relative = os.path.relpath(path, base).replace("\\", "/")
                derived_id = f"{LINK_NAME}/{relative}"
                if parse(derived_id) is None:
                    continue
                try:
                    size = os.path.getsize(path)
                except OSError:
                    continue
                entry = manifest.get(derived_id) if isinstance(manifest.get(derived_id), dict) else {}
                found.append(dict(entry, id=derived_id, bytes=size))
        found.sort(key=lambda item: item["id"])
        return found

    def delete(self, derived_ids) -> int:
        """Remove built files; returns how many went.  Their recipes stay."""
        removed = 0
        with self._lock:
            data = self.manifest()
            for derived_id in derived_ids:
                value = normalize_id(derived_id)
                if parse(value) is None:
                    continue
                try:
                    os.remove(self.write_path(value))
                    removed += 1
                except (OSError, ValueError):
                    pass
                data.pop(value, None)
            self._write_manifest(data)
            # Folders a deletion emptied go too, so the picker does not keep
            # listing empty folders.
            base = self.base()
            for folder, _, _ in sorted(os.walk(base), key=lambda item: -len(item[0])):
                if folder != base and not os.listdir(folder):
                    shutil.rmtree(folder, ignore_errors=True)
        return removed


def recipe_of(entry: dict) -> dict:
    """What to keep of a manifest entry to build its file again."""
    kind = entry.get("kind")
    if kind == "split":
        sources = entry.get("sources") or [{}]
        source = sources[0].get("id", "") if isinstance(sources[0], dict) else ""
        return {"kind": "split", "mode": entry.get("mode", ""), "source": source, "label": entry.get("label", "")}
    if kind == "bake":
        members = []
        for member in entry.get("members") or []:
            if isinstance(member, dict) and member.get("id"):
                members.append({key: member[key] for key in ("id", "mode", "name", "weight") if key in member})
        found = {"kind": "bake", "rank": entry.get("rank", 0), "members": members, "label": entry.get("label", "")}
        if entry.get("quality"):
            found["quality"] = entry["quality"]
        if entry.get("shrink"):
            found["shrink"] = True
        if entry.get("track"):
            found["track"] = entry["track"]
        return found
    return {}


def _is_junction(path: str) -> bool:
    checker = getattr(os.path, "isjunction", None)
    try:
        return bool(checker and checker(path))
    except OSError:
        return False


def _remove_link(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        try:
            os.rmdir(path)              # a junction on older Pythons
        except OSError:
            pass
