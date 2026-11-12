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

    _lora_browser/<original without its extension>.<visual|audio>.safetensors
    _lora_browser/_bakes/<hash of the members>.r<rank>.safetensors

A copy keeps its original's file name, with the mode added, because WanGP
recognises some LoRAs by name (LTX-2's IC-LoRAs).

``manifest.json`` in the store records what each file was built from -- the
originals' size and modification time, and the SHA-256 of a split's source --
so a copy whose original has been replaced is rebuilt rather than used.
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

from .utils import is_within, normalize_id

LINK_NAME = "_lora_browser"
BAKES = "_bakes"
MANIFEST = "manifest.json"
MODES = ("visual", "audio")

_SPLIT = re.compile(r"^_lora_browser/(?!_bakes/)(?P<stem>.+)\.(?P<mode>visual|audio)\.safetensors$")
_BAKE = re.compile(r"^_lora_browser/_bakes/(?P<hash>[0-9a-f]{6,64})\.r(?P<rank>[1-9]\d{0,4})\.safetensors$")
_ORIGINAL_EXTENSIONS = (".safetensors", ".sft")


@dataclass(frozen=True)
class Derived:
    """A built file's id, read back."""

    kind: str                 # "split" or "bake"
    id: str
    #: Split: the original's id without its extension, and the mode.
    stem: str = ""
    mode: str = ""
    #: Bake: the members' hash and the rank.
    hash: str = ""
    rank: int = 0


def is_derived(lora_id: str) -> bool:
    return normalize_id(lora_id).startswith(LINK_NAME + "/")


def parse(lora_id: str) -> Derived | None:
    value = normalize_id(lora_id)
    match = _SPLIT.match(value)
    if match:
        return Derived("split", value, stem=match.group("stem"), mode=match.group("mode"))
    match = _BAKE.match(value)
    if match:
        return Derived("bake", value, hash=match.group("hash"), rank=int(match.group("rank")))
    return None


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


def bake_hash(members: list[tuple[str, str]]) -> str:
    """One name per set of members and what each brings.

    ``members`` is ``(original id, sides)`` pairs, ``sides`` saying which of
    its streams a member contributes.  Two groups that would bake the same
    thing share a file; changing what any member brings changes the name, so
    a bake is never mistaken for one built from something else.
    """
    text = "\n".join(f"{normalize_id(lora_id)}\t{sides}" for lora_id, sides in members)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def bake_id(members: list[tuple[str, str]], rank: int) -> str:
    return f"{LINK_NAME}/{BAKES}/{bake_hash(members)}.r{int(rank)}.safetensors"


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
        path = self._manifest_path()
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
        """Remove built files; returns how many went."""
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
