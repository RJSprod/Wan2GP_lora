"""Read LoRA files and write exact copies of part of one, without torch.

A ``.safetensors`` file is an 8-byte little-endian header length, a JSON header
naming every tensor's dtype, shape and byte range, and then the raw bytes.
That is all a visual-only or audio-only copy needs: it keeps some of the
tensors byte for byte and leaves the rest out, so the copy is exact and needs
no tensor library.  WanGP ships torch, but the plugin does not need it here.

LTX-2 carries picture and sound as two streams, and every module a LoRA can
change writes into one of them (``models/ltx2/ltx_core/model/transformer`` in
WanGP): ``attn1``, ``attn2``, ``ff`` and ``audio_to_video_attn`` -- the
picture attending to the sound -- write into the picture; ``audio_attn1``,
``audio_attn2``, ``audio_ff`` and ``video_to_audio_attn`` into the sound.
:func:`ltx2_side` sorts a tensor by its module name, in any of the naming
styles WanGP's LTX-2 loader accepts (ComfyUI, diffusers, kohya).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field

#: safetensors refuses headers larger than this; so does this reader.
HEADER_LIMIT = 100 * 1024 * 1024

#: How much of a file is read or written per call while copying.
CHUNK = 8 * 1024 * 1024

PICTURE = "picture"
SOUND = "sound"
#: Modules that feed both streams -- the text projection in front of both
#: connectors.  They cannot be given to one side, so both copies keep them.
SHARED = "shared"
SIDES = (PICTURE, SOUND, SHARED)

#: Bytes per element for the dtypes safetensors writes.
DTYPE_SIZES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
    "F8_E4M3": 1, "F8_E5M2": 1,
}

#: What a key does inside its module.  ``down``/``up`` are the two low-rank
#: factors, under every name WanGP's loader reads them by (``lora_A``,
#: ``lora_down``, PEFT's ``lora_A.default``); the rest are kept or refused as
#: they come.
_KEY = re.compile(
    r"^(?P<module>.+?)\."
    r"(?:lora[._](?P<factor>A|B|down|up)(?:\.[^.]+)?\.weight"
    r"|(?P<alpha>alpha)"
    r"|(?P<diff_b>diff_b)"
    r"|(?P<diff>diff)"
    r"|(?P<dora>dora_scale)"
    r"|(?P<lokr>lokr_\w+))$"
)
_FACTOR_ROLE = {"A": "down", "down": "down", "B": "up", "up": "up"}


class LoraFileError(ValueError):
    """The file is not a safetensors file this module can read."""


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: str
    shape: tuple[int, ...]
    #: Absolute byte offsets in the file, end exclusive.
    start: int
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.start


@dataclass
class LoraFile:
    path: str
    #: Tensors in the order their bytes appear in the file.
    tensors: list[TensorInfo] = field(default_factory=list)
    #: The header's ``__metadata__``: strings only, as safetensors requires.
    metadata: dict[str, str] = field(default_factory=dict)
    data_start: int = 0
    size: int = 0

    def get(self, name: str) -> TensorInfo | None:
        for tensor in self.tensors:
            if tensor.name == name:
                return tensor
        return None


def read_header(path: str) -> LoraFile:
    """The tensor table of a safetensors file; no tensor bytes are read."""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise LoraFileError("too short to be a safetensors file")
            length = int.from_bytes(prefix, "little")
            if length <= 1 or length > HEADER_LIMIT or 8 + length > size:
                raise LoraFileError("not a safetensors file")
            header = json.loads(handle.read(length).decode("utf-8"))
    except OSError as error:
        raise LoraFileError(str(error)) from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LoraFileError("unreadable safetensors header") from error
    if not isinstance(header, dict):
        raise LoraFileError("unreadable safetensors header")

    data_start = 8 + length
    metadata = header.pop("__metadata__", None) or {}
    if not isinstance(metadata, dict):
        metadata = {}
    tensors = []
    for name, info in header.items():
        try:
            begin, end = (int(value) for value in info["data_offsets"])
            shape = tuple(int(value) for value in info["shape"])
            dtype = str(info["dtype"])
        except (KeyError, TypeError, ValueError) as error:
            raise LoraFileError(f"bad entry for tensor '{name}'") from error
        if begin < 0 or end < begin or data_start + end > size:
            raise LoraFileError(f"tensor '{name}' lies outside the file")
        tensors.append(TensorInfo(str(name), dtype, shape, data_start + begin, data_start + end))
    tensors.sort(key=lambda tensor: (tensor.start, tensor.end))
    return LoraFile(
        path=path,
        tensors=tensors,
        metadata={str(key): str(value) for key, value in metadata.items()},
        data_start=data_start,
        size=size,
    )


# ------------------------------------------------------------------ keys


def split_key(name: str) -> tuple[str, str]:
    """``(module, role)`` for one tensor name.

    ``role`` is ``down``/``up`` for the low-rank factors, ``alpha``,
    ``diff_b`` (a bias delta), ``diff`` (a dense weight delta), ``dora`` or
    ``lokr``; ``other`` with the whole name as the module for anything else.
    """
    match = _KEY.match(str(name))
    if not match:
        return str(name), "other"
    if match.group("factor"):
        return match.group("module"), _FACTOR_ROLE[match.group("factor")]
    for role in ("alpha", "diff_b", "diff", "dora", "lokr"):
        if match.group(role):
            return match.group("module"), role
    return str(name), "other"


def ltx2_side(name: str) -> str:
    """Which LTX-2 stream a tensor writes into: picture, sound, or both.

    The cross-attention between the streams belongs to the stream it writes
    into: ``audio_to_video_attn`` is the picture attending to the sound, so
    it is picture; ``video_to_audio_attn`` is sound.  The modulation tables
    for that cross-attention follow the same rule (``av_ca_a2v_*`` and
    ``av_ca_video_*`` picture, ``av_ca_v2a_*`` and ``av_ca_audio_*`` sound).
    Anything else named for audio is sound, the text projection in front of
    both connectors feeds both, and everything left is the picture stream.
    """
    key = str(name).lower()
    if "video_to_audio" in key or "av_ca_v2a" in key or "av_ca_audio" in key:
        return SOUND
    if "audio_to_video" in key or "av_ca_a2v" in key or "av_ca_video" in key:
        return PICTURE
    if "feature_extractor_linear" in key or "text_embedding_projection" in key:
        return SHARED
    if "audio" in key:
        return SOUND
    return PICTURE


def keeps(side: str, mode: str) -> bool:
    """Whether a copy in ``mode`` keeps a tensor on ``side``.

    ``sound`` is the sound half of a LoRA whose picture and sound have
    strengths of their own: it is selected beside the ``visual`` copy, which
    keeps the shared layers, so it leaves them out -- applied twice, they
    would take the sum of the two strengths.
    """
    if mode == "visual":
        return side in (PICTURE, SHARED)
    if mode == "audio":
        return side in (SOUND, SHARED)
    if mode == "sound":
        return side == SOUND
    return True


# --------------------------------------------------------------- summary


@dataclass(frozen=True)
class SideSummary:
    #: Distinct modules the LoRA changes on this side.
    modules: int = 0
    tensors: int = 0
    bytes: int = 0

    def to_dict(self) -> dict:
        return {"modules": self.modules, "tensors": self.tensors, "bytes": self.bytes}


@dataclass(frozen=True)
class StreamSummary:
    """What one LTX-2 LoRA changes on each side, and how big it is."""

    picture: SideSummary = SideSummary()
    sound: SideSummary = SideSummary()
    shared: SideSummary = SideSummary()
    #: The rank most of its low-rank factors have, or 0 when it has none.
    rank: int = 0

    def has(self, mode: str) -> bool:
        """Whether a copy in ``mode`` would change anything at all.

        Tensors shared by both streams do not count: a LoRA whose only
        sound-side change is the shared text projection has no sound of its
        own to keep.
        """
        if mode == "visual":
            return self.picture.tensors > 0
        if mode in ("audio", "sound"):
            return self.sound.tensors > 0
        return self.picture.tensors + self.sound.tensors + self.shared.tensors > 0

    @property
    def both(self) -> bool:
        """It changes picture and sound, so a split copy differs from it."""
        return self.picture.tensors > 0 and self.sound.tensors > 0

    def to_dict(self) -> dict:
        return {
            "picture": self.picture.to_dict(),
            "sound": self.sound.to_dict(),
            "shared": self.shared.to_dict(),
            "rank": self.rank,
        }


def summarize(lora: LoraFile) -> StreamSummary:
    sides: dict[str, dict] = {side: {"modules": set(), "tensors": 0, "bytes": 0} for side in SIDES}
    ranks: Counter = Counter()
    for tensor in lora.tensors:
        module, role = split_key(tensor.name)
        entry = sides[ltx2_side(module)]
        entry["modules"].add(module)
        entry["tensors"] += 1
        entry["bytes"] += tensor.nbytes
        if role == "down" and tensor.shape:
            ranks[int(tensor.shape[0])] += 1
    made = {
        side: SideSummary(len(entry["modules"]), entry["tensors"], entry["bytes"])
        for side, entry in sides.items()
    }
    rank = ranks.most_common(1)[0][0] if ranks else 0
    return StreamSummary(made[PICTURE], made[SOUND], made[SHARED], rank)


class SummaryCache:
    """Stream summaries keyed on path, size and modification time.

    Reading a header is cheap, but the panel asks on every payload; a file
    that has not changed is never read twice.
    """

    def __init__(self) -> None:
        self._cache: dict[str, tuple[tuple[int, float], StreamSummary | None]] = {}

    def get(self, path: str | None) -> StreamSummary | None:
        if not path:
            return None
        try:
            stat = os.stat(path)
        except OSError:
            return None
        stamp = (stat.st_size, stat.st_mtime)
        cached = self._cache.get(path)
        if cached is not None and cached[0] == stamp:
            return cached[1]
        try:
            summary = summarize(read_header(path))
        except LoraFileError:
            summary = None
        self._cache[path] = (stamp, summary)
        return summary


# --------------------------------------------------------------- writing


def _header_bytes(entries: list[tuple[str, str, tuple[int, ...], int]], metadata: dict[str, str]) -> bytes:
    """A safetensors header for tensors laid out back to back, in order,
    padded with spaces so the tensor bytes start on an 8-byte boundary."""
    header: dict = {}
    if metadata:
        header["__metadata__"] = {str(key): str(value) for key, value in metadata.items()}
    offset = 0
    for name, dtype, shape, nbytes in entries:
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((-(8 + len(encoded))) % 8)
    return len(encoded).to_bytes(8, "little") + encoded


def _replace(part: str, path: str) -> None:
    os.replace(part, path)


class StreamWriter:
    """Writes a safetensors file tensor by tensor, without holding it in memory.

    The layout -- every name, dtype and shape -- is declared up front so the
    header can be written first; tensors must then arrive in that order.  The
    file appears under its own name only once it is complete.
    """

    def __init__(self, path: str, entries: list[tuple[str, str, tuple[int, ...]]], metadata: dict[str, str]):
        self.path = path
        self._entries = []
        for name, dtype, shape in entries:
            if dtype not in DTYPE_SIZES:
                raise LoraFileError(f"cannot write dtype {dtype}")
            count = 1
            for dim in shape:
                count *= int(dim)
            self._entries.append((name, dtype, tuple(int(dim) for dim in shape), count * DTYPE_SIZES[dtype]))
        self._next = 0
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._part = path + ".part"
        self._handle = open(self._part, "wb")
        self._handle.write(_header_bytes(self._entries, metadata))

    def write(self, name: str, data: bytes) -> None:
        expected_name, _, _, nbytes = self._entries[self._next]
        if name != expected_name or len(data) != nbytes:
            self.abort()
            raise LoraFileError(f"tensor '{name}' does not match the declared layout")
        self._handle.write(data)
        self._next += 1

    def close(self) -> None:
        if self._next != len(self._entries):
            self.abort()
            raise LoraFileError("not every declared tensor was written")
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()
        _replace(self._part, self.path)

    def abort(self) -> None:
        try:
            self._handle.close()
        finally:
            try:
                os.remove(self._part)
            except OSError:
                pass


def copy_subset(source: LoraFile, keep: list[str], path: str, metadata: dict[str, str], progress=None) -> str:
    """Write the tensors named in ``keep`` from ``source`` to ``path``, exactly.

    The whole source is read once, in order, so its SHA-256 comes out of the
    same pass -- returned, for the manifest that notices an original being
    replaced.  ``progress(done, total)`` is called with bytes read.
    """
    wanted = set(keep)
    kept = [tensor for tensor in source.tensors if tensor.name in wanted]
    writer = StreamWriter(path, [(tensor.name, tensor.dtype, tensor.shape) for tensor in kept], metadata)
    digest = hashlib.sha256()
    done = 0
    try:
        with open(source.path, "rb") as handle:
            head = handle.read(source.data_start)
            digest.update(head)
            done += len(head)
            for tensor in source.tensors:
                position = handle.tell()
                if position < tensor.start:                      # padding between tensors
                    gap = handle.read(tensor.start - position)
                    digest.update(gap)
                    done += len(gap)
                elif position > tensor.start:                    # overlapping views: re-read
                    handle.seek(tensor.start)
                remaining = tensor.end - handle.tell()
                pieces = []
                while remaining > 0:
                    piece = handle.read(min(CHUNK, remaining))
                    if not piece:
                        raise LoraFileError(f"tensor '{tensor.name}' is cut short")
                    digest.update(piece)
                    done += len(piece)
                    remaining -= len(piece)
                    if tensor.name in wanted:
                        pieces.append(piece)
                    if progress:
                        progress(done, source.size)
                if tensor.name in wanted:
                    writer.write(tensor.name, b"".join(pieces))
            while True:
                tail = handle.read(CHUNK)
                if not tail:
                    break
                digest.update(tail)
                done += len(tail)
        writer.close()
    except BaseException:
        writer.abort()
        raise
    if progress:
        progress(source.size, source.size)
    return digest.hexdigest()


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            piece = handle.read(CHUNK)
            if not piece:
                break
            digest.update(piece)
    return digest.hexdigest()
