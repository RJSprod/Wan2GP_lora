"""Tiny LoRA files for tests, written without torch or the plugin's own writer.

``write_lora`` lays tensors out the way safetensors does, independently of
``lora_files``, so the reader is tested against files it did not produce.
``effective_delta`` reads a LoRA back the way mmgp applies one -- per module,
``alpha / rank`` (or the header ``alpha``, or 1) times ``up @ down`` -- so a
bake can be checked against the dense sum it stands for.
"""

from __future__ import annotations

import json
import re

import numpy as np

_NUMPY = {"F32": "<f4", "F16": "<f2", "F64": "<f8"}


def encode(array, dtype: str) -> bytes:
    data = np.ascontiguousarray(array, dtype=np.float32)
    if dtype == "BF16":
        bits = data.view(np.uint32)
        return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype("<u2").tobytes()
    return data.astype(_NUMPY[dtype]).tobytes()


def decode(raw: bytes, dtype: str, shape) -> np.ndarray:
    if dtype == "BF16":
        values = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
    else:
        values = np.frombuffer(raw, dtype=_NUMPY[dtype]).astype(np.float32)
    return values.reshape(shape) if shape else values.reshape(())


def write_lora(path, tensors, metadata=None, align=True) -> None:
    """``tensors``: ``[(name, dtype, array)]`` in file order."""
    header, blobs, offset = {}, [], 0
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    for name, dtype, array in tensors:
        array = np.asarray(array, dtype=np.float32)
        blob = encode(array, dtype)
        header[name] = {"dtype": dtype, "shape": list(array.shape), "data_offsets": [offset, offset + len(blob)]}
        blobs.append(blob)
        offset += len(blob)
    encoded = json.dumps(header).encode("utf-8")
    if align:
        encoded += b" " * ((-(8 + len(encoded))) % 8)
    with open(path, "wb") as handle:
        handle.write(len(encoded).to_bytes(8, "little"))
        handle.write(encoded)
        for blob in blobs:
            handle.write(blob)


def read_tensors(path) -> tuple[dict[str, np.ndarray], dict[str, str], dict[str, bytes]]:
    """``(arrays, metadata, raw bytes)`` of a whole file."""
    with open(path, "rb") as handle:
        length = int.from_bytes(handle.read(8), "little")
        header = json.loads(handle.read(length))
        data = handle.read()
    metadata = header.pop("__metadata__", {}) or {}
    arrays, raw = {}, {}
    for name, info in header.items():
        begin, end = info["data_offsets"]
        raw[name] = data[begin:end]
        arrays[name] = decode(data[begin:end], info["dtype"], tuple(info["shape"]))
    return arrays, metadata, raw


_FACTOR = re.compile(r"^(?P<module>.+?)\.lora[._](?P<factor>A|B|down|up)(?:\.[^.]+)?\.weight$")


def effective_delta(path, sides=None, canonical=None) -> dict[str, np.ndarray]:
    """The dense change mmgp applies per module, keyed by ``canonical(module)``."""
    arrays, metadata, _ = read_tensors(path)
    header_alpha = float(metadata["alpha"]) if "alpha" in metadata else None
    factors: dict[str, dict] = {}
    for name, array in arrays.items():
        match = _FACTOR.match(name)
        if match:
            role = "down" if match.group("factor") in ("A", "down") else "up"
            factors.setdefault(match.group("module"), {})[role] = array
    deltas = {}
    for module, pair in factors.items():
        down, up = pair["down"], pair["up"]
        rank = down.shape[0]
        alpha = arrays.get(module + ".alpha")
        if alpha is not None:
            scale = float(np.asarray(alpha).reshape(-1)[0]) / rank
        elif header_alpha is not None:
            scale = header_alpha / rank
        else:
            scale = 1.0
        key = canonical(module) if canonical else module
        deltas[key] = deltas.get(key, 0) + scale * (up.astype(np.float64) @ down.astype(np.float64))
    return deltas


def lora_pair(rng, out_dim, in_dim, rank, scale=0.1):
    """Random factors ``(down, up)`` of one rank-``rank`` LoRA module."""
    down = rng.standard_normal((rank, in_dim)).astype(np.float32) * scale
    up = rng.standard_normal((out_dim, rank)).astype(np.float32) * scale
    return down, up
