"""Small shared helpers: path normalisation, escaping, numeric formatting."""

from __future__ import annotations

import html
import math
import os
import re
from typing import Any, Iterable


_TRAILING_ZEROS = re.compile(r"\.?0+$")


def normalize_id(value: Any) -> str:
    """Normalise a WanGP LoRA value into the plugin's stable identity.

    WanGP stores activated LoRAs as paths relative to the model's LoRA
    directory (see ``get_lora_local_path`` in ``wgp.py``).  Windows and Linux
    installs can disagree on the separator, and repeated separators show up in
    hand-edited ``.lset`` files, so both are normalised away.  Absolute paths
    and URLs are left alone apart from separator normalisation: they are still
    valid WanGP values and must round-trip untouched.
    """
    text = str(value or "").strip().replace("\\", "/")
    while "//" in text:
        # Preserve the "https://" style prefix while collapsing the rest.
        head, sep, tail = text.partition("://")
        if sep:
            text = head + sep + tail.replace("//", "/")
            break
        text = text.replace("//", "/")
    if len(text) > 1 and text.endswith("/"):
        text = text[:-1]
    return text


def display_name(lora_id: str) -> str:
    """Basename shown on a tile."""
    return normalize_id(lora_id).rsplit("/", 1)[-1]


def parent_name(lora_id: str) -> str:
    """Relative parent folder, used only to disambiguate identical basenames."""
    normalized = normalize_id(lora_id)
    return normalized.rsplit("/", 1)[0] if "/" in normalized else ""


def stem(name: str) -> str:
    return os.path.splitext(name)[0]


def escape(value: Any) -> str:
    """HTML-escape any user-controlled string before it reaches the browser."""
    return html.escape(str(value if value is not None else ""), quote=True)


def is_finite_number(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def format_number(value: float, decimals: int = 4) -> str:
    """Format a multiplier without floating point junk.

    ``1.0`` becomes ``"1"``, ``0.8000000001`` becomes ``"0.8"``.
    """
    number = round(float(value), decimals)
    if number == int(number) and abs(number) < 1e15:
        return str(int(number))
    text = f"{number:.{decimals}f}"
    return _TRAILING_ZEROS.sub("", text) if "." in text else text


def clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


def unique(items: Iterable[str]) -> list[str]:
    """Order-preserving de-duplication."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def is_within(root: str, candidate: str) -> bool:
    """True when ``candidate`` resolves inside ``root``.

    Used to keep preview lookups from escaping the model's LoRA directory via
    symlinks or ``..`` segments smuggled in through a LoRA value.
    """
    try:
        root_real = os.path.realpath(root)
        candidate_real = os.path.realpath(candidate)
    except OSError:
        return False
    return os.path.commonpath([root_real, candidate_real]) == root_real
