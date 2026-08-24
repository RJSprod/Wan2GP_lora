"""Preview discovery and derived-thumbnail generation.

Rules the rest of the plugin relies on:

* A preview must sit in the **same directory** with the **same stem** as the
  ``.safetensors``/``.sft`` file.  No broader search — it keeps lookups fast
  and predictable on libraries with hundreds of entries.
* Images always win over videos, even when the video is newer.
* A video preview is never handed to the browser.  Only its first decodable
  frame is decoded server-side, downscaled, and cached as a still image.
* Nothing under the LoRA directory is ever exposed as a static route.  The
  browser receives a ``data:`` URI for a small derived thumbnail, so
  ``.safetensors`` files and private media stay unreachable.
"""

from __future__ import annotations

import base64
import hashlib
import os
from dataclasses import dataclass

from .utils import is_within

#: Bumped whenever the derived output changes, which invalidates every cache
#: entry without needing to clear the directory by hand.
THUMB_SCHEMA = "thumb-v1"

IMAGE_EXTENSIONS = (".png", ".webp", ".jpg", ".jpeg")
VIDEO_EXTENSIONS = (".mp4", ".webm", ".mov", ".mkv")

DEFAULT_MAX_DIM = 320


@dataclass
class Preview:
    path: str
    #: ``"image"`` or ``"video"`` — decides how the thumbnail is derived.
    kind: str


def find_preview(lora_path: str, lora_root: str = "", listing=None) -> Preview | None:
    """Find the same-stem preview next to ``lora_path``.

    ``listing`` lets callers pass a cached ``os.listdir`` result so a large
    library does not stat every candidate extension per tile.  Extension
    comparison is case-insensitive, which matters on case-sensitive
    filesystems where ``foo.PNG`` is a perfectly normal filename.
    """
    if not lora_path:
        return None

    directory = os.path.dirname(lora_path)
    stem = os.path.splitext(os.path.basename(lora_path))[0]
    if not directory or not stem:
        return None

    if lora_root and not is_within(lora_root, directory):
        # A LoRA value that escapes the model's LoRA root must not be used to
        # read arbitrary files from disk.
        return None

    if listing is None:
        try:
            listing = os.listdir(directory)
        except OSError:
            return None

    stem_lower = stem.lower()
    candidates: dict[str, str] = {}
    for name in listing:
        base, extension = os.path.splitext(name)
        if base.lower() != stem_lower:
            continue
        extension = extension.lower()
        candidates.setdefault(extension, name)

    for extension in IMAGE_EXTENSIONS:
        if extension in candidates:
            return Preview(os.path.join(directory, candidates[extension]), "image")
    for extension in VIDEO_EXTENSIONS:
        if extension in candidates:
            return Preview(os.path.join(directory, candidates[extension]), "video")
    return None


def cache_key(preview: Preview, max_dim: int = DEFAULT_MAX_DIM) -> str:
    """Key on content identity so replacing a preview file invalidates it."""
    try:
        stat = os.stat(preview.path)
        signature = f"{stat.st_mtime_ns}:{stat.st_size}"
    except OSError:
        signature = "0:0"
    raw = f"{os.path.abspath(preview.path)}|{signature}|{THUMB_SCHEMA}|{int(max_dim)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class ThumbnailCache:
    """Derived thumbnails, keyed by source identity, stored as small WebP."""

    def __init__(self, cache_dir: str, max_dim: int = DEFAULT_MAX_DIM):
        self.cache_dir = cache_dir
        self.max_dim = int(max_dim)
        self._failures: set[str] = set()

    def _ensure_dir(self) -> bool:
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            return True
        except OSError:
            return False

    def cached_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.webp")

    def get(self, preview: Preview) -> str | None:
        """Return a ``data:`` URI for the preview, generating it if needed.

        Returns ``None`` on any failure so the caller can fall back to a
        placeholder tile; a broken preview must never break the browser.
        """
        key = cache_key(preview, self.max_dim)
        if key in self._failures:
            return None

        path = self.cached_path(key)
        if os.path.isfile(path):
            return _to_data_uri(path)

        if not self._ensure_dir():
            self._failures.add(key)
            return None

        try:
            produced = (
                _write_image_thumbnail(preview.path, path, self.max_dim)
                if preview.kind == "image"
                else _write_video_thumbnail(preview.path, path, self.max_dim)
            )
        except Exception:
            produced = False

        if not produced:
            self._failures.add(key)
            return None
        return _to_data_uri(path)

    def prune(self, keep_keys: set[str], limit: int = 4000) -> int:
        """Drop stale cache files once the directory grows past ``limit``."""
        if not os.path.isdir(self.cache_dir):
            return 0
        try:
            names = os.listdir(self.cache_dir)
        except OSError:
            return 0
        if len(names) <= limit:
            return 0

        removed = 0
        for name in names:
            key = os.path.splitext(name)[0]
            if key in keep_keys:
                continue
            try:
                os.remove(os.path.join(self.cache_dir, name))
                removed += 1
            except OSError:
                pass
        return removed


def _to_data_uri(path: str) -> str | None:
    try:
        with open(path, "rb") as handle:
            payload = base64.b64encode(handle.read()).decode("ascii")
    except OSError:
        return None
    return f"data:image/webp;base64,{payload}"


def _write_image_thumbnail(source: str, destination: str, max_dim: int) -> bool:
    try:
        from PIL import Image
    except ImportError:
        return False

    with Image.open(source) as image:
        image.load()
        if image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGB")
        image.thumbnail((max_dim, max_dim), Image.LANCZOS)
        image.convert("RGB").save(destination, "WEBP", quality=82, method=4)
    return True


def _write_video_thumbnail(source: str, destination: str, max_dim: int) -> bool:
    """Decode only the first usable frame, then release the video handle.

    WanGP already ships OpenCV and Pillow, so no new dependency is introduced.
    The video is never streamed to the browser.
    """
    try:
        import cv2
        from PIL import Image
    except ImportError:
        return False

    capture = cv2.VideoCapture(source)
    try:
        if not capture.isOpened():
            return False
        ok, frame = capture.read()
        if not ok or frame is None:
            # Some containers cannot serve frame 0; nudge forward once before
            # giving up and falling back to a placeholder.
            capture.set(cv2.CAP_PROP_POS_FRAMES, 1)
            ok, frame = capture.read()
        if not ok or frame is None:
            return False
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()

    image = Image.fromarray(rgb)
    image.thumbnail((max_dim, max_dim), Image.LANCZOS)
    image.save(destination, "WEBP", quality=82, method=4)
    return True
