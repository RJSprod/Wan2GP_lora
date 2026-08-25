"""Read the Civitai sidecar folders produced by process_minimaxH3_lora.py.

That script writes, next to each ``<stem>.safetensors``:

    <stem>/
        <stem>.json                  combined {modelVersion, model, sha256, ...}
        modelVersion.json
        model.json
        <stem>.txt                   trainedWords, one per line
        summary.txt                  readable summary
        media/
            001.jpg | 001.mp4        downloaded Civitai media
            001.json                 {index, source_url, civitai: {...}}

Two different read paths on purpose:

* :func:`read_index` is the cheap one, used to build the browser's search
  index for every LoRA on every payload. It reads ``summary.txt`` (a few
  hundred bytes) rather than ``model.json`` (which can be megabytes), so
  indexing a large library stays fast.
* :func:`read_detail` is the expensive one, used only when the user opens
  Inspect on a single LoRA.
"""

from __future__ import annotations

import html
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

MEDIA_DIRNAME = "media"
SUMMARY_FILENAME = "summary.txt"

IMAGE_EXTENSIONS = (".png", ".webp", ".jpg", ".jpeg", ".gif", ".avif")
VIDEO_EXTENSIONS = (".mp4", ".webm", ".mov", ".m4v", ".mkv")

#: ``summary.txt`` field labels we care about.
_SUMMARY_FIELDS = {
    "Civitai model name": "civitai_name",
    "Civitai model ID": "model_id",
    "Civitai version name": "version_name",
    "Civitai version ID": "version_id",
    "Base model": "base_model",
    "Creator": "creator",
}

#: Opening tags count as breaks too, otherwise "Line 2<ul><li>a" would run
#: the list item straight onto the preceding sentence.
_TAG_BREAK = re.compile(
    r"(?i)<\s*/?\s*(br|p|div|li|ul|ol|tr|h[1-6]|blockquote)\b[^>]*>"
)
#: Drop script/style bodies outright; their text is meaningless here and
#: carrying it forward would put remote code text into the panel.
_TAG_DROP = re.compile(r"(?is)<\s*(script|style)\b[^>]*>.*?<\s*/\s*\1\s*>")
_TAG_ANY = re.compile(r"<[^>]+>")
_BLANK_RUN = re.compile(r"\n{3,}")


@dataclass
class CatalogueIndex:
    """Cheap, searchable facts about one LoRA."""

    civitai_name: str = ""
    version_name: str = ""
    creator: str = ""
    base_model: str = ""
    trained_words: list[str] = field(default_factory=list)
    has_sidecar: bool = False

    @property
    def searchable(self) -> str:
        parts = [self.civitai_name, self.version_name, self.creator] + self.trained_words
        return " ".join(part for part in parts if part).lower()


@dataclass
class MediaItem:
    index: int
    kind: str          # "image" or "video"
    path: str          # absolute; never serialised to the browser
    prompt: str = ""
    negative_prompt: str = ""
    width: int = 0
    height: int = 0
    source_url: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class CatalogueDetail:
    """Everything the Inspect view shows for one LoRA."""

    civitai_name: str = ""
    version_name: str = ""
    creator: str = ""
    base_model: str = ""
    description: str = ""
    version_description: str = ""
    trained_words: list[str] = field(default_factory=list)
    civitai_url: str = ""
    sha256: str = ""
    detection: str = ""
    media: list[MediaItem] = field(default_factory=list)
    error: str = ""


def sidecar_dir(lora_path: str) -> str | None:
    """The ``<stem>/`` folder beside a ``.safetensors``, when it exists."""
    if not lora_path:
        return None
    stem = os.path.splitext(os.path.basename(lora_path))[0]
    candidate = os.path.join(os.path.dirname(lora_path), stem)
    return candidate if os.path.isdir(candidate) else None


def strip_html(value: Any) -> str:
    """Civitai descriptions are HTML; the panel renders them as plain text.

    Block-level tags become newlines so paragraphs and lists stay readable,
    everything else is dropped, and entities are unescaped. Rendering the
    original markup would mean trusting remote HTML inside the WanGP page.
    """
    text = str(value or "")
    if not text:
        return ""
    text = _TAG_DROP.sub("", text)
    text = _TAG_BREAK.sub("\n", text)
    text = _TAG_ANY.sub("", text)
    text = html.unescape(text)
    text = "\n".join(line.rstrip() for line in text.replace("\r", "").split("\n"))
    return _BLANK_RUN.sub("\n\n", text).strip()


def _load_json(path: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def parse_summary(text: str) -> tuple[dict[str, str], list[str]]:
    """Pull the labelled fields and the trained-word bullets out of summary.txt."""
    fields: dict[str, str] = {}
    words: list[str] = []
    in_words = False

    for raw_line in str(text or "").replace("\r", "").split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("Trained words:"):
            in_words = True
            continue
        if in_words:
            if line.startswith("- "):
                word = line[2:].strip()
                if word:
                    words.append(word)
                continue
            if line.startswith("("):
                continue
            in_words = False

        label, separator, value = line.partition(":")
        if separator and label.strip() in _SUMMARY_FIELDS:
            fields[_SUMMARY_FIELDS[label.strip()]] = value.strip()

    return fields, words


def read_index(lora_path: str) -> CatalogueIndex:
    """Cheap per-LoRA index used for display names and search."""
    directory = sidecar_dir(lora_path)
    if directory is None:
        return CatalogueIndex()

    stem = os.path.splitext(os.path.basename(lora_path))[0]
    summary_path = os.path.join(directory, SUMMARY_FILENAME)

    fields: dict[str, str] = {}
    words: list[str] = []
    if os.path.isfile(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as handle:
                fields, words = parse_summary(handle.read())
        except OSError:
            pass

    if not words:
        # Fall back to the trained-words text file the script also writes.
        words_path = os.path.join(directory, f"{stem}.txt")
        try:
            with open(words_path, "r", encoding="utf-8") as handle:
                words = [line.strip() for line in handle if line.strip()]
        except OSError:
            words = []

    if not fields.get("civitai_name"):
        # summary.txt missing or older format: pay for model.json just this once.
        model = _load_json(os.path.join(directory, "model.json"))
        if isinstance(model, dict):
            fields["civitai_name"] = str(model.get("name", "") or "")

    return CatalogueIndex(
        civitai_name=fields.get("civitai_name", ""),
        version_name=fields.get("version_name", ""),
        creator=fields.get("creator", ""),
        base_model=fields.get("base_model", ""),
        trained_words=words,
        has_sidecar=True,
    )


def _media_files(media_dir: str) -> dict[int, str]:
    """Map media index -> file path, ignoring the .json sidecars."""
    found: dict[int, str] = {}
    try:
        names = sorted(os.listdir(media_dir))
    except OSError:
        return found

    for name in names:
        stem, extension = os.path.splitext(name)
        extension = extension.lower()
        if extension in (".json", ".part") or not stem.isdigit():
            continue
        if extension not in IMAGE_EXTENSIONS + VIDEO_EXTENSIONS:
            continue
        found.setdefault(int(stem), os.path.join(media_dir, name))
    return found


def _media_kind(path: str) -> str:
    extension = os.path.splitext(path)[1].lower()
    return "video" if extension in VIDEO_EXTENSIONS else "image"


def read_detail(lora_path: str) -> CatalogueDetail:
    """Full catalogue for the Inspect view. Only called for one LoRA at a time."""
    directory = sidecar_dir(lora_path)
    if directory is None:
        return CatalogueDetail(error="No catalogue folder was found next to this LoRA.")

    stem = os.path.splitext(os.path.basename(lora_path))[0]

    combined = _load_json(os.path.join(directory, f"{stem}.json"))
    if isinstance(combined, dict):
        version = combined.get("modelVersion") or {}
        model = combined.get("model") or {}
        sha256 = str(combined.get("sha256", "") or "")
        detection = str(combined.get("minimaxH3Detection", "") or "")
    else:
        version = _load_json(os.path.join(directory, "modelVersion.json")) or {}
        model = _load_json(os.path.join(directory, "model.json")) or {}
        sha256 = ""
        detection = ""

    if not isinstance(version, dict):
        version = {}
    if not isinstance(model, dict):
        model = {}

    creator = ""
    if isinstance(model.get("creator"), dict):
        creator = str(model["creator"].get("username", "") or "")

    words = [str(word) for word in version.get("trainedWords", []) or [] if str(word).strip()]
    index = read_index(lora_path)
    if not words:
        words = index.trained_words

    detail = CatalogueDetail(
        civitai_name=str(model.get("name", "") or index.civitai_name),
        version_name=str(version.get("name", "") or index.version_name),
        creator=creator or index.creator,
        base_model=str(version.get("baseModel", "") or index.base_model),
        description=strip_html(model.get("description")),
        version_description=strip_html(version.get("description")),
        trained_words=words,
        sha256=sha256,
        detection=detection,
        media=_read_media(directory),
    )

    model_id = model.get("id") or version.get("modelId")
    version_id = version.get("id")
    if model_id and version_id:
        detail.civitai_url = f"https://civitai.com/models/{model_id}?modelVersionId={version_id}"
    elif model_id:
        detail.civitai_url = f"https://civitai.com/models/{model_id}"

    return detail


def _read_media(directory: str) -> list[MediaItem]:
    media_dir = os.path.join(directory, MEDIA_DIRNAME)
    files = _media_files(media_dir)
    items: list[MediaItem] = []

    for index in sorted(files):
        path = files[index]
        sidecar = _load_json(os.path.join(media_dir, f"{index:03d}.json"))
        civitai = {}
        source_url = ""
        if isinstance(sidecar, dict):
            source_url = str(sidecar.get("source_url", "") or "")
            if isinstance(sidecar.get("civitai"), dict):
                civitai = sidecar["civitai"]

        meta = civitai.get("meta") if isinstance(civitai.get("meta"), dict) else {}
        items.append(
            MediaItem(
                index=index,
                kind=_media_kind(path),
                path=path,
                prompt=str(meta.get("prompt", "") or ""),
                negative_prompt=str(meta.get("negativePrompt", "") or ""),
                width=int(civitai.get("width") or 0),
                height=int(civitai.get("height") or 0),
                source_url=source_url,
                meta=meta,
            )
        )
    return items
