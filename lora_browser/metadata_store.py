"""Versioned, atomically-written plugin preferences.

Holds only presentation metadata — favourites, tags, zoom, stack profiles, and
the optional Civitai API key used by catalogue fetches.  Nothing here influences
generation; WanGP's own state remains canonical.

The API key is stored as written, in the same plain JSON as everything else, so
treat this file as a credential file if you set one.  ``CIVITAI_API_KEY`` in the
environment is read first and never written here, which is the better option on
a shared machine.

The file is written next to WanGP's ``wgp_config.json`` when the plugin can
discover that location, so plugin state is never committed into the plugin's
own git checkout (which the Plugin Manager may update or reinstall).
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from typing import Any

SCHEMA_VERSION = 1
STORE_FILENAME = "wan2gp_lora_browser.json"

DEFAULT_ZOOM_PX = 104
MIN_ZOOM_PX = 72
MAX_ZOOM_PX = 176


#: How the grid is ordered. Purely presentational; never affects the native
#: multiplier order, which stays positional.
SORT_MODES = ("name", "civitai", "recent", "active", "favorite")
DEFAULT_SORT = "name"

#: Which name a tile shows. Civitai names are usually more coherent than the
#: downloaded filename, so they are preferred when a sidecar supplies one.
NAME_MODES = ("civitai", "file")
DEFAULT_NAME_MODE = "civitai"


def default_document() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "zoom_px": DEFAULT_ZOOM_PX,
        "sort_mode": DEFAULT_SORT,
        "name_mode": DEFAULT_NAME_MODE,
        "favorites": {},
        "tags": {},
        "profiles": {},
        "default_profiles": {},
        "civitai_api_key": "",
    }


def scope_key(model_key: str, lora_id: str) -> str:
    """Favourites and tags are per model family, so the same filename in two
    model directories does not share a star."""
    return f"{model_key or 'default'}|{lora_id}"


class MetadataStore:
    def __init__(self, path: str):
        self.path = path
        self.data = default_document()
        self._dirty = False
        self.load()

    # ------------------------------------------------------------------ io

    def load(self) -> None:
        """Read the store, quarantining a corrupt file instead of crashing."""
        if not self.path or not os.path.isfile(self.path):
            self.data = default_document()
            return
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if not isinstance(payload, dict):
                raise ValueError("metadata root must be an object")
        except Exception as error:
            print(f"[LoRA Browser] Unreadable metadata file ({error}); starting from defaults.")
            self._quarantine()
            self.data = default_document()
            return
        self.data = self._migrate(payload)

    def _quarantine(self) -> None:
        try:
            shutil.copyfile(self.path, f"{self.path}.corrupt-{int(time.time())}")
        except OSError:
            pass

    def _migrate(self, payload: dict[str, Any]) -> dict[str, Any]:
        document = default_document()
        version = int(payload.get("schema_version", 0) or 0)
        if version > SCHEMA_VERSION:
            # Written by a newer plugin: keep unknown keys so downgrading and
            # upgrading again does not throw the user's data away.
            document.update(payload)
            return document
        for key in ("favorites", "tags", "profiles", "default_profiles"):
            value = payload.get(key)
            if isinstance(value, dict):
                document[key] = value
        document["zoom_px"] = self.sanitize_zoom(payload.get("zoom_px"))
        document["sort_mode"] = self.sanitize_choice(
            payload.get("sort_mode"), SORT_MODES, DEFAULT_SORT
        )
        document["name_mode"] = self.sanitize_choice(
            payload.get("name_mode"), NAME_MODES, DEFAULT_NAME_MODE
        )
        document["civitai_api_key"] = str(payload.get("civitai_api_key", "") or "").strip()
        document["schema_version"] = SCHEMA_VERSION
        return document

    def save(self) -> bool:
        """Atomic write: temp file in the same directory, then replace."""
        if not self.path:
            return False
        directory = os.path.dirname(self.path) or "."
        try:
            os.makedirs(directory, exist_ok=True)
            handle = tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=directory, prefix=".lora_browser-", suffix=".tmp", delete=False
            )
            try:
                json.dump(self.data, handle, indent=2, ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
            finally:
                handle.close()
            os.replace(handle.name, self.path)
        except OSError as error:
            print(f"[LoRA Browser] Could not save metadata: {error}")
            return False
        self._dirty = False
        return True

    # -------------------------------------------------------------- values

    @staticmethod
    def sanitize_zoom(value: Any) -> int:
        try:
            zoom = int(round(float(value)))
        except (TypeError, ValueError):
            return DEFAULT_ZOOM_PX
        return max(MIN_ZOOM_PX, min(MAX_ZOOM_PX, zoom))

    @property
    def zoom_px(self) -> int:
        return self.sanitize_zoom(self.data.get("zoom_px"))

    def set_zoom(self, value: Any) -> int:
        zoom = self.sanitize_zoom(value)
        if zoom != self.data.get("zoom_px"):
            self.data["zoom_px"] = zoom
            self.save()
        return zoom

    @staticmethod
    def sanitize_choice(value: Any, allowed: tuple[str, ...], fallback: str) -> str:
        text = str(value or "").strip().lower()
        return text if text in allowed else fallback

    @property
    def sort_mode(self) -> str:
        return self.sanitize_choice(self.data.get("sort_mode"), SORT_MODES, DEFAULT_SORT)

    def set_sort_mode(self, value: Any) -> str:
        mode = self.sanitize_choice(value, SORT_MODES, DEFAULT_SORT)
        if mode != self.data.get("sort_mode"):
            self.data["sort_mode"] = mode
            self.save()
        return mode

    @property
    def name_mode(self) -> str:
        return self.sanitize_choice(self.data.get("name_mode"), NAME_MODES, DEFAULT_NAME_MODE)

    def set_name_mode(self, value: Any) -> str:
        mode = self.sanitize_choice(value, NAME_MODES, DEFAULT_NAME_MODE)
        if mode != self.data.get("name_mode"):
            self.data["name_mode"] = mode
            self.save()
        return mode

    @property
    def civitai_api_key(self) -> str:
        return str(self.data.get("civitai_api_key", "") or "")

    def set_civitai_api_key(self, value: Any) -> str:
        """Store (or clear) the key. Returns it so the caller can report state."""
        text = str(value or "").strip()[:200]
        if text != self.civitai_api_key:
            self.data["civitai_api_key"] = text
            self.save()
        return text

    def default_profile(self, model_key: str) -> str:
        return str(self.data.get("default_profiles", {}).get(model_key or "default", "") or "")

    def set_default_profile(self, model_key: str, name: str) -> str:
        defaults = self.data.setdefault("default_profiles", {})
        key = model_key or "default"
        text = str(name or "").strip()
        if text:
            defaults[key] = text
        else:
            defaults.pop(key, None)
        self.save()
        return text

    def is_favorite(self, model_key: str, lora_id: str) -> bool:
        return bool(self.data.get("favorites", {}).get(scope_key(model_key, lora_id)))

    def set_favorite(self, model_key: str, lora_id: str, value: bool) -> None:
        favorites = self.data.setdefault("favorites", {})
        key = scope_key(model_key, lora_id)
        if value:
            favorites[key] = True
        else:
            favorites.pop(key, None)
        self.save()

    def tag(self, model_key: str, lora_id: str) -> str:
        return str(self.data.get("tags", {}).get(scope_key(model_key, lora_id), "") or "")

    def set_tag(self, model_key: str, lora_id: str, value: str) -> str:
        tags = self.data.setdefault("tags", {})
        key = scope_key(model_key, lora_id)
        text = str(value or "").strip()[:64]
        if text:
            tags[key] = text
        else:
            tags.pop(key, None)
        self.save()
        return text


def resolve_store_path(config_filename: str = "", fallback_dir: str = "") -> str:
    """Prefer WanGP's config directory; fall back to a plugin data folder.

    ``config_filename`` is WanGP's ``server_config_filename`` global when the
    running build exposes it.
    """
    if config_filename:
        directory = os.path.dirname(os.path.abspath(config_filename))
        if directory:
            return os.path.join(directory, STORE_FILENAME)
    base = fallback_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    return os.path.join(base, STORE_FILENAME)
