"""Stack profiles: lightweight, LoRA-only recalls.

A profile deliberately stores nothing but the user's LoRA stack — no prompt,
resolution, steps or seed.  Those belong to WanGP presets and settings files;
duplicating them here would create a second, competing configuration system.

Profiles are scoped to the model they were saved under and never carry
accelerator/system LoRAs, which WanGP manages on its own.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .metadata_store import MetadataStore
from .utils import normalize_id

MAX_NAME_LENGTH = 64

#: Anything that could turn a display name into a filesystem path.  Profiles
#: live inside one JSON document, but the name is echoed into the UI and must
#: never be usable as a path fragment.
_UNSAFE_NAME = re.compile(r"[\\/:*?\"<>|\x00-\x1f]")


@dataclass
class ProfileEntry:
    id: str
    multiplier: str = "1"
    linked: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "multiplier": self.multiplier, "linked": bool(self.linked)}


@dataclass
class Profile:
    name: str
    model_key: str = ""
    loras: list[ProfileEntry] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_key": self.model_key,
            "loras": [entry.to_dict() for entry in self.loras],
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, name: str, payload: dict[str, Any]) -> "Profile":
        entries = []
        for item in payload.get("loras", []) or []:
            if not isinstance(item, dict):
                continue
            lora_id = normalize_id(item.get("id"))
            if not lora_id:
                continue
            entries.append(
                ProfileEntry(
                    id=lora_id,
                    multiplier=str(item.get("multiplier", "1") or "1"),
                    linked=bool(item.get("linked", False)),
                )
            )
        return cls(
            name=name,
            model_key=str(payload.get("model_key", "") or ""),
            loras=entries,
            note=str(payload.get("note", "") or ""),
        )


class ProfileError(Exception):
    """Raised for a rejected name or a missing profile."""


def sanitize_name(name: str) -> str:
    text = _UNSAFE_NAME.sub("", str(name or "")).strip()
    # Unicode display names are preserved; only path-dangerous characters and
    # leading dots (which would hide a file if a name ever became one) go.
    text = text.lstrip(".").strip()
    if not text:
        raise ProfileError("Profile name cannot be empty.")
    return text[:MAX_NAME_LENGTH]


class ProfileStore:
    def __init__(self, metadata: MetadataStore):
        self.metadata = metadata

    @property
    def _profiles(self) -> dict[str, Any]:
        return self.metadata.data.setdefault("profiles", {})

    def names(self, model_key: str = "") -> list[str]:
        """Profile names, compatible ones first, each sorted case-insensitively."""
        compatible, others = [], []
        for name, payload in self._profiles.items():
            if not isinstance(payload, dict):
                continue
            saved_key = str(payload.get("model_key", "") or "")
            (compatible if not model_key or saved_key == model_key else others).append(name)
        return sorted(compatible, key=str.casefold) + sorted(others, key=str.casefold)

    def get(self, name: str) -> Profile | None:
        payload = self._profiles.get(str(name or ""))
        if not isinstance(payload, dict):
            return None
        return Profile.from_dict(str(name), payload)

    def exists(self, name: str) -> bool:
        return str(name or "") in self._profiles

    def save(self, name: str, model_key: str, entries: list[ProfileEntry], overwrite: bool = False) -> Profile:
        clean = sanitize_name(name)
        if self.exists(clean) and not overwrite:
            raise ProfileError(f"A profile named '{clean}' already exists.")
        profile = Profile(name=clean, model_key=model_key or "", loras=list(entries))
        self._profiles[clean] = profile.to_dict()
        self.metadata.save()
        return profile

    def rename(self, old_name: str, new_name: str) -> Profile:
        if not self.exists(old_name):
            raise ProfileError(f"Profile '{old_name}' not found.")
        clean = sanitize_name(new_name)
        if clean != old_name and self.exists(clean):
            raise ProfileError(f"A profile named '{clean}' already exists.")
        self._profiles[clean] = self._profiles.pop(str(old_name))
        self.metadata.save()
        return Profile.from_dict(clean, self._profiles[clean])

    def delete(self, name: str) -> None:
        if not self.exists(name):
            raise ProfileError(f"Profile '{name}' not found.")
        self._profiles.pop(str(name))
        self.metadata.save()

    def is_compatible(self, profile: Profile, model_key: str) -> bool:
        """An unscoped legacy profile stays usable; a mismatched one does not."""
        return not profile.model_key or not model_key or profile.model_key == model_key

    def match(self, model_key: str, entries: list[ProfileEntry]) -> str:
        """Name of the profile that exactly equals the current stack, else "".

        Used so the dropdown never claims profile "X" is active once external
        state (a preset, an .lset, imported media settings) has diverged.
        """
        signature = _signature(entries)
        for name in self._profiles:
            profile = self.get(name)
            if profile is None or not self.is_compatible(profile, model_key):
                continue
            if _signature(profile.loras) == signature:
                return name
        return ""


def _signature(entries: list[ProfileEntry]) -> tuple:
    """Order matters: WanGP multiplier tokens are positional."""
    return tuple((entry.id, str(entry.multiplier).strip()) for entry in entries)
