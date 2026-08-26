"""Stack profiles: lightweight, LoRA-only recalls.

A profile deliberately stores nothing but the user's LoRA stack — no prompt,
resolution, steps or seed.  Those belong to WanGP presets and settings files;
duplicating them here would create a second, competing configuration system.

A profile is **not** tied to the model it was saved under.  Whole model
families (every LTX 2 variant, say) share one LoRA folder and can load each
other's LoRAs, so the model key would be a false barrier.  Availability is the
only rule: a profile applies as far as the LoRAs in it are visible to the
current model.  ``model_key`` is kept purely as provenance.

Profiles never carry accelerator/system LoRAs, which WanGP manages on its own.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
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

    def names(self, inventory_ids: Iterable[str] | None = None) -> list[str]:
        """Profile names, the ones this model can apply in full listed first."""
        complete, incomplete = self.partition(inventory_ids)
        return complete + incomplete

    def partition(self, inventory_ids: Iterable[str] | None = None) -> tuple[list[str], list[str]]:
        """Split the profiles into (fully available, partly missing) names.

        The model a profile was saved under plays no part: only whether the
        current inventory can supply its LoRAs.  Passing ``None`` means "the
        inventory is unknown", and nothing is called incomplete.
        """
        stored = [name for name, payload in self._profiles.items() if isinstance(payload, dict)]
        if inventory_ids is None:
            return sorted(stored, key=str.casefold), []

        known = {normalize_id(value) for value in inventory_ids}
        complete, incomplete = [], []
        for name in stored:
            profile = self.get(name)
            missing = profile is not None and any(entry.id not in known for entry in profile.loras)
            (incomplete if missing else complete).append(name)
        return sorted(complete, key=str.casefold), sorted(incomplete, key=str.casefold)

    def missing_ids(self, profile: Profile, inventory_ids: Iterable[str]) -> list[str]:
        """LoRA ids in ``profile`` that the current model cannot see."""
        known = {normalize_id(value) for value in inventory_ids}
        return [entry.id for entry in profile.loras if entry.id not in known]

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

    def match(self, entries: list[ProfileEntry]) -> str:
        """Name of the profile that exactly equals the current stack, else "".

        Used so the dropdown never claims profile "X" is active once external
        state (a preset, an .lset, imported media settings) has diverged.  A
        stack that is on screen is by definition available, so the model a
        matching profile was saved under is irrelevant here too.
        """
        signature = _signature(entries)
        for name in self._profiles:
            profile = self.get(name)
            if profile is None:
                continue
            if _signature(profile.loras) == signature:
                return name
        return ""


def _signature(entries: list[ProfileEntry]) -> tuple:
    """Order matters: WanGP multiplier tokens are positional."""
    return tuple((entry.id, str(entry.multiplier).strip()) for entry in entries)
