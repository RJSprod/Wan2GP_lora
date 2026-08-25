"""Turn WanGP's own LoRA list into stable, displayable inventory entries.

The plugin never performs its own authoritative filesystem scan.  WanGP builds
``state["loras"]`` (via ``setup_loras`` / ``refresh_lora_list``) and that list
decides what can be activated; this module only adds display metadata and
resolves an absolute path so previews can be looked up next to each file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from .utils import display_name, normalize_id, parent_name


@dataclass
class InventoryEntry:
    #: Stable plugin identity — the normalised relative path WanGP itself uses.
    id: str
    #: The exact string WanGP put in its choices, written back verbatim.
    native_value: str
    #: Basename shown on the tile.
    name: str
    #: Relative parent folder; used for disambiguation and search only.
    parent: str = ""
    #: Tile label, equal to ``name`` unless a basename collides.
    label: str = ""
    #: Absolute path on disk. Stays on the Python side — never serialised.
    path: str | None = None
    #: Modification time, used only for the "recently added" sort order.
    mtime: float = 0.0
    #: True when a same-stem video preview exists, so the tile can offer a
    #: play control instead of only a still frame.
    video_preview: bool = False

    @property
    def exists(self) -> bool:
        return bool(self.path) and os.path.isfile(self.path)


@dataclass
class Inventory:
    entries: list[InventoryEntry] = field(default_factory=list)
    lora_dir: str = ""

    def __post_init__(self) -> None:
        self._by_id = {entry.id: entry for entry in self.entries}

    @property
    def ids(self) -> list[str]:
        return [entry.id for entry in self.entries]

    def get(self, value: str) -> InventoryEntry | None:
        """Look an entry up by stable ID or by any equivalent native value."""
        return self._by_id.get(normalize_id(value))

    def native_value(self, lora_id: str) -> str | None:
        entry = self.get(lora_id)
        return entry.native_value if entry else None

    def to_native(self, ids: list[str]) -> list[str]:
        """Map plugin IDs back to native values, dropping unknown ones.

        Unknown IDs are dropped rather than passed through: a frontend must
        never be able to introduce a LoRA that WanGP did not offer.
        """
        values = []
        for lora_id in ids:
            entry = self.get(lora_id)
            if entry is not None:
                values.append(entry.native_value)
        return values


def _resolve_path(lora_dir: str, lora_id: str) -> str | None:
    """Absolute path for a LoRA, or ``None`` when it cannot be placed on disk."""
    if not lora_dir:
        return None
    if os.path.isabs(lora_id):
        return lora_id
    if lora_id.startswith(("http:", "https:")):
        return None
    return os.path.normpath(os.path.join(lora_dir, lora_id))


def build_inventory(native_values, lora_dir: str = "") -> Inventory:
    """Build the flat, navigation-free inventory shown in the grid.

    Subdirectories are flattened: WanGP can nest LoRAs, but the browser has no
    folder navigation, so the parent folder survives only as a disambiguator
    and as a searchable (though not displayed) field.
    """
    entries: list[InventoryEntry] = []
    seen: set[str] = set()

    for value in native_values or []:
        lora_id = normalize_id(value)
        if not lora_id or lora_id in seen:
            continue
        seen.add(lora_id)
        entries.append(
            InventoryEntry(
                id=lora_id,
                native_value=str(value),
                name=display_name(lora_id),
                parent=parent_name(lora_id),
                label=display_name(lora_id),
                path=_resolve_path(lora_dir, lora_id),
            )
        )

    _apply_disambiguators(entries)
    _annotate_files(entries)
    return Inventory(entries=entries, lora_dir=lora_dir)


def _annotate_files(entries: list[InventoryEntry]) -> None:
    """Fill in mtime and video-preview presence with one listing per folder."""
    from .thumbnails import VIDEO_EXTENSIONS

    listings: dict[str, dict[str, str]] = {}
    for entry in entries:
        if not entry.path:
            continue
        directory = os.path.dirname(entry.path)
        if directory not in listings:
            try:
                listings[directory] = {name.lower(): name for name in os.listdir(directory)}
            except OSError:
                listings[directory] = {}
        listing = listings[directory]

        try:
            entry.mtime = os.path.getmtime(entry.path)
        except OSError:
            entry.mtime = 0.0

        stem = os.path.splitext(os.path.basename(entry.path))[0].lower()
        entry.video_preview = any(
            f"{stem}{extension}" in listing for extension in VIDEO_EXTENSIONS
        )


def _apply_disambiguators(entries: list[InventoryEntry]) -> None:
    """Only collided basenames get a parent-folder suffix; the rest stay clean."""
    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry.name] = counts.get(entry.name, 0) + 1

    for entry in entries:
        if counts.get(entry.name, 0) > 1 and entry.parent:
            entry.label = f"{entry.name}  ·  {entry.parent.rsplit('/', 1)[-1]}"
        else:
            entry.label = entry.name


def diff_ids(old_ids, new_ids) -> tuple[list[str], list[str]]:
    """Added and removed IDs, for the Refresh status line."""
    old_set = {normalize_id(value) for value in old_ids or []}
    new_set = {normalize_id(value) for value in new_ids or []}
    added = [value for value in new_ids or [] if normalize_id(value) not in old_set]
    removed = [value for value in old_ids or [] if normalize_id(value) not in new_set]
    return added, removed


def missing_ids(selected, inventory: Inventory) -> list[str]:
    """Selected LoRAs that the current model's inventory cannot resolve.

    These stay in native state and are shown as "missing locally" rows rather
    than being dropped, so imported settings are never silently damaged.
    """
    return [
        normalize_id(value)
        for value in selected or []
        if inventory.get(value) is None
    ]
