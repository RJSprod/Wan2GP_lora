"""The state model shared by the panel and WanGP's native components.

Everything in here is pure: it takes native LoRA state in, applies a typed
frontend action, and returns the native state to write back.  Keeping it free
of Gradio makes the reconciliation rules -- which are where multiplier
corruption would come from -- directly testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import multiplier_codec as codec
from .inventory import Inventory
from .utils import format_number, is_finite_number, normalize_id

PHASE_2_TILING_GUIDANCE_VALUE = "2~"

#: The slider covers the common 0..1 range at 0.01 steps. Values outside it are
#: entered in the numeric field, which accepts the full supported range; the
#: slider disables itself rather than silently clamping a value it cannot show.
SLIDER_MIN = 0.0
SLIDER_MAX = 1.0
SLIDER_STEP = 0.01

#: Hard bounds for the numeric field. Negative multipliers are legitimate
#: (they invert a LoRA's effect) and WanGP does not cap the upper end either.
VALUE_MIN = -10.0
VALUE_MAX = 10.0

#: Field separator used only inside the canonical fingerprint below.
_SIGNATURE_SEP = "\x1f"


@dataclass(frozen=True)
class PhaseConfig:
    """How many multiplier phases exist, and how many are editable right now.

    These are deliberately two numbers.  MiniMax H3 declares
    ``lora_multiplier_phases = 2`` in its model definition whatever the
    guidance mode is, so in One Phase mode the token may still legitimately
    carry two values -- which is how a phase 2 the user tuned survives a trip
    through One Phase without the plugin having to hide it somewhere.
    """

    #: Values a native token may declare; WanGP rejects a token with more.
    capacity: int = 1
    #: Sliders the editor shows for the current guidance mode.
    effective: int = 1

    @property
    def hidden(self) -> int:
        return max(0, self.capacity - self.effective)


def resolve_phases(model_def: dict | None, guidance_phases_value: Any) -> PhaseConfig:
    """Resolve phase capability from current WanGP data, not from a model name.

    Mirrors ``wgp.py``: the guidance dropdown can hold the string ``"2~"``
    ("Two Phases with Tiling"), which is still two multiplier phases -- tiling
    changes how phase 2 is processed, it does not add a third phase.
    """
    model_def = model_def or {}

    if guidance_phases_value == PHASE_2_TILING_GUIDANCE_VALUE:
        guidance_count = 2
    else:
        try:
            guidance_count = int(guidance_phases_value or 0)
        except (TypeError, ValueError):
            guidance_count = 0

    max_phases = int(model_def.get("guidance_max_phases", 0) or 0)
    if model_def.get("lock_guidance_phases", False):
        guidance_count = max_phases
    elif max_phases:
        guidance_count = min(guidance_count, max_phases)

    capacity = model_def.get("lora_multiplier_phases", guidance_count)
    try:
        capacity = int(capacity or guidance_count or 1)
    except (TypeError, ValueError):
        capacity = guidance_count or 1
    capacity = max(1, capacity)

    return PhaseConfig(capacity=capacity, effective=max(1, min(capacity, guidance_count or 1)))


def phase_labels(phases: int) -> list[str]:
    """Generic labels only -- mislabelling a phase is worse than not naming it."""
    return ["Strength"] if phases <= 1 else [f"Phase {index + 1}" for index in range(phases)]


def sanitize_value(value: Any, fallback: float = 1.0) -> float:
    """Reject NaN/Infinity and out-of-range values coming from the frontend.

    Anything outside the supported range falls back to the previous value
    rather than being clamped, so a typo cannot quietly become -10.
    """
    if not is_finite_number(value):
        return fallback
    number = round(float(value), 2)
    if number < VALUE_MIN or number > VALUE_MAX:
        return fallback
    return number


@dataclass
class Stack:
    """The selected LoRAs and their multiplier tokens, in native order."""

    ids: list[str] = field(default_factory=list)
    tokens: list[str] = field(default_factory=list)
    separator_index: int = -1

    @classmethod
    def from_native(cls, selected, multipliers: str) -> "Stack":
        ids = [normalize_id(value) for value in (selected or []) if normalize_id(value)]
        parsed = codec.parse_multipliers(multipliers)
        return cls(
            ids=ids,
            tokens=codec.align_tokens(parsed.tokens, len(ids)),
            separator_index=min(parsed.separator_index, len(ids)) if parsed.has_separator else -1,
        )

    def index_of(self, lora_id: str) -> int:
        target = normalize_id(lora_id)
        for position, value in enumerate(self.ids):
            if value == target:
                return position
        return -1

    def is_system(self, position: int) -> bool:
        """LoRAs left of the bar were placed by an accelerator profile."""
        return self.separator_index >= 0 and 0 <= position < self.separator_index

    def token_for(self, lora_id: str) -> str:
        position = self.index_of(lora_id)
        return self.tokens[position] if position >= 0 else codec.DEFAULT_TOKEN

    def add(self, lora_id: str, capacity: int) -> bool:
        """Include a LoRA on the user side of the stack, at 1.0 per phase.

        A newly included LoRA never inherits the strength of whichever row
        previously occupied its index.
        """
        lora_id = normalize_id(lora_id)
        if not lora_id or self.index_of(lora_id) >= 0:
            return False
        self.ids.append(lora_id)
        self.tokens.append(codec.default_token(capacity))
        return True

    def remove(self, lora_id: str) -> bool:
        position = self.index_of(lora_id)
        if position < 0:
            return False
        self.ids.pop(position)
        self.tokens.pop(position)
        self.separator_index = codec.separator_after_removal(self.separator_index, [position])
        return True

    def set_token(self, lora_id: str, token: str) -> bool:
        position = self.index_of(lora_id)
        if position < 0:
            return False
        self.tokens[position] = str(token or codec.DEFAULT_TOKEN).strip() or codec.DEFAULT_TOKEN
        return True

    def to_native(self, inventory: Inventory | None = None) -> tuple[list[str], str]:
        """Selections and multiplier tokens, emitted in exactly the same order.

        A selected LoRA that the inventory cannot resolve keeps its original
        value: imported settings may legitimately reference a file that is not
        installed locally, and dropping it would silently damage them.
        """
        values = []
        for lora_id in self.ids:
            entry = inventory.get(lora_id) if inventory else None
            values.append(entry.native_value if entry else lora_id)
        return values, codec.serialize(self.tokens, self.separator_index)


def build_items(
    inventory: Inventory,
    stack: "Stack",
    phases: PhaseConfig,
    metadata=None,
    model_key: str = "",
    memory: dict[str, list[float]] | None = None,
    catalogue: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Browser tiles for everything WanGP can activate for this model.

    ``catalogue`` maps LoRA id -> CatalogueIndex, supplying the Civitai name and
    trigger words. Those travel with the payload so naming, search and sort all
    happen client-side with no round trip.
    """
    memory = memory or {}
    catalogue = catalogue or {}
    items = []
    for entry in inventory.entries:
        position = stack.index_of(entry.id)
        index = catalogue.get(entry.id)
        item: dict[str, Any] = {
            "id": entry.id,
            "name": entry.name,
            "label": entry.label,
            "parent": entry.parent,
            "active": position >= 0,
            "favorite": bool(metadata.is_favorite(model_key, entry.id)) if metadata else False,
            "tag": metadata.tag(model_key, entry.id) if metadata else "",
            "system_managed": stack.is_system(position),
            "missing": False,
            "civitai_name": getattr(index, "civitai_name", "") if index else "",
            "words": list(getattr(index, "trained_words", []) or []) if index else [],
            "has_catalogue": bool(getattr(index, "has_sidecar", False)) if index else False,
            "has_video": bool(entry.video_preview) if entry else False,
            "mtime": entry.mtime,
        }
        if position >= 0:
            item.update(multiplier_fields(stack.tokens[position], phases, entry.id, memory))
        items.append(item)
    return items


def build_active_rows(
    inventory: Inventory,
    stack: "Stack",
    phases: PhaseConfig,
    memory: dict[str, list[float]] | None = None,
    catalogue: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Editable rows, in native order, including LoRAs missing from disk."""
    memory = memory or {}
    catalogue = catalogue or {}
    rows = []
    for position, lora_id in enumerate(stack.ids):
        entry = inventory.get(lora_id)
        index = catalogue.get(lora_id)
        fallback_name = lora_id.rsplit("/", 1)[-1]
        row: dict[str, Any] = {
            "id": lora_id,
            "name": entry.name if entry else fallback_name,
            "label": entry.label if entry else fallback_name,
            "civitai_name": getattr(index, "civitai_name", "") if index else "",
            "has_catalogue": bool(getattr(index, "has_sidecar", False)) if index else False,
            "system_managed": stack.is_system(position),
            "missing": entry is None,
        }
        row.update(multiplier_fields(stack.tokens[position], phases, lora_id, memory))
        rows.append(row)
    return rows


def multiplier_fields(
    raw: str,
    phases: PhaseConfig,
    lora_id: str,
    memory: dict[str, list[float]] | None = None,
) -> dict[str, Any]:
    """Editor fields for one token: sliders when simple, raw text when not."""
    memory = memory if memory is not None else {}
    info = codec.classify(raw, phases.capacity)
    if info.kind == codec.ADVANCED:
        return {
            "multiplier_raw": info.raw,
            "multiplier_kind": codec.ADVANCED,
            "phase_values": [],
            "hidden_values": [],
            "linked": False,
        }

    values = full_values(info, phases, lora_id, memory)
    visible = values[: phases.effective]
    return {
        "multiplier_raw": info.raw,
        "multiplier_kind": codec.SIMPLE,
        "phase_values": [round(value, 4) for value in visible],
        "hidden_values": [round(value, 4) for value in values[phases.effective:]],
        # Linking is a UI preference the user sets, never inferred from the
        # values happening to be equal -- otherwise a freshly added LoRA (1;1)
        # would start linked and dragging phase 1 would silently move phase 2.
        "linked": False,
    }


def full_values(
    info: codec.TokenInfo,
    phases: PhaseConfig,
    lora_id: str,
    memory: dict[str, list[float]],
) -> list[float]:
    """Capacity-length phase vector, filling gaps from remembered values.

    Only the phases the token actually declared count as known; a token that
    was padded up to capacity must still defer to a remembered value.
    """
    return codec.rescale_values(list(info.declared), phases.capacity, memory.get(lora_id))


def remember_hidden_phases(stack: "Stack", phases: PhaseConfig, memory: dict[str, list[float]]) -> None:
    """Keep phase values the current mode cannot show.

    When a model's capacity is wide enough (H3 keeps two even in One Phase),
    the hidden value simply stays in the native token.  When capacity shrinks
    with the guidance mode, the token has to lose it -- WanGP rejects a token
    declaring more phases than the model supports -- so it is held here and
    restored the moment the mode comes back.
    """
    for position, lora_id in enumerate(stack.ids):
        info = codec.classify(stack.tokens[position], phases.capacity)
        if info.kind != codec.SIMPLE:
            continue
        values = list(info.declared)
        remembered = memory.get(lora_id, [])
        if len(remembered) > len(values):
            # Keep the remembered tail; refresh the head from the live token.
            values = values + remembered[len(values):]
        memory[lora_id] = values


def normalize_stack_tokens(stack: "Stack", phases: PhaseConfig, memory: dict[str, list[float]]) -> bool:
    """Rewrite simple tokens so they declare exactly ``phases.capacity`` values.

    Advanced tokens are left byte-for-byte alone.  Returns ``True`` when
    anything actually changed, so the caller can avoid a pointless native write
    (and the change-event loop it would start).
    """
    changed = False
    for position, lora_id in enumerate(stack.ids):
        info = codec.classify(stack.tokens[position], phases.capacity)
        if info.kind != codec.SIMPLE:
            continue
        values = full_values(info, phases, lora_id, memory)
        token = codec.build_token(values, phases.capacity)
        if token != stack.tokens[position]:
            stack.tokens[position] = token
            changed = True
    return changed


def set_phase_value(
    stack: "Stack",
    lora_id: str,
    phase: int,
    value: Any,
    phases: PhaseConfig,
    memory: dict[str, list[float]],
    linked: bool = False,
) -> bool:
    """Write one slider back into the token, leaving hidden phases untouched.

    Editing a LoRA whose token is an advanced schedule is refused here; the
    frontend has to ask for an explicit conversion first, so a comma-based
    schedule is never flattened by an accidental drag.
    """
    position = stack.index_of(lora_id)
    if position < 0:
        return False

    info = codec.classify(stack.tokens[position], phases.capacity)
    if info.kind != codec.SIMPLE:
        return False

    values = full_values(info, phases, lora_id, memory)
    number = sanitize_value(value, values[0] if values else 1.0)

    if linked and phases.effective > 1:
        for index in range(phases.effective):
            values[index] = number
    elif 0 <= int(phase) < len(values):
        values[int(phase)] = number
    else:
        return False

    stack.tokens[position] = codec.build_token(values, phases.capacity)
    memory[lora_id] = list(values)
    return True


def convert_to_simple(stack: "Stack", lora_id: str, phases: PhaseConfig) -> bool:
    """Explicitly replace an advanced schedule with a plain multiplier."""
    position = stack.index_of(lora_id)
    if position < 0:
        return False
    stack.tokens[position] = codec.default_token(phases.capacity)
    return True


def stack_signature(selected, multipliers: str) -> str:
    """Canonical fingerprint of native state.

    The frontend compares this against what it last sent; when they match it
    knows the incoming payload is the echo of its own write and skips the
    resend that would otherwise loop.
    """
    ids = [normalize_id(value) for value in (selected or [])]
    parsed = codec.parse_multipliers(multipliers)
    tokens = codec.align_tokens(parsed.tokens, len(ids))
    return _SIGNATURE_SEP.join(ids) + _SIGNATURE_SEP + codec.serialize(tokens, parsed.separator_index)
