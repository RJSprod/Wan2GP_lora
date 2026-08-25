"""WanGP LoRA Browser -- thumbnail LoRA browser and real-time multiplier editor.

This file owns only the WanGP/Gradio integration: requesting native components
and globals, injecting the panel after ``loras_multipliers``, and bridging JSON
between the browser and the helper modules in ``lora_browser/``.

Design rule for the whole plugin: WanGP stays the source of truth.  The panel
is a presentation and editing layer over ``loras_choices`` /
``loras_multipliers``; those components keep working underneath it, which is
what lets presets, ``.lset`` files, accelerator profiles, settings recovered
from generated media, queue edits and model switches continue to behave
normally.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import traceback
from typing import Any

import gradio as gr

from shared.utils.plugins import WAN2GPPlugin

from .lora_browser import catalogue as cat
from .lora_browser import ui_payloads as up
from .lora_browser.inventory import build_inventory, diff_ids
from .lora_browser.metadata_store import MetadataStore, resolve_store_path
from .lora_browser.profile_store import ProfileEntry, ProfileError, ProfileStore
from .lora_browser.thumbnails import (
    VIDEO_EXTENSIONS,
    Preview,
    ThumbnailCache,
    find_preview,
)
from .lora_browser.utils import is_within, normalize_id

PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(PLUGIN_DIR, "assets")

#: Prefixes for the elem_ids stamped on the components the panel replaces.
#: WanGP builds the media-generator tab more than once (the generation form and
#: the queue-edit form), so these must be per-instance -- a shared id would put
#: duplicate ids in the DOM and getElementById would resolve the second tab's
#: lookups to the first tab's controls.
#: Catalogue images are re-encoded to this size for the Inspect view: large
#: enough to judge a preview, small enough to inline.
MODAL_IMAGE_MAX_DIM = 1400

#: Hard ceiling on any single inlined file. Videos are fetched one at a time on
#: an explicit click, so this bounds the cost without exposing a static route.
MEDIA_INLINE_LIMIT = 32 * 1024 * 1024

NATIVE_CHOICES_ELEM_ID = "wgp_lora_browser_native_choices"
NATIVE_MULTIPLIERS_ELEM_ID = "wgp_lora_browser_native_multipliers"


def _read_asset(name: str) -> str:
    try:
        with open(os.path.join(ASSETS_DIR, name), "r", encoding="utf-8") as handle:
            return handle.read()
    except OSError as error:
        print(f"[LoRA Browser] Could not read asset '{name}': {error}")
        return ""


class InstanceState:
    """Per media-generator tab state. WanGP can build the tab more than once."""

    def __init__(self) -> None:
        self.revision = 0
        self.model_key = ""
        self.lora_dir = ""
        self.inventory_ids: list[str] = []
        #: Phase values the current mode cannot show, kept so switching modes
        #: does not erase a carefully chosen phase 2.
        self.phase_memory: dict[str, list[float]] = {}
        #: Snapshot taken by "Disable all"; invalidated when native state moves
        #: on its own, so Restore never overwrites newer settings unnoticed.
        self.restore_snapshot: tuple[list[str], str] | None = None
        self.restore_signature = ""
        self.pending_status = ""
        self.pending_warn = False
        self.ready = False
        #: lora id -> (sidecar mtime, CatalogueIndex). Reading summary.txt for
        #: every LoRA on every payload would be wasteful; the sidecar only
        #: changes when the enrichment script runs again.
        self.catalogue_cache: dict[str, tuple[float, Any]] = {}

    def next_revision(self) -> int:
        self.revision += 1
        return self.revision

    def take_status(self) -> tuple[str, bool]:
        status, warn = self.pending_status, self.pending_warn
        self.pending_status, self.pending_warn = "", False
        return status, warn

    def note(self, message: str, warn: bool = False) -> None:
        self.pending_status = message
        self.pending_warn = warn


class LoraBrowserPlugin(WAN2GPPlugin):
    def __init__(self) -> None:
        super().__init__()
        self.name = "LoRA Browser"
        self.version = "1.0.0"
        self.description = "Thumbnail LoRA browser and real-time multiplier editor for WanGP."
        self.type = ["extension"]

        self._instances: dict[Any, InstanceState] = {}
        self._metadata: MetadataStore | None = None
        self._profiles: ProfileStore | None = None
        self._thumbnails: ThumbnailCache | None = None

    # ------------------------------------------------------------- setup

    def setup_ui(self) -> None:
        # Component names are the local variable names in wgp.py's
        # generate_media_tab; the plugin manager resolves them from that scope.
        for component in ("loras_choices", "loras_multipliers", "guidance_phases", "state", "lset_name", "main"):
            self.request_component(component)

        # Optional globals: names have moved between WanGP builds, so every
        # read below goes through getattr with a fallback.
        for name in ("get_lora_dir", "get_model_def", "get_state_model_type", "refresh_lora_list", "server_config_filename"):
            self.request_global(name)

    def post_ui_setup(self, components: dict) -> dict:
        try:
            required = ("loras_choices", "loras_multipliers", "state", "main")
            missing = [name for name in required if components.get(name) is None]
            if missing:
                print(f"[LoRA Browser] Missing WanGP components {missing}; leaving the native LoRA UI in place.")
                return {}

            self._init_stores()
            self.insert_after(
                target_component_id="loras_multipliers",
                new_component_constructor=lambda: self._build_panel(components),
            )
            resolved = [name for name in self._global_requests if callable(getattr(self, name, None))
                        or isinstance(getattr(self, name, None), str)]
            print(f"[LoRA Browser] panel queued for injection after loras_multipliers; "
                  f"WanGP globals resolved: {resolved or 'none'}")
        except Exception:
            # Fail open: any setup error leaves the native controls usable.
            print("[LoRA Browser] Setup failed; the native LoRA controls remain visible.")
            traceback.print_exc()
        return {}

    def on_model_change(self, state: dict, model_type: str) -> None:
        """Drop per-model caches so the browser cannot render stale inventory."""
        for instance in self._instances.values():
            instance.model_key = ""
            instance.inventory_ids = []
            instance.phase_memory.clear()
            instance.catalogue_cache.clear()
            instance.restore_snapshot = None
            instance.next_revision()

    # ------------------------------------------------------------ stores

    def _init_stores(self) -> None:
        if self._metadata is not None:
            return
        config_filename = getattr(self, "server_config_filename", "") or ""
        store_path = resolve_store_path(config_filename, os.path.join(PLUGIN_DIR, "data"))
        self._metadata = MetadataStore(store_path)
        self._profiles = ProfileStore(self._metadata)
        self._thumbnails = ThumbnailCache(
            os.path.join(os.path.dirname(store_path), "lora_browser_thumbs")
        )

    def _instance(self, key: Any) -> InstanceState:
        if key not in self._instances:
            self._instances[key] = InstanceState()
        return self._instances[key]

    # -------------------------------------------------- WanGP accessors

    def _model_type(self, state: dict) -> str:
        getter = getattr(self, "get_state_model_type", None)
        if callable(getter):
            try:
                return str(getter(state) or "")
            except Exception:
                pass
        if isinstance(state, dict):
            return str(state.get("model_type", "") or "")
        return ""

    def _model_def(self, model_type: str) -> dict:
        getter = getattr(self, "get_model_def", None)
        if callable(getter) and model_type:
            try:
                return getter(model_type) or {}
            except Exception:
                pass
        return {}

    def _lora_dir(self, model_type: str) -> str:
        getter = getattr(self, "get_lora_dir", None)
        if callable(getter) and model_type:
            try:
                return str(getter(model_type) or "")
            except Exception:
                # A model without a LoRA directory is not an error worth
                # breaking the panel over; previews simply stay unresolved.
                pass
        return ""

    def _native_loras(self, state: dict, selected) -> list[str]:
        """WanGP's own selectable list -- never an independent filesystem scan."""
        loras = []
        if isinstance(state, dict):
            loras = list(state.get("loras", []) or [])
        if not loras:
            # Before setup_loras has populated state, at least show what is
            # already selected rather than an empty grid.
            loras = list(selected or [])
        return loras

    # ------------------------------------------------------------ panel

    def _build_panel(self, components: dict):
        loras_choices = components["loras_choices"]
        loras_multipliers = components["loras_multipliers"]
        state = components["state"]
        main = components["main"]
        guidance_phases = components.get("guidance_phases")
        lset_name = components.get("lset_name")

        instance_id = getattr(loras_multipliers, "_id", id(loras_multipliers))
        instance = self._instance(instance_id)

        namespace = f"wgp-lora-browser-{instance_id}"
        ids = {
            "__INST__": str(instance_id),
            "__NS__": namespace,
            "__ROOT__": f"lb_root_{instance_id}",
            "__PAYLOAD_ID__": f"lb_payload_{instance_id}",
            "__ACTION_ID__": f"lb_action_{instance_id}",
            "__ACTION_BTN__": f"lb_action_btn_{instance_id}",
            "__THUMB_REQ_ID__": f"lb_thumb_req_{instance_id}",
            "__THUMB_BTN__": f"lb_thumb_btn_{instance_id}",
            "__THUMB_RES_ID__": f"lb_thumb_res_{instance_id}",
            "__REFRESH_BTN__": f"lb_refresh_btn_{instance_id}",
            "__NATIVE_CHOICES__": f"{NATIVE_CHOICES_ELEM_ID}_{instance_id}",
            "__NATIVE_MULTIPLIERS__": f"{NATIVE_MULTIPLIERS_ELEM_ID}_{instance_id}",
        }

        # Give the native components stable DOM ids so the frontend can hide
        # their wrappers by id instead of guessing at Gradio's hashed classes.
        if not getattr(loras_choices, "elem_id", None):
            loras_choices.elem_id = ids["__NATIVE_CHOICES__"]
        if not getattr(loras_multipliers, "elem_id", None):
            loras_multipliers.elem_id = ids["__NATIVE_MULTIPLIERS__"]
        ids["__NATIVE_CHOICES__"] = loras_choices.elem_id
        ids["__NATIVE_MULTIPLIERS__"] = loras_multipliers.elem_id

        css = _read_asset("lora_browser.css").replace("__NS__", namespace)
        script = _read_asset("lora_browser.js")
        for placeholder, value in ids.items():
            script = script.replace(placeholder, value)

        sync_inputs = [state, loras_choices, loras_multipliers]
        if guidance_phases is not None:
            sync_inputs.append(guidance_phases)

        def sync(state_value, selected, multipliers, guidance=None):
            return self._build_payload(instance, state_value, selected, multipliers, guidance)

        def act(action_json, state_value, selected, multipliers, guidance=None):
            return self._apply_action(instance, action_json, state_value, selected, multipliers, guidance)

        def serve_media(request_json, state_value):
            return self._serve_media(instance, request_json, state_value)

        with gr.Column(elem_classes=["wgp-lora-browser-host"]) as panel:
            gr.HTML(f"<style>{css}</style>", visible=True)
            gr.HTML(f"<div id='{ids['__ROOT__']}'></div>")

            with gr.Row(visible=False):
                payload_box = gr.Text(elem_id=ids["__PAYLOAD_ID__"], value="")
                action_box = gr.Text(elem_id=ids["__ACTION_ID__"], value="")
                action_btn = gr.Button(elem_id=ids["__ACTION_BTN__"])
                thumb_req = gr.Text(elem_id=ids["__THUMB_REQ_ID__"], value="")
                thumb_btn = gr.Button(elem_id=ids["__THUMB_BTN__"])
                thumb_res = gr.Text(elem_id=ids["__THUMB_RES_ID__"], value="")
                refresh_btn = gr.Button(elem_id=ids["__REFRESH_BTN__"])

        # Install the frontend, then push the first payload into it.
        main.load(fn=None, js=f"() => {{ {script} }}")
        main.load(fn=sync, inputs=sync_inputs, outputs=[payload_box], show_progress="hidden")

        def bridge_js(method: str, element_id: str) -> str:
            return (
                "() => { const api = window.wgpLoraBrowser && window.wgpLoraBrowser['%s'];"
                " if (!api) { return; }"
                " const host = document.getElementById('%s');"
                " const field = host && host.querySelector('textarea, input');"
                " if (field) { api.%s(field.value); } }" % (instance_id, element_id, method)
            )

        payload_box.change(
            fn=None, js=bridge_js("apply", ids["__PAYLOAD_ID__"]), show_progress="hidden"
        )
        thumb_res.change(
            fn=None, js=bridge_js("applyMedia", ids["__THUMB_RES_ID__"]), show_progress="hidden"
        )

        # Native -> plugin. Component change events are the common trigger, so
        # presets, .lset files, imported media settings, accelerator profiles
        # and queue edits all resynchronise without being special-cased.
        watched = [loras_choices, loras_multipliers] + ([guidance_phases] if guidance_phases is not None else [])
        for component in watched:
            if hasattr(component, "change"):
                component.change(fn=sync, inputs=sync_inputs, outputs=[payload_box], show_progress="hidden")

        # Plugin -> native, as one transaction so selection and multipliers can
        # never be written out of step with each other.
        action_btn.click(
            fn=act,
            inputs=[action_box] + sync_inputs,
            outputs=[loras_choices, loras_multipliers, payload_box],
            show_progress="hidden",
        )

        thumb_btn.click(fn=serve_media, inputs=[thumb_req, state], outputs=[thumb_res], show_progress="hidden")

        self._wire_refresh(instance, refresh_btn, state, lset_name, loras_choices, sync, sync_inputs, payload_box)
        print(f"[LoRA Browser] panel built (instance {instance_id}); "
              f"native ids: {ids['__NATIVE_CHOICES__']}, {ids['__NATIVE_MULTIPLIERS__']}")
        return panel

    def _wire_refresh(self, instance, refresh_btn, state, lset_name, loras_choices, sync, sync_inputs, payload_box):
        """Reuse WanGP's own refresh so both lists rebuild the way it expects."""
        native_refresh = getattr(self, "refresh_lora_list", None)
        if not callable(native_refresh) or lset_name is None:
            refresh_btn.click(
                fn=lambda: instance.note("Refresh is unavailable in this WanGP build.", True),
                show_progress="hidden",
            ).then(fn=sync, inputs=sync_inputs, outputs=[payload_box], show_progress="hidden")
            return

        def refresh(state_value, lset_value, selected):
            before = list(self._native_loras(state_value, selected))
            try:
                updates = native_refresh(state_value, lset_value, selected)
            except Exception as error:
                traceback.print_exc()
                instance.note(f"Refresh failed: {error}", True)
                return gr.update(), gr.update()

            after = list(self._native_loras(state_value, selected))
            added, removed = diff_ids(before, after)
            instance.note(
                f"Refreshed - {len(added)} added - {len(removed)} removed - {len(after)} available"
            )
            return updates

        refresh_btn.click(
            fn=refresh,
            inputs=[state, lset_name, loras_choices],
            outputs=[lset_name, loras_choices],
            show_progress="hidden",
        ).then(fn=sync, inputs=sync_inputs, outputs=[payload_box], show_progress="hidden")

    # ---------------------------------------------------------- payload

    def _catalogue_index(self, instance: InstanceState, inventory) -> dict:
        """Civitai name and trigger words per LoRA, cached on sidecar mtime."""
        index: dict[str, Any] = {}
        for entry in inventory.entries:
            if not entry.path:
                continue
            directory = cat.sidecar_dir(entry.path)
            if directory is None:
                continue
            try:
                stamp = os.path.getmtime(directory)
            except OSError:
                stamp = 0.0

            cached = instance.catalogue_cache.get(entry.id)
            if cached is not None and cached[0] == stamp:
                index[entry.id] = cached[1]
                continue

            record = cat.read_index(entry.path)
            instance.catalogue_cache[entry.id] = (stamp, record)
            index[entry.id] = record
        return index

    def _context(self, instance: InstanceState, state_value, selected, multipliers, guidance):
        model_type = self._model_type(state_value)
        model_def = self._model_def(model_type)
        phases = up.resolve_phases(model_def, guidance)
        lora_dir = self._lora_dir(model_type)
        inventory = build_inventory(self._native_loras(state_value, selected), lora_dir)
        stack = up.Stack.from_native(selected, multipliers)

        instance.model_key = model_type
        instance.lora_dir = lora_dir
        instance.inventory_ids = inventory.ids
        up.remember_hidden_phases(stack, phases, instance.phase_memory)
        return model_type, inventory, stack, phases

    def _build_payload(self, instance: InstanceState, state_value, selected, multipliers, guidance) -> str:
        try:
            model_type, inventory, stack, phases = self._context(
                instance, state_value, selected, multipliers, guidance
            )
            status, warn = instance.take_status()

            entries = self._user_entries(stack, phases)
            signature = up.stack_signature(selected, multipliers)
            if instance.restore_snapshot and instance.restore_signature != signature:
                # Native state moved on its own; a stale snapshot must not be
                # applied over newer settings.
                instance.restore_snapshot = None

            missing = [row for row in up.build_active_rows(inventory, stack, phases, instance.phase_memory) if row["missing"]]
            if missing and not status:
                names = ", ".join(row["name"] for row in missing[:3])
                status = f"{len(missing)} selected LoRA(s) missing locally: {names}"
                warn = True

            catalogue = self._catalogue_index(instance, inventory)
            payload = {
                "revision": instance.next_revision(),
                "model_key": model_type,
                "phases": {"capacity": phases.capacity, "effective": phases.effective},
                "phase_labels": up.phase_labels(phases.effective),
                "items": up.build_items(
                    inventory, stack, phases, self._metadata, model_type,
                    instance.phase_memory, catalogue,
                ),
                "active": up.build_active_rows(
                    inventory, stack, phases, instance.phase_memory, catalogue,
                ),
                "profiles": self._profiles.names(model_type) if self._profiles else [],
                "active_profile": self._profiles.match(model_type, entries) if self._profiles else "",
                "default_profile": self._metadata.default_profile(model_type) if self._metadata else "",
                "zoom_px": self._metadata.zoom_px if self._metadata else 104,
                "sort_mode": self._metadata.sort_mode if self._metadata else "name",
                "name_mode": self._metadata.name_mode if self._metadata else "civitai",
                "slider_min": up.SLIDER_MIN,
                "slider_max": up.SLIDER_MAX,
                "slider_step": up.SLIDER_STEP,
                "value_min": up.VALUE_MIN,
                "value_max": up.VALUE_MAX,
                "can_restore": instance.restore_snapshot is not None,
                "signature": signature,
                "status": status,
                "status_warn": warn,
            }
            return json.dumps(payload)
        except Exception as error:
            traceback.print_exc()
            # The frontend shows the native controls again when it sees this.
            return json.dumps({"error": f"LoRA Browser could not render: {error}"})

    def _user_entries(self, stack: up.Stack, phases: up.PhaseConfig) -> list[ProfileEntry]:
        """User-editable part of the stack -- accelerator LoRAs are excluded."""
        entries = []
        for position, lora_id in enumerate(stack.ids):
            if stack.is_system(position):
                continue
            entries.append(ProfileEntry(id=lora_id, multiplier=stack.tokens[position]))
        return entries

    # ----------------------------------------------------------- actions

    def _apply_action(self, instance: InstanceState, action_json, state_value, selected, multipliers, guidance):
        no_change = (gr.update(), gr.update())
        try:
            action = json.loads(action_json or "{}")
        except (TypeError, ValueError):
            return no_change + (gr.update(),)

        try:
            model_type, inventory, stack, phases = self._context(
                instance, state_value, selected, multipliers, guidance
            )
            changed = self._dispatch(instance, action, stack, inventory, phases, model_type)

            if not changed:
                payload = self._build_payload(instance, state_value, selected, multipliers, guidance)
                return no_change + (payload,)

            up.normalize_stack_tokens(stack, phases, instance.phase_memory)
            values, multiplier_text = stack.to_native(inventory)
            payload = self._build_payload(instance, state_value, values, multiplier_text, guidance)
            return gr.update(value=values), gr.update(value=multiplier_text), payload
        except Exception as error:
            traceback.print_exc()
            instance.note(f"Action failed: {error}", True)
            payload = self._build_payload(instance, state_value, selected, multipliers, guidance)
            return no_change + (payload,)

    def _dispatch(self, instance, action, stack, inventory, phases, model_type) -> bool:
        """Apply one typed frontend action. Returns True when native state moved."""
        kind = str(action.get("type", ""))

        if kind == "ready":
            instance.ready = True
            return False

        if kind == "batch":
            changed = False
            for item in action.get("actions", []) or []:
                if isinstance(item, dict):
                    changed = self._dispatch(instance, item, stack, inventory, phases, model_type) or changed
            return changed

        if kind == "toggle":
            lora_id = normalize_id(action.get("id"))
            # Only LoRAs WanGP actually offers may be activated.
            if action.get("enabled"):
                if inventory.get(lora_id) is None:
                    instance.note("That LoRA is not available for the current model.", True)
                    return False
                return stack.add(lora_id, phases.capacity)
            return stack.remove(lora_id)

        if kind == "remove":
            return stack.remove(normalize_id(action.get("id")))

        if kind == "set_strength":
            return up.set_phase_value(
                stack,
                normalize_id(action.get("id")),
                int(action.get("phase", 0) or 0),
                action.get("value"),
                phases,
                instance.phase_memory,
                bool(action.get("linked", False)),
            )

        if kind == "convert_simple":
            return up.convert_to_simple(stack, normalize_id(action.get("id")), phases)

        if kind == "favorite":
            if self._metadata:
                self._metadata.set_favorite(model_type, normalize_id(action.get("id")), bool(action.get("value")))
            return False

        if kind == "tag":
            if self._metadata:
                self._metadata.set_tag(model_type, normalize_id(action.get("id")), action.get("value", ""))
            return False

        if kind == "zoom":
            if self._metadata:
                self._metadata.set_zoom(action.get("value"))
            return False

        if kind == "sort":
            if self._metadata:
                self._metadata.set_sort_mode(action.get("value"))
            return False

        if kind == "name_mode":
            if self._metadata:
                self._metadata.set_name_mode(action.get("value"))
            return False

        if kind == "disable_all":
            return self._disable_all(instance, stack, inventory)

        if kind == "restore":
            return self._restore(instance, stack, inventory)

        if kind.startswith("profile_"):
            return self._profile_action(instance, action, stack, inventory, phases, model_type)

        return False

    def _disable_all(self, instance, stack, inventory) -> bool:
        removable = [lora_id for position, lora_id in enumerate(stack.ids) if not stack.is_system(position)]
        if not removable:
            instance.note("No user LoRAs to disable.")
            return False

        values, text = stack.to_native(inventory)
        instance.restore_snapshot = (list(values), text)
        for lora_id in removable:
            stack.remove(lora_id)
        remaining_values, remaining_text = stack.to_native(inventory)
        instance.restore_signature = up.stack_signature(remaining_values, remaining_text)
        instance.note(f"Disabled {len(removable)} LoRA(s).")
        return True

    def _restore(self, instance, stack, inventory) -> bool:
        if not instance.restore_snapshot:
            instance.note("Nothing to restore.", True)
            return False
        values, text = instance.restore_snapshot
        restored = up.Stack.from_native(values, text)
        stack.ids = restored.ids
        stack.tokens = restored.tokens
        stack.separator_index = restored.separator_index
        instance.restore_snapshot = None
        instance.note("Restored the previous LoRA stack.")
        return True

    # ---------------------------------------------------------- profiles

    def _profile_action(self, instance, action, stack, inventory, phases, model_type) -> bool:
        if not self._profiles:
            return False
        kind = action["type"]
        name = str(action.get("name", "") or "")

        try:
            if kind in ("profile_save", "profile_update"):
                entries = self._user_entries(stack, phases)
                if not entries:
                    instance.note("Select at least one LoRA before saving a profile.", True)
                    return False
                saved = self._profiles.save(name, model_type, entries, overwrite=(kind == "profile_update"))
                instance.note(f"Profile '{saved.name}' saved.")
                return False

            if kind == "profile_rename":
                renamed = self._profiles.rename(name, str(action.get("new_name", "") or ""))
                if self._metadata and self._metadata.default_profile(model_type) == name:
                    self._metadata.set_default_profile(model_type, renamed.name)
                instance.note(f"Profile renamed to '{renamed.name}'.")
                return False

            if kind == "profile_delete":
                self._profiles.delete(name)
                if self._metadata and self._metadata.default_profile(model_type) == name:
                    self._metadata.set_default_profile(model_type, "")
                instance.note(f"Profile '{name}' deleted.")
                return False

            if kind == "profile_default":
                # Toggle: choosing the current default clears it.
                if self._metadata:
                    current = self._metadata.default_profile(model_type)
                    if not name or current == name:
                        self._metadata.set_default_profile(model_type, "")
                        instance.note("Default profile cleared.")
                    elif self._profiles.get(name) is None:
                        instance.note(f"Profile '{name}' not found.", True)
                    else:
                        self._metadata.set_default_profile(model_type, name)
                        instance.note(
                            f"'{name}' is now the default for this model; "
                            "it is applied when you switch to it with no LoRAs selected."
                        )
                return False

            if kind == "profile_recall":
                return self._recall_profile(instance, name, stack, inventory, phases, model_type)
        except ProfileError as error:
            instance.note(str(error), True)
        return False

    def _recall_profile(self, instance, name, stack, inventory, phases, model_type) -> bool:
        profile = self._profiles.get(name)
        if profile is None:
            instance.note(f"Profile '{name}' not found.", True)
            return False
        if not self._profiles.is_compatible(profile, model_key=model_type):
            # Never blindly apply a stack saved under a different model.
            instance.note(
                f"Profile '{name}' was saved for '{profile.model_key}' and was not applied.", True
            )
            return False

        # WanGP-managed LoRAs (left of the accelerator bar) are preserved; a
        # profile only ever replaces the user side of the stack.
        keep = max(0, stack.separator_index) if stack.separator_index >= 0 else 0
        system_ids = stack.ids[:keep]
        system_tokens = stack.tokens[:keep]

        applied, missing = [], []
        for entry in profile.loras:
            if inventory.get(entry.id) is None:
                missing.append(entry.id)
                continue
            applied.append(entry)

        stack.ids = system_ids + [entry.id for entry in applied]
        stack.tokens = system_tokens + [entry.multiplier for entry in applied]
        stack.separator_index = keep if keep else -1

        if missing:
            instance.note(
                f"Recalled '{name}' - {len(missing)} LoRA(s) not found locally: "
                + ", ".join(value.rsplit("/", 1)[-1] for value in missing[:3]),
                True,
            )
        else:
            instance.note(f"Profile '{name}' recalled.")
        return True

    # -------------------------------------------------------- thumbnails

    def _serve_media(self, instance: InstanceState, request_json, state_value) -> str:
        """One bridge for every on-demand asset the browser asks for.

        Kinds: ``thumbs`` (grid tiles), ``inspect`` (catalogue for one LoRA),
        ``media`` (one catalogue image/video), ``video`` (a tile's preview
        video). Everything is addressed by stable LoRA id -- the frontend never
        supplies a filesystem path.
        """
        try:
            request = json.loads(request_json or "{}")
        except (TypeError, ValueError):
            return json.dumps({"kind": "none"})
        if not isinstance(request, dict):
            return json.dumps({"kind": "none"})

        kind = str(request.get("kind", "thumbs"))
        try:
            if kind == "inspect":
                return self._serve_inspect(instance, request, state_value)
            if kind == "media":
                return self._serve_catalogue_media(instance, request, state_value)
            if kind == "video":
                return self._serve_preview_video(instance, request, state_value)
            return self._serve_thumbnails(instance, request, state_value)
        except Exception as error:
            traceback.print_exc()
            return json.dumps({"kind": kind, "error": str(error)})

    def _entry_for(self, state_value, lora_id):
        """Resolve a frontend-supplied id against the current native inventory."""
        model_type = self._model_type(state_value)
        lora_dir = self._lora_dir(model_type)
        inventory = build_inventory(self._native_loras(state_value, []), lora_dir)
        return inventory.get(normalize_id(lora_id)), lora_dir

    def _serve_inspect(self, instance, request, state_value) -> str:
        entry, lora_dir = self._entry_for(state_value, request.get("id"))
        if entry is None or not entry.path:
            return json.dumps({"kind": "inspect", "error": "That LoRA is not in the current inventory."})

        detail = cat.read_detail(entry.path)
        return json.dumps({
            "kind": "inspect",
            "id": entry.id,
            "name": entry.name,
            "civitai_name": detail.civitai_name,
            "version_name": detail.version_name,
            "creator": detail.creator,
            "base_model": detail.base_model,
            "description": detail.description,
            "version_description": detail.version_description,
            "trained_words": detail.trained_words,
            "civitai_url": detail.civitai_url,
            "sha256": detail.sha256,
            "error": detail.error,
            # Paths stay server-side; the browser asks for media by index.
            "media": [
                {
                    "index": item.index,
                    "kind": item.kind,
                    "prompt": item.prompt,
                    "negative_prompt": item.negative_prompt,
                    "width": item.width,
                    "height": item.height,
                }
                for item in detail.media
            ],
        })

    def _serve_catalogue_media(self, instance, request, state_value) -> str:
        entry, _ = self._entry_for(state_value, request.get("id"))
        if entry is None or not entry.path:
            return json.dumps({"kind": "media", "error": "unknown LoRA"})

        try:
            wanted = int(request.get("index", 0))
        except (TypeError, ValueError):
            return json.dumps({"kind": "media", "error": "bad index"})

        detail = cat.read_detail(entry.path)
        item = next((media for media in detail.media if media.index == wanted), None)
        if item is None:
            return json.dumps({"kind": "media", "error": "no such media"})

        data = (
            self._image_data_uri(item.path)
            if item.kind == "image"
            else self._file_data_uri(item.path)
        )
        if not data:
            return json.dumps({"kind": "media", "id": entry.id, "index": wanted,
                               "error": "could not read media"})
        return json.dumps({"kind": "media", "id": entry.id, "index": wanted,
                           "media_kind": item.kind, "data": data})

    def _serve_preview_video(self, instance, request, state_value) -> str:
        """The same-stem video beside the .safetensors, for tile playback."""
        entry, lora_dir = self._entry_for(state_value, request.get("id"))
        if entry is None or not entry.path:
            return json.dumps({"kind": "video", "error": "unknown LoRA"})

        directory = os.path.dirname(entry.path)
        stem = os.path.splitext(os.path.basename(entry.path))[0].lower()
        try:
            listing = os.listdir(directory)
        except OSError:
            listing = []

        match = None
        for name in listing:
            base, extension = os.path.splitext(name)
            if base.lower() == stem and extension.lower() in VIDEO_EXTENSIONS:
                match = os.path.join(directory, name)
                break
        if match is None or (lora_dir and not is_within(lora_dir, match)):
            return json.dumps({"kind": "video", "id": entry.id, "error": "no preview video"})

        data = self._file_data_uri(match)
        if not data:
            return json.dumps({"kind": "video", "id": entry.id, "error": "video too large to preview"})
        return json.dumps({"kind": "video", "id": entry.id, "data": data})

    def _image_data_uri(self, path: str) -> str:
        """Re-encode a catalogue image down to a sane size for the modal."""
        if self._thumbnails is None:
            return ""
        preview = Preview(path, "image")
        return ThumbnailCache(self._thumbnails.cache_dir, MODAL_IMAGE_MAX_DIM).get(preview) or ""

    @staticmethod
    def _file_data_uri(path: str) -> str:
        """Raw bytes as a data: URI, refused above the size cap.

        Videos are only ever fetched one at a time, on an explicit click, which
        is what keeps this affordable and avoids exposing the LoRA folder as a
        static route.
        """
        try:
            if os.path.getsize(path) > MEDIA_INLINE_LIMIT:
                return ""
            with open(path, "rb") as handle:
                payload = base64.b64encode(handle.read()).decode("ascii")
        except OSError:
            return ""
        mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
        return f"data:{mime};base64,{payload}"

    def _serve_thumbnails(self, instance: InstanceState, request, state_value) -> str:
        """Resolve previews for the tiles the browser can actually see.

        The frontend only ever sends stable LoRA IDs; paths are resolved here
        against the current model's LoRA directory, and nothing under it is
        exposed as a static route -- each tile gets a small derived WebP as a
        data URI instead.
        """
        requested = [normalize_id(value) for value in request.get("ids", []) or []]
        if not requested or self._thumbnails is None:
            return json.dumps({"kind": "thumbs", "thumbs": {}})

        model_type = self._model_type(state_value)
        lora_dir = self._lora_dir(model_type) or instance.lora_dir
        inventory = build_inventory(self._native_loras(state_value, []), lora_dir)

        thumbs: dict[str, str] = {}
        listings: dict[str, list[str]] = {}
        for lora_id in requested[:32]:
            entry = inventory.get(lora_id)
            if entry is None or not entry.path:
                continue
            directory = os.path.dirname(entry.path)
            if directory not in listings:
                try:
                    listings[directory] = os.listdir(directory)
                except OSError:
                    listings[directory] = []
            preview = find_preview(entry.path, lora_dir, listings[directory])
            if preview is None:
                continue
            data_uri = self._thumbnails.get(preview)
            if data_uri:
                thumbs[lora_id] = data_uri
        return json.dumps({"kind": "thumbs", "thumbs": thumbs})
