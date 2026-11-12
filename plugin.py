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

from .lora_browser import builder as bd
from .lora_browser import catalogue as cat
from .lora_browser import civitai
from .lora_browser import derived as dv
from .lora_browser import lora_files as lf
from .lora_browser import schedule as sch
from .lora_browser import stack_modes as sm
from .lora_browser import stack_text
from .lora_browser import ui_payloads as up
from .lora_browser.stack_files import StackFiles
from .lora_browser.inventory import build_inventory, diff_ids
from .lora_browser.metadata_store import MetadataStore, resolve_store_path
from .lora_browser.profile_store import ProfileEntry, ProfileError, ProfileStore
from .lora_browser.thumbnails import (
    VIDEO_EXTENSIONS,
    Preview,
    ThumbnailCache,
    find_preview,
)
from .lora_browser.utils import display_name, is_within, normalize_id

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

#: Data-URI types for formats ``catalogue.sniff_media`` recognises.
_SNIFFED_MIME = {
    ".jpg": "image/jpeg", ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp",
    ".avif": "image/avif", ".mp4": "video/mp4", ".mov": "video/mp4", ".webm": "video/webm",
    ".mkv": "video/webm",
}

NATIVE_CHOICES_ELEM_ID = "wgp_lora_browser_native_choices"
NATIVE_MULTIPLIERS_ELEM_ID = "wgp_lora_browser_native_multipliers"


def _read_asset(name: str) -> str:
    try:
        with open(os.path.join(ASSETS_DIR, name), "r", encoding="utf-8") as handle:
            return handle.read()
    except OSError as error:
        print(f"[LoRA Browser] Could not read asset '{name}': {error}")
        return ""


def _follow_note(context: up.ScheduleContext) -> str:
    """What a resync that moved some timeline says it did."""
    if len(context.phase_steps) >= 2:
        first, second = context.phase_steps[0], context.phase_steps[1]
        return f"Schedules follow the stages: {first} steps, then {second} to refine."
    return f"Schedules follow {context.phase_length(0)} steps."


def _fit_note(
    phases: up.PhaseConfig,
    scheduled: bool,
    context: up.ScheduleContext | None = None,
    entering: bool = False,
) -> str:
    """What fitting the multipliers to a new phase mode says it did.

    ``scheduled`` is whether some token scheduled a phase before the fit: on
    LTX-2 those are the ones whose stage 2 does not come back, but is filled
    from stage 1 again (``up.fill_stages_from_stage_1``).  ``entering`` is
    whether some stage-1 schedule just entered Two Phases to be filled.
    """
    generic = "Multipliers now match the phase mode; anything it hides comes back with it."
    if not phases.staged or not (entering if phases.effective > 1 else scheduled):
        return generic
    if phases.effective > 1:
        if context is not None and context.stage_2_match:
            return (
                "Two Phases: stage 2 of each scheduled LoRA copies its stage 1 steps "
                f"{up.step_span(context.stage_2_match)}, matched by noise level."
            )
        if context is not None and len(context.phase_steps) > 1:
            return "Two Phases: stage 2 of each scheduled LoRA holds the strength its stage 1 ends at."
        return "Two Phases: scheduled LoRAs keep their stage 1 schedule; stage 2 starts empty."
    return (
        "One Phase runs no refinement stage. Back in Two Phases, plain LoRAs get their "
        "stage 2 strength back and scheduled LoRAs fill stage 2 from stage 1 again."
    )


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
        #: lora id -> the ";" parts a narrower phase mode cut from its token,
        #: verbatim, so the mode coming back restores them -- never an LTX-2
        #: stage 2 that was scheduled, which starts empty instead. See
        #: ui_payloads.fit_phase_structure.
        self.hidden_phases: up.HiddenPhases = {}
        self.pending_status = ""
        self.pending_warn = False
        self.ready = False
        #: lora id -> (sidecar mtime, CatalogueIndex). Reading summary.txt for
        #: every LoRA on every payload would be wasteful; the sidecar only
        #: changes when the enrichment script runs again.
        self.catalogue_cache: dict[str, tuple[float, Any]] = {}
        #: (lora id, phase) -> PhaseSchedule. The editor's region decomposition
        #: of a scheduled multiplier; re-derived from the native token whenever
        #: the two disagree, so WanGP stays the truth.
        self.schedules: dict[tuple[str, int], Any] = {}
        #: What the last pasted stack did, line by line, for the paste box --
        #: carried by the next payload only, the one answering the paste.
        self.paste_report: dict | None = None
        #: LTX-2: the model's LoRAs have a picture side and a sound side.
        self.sided = False

    def next_revision(self) -> int:
        self.revision += 1
        return self.revision

    def take_status(self) -> tuple[str, bool]:
        status, warn = self.pending_status, self.pending_warn
        self.pending_status, self.pending_warn = "", False
        return status, warn

    def take_paste_report(self) -> dict | None:
        report, self.paste_report = self.paste_report, None
        return report

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
        #: At most one "Fetch all" run, shared by every media-generator tab.
        self._bulk: civitai.BulkFetch | None = None
        self._metadata: MetadataStore | None = None
        self._profiles: ProfileStore | None = None
        self._thumbnails: ThumbnailCache | None = None
        #: One queue builds visual-only/audio-only copies and bakes for every
        #: tab, one file at a time.
        self._builder = bd.Builder()
        #: LoRA folder -> its store of built files.
        self._stores: dict[str, dv.Store] = {}
        self._summaries = lf.SummaryCache()

    # ------------------------------------------------------------- setup

    def setup_ui(self) -> None:
        # Component names are the local variable names in wgp.py's
        # generate_media_tab; the plugin manager resolves them from that scope.
        # num_inference_steps is optional: it only decides whether a shared
        # schedule may label its slots with real step numbers, so a WanGP build
        # that does not expose it still gets a working timeline. sample_solver
        # is optional for the same reason: on LTX-2 Dev a distilled sampler
        # runs eight first-stage steps whatever the counter says. And
        # num_inference_steps2 ("Phase 2 Steps") is how long MiniMax H3's
        # second stage runs.
        for component in (
            "loras_choices", "loras_multipliers", "guidance_phases", "state",
            "lset_name", "main", "num_inference_steps", "sample_solver",
            "num_inference_steps2",
        ):
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
            instance.hidden_phases.clear()
            instance.catalogue_cache.clear()
            instance.schedules.clear()
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

    def _store(self, lora_dir: str) -> dv.Store | None:
        if not lora_dir:
            return None
        if lora_dir not in self._stores:
            self._stores[lora_dir] = dv.Store(PLUGIN_DIR, lora_dir)
        return self._stores[lora_dir]

    def _stack_files(self, instance: InstanceState, inventory) -> StackFiles:
        return StackFiles(self._store(instance.lora_dir), self._builder, self._summaries, inventory, instance.sided)

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
        num_inference_steps = components.get("num_inference_steps")
        sample_solver = components.get("sample_solver")
        num_inference_steps2 = components.get("num_inference_steps2")

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

        # Gradio passes inputs positionally, and any optional component may be
        # absent in a given WanGP build, so the tail is unpacked by name.
        sync_inputs = [state, loras_choices, loras_multipliers]
        optional: list[str] = []
        if guidance_phases is not None:
            sync_inputs.append(guidance_phases)
            optional.append("guidance")
        if num_inference_steps is not None:
            sync_inputs.append(num_inference_steps)
            optional.append("steps")
        if sample_solver is not None:
            sync_inputs.append(sample_solver)
            optional.append("sampler")
        if num_inference_steps2 is not None:
            sync_inputs.append(num_inference_steps2)
            optional.append("steps2")

        def unpack(extra):
            values = {"guidance": None, "steps": None, "sampler": None, "steps2": None}
            for name, value in zip(optional, extra):
                values[name] = value
            return values["guidance"], values["steps"], values["sampler"], values["steps2"]

        def sync(state_value, selected, multipliers, *extra):
            return self._build_payload(instance, state_value, selected, multipliers, *unpack(extra))

        def act(action_json, state_value, selected, multipliers, *extra):
            return self._apply_action(
                instance, action_json, state_value, selected, multipliers, *unpack(extra)
            )

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
        watched = [loras_choices, loras_multipliers]
        watched += [
            component
            for component in (guidance_phases, num_inference_steps, sample_solver, num_inference_steps2)
            if component is not None
        ]
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

    def _context(
        self, instance: InstanceState, state_value, selected, multipliers, guidance, steps=None,
        sampler=None, steps2=None,
    ):
        model_type = self._model_type(state_value)
        model_def = self._model_def(model_type)
        phases = up.resolve_phases(model_def, guidance)
        lora_dir = self._lora_dir(model_type)
        inventory = build_inventory(self._native_loras(state_value, selected), lora_dir)
        stack = up.Stack.from_native(selected, multipliers)
        context = up.ScheduleContext.from_native(steps, model_def, phases, sampler, steps2)

        instance.model_key = model_type
        instance.lora_dir = lora_dir
        instance.inventory_ids = inventory.ids
        instance.sided = up.is_ltx2(model_def)
        up.remember_hidden_phases(stack, phases, instance.phase_memory)
        up.forget_superseded_phases(stack, instance.hidden_phases)
        # Bring the region editor back in line with the native tokens before
        # anything reads or writes them.
        up.sync_schedules(stack, phases, instance.schedules, context)
        return model_type, inventory, stack, phases, context

    def _build_payload(
        self, instance: InstanceState, state_value, selected, multipliers, guidance, steps=None,
        sampler=None, steps2=None,
    ) -> str:
        try:
            model_type, inventory, stack, phases, context = self._context(
                instance, state_value, selected, multipliers, guidance, steps, sampler, steps2
            )
            status, warn = instance.take_status()

            # Build whatever the stack selects and does not have yet: a copy
            # starts the moment a setting needs it, however the setting came.
            files = self._stack_files(instance, inventory)
            files.ensure(stack)

            entries, bakes = self._profile_entries(stack, files)
            signature = up.stack_signature(selected, multipliers)

            catalogue = self._catalogue_index(instance, inventory)
            name_index = stack_text.NameIndex(inventory, self._civitai_names(catalogue))
            rows = up.build_active_rows(
                inventory, stack, phases, instance.phase_memory, catalogue,
                instance.schedules, context,
            )
            rows = sm.fold_rows(rows, stack, lambda lora_id: files.describe(lora_id, catalogue))
            rows = files.decorate(rows, stack, catalogue)
            missing = [row for row in rows if row["missing"]]
            if missing and not status:
                names = ", ".join(row["name"] for row in missing[:3])
                status = f"{len(missing)} selected LoRA(s) missing locally: {names}"
                warn = True

            complete, incomplete = (
                self._profiles.partition(inventory.ids) if self._profiles else ([], [])
            )
            payload = {
                "revision": instance.next_revision(),
                "model_key": model_type,
                "phases": {"capacity": phases.capacity, "effective": phases.effective},
                "phase_labels": up.phase_labels(phases.effective),
                "items": up.build_items(
                    inventory, stack, phases, self._metadata, model_type,
                    instance.phase_memory, catalogue, files.aliases(stack),
                ),
                "active": rows,
                "profiles": complete + incomplete,
                # Named so the dropdown can flag a profile this model can only
                # apply in part -- it is still selectable.
                "profiles_incomplete": incomplete,
                "active_profile": self._profiles.match(entries, bakes) if self._profiles else "",
                "default_profile": self._metadata.default_profile(model_type) if self._metadata else "",
                "civitai_key_set": bool(self._civitai_key()),
                "zoom_px": self._metadata.zoom_px if self._metadata else 104,
                "sort_mode": self._metadata.sort_mode if self._metadata else "name",
                "name_mode": self._metadata.name_mode if self._metadata else "civitai",
                "stack_view": self._metadata.stack_view if self._metadata else "list",
                # Visual only / audio only: LTX-2 LoRAs have a picture side
                # and a sound side, and a copy of one can be built.
                "av_supported": files.sided and files.store is not None,
                # What is building for this stack right now, if anything.
                "builds": self._build_summary(files, stack),
                "built": self._built_summary(files),
                "slider_min": up.SLIDER_MIN,
                "slider_max": up.SLIDER_MAX,
                "slider_step": up.SLIDER_STEP,
                "nudge_step": up.NUDGE_STEP,
                "value_min": up.VALUE_MIN,
                "value_max": up.VALUE_MAX,
                "value_decimals": up.VALUE_DECIMALS,
                # Schedule coordinates: how long the run is, and whether the
                # plugin may claim exact step numbers for a phase-specific
                # schedule -- only where each phase is a stage of known length
                # (LTX-2, MiniMax H3), and then how long each one is.
                "steps": context.steps,
                "phase_boundaries_known": context.boundaries_known,
                "phase_steps": list(context.phase_steps),
                # The frontend commits a resync when it sees this, so the
                # timelines follow the step counter, and the tokens the phase
                # mode, without the user asking.
                "schedules_out_of_sync": up.schedules_need_resync(
                    stack, phases, instance.schedules, context
                ) or up.phase_structure_needs_fit(stack, phases, instance.hidden_phases),
                # Step schedules only mean what they show in One Phase guidance,
                # or on a model whose phases are stages of known length.
                "scheduling_enabled": up.scheduling_allowed(phases),
                "scheduling_disabled_reason": up.SCHEDULING_DISABLED_REASON,
                "schedule_slot_limits": {
                    "min": sch.MIN_SLOTS, "max": sch.MAX_SLOTS, "default": sch.DEFAULT_SLOTS,
                },
                # What the two bulk switches would change right now.
                "can_disable_all": any(on for _, on, _ in stack.user_entries()),
                "can_enable_all": bool(stack.disabled),
                # The LoRAs that run, as text the paste box reads back.
                "stack_text": stack_text.format_stack(
                    [
                        (entry.id, entry.multiplier, entry.mode, entry.group, bakes.get(entry.group, 0))
                        for entry in entries if entry.enabled
                    ],
                    name_index,
                ),
                # Set only on the payload answering a pasted stack.
                "paste_report": instance.take_paste_report(),
                "signature": signature,
                "status": status,
                "status_warn": warn,
            }
            return json.dumps(payload)
        except Exception as error:
            traceback.print_exc()
            # The frontend shows the native controls again when it sees this.
            return json.dumps({"error": f"LoRA Browser could not render: {error}"})

    def _profile_entries(self, stack: up.Stack, files: StackFiles) -> tuple[list[ProfileEntry], dict[str, int]]:
        """User-editable part of the stack, switched-off LoRAs included where
        they sit -- accelerator LoRAs are excluded -- each by its original's id
        with its mode and group, and the rank each baked group is baked at.
        A baked group is listed as its members, where the group sits."""
        found = sm.groups(stack)
        member_of = {member: group.letter for group in found.values() for member in group.members}
        baked = {group.bake: group for group in found.values() if group.baked}
        skipped = stack.skipped()
        entries: list[ProfileEntry] = []
        bakes: dict[str, int] = {}
        for lora_id, on, token in stack.user_entries():
            group = baked.get(lora_id)
            members = group.members if group else [lora_id]
            if group:
                bakes[group.letter] = dv.parse(group.bake).rank
            for member in members:
                source, mode = files.source(member)
                entries.append(ProfileEntry(
                    id=source, multiplier=token, enabled=on, mode=mode,
                    group=group.letter if group else member_of.get(member, ""),
                    skipped=(not on) and lora_id in skipped,
                ))
        return entries, bakes

    def _apply_entries(self, stack: up.Stack, files: StackFiles, entries: list[ProfileEntry],
                       bakes: dict[str, int]) -> None:
        """Rebuild the user side of the stack from profile-style entries:
        modes as built copies, groups as group lines, bakes as baked files.
        Anything not on disk yet is built by the next payload."""
        native, skipped, groups = [], set(), {}
        for entry in entries:
            lora_id = dv.split_id(entry.id, entry.mode) if files.sided else entry.id
            native.append((lora_id, entry.enabled, entry.multiplier))
            if entry.skipped and not entry.enabled:
                skipped.add(lora_id)
            if entry.group:
                groups.setdefault(entry.group, []).append(lora_id)
        stack.set_user_entries(native, skipped)
        stack.groups = {letter: members for letter, members in groups.items() if len(members) >= 2}
        for letter, rank in bakes.items():
            group = sm.groups(stack).get(letter)
            if group is not None and not group.baked:
                sm.bake(stack, letter, files.bake_id(group, rank))
        sm.sync_tokens(stack)
        sm.settle(stack, files.empty)

    def _build_summary(self, files: StackFiles, stack: up.Stack) -> dict:
        status = self._builder.status()
        waiting = files.waiting(stack)
        failed = {key: message for key, message in status["failed"].items() if key in waiting}
        return {
            "running": status["running"],
            "current": status["current"],
            "queued": len(status["queued"]),
            "waiting": len(waiting),
            "failed": failed,
            "generation": status["generation"],
        }

    def _built_summary(self, files: StackFiles) -> dict:
        store = files.store
        if store is None or not (os.path.isdir(store.root) or os.path.isdir(store.link)):
            return {"count": 0, "bytes": 0}
        try:
            listed = store.files()
        except OSError:
            return {"count": 0, "bytes": 0}
        return {"count": len(listed), "bytes": sum(item.get("bytes", 0) for item in listed)}

    # ----------------------------------------------------------- actions

    def _apply_action(
        self, instance: InstanceState, action_json, state_value, selected, multipliers, guidance,
        steps=None, sampler=None, steps2=None,
    ):
        no_change = (gr.update(), gr.update())
        try:
            action = json.loads(action_json or "{}")
        except (TypeError, ValueError):
            return no_change + (gr.update(),)

        native = (guidance, steps, sampler, steps2)
        try:
            model_type, inventory, stack, phases, context = self._context(
                instance, state_value, selected, multipliers, *native
            )
            changed = self._dispatch(instance, action, stack, inventory, phases, model_type, context)
            # Every member of a group carries its lead's multiplier.
            changed = sm.sync_tokens(stack) or changed

            if not changed:
                payload = self._build_payload(instance, state_value, selected, multipliers, *native)
                return no_change + (payload,)

            up.normalize_stack_tokens(stack, phases, instance.phase_memory)
            values, multiplier_text = stack.to_native(inventory)
            payload = self._build_payload(instance, state_value, values, multiplier_text, *native)
            return gr.update(value=values), gr.update(value=multiplier_text), payload
        except Exception as error:
            traceback.print_exc()
            instance.note(f"Action failed: {error}", True)
            payload = self._build_payload(instance, state_value, selected, multipliers, *native)
            return no_change + (payload,)

    def _dispatch(self, instance, action, stack, inventory, phases, model_type, context=None) -> bool:
        """Apply one typed frontend action. Returns True when native state moved."""
        kind = str(action.get("type", ""))

        if kind == "ready":
            instance.ready = True
            return False

        if kind == "batch":
            changed = False
            for item in action.get("actions", []) or []:
                if isinstance(item, dict):
                    changed = self._dispatch(
                        instance, item, stack, inventory, phases, model_type, context
                    ) or changed
            return changed

        if kind == "toggle":
            lora_id = normalize_id(action.get("id"))
            # A LoRA in a mode, or baked into a group, is still its tile.
            files = self._stack_files(instance, inventory)
            alias = files.aliases(stack).get(lora_id)
            if alias is not None and alias[0] != lora_id:
                native, on = alias
                if action.get("enabled"):
                    return False if on else self._enable(instance, stack, files, native, True)
                return self._remove(instance, stack, files, native)
            # Only LoRAs WanGP actually offers may be activated.
            if action.get("enabled"):
                if stack.disabled_entry(lora_id) is not None:
                    # Switched off, not gone: its tile brings it back as it was.
                    return stack.add(lora_id, phases.capacity)
                if inventory.get(lora_id) is None:
                    instance.note("That LoRA is not available for the current model.", True)
                    return False
                return stack.add(lora_id, phases.capacity)
            return stack.remove(lora_id)

        if kind == "remove":
            files = self._stack_files(instance, inventory)
            letter = str(action.get("group", "") or "")
            if letter:
                return sm.remove_group(stack, letter)
            return self._remove(instance, stack, files, normalize_id(action.get("id")))

        if kind == "set_enabled":
            # Off keeps the multiplier -- schedule and all -- where WanGP keeps
            # the rest of the stack, and applies none of it.
            return self._enable(
                instance, stack, self._stack_files(instance, inventory),
                normalize_id(action.get("id")), bool(action.get("value")), str(action.get("group", "") or ""),
            )

        if kind in ("set_mode", "set_mode_all", "group_create", "group_add", "group_leave",
                    "ungroup", "bake", "unbake"):
            return self._mode_action(instance, action, stack, self._stack_files(instance, inventory))

        if kind == "build_retry":
            self._builder.retry(normalize_id(action.get("id")))
            return False

        if kind == "view":
            if self._metadata:
                self._metadata.set_stack_view(action.get("value"))
            return False

        if kind == "set_strength":
            # Routes itself: a scheduled phase moves its schedule base, an
            # unscheduled one moves the plain scalar.
            return up.set_phase_value(
                stack,
                normalize_id(action.get("id")),
                int(action.get("phase", 0) or 0),
                action.get("value"),
                phases,
                instance.phase_memory,
                bool(action.get("linked", False)),
                instance.schedules,
            )

        if kind.startswith("schedule_"):
            return self._schedule_action(instance, action, stack, phases, context)

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

        if kind == "civitai_key":
            if self._metadata:
                if os.environ.get("CIVITAI_API_KEY", "").strip():
                    instance.note(
                        "CIVITAI_API_KEY is set in the environment and takes precedence.", True
                    )
                    return False
                stored = self._metadata.set_civitai_api_key(action.get("value"))
                instance.note("Civitai API key saved." if stored else "Civitai API key cleared.")
            return False

        if kind == "disable_all":
            return self._set_all_enabled(instance, stack, False)

        if kind == "enable_all":
            return self._set_all_enabled(instance, stack, True, self._stack_files(instance, inventory))

        if kind == "stack_paste":
            return self._paste_stack(instance, action, stack, inventory, phases, context)

        if kind.startswith("profile_"):
            return self._profile_action(instance, action, stack, inventory, phases, model_type)

        return False

    def _schedule_action(self, instance, action, stack, phases, context) -> bool:
        """Apply one timeline action. Python validates every committed range.

        The frontend never writes a raw token: it says what the user did and
        this decides whether the native multiplier moves at all.
        """
        kind = str(action.get("type", ""))
        lora_id = normalize_id(action.get("id"))
        phase = action.get("phase", 0)
        args = (phases, instance.phase_memory, instance.schedules, context)

        try:
            if kind == "schedule_enable":
                return up.schedule_enable(stack, lora_id, phase, *args)

            if kind == "schedule_set_base":
                return up.schedule_set_base(
                    stack, lora_id, phase, action.get("value"), *args,
                    linked=bool(action.get("linked", False)),
                )

            if kind == "schedule_add_region":
                return up.schedule_add_region(
                    stack, lora_id, phase, *args,
                    start=action.get("start"), end=action.get("end"),
                )

            if kind == "schedule_commit_region":
                return up.schedule_commit_region(
                    stack, lora_id, phase, str(action.get("region_id", "")),
                    action.get("start"), action.get("end"), *args,
                    keep_width=bool(action.get("keep_width", False)),
                )

            if kind == "schedule_set_region_strength":
                return up.schedule_set_region_strength(
                    stack, lora_id, phase, str(action.get("region_id", "")),
                    action.get("value"), *args,
                )

            if kind == "schedule_delete_region":
                return up.schedule_delete_region(
                    stack, lora_id, phase, str(action.get("region_id", "")), *args
                )

            if kind == "schedule_clear_phase":
                return up.schedule_clear_phase(stack, lora_id, phase, *args)

            if kind == "schedule_fill_stage_2":
                # Described before it writes: afterwards stage 2 is drawn, and
                # there is nothing left on offer to describe.
                fill = up.stage_2_fill(
                    up.codec.classify(stack.token_for(lora_id), phases.capacity), phases, context
                )
                changed = up.schedule_fill_stage_2(stack, lora_id, *args)
                if changed and fill is not None:
                    instance.note(fill.done)
                return changed

            if kind == "schedule_resync":
                # First the phase structure: a token declaring more phases than
                # this mode runs is one WanGP refuses outright, and on a staged
                # model (LTX-2, MiniMax H3) a stage-1 schedule entering Two
                # Phases gets a stage 2 of its own. The timelines are then
                # re-derived from what it became before they move.
                scheduled = up.has_step_schedule(stack)
                entering = up.entering_two_phases(stack, phases)
                fitted = up.fit_phase_structure(stack, phases, instance.hidden_phases)
                if fitted:
                    up.sync_schedules(stack, phases, instance.schedules, context)
                    # Entering Two Phases, stage 2 carries on from where stage
                    # 1 ends -- what Fill from stage 1 draws.
                    up.fill_stages_from_stage_1(stack, entering, *args)
                changed = up.refit_schedules(
                    stack, phases, instance.schedules, instance.phase_memory, context
                )
                if changed:
                    instance.note(_follow_note(context))
                elif fitted:
                    instance.note(_fit_note(phases, scheduled, context, bool(entering)))
                return changed or fitted

            if kind == "schedule_normalize":
                changed = up.schedule_normalize(stack, lora_id, phase, action.get("slots"), *args)
                if changed:
                    instance.note("Schedule re-gridded; the native multiplier was rewritten.")
                return changed
        except sch.ScheduleError as error:
            # A refusal is a normal outcome (a full timeline, a preserved
            # token), so it is reported in the status bar, not raised.
            instance.note(str(error), True)
            return False

        return False

    def _set_all_enabled(self, instance, stack, on: bool, files: StackFiles | None = None) -> bool:
        """Every user LoRA on, or every one off -- each keeping its multiplier
        and its place.  WanGP-managed LoRAs are not the user's to switch.

        One its mode left nothing to apply comes back as Normal: switched on,
        it has to apply something."""
        reset = 0
        if on and files is not None:
            for lora_id in sorted(stack.skipped()):
                group = sm.group_of(stack, lora_id)
                for member in (group.members if group and group.bake == lora_id else [lora_id]):
                    if files.empty(member) and self._rename_mode(instance, stack, files, member, sm.NORMAL):
                        reset += 1
            self._rebake_changed(stack, files)
        changed = stack.set_all_enabled(on)
        if not changed:
            instance.note("Every LoRA is already on." if on else "No LoRAs are on.")
            return False
        note = f"Switched {changed} LoRA(s) {'on' if on else 'off'}."
        if reset:
            note += f" {reset} had nothing to apply in their mode, so they are back to normal."
        instance.note(note)
        return True

    # -------------------------------------------- modes, groups and bakes

    def _migrate(self, instance: InstanceState, old: str, new: str) -> None:
        """What the panel remembers about a LoRA follows it to its new id."""
        for memory in (instance.phase_memory, instance.hidden_phases):
            if old in memory:
                memory[new] = memory.pop(old)
        for key in [key for key in instance.schedules if key[0] == old]:
            instance.schedules[(new, key[1])] = instance.schedules.pop(key)

    def _rename_mode(self, instance, stack, files: StackFiles, lora_id: str, mode: str) -> str:
        new_id = sm.set_mode(stack, lora_id, mode, files.known)
        if new_id:
            self._migrate(instance, normalize_id(lora_id), new_id)
        return new_id

    @staticmethod
    def _rebake_changed(stack, files: StackFiles) -> None:
        """A baked group whose members now bring something else points at the
        bake of what they bring now -- built by the next payload, or already
        on disk if it was baked that way before."""
        for group in sm.groups(stack).values():
            if group.baked:
                wanted = files.bake_id(group, dv.parse(group.bake).rank)
                if wanted != group.bake:
                    sm.rebake(stack, group.letter, wanted)

    def _enable(self, instance, stack, files: StackFiles, lora_id: str, on: bool, letter: str = "") -> bool:
        if letter:
            group = sm.groups(stack).get(letter)
            if group is None:
                return False
            if on and all(files.empty(member) for member in group.members):
                for member in list(group.members):
                    self._rename_mode(instance, stack, files, member, sm.NORMAL)
                self._rebake_changed(stack, files)
                instance.note(f"Group {letter} is on again and back to normal, since its mode left it nothing to apply.")
            changed = sm.set_group_enabled(stack, letter, on)
            sm.settle(stack, files.empty)
            return changed
        if on and files.empty(lora_id):
            source, mode = files.source(lora_id)
            new_id = self._rename_mode(instance, stack, files, lora_id, sm.NORMAL)
            if new_id:
                lora_id = new_id
                instance.note(f"{display_name(source)} is on again and back to normal, since {mode} only left it nothing to apply.")
        return stack.set_enabled(lora_id, on)

    def _remove(self, instance, stack, files: StackFiles, lora_id: str) -> bool:
        group = sm.group_of(stack, lora_id)
        if group is not None:
            if lora_id == group.bake:
                return sm.remove_group(stack, group.letter)
            sm.leave_group(stack, group.letter, lora_id)
            self._rebake_changed(stack, files)
        return stack.remove(lora_id)

    def _mode_action(self, instance, action, stack, files: StackFiles) -> bool:
        kind = action["type"]
        letter = str(action.get("group", "") or "").upper()
        name = lambda lora_id: display_name(files.source(lora_id)[0])  # noqa: E731

        if kind in ("set_mode", "set_mode_all"):
            mode = str(action.get("mode", ""))
            if mode not in sm.MODES:
                return False
            if not files.sided:
                instance.note("Visual only and audio only are for LTX-2 models, whose LoRAs change picture and sound separately.", True)
                return False
            if files.store is None:
                instance.note("This model has no LoRA folder to build copies in.", True)
                return False
            if kind == "set_mode":
                targets = [normalize_id(action.get("id"))]
            else:
                targets = self._mode_targets(stack, action.get("ids"))
            renamed = [new_id for new_id in (self._rename_mode(instance, stack, files, lora_id, mode) for lora_id in targets) if new_id]
            if not renamed:
                instance.note("Nothing to change." if kind == "set_mode_all" else "")
                return False
            self._rebake_changed(stack, files)
            moved = sm.settle(stack, files.empty)
            instance.note(self._mode_note(files, renamed, mode, moved))
            return True

        if kind == "group_create":
            ids = [normalize_id(value) for value in (action.get("ids") or [])]
            made = sm.create_group(stack, ids, normalize_id(action.get("lead")))
            if not made:
                instance.note("Pick two or more LoRAs that are not in a group yet.", True)
                return False
            sm.sync_tokens(stack)
            sm.settle(stack, files.empty)
            members = sm.groups(stack)[made].members
            instance.note(f"Grouped {', '.join(name(member) for member in members)} as Group {made}. "
                          "They share one strength and schedule now; nothing was built.")
            return True

        if kind == "group_add":
            lora_id = normalize_id(action.get("id"))
            if not sm.add_to_group(stack, letter, lora_id):
                return False
            self._rebake_changed(stack, files)
            sm.sync_tokens(stack)
            sm.settle(stack, files.empty)
            instance.note(f"{name(lora_id)} joined Group {letter} and takes its strength and schedule.")
            return True

        if kind == "group_leave":
            lora_id = normalize_id(action.get("id"))
            if not sm.leave_group(stack, letter, lora_id):
                return False
            self._rebake_changed(stack, files)
            sm.settle(stack, files.empty)
            instance.note(f"{name(lora_id)} left Group {letter} and keeps its strength and schedule.")
            return True

        if kind == "ungroup":
            if not sm.ungroup(stack, letter):
                return False
            sm.settle(stack, files.empty)
            instance.note(f"Group {letter} is separate LoRAs again; each keeps the group's strength and schedule.")
            return True

        if kind == "bake":
            group = sm.groups(stack).get(letter)
            if group is None or files.store is None:
                return False
            try:
                rank = int(action.get("rank") or files.default_rank(group) or 0)
            except (TypeError, ValueError):
                rank = 0
            if rank < 1:
                instance.note("Choose a rank to bake at.", True)
                return False
            bake_id = files.bake_id(group, rank)
            changed = sm.rebake(stack, letter, bake_id) if group.baked else sm.bake(stack, letter, bake_id)
            if changed:
                instance.note(f"Baking Group {letter} at rank {rank}. Its LoRAs are loaded separately until it is ready.")
            return changed

        if kind == "unbake":
            if not sm.unbake(stack, letter):
                return False
            sm.settle(stack, files.empty)
            instance.note(f"Group {letter} is unbaked; WanGP loads its LoRAs separately again. "
                          "The baked file stays under Built files until you delete it.")
            return True
        return False

    @staticmethod
    def _mode_targets(stack, ids) -> list[str]:
        """The LoRAs a bulk change reaches: those named -- a group's row names
        every member -- or, with none named, every user LoRA."""
        found = sm.groups(stack)
        if ids:
            wanted, targets = [normalize_id(value) for value in ids], []
            for lora_id in wanted:
                group = next((g for g in found.values() if lora_id in g.members or lora_id == g.bake), None)
                for member in (group.members if group else [lora_id]):
                    if member not in targets:
                        targets.append(member)
            return targets
        targets = []
        for lora_id, _, _ in stack.user_entries():
            group = next((g for g in found.values() if g.bake == lora_id), None)
            targets.extend(group.members if group else [lora_id])
        return targets

    @staticmethod
    def _mode_note(files: StackFiles, renamed: list[str], mode: str, moved: dict) -> str:
        names = [display_name(files.source(lora_id)[0]) for lora_id in renamed]
        label = "normal" if mode == sm.NORMAL else f"{mode} only"
        if len(renamed) == 1:
            if renamed[0] in moved["off"]:
                other = "picture" if mode == sm.AUDIO else "sound"
                back = "Visual" if mode == sm.AUDIO else "Audio"
                return f"{names[0]} only changes the {other}, so {label} switches it off. Normal or {back} only switches it back on."
            message = f"{names[0]} is back to normal." if mode == sm.NORMAL else f"{names[0]}: {label}."
        else:
            message = f"{len(renamed)} LoRAs set to {label}."
            if moved["off"]:
                other = "picture" if mode == sm.AUDIO else "sound"
                message += f" Switched off {len(moved['off'])} that only change the {other}."
        if moved["on"]:
            message += f" Switched {len(moved['on'])} back on."
        waiting = [lora_id for lora_id in renamed if files.state(lora_id)["state"] in ("pending", "queued", "building")]
        if waiting:
            message += f" Building {len(waiting)} {'copy' if len(waiting) == 1 else 'copies'}."
        return message

    # ----------------------------------------------------- pasted stacks

    @staticmethod
    def _civitai_names(catalogue: dict) -> dict[str, str]:
        return {
            lora_id: str(getattr(record, "civitai_name", "") or "")
            for lora_id, record in (catalogue or {}).items()
        }

    def _paste_stack(self, instance, action, stack, inventory, phases, context) -> bool:
        """Load a stack pasted as text -- ``name: x<multiplier>`` per line.

        It replaces the user's side of the stack, switched-off LoRAs included,
        the way recalling a profile does; WanGP-managed LoRAs stay.  A line
        loads when its name is a LoRA this model offers and its multiplier is
        one WanGP can read.  Every other line is reported, and when no line
        can load nothing changes.

        What loaded is then settled the way a preset arriving in this phase
        mode would be (see ``schedule_resync``), so WanGP is never handed a
        token the mode refuses, not even until the next round trip.
        """
        report = {"seq": action.get("seq"), "ok": False, "loaded": 0, "lines": 0,
                  "problems": [], "notes": []}
        instance.paste_report = report
        lines = stack_text.parse(action.get("text", ""))
        report["lines"] = len(lines)
        if not lines:
            instance.note("Nothing to load - paste one LoRA per line, as name: x multiplier.", True)
            return False

        names = stack_text.NameIndex(
            inventory, self._civitai_names(self._catalogue_index(instance, inventory))
        )
        managed = {lora_id for position, lora_id in enumerate(stack.ids) if stack.is_system(position)}
        files = self._stack_files(instance, inventory)
        entries, problems, first_line = [], [], {}
        groups: dict[str, list[str]] = {}
        group_token: dict[str, tuple[str, int]] = {}
        bakes: dict[str, int] = {}
        ignored_modes = 0
        for line in lines:
            lora_id, candidates = names.resolve(line.name)
            token = line.token or up.codec.default_token(phases.capacity)
            if candidates:
                reason = (f"'{line.name}' matches {len(candidates)} LoRAs ("
                          + ", ".join(candidates[:3]) + ") - name one by its file or folder path")
            elif not lora_id:
                reason = f"no LoRA named '{line.name}' for this model"
            elif lora_id in managed:
                reason = f"{names.label(lora_id)} is managed by WanGP"
            elif lora_id in first_line:
                reason = f"{names.label(lora_id)} is already on line {first_line[lora_id]}"
            else:
                reason = stack_text.token_problem(token) or self._paste_phase_problem(token, phases)
            if not reason and line.group and line.group in group_token and group_token[line.group][0] != token:
                reason = (f"group {line.group} moves as one, so its lines share one multiplier; "
                          f"line {group_token[line.group][1]} gives x{group_token[line.group][0]}")
            if reason:
                problems.append(stack_text.LineProblem(line.number, line.text, reason))
                continue
            first_line[lora_id] = line.number
            native = lora_id
            if line.mode in (sm.VISUAL, sm.AUDIO):
                if files.sided:
                    native = dv.split_id(lora_id, line.mode)
                else:
                    ignored_modes += 1
            if line.group:
                group_token.setdefault(line.group, (token, line.number))
                groups.setdefault(line.group, []).append(native)
                if line.baked:
                    bakes[line.group] = max(bakes.get(line.group, 0), line.baked)
            entries.append((native, True, token))

        report["problems"] = [problem.to_dict() for problem in problems]
        if not entries:
            instance.note(
                f"Nothing loaded - line {problems[0].number}: {problems[0].reason}." if len(problems) == 1
                else f"Nothing loaded - none of the {len(problems)} lines names a LoRA this model can load.",
                True,
            )
            return False

        before = self._stack_snapshot(stack)
        for lora_id, _, _ in entries:
            # The text is the truth for these now: nothing remembered about how
            # they were set before -- a phase a narrower mode hid, a timeline
            # -- may carry over into what was pasted.
            instance.phase_memory.pop(lora_id, None)
            instance.hidden_phases.pop(lora_id, None)
            for key in [key for key in instance.schedules if key[0] == lora_id]:
                del instance.schedules[key]
        stack.set_user_entries(entries, set())
        stack.groups = {letter: members for letter, members in groups.items() if len(members) >= 2}
        for letter, rank in bakes.items():
            group = sm.groups(stack).get(letter)
            if group is not None:
                sm.bake(stack, letter, files.bake_id(group, rank))
        sm.settle(stack, files.empty)

        unrun = sum(1 for _, _, token in entries if stack_text.phase_count(token) > phases.effective)
        pasted = {lora_id for lora_id, _, _ in entries}
        entering = up.entering_two_phases(stack, phases)
        up.fit_phase_structure(stack, phases, instance.hidden_phases)
        up.sync_schedules(stack, phases, instance.schedules, context)
        up.fill_stages_from_stage_1(
            stack, entering, phases, instance.phase_memory, instance.schedules, context
        )
        up.refit_schedules(stack, phases, instance.schedules, instance.phase_memory, context)

        notes = []
        if unrun:
            notes.append(self._unrun_note(unrun, phases))
        if ignored_modes:
            notes.append(
                f"{ignored_modes} line{'s' if ignored_modes != 1 else ''} asked for visual or audio only, "
                "which only LTX-2 models have; loaded as normal."
            )
        lonely = [letter for letter, members in groups.items() if len(members) < 2]
        if lonely:
            notes.append(f"Group {', '.join(lonely)} has one LoRA, so it is not a group.")
        carried = sum(1 for lora_id in entering if lora_id in pasted)
        if carried:
            notes.append(
                f"{carried} line{'s' if carried != 1 else ''} gave stage 1 only; "
                "stage 2 carries on from where stage 1 ends."
            )

        loaded = len(entries)
        if problems:
            skipped = "; ".join(f"line {problem.number}: {problem.reason}" for problem in problems[:2])
            more = f" (+{len(problems) - 2} more)" if len(problems) > 2 else ""
            message = f"Loaded {loaded} of {len(lines)} LoRAs from text. Skipped {skipped}{more}."
        else:
            message = f"Loaded {loaded} LoRA{'s' if loaded != 1 else ''} from text."
        # A phase the text sets but the mode will not run is worth a warning
        # too: what renders is not everything that was pasted.
        instance.note(" ".join([message] + notes), bool(problems) or bool(unrun))
        report.update(ok=not problems, loaded=loaded, notes=notes)
        return self._stack_snapshot(stack) != before

    @staticmethod
    def _paste_phase_problem(token: str, phases: up.PhaseConfig) -> str:
        """A pasted token naming more phases than WanGP accepts here.

        Plain lists are cut to fit with the rest of the stack, as a preset's
        are; branch syntax never is (``fit_phase_structure``), so a branch
        token naming too many phases could only be refused by WanGP.
        """
        count = stack_text.phase_count(token)
        if ":" in token and count > phases.capacity:
            return f"names {count} phases; this model takes {phases.capacity} here"
        return ""

    @staticmethod
    def _unrun_note(count: int, phases: up.PhaseConfig) -> str:
        lines = f"{count} line{'s' if count != 1 else ''}"
        if phases.effective <= 1:
            # On LTX-2 a stage 2 One Phase cut does not come back as it was
            # pasted, so say how to load it as written.
            advice = " - switch to Two Phases and paste again to use it" if phases.staged else ""
            return f"One Phase runs phase 1 only: phase 2 from {lines} is not applied{advice}."
        return f"This mode runs {phases.effective} phases: later phases from {lines} are not applied."

    @staticmethod
    def _stack_snapshot(stack: up.Stack) -> tuple:
        return (
            tuple(stack.ids), tuple(stack.tokens), stack.separator_index,
            tuple((entry.id, entry.token, entry.at, entry.skip) for entry in stack.disabled),
            tuple(sorted((letter, tuple(members)) for letter, members in stack.groups.items())),
        )

    # ---------------------------------------------------------- profiles

    def _profile_action(self, instance, action, stack, inventory, phases, model_type) -> bool:
        if not self._profiles:
            return False
        kind = action["type"]
        name = str(action.get("name", "") or "")

        try:
            if kind in ("profile_save", "profile_update"):
                entries, bakes = self._profile_entries(stack, self._stack_files(instance, inventory))
                if not entries:
                    instance.note("Select at least one LoRA before saving a profile.", True)
                    return False
                saved = self._profiles.save(
                    name, model_type, entries, overwrite=(kind == "profile_update"), bakes=bakes
                )
                extras = []
                modes = sum(1 for entry in entries if entry.mode != "normal")
                if modes:
                    extras.append(f"{modes} visual or audio only")
                letters = {entry.group for entry in entries if entry.group}
                if letters:
                    extras.append(f"{len(letters)} group{'s' if len(letters) != 1 else ''}")
                if bakes:
                    extras.append(f"{len(bakes)} baked")
                instance.note(f"Profile '{saved.name}' saved" + (f" ({', '.join(extras)})." if extras else "."))
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
                return self._recall_profile(instance, name, stack, inventory)
        except ProfileError as error:
            instance.note(str(error), True)
        return False

    def _recall_profile(self, instance, name, stack, inventory) -> bool:
        profile = self._profiles.get(name)
        if profile is None:
            instance.note(f"Profile '{name}' not found.", True)
            return False
        if not profile.loras:
            instance.note(f"Profile '{name}' has no LoRAs to apply.", True)
            return False

        # The model a profile was saved under is provenance, not a gate: model
        # families that share a LoRA folder (every LTX 2 variant, say) see each
        # other's LoRAs, so availability is the only thing that can block a
        # recall.
        applied, missing = [], []
        for entry in profile.loras:
            if inventory.get(entry.id) is None:
                missing.append(entry.id)
                continue
            applied.append(entry)

        if not applied:
            instance.note(
                f"Profile '{name}' was not applied - none of its "
                f"{len(missing)} LoRA(s) are available to this model.", True
            )
            return False

        # WanGP-managed LoRAs (left of the accelerator bar) are preserved; a
        # profile only ever replaces the user side of the stack, switched-off
        # LoRAs included -- with their modes, groups and bakes, any copy or
        # bake this folder lacks being built by the next payload.
        files = self._stack_files(instance, inventory)
        present = {entry.group for entry in applied if entry.group}
        bakes = {letter: rank for letter, rank in profile.bakes.items() if letter in present}
        self._apply_entries(stack, files, applied, bakes)

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

        Every reply echoes the request's ``seq``. The browser only hears a
        reply through the result textbox's change event, and Gradio raises none
        when the new text equals the old -- two preview-less thumbnail batches,
        or reopening a LoRA with no media, would otherwise leave the request
        unanswered until its timeout, with everything queued behind it.
        """
        try:
            request = json.loads(request_json or "{}")
        except (TypeError, ValueError):
            request = None
        if not isinstance(request, dict):
            return json.dumps({"kind": "none"})

        kind = str(request.get("kind", "thumbs"))
        try:
            if kind == "inspect":
                reply = self._serve_inspect(instance, request, state_value)
            elif kind == "fetch":
                reply = self._serve_fetch(instance, request, state_value)
            elif kind == "fetch_all":
                reply = self._serve_fetch_all(instance, request, state_value)
            elif kind == "media":
                reply = self._serve_catalogue_media(instance, request, state_value)
            elif kind == "video":
                reply = self._serve_preview_video(instance, request, state_value)
            elif kind == "builds":
                reply = {"kind": "builds", **self._builder.status()}
            elif kind == "built":
                reply = self._serve_built(instance, request, state_value)
            elif kind == "bake_plan":
                reply = self._serve_bake_plan(instance, request, state_value)
            else:
                reply = self._serve_thumbnails(instance, request, state_value)
        except Exception as error:
            traceback.print_exc()
            reply = {"kind": kind, "error": str(error)}
        reply["seq"] = request.get("seq")
        return json.dumps(reply)

    def _request_files(self, instance: InstanceState, state_value) -> StackFiles:
        model_type = self._model_type(state_value)
        lora_dir = self._lora_dir(model_type)
        inventory = build_inventory(self._native_loras(state_value, []), lora_dir)
        instance.lora_dir = lora_dir
        instance.sided = up.is_ltx2(self._model_def(model_type))
        return self._stack_files(instance, inventory)

    def _profile_references(self, files: StackFiles) -> dict[str, list[str]]:
        """Built file id -> the profiles that use it."""
        used: dict[str, list[str]] = {}
        if not self._profiles:
            return used
        for name in self._profiles.names():
            profile = self._profiles.get(name)
            if profile is None:
                continue
            stack = up.Stack()
            self._apply_entries(stack, files, profile.loras, profile.bakes)
            for lora_id, _, _ in stack.user_entries():
                if dv.parse(lora_id) is not None:
                    used.setdefault(lora_id, []).append(name)
        return used

    def _serve_built(self, instance, request, state_value) -> dict:
        """The built files for this model's LoRA folder, what uses each, and
        deleting the ones nothing uses.

        The frontend says which built files its stack uses -- it is the one
        that knows the stack -- and the profiles are read here.
        """
        files = self._request_files(instance, state_value)
        store = files.store
        if store is None:
            return {"kind": "built", "files": [], "error": "This model has no LoRA folder."}
        in_stack = {normalize_id(value) for value in (request.get("in_use") or []) if dv.parse(value)}
        used = self._profile_references(files)
        listed = store.files() if (os.path.isdir(store.root) or os.path.isdir(store.link)) else []
        unused = [item["id"] for item in listed if item["id"] not in in_stack and item["id"] not in used]
        if request.get("action") == "delete_unused":
            removed = store.delete(unused)
            instance.note(f"Deleted {removed} unused built file{'s' if removed != 1 else ''}.")
            listed = store.files()
            unused = []
        rows = []
        for item in listed:
            users = (["this stack"] if item["id"] in in_stack else []) + used.get(item["id"], [])
            rows.append({
                "id": item["id"], "bytes": item.get("bytes", 0), "kind": item.get("kind", ""),
                "mode": item.get("mode", ""), "rank": item.get("rank", 0), "kept": item.get("kept"),
                "exact": item.get("exact"), "label": item.get("label", ""), "linked": bool(item.get("linked")),
                "members": [member.get("name", "") for member in item.get("members", []) or []],
                "used_by": users,
            })
        folder = os.path.relpath(store.root, os.path.dirname(PLUGIN_DIR)).replace("\\", "/")
        return {
            "kind": "built", "files": rows, "unused": len(unused), "folder": folder,
            "link": f"{dv.LINK_NAME}/", "linked": store.linked if listed else True,
        }

    def _serve_bake_plan(self, instance, request, state_value) -> dict:
        """Sizes for the bake dialog, from the members' headers only."""
        files = self._request_files(instance, state_value)
        members = [normalize_id(value) for value in (request.get("members") or []) if normalize_id(value)]
        if len(members) < 2:
            return {"kind": "bake_plan", "error": "A bake needs at least two LoRAs."}
        stack = up.Stack()
        stack.set_user_entries([(member, True, "1") for member in members], set())
        stack.groups = {"A": list(members)}
        reply = files.bake_options(stack, "A")
        reply["kind"] = "bake_plan"
        reply["group"] = str(request.get("group", "") or "")
        return reply

    def _entry_for(self, state_value, lora_id):
        """Resolve a frontend-supplied id against the current native inventory."""
        model_type = self._model_type(state_value)
        lora_dir = self._lora_dir(model_type)
        inventory = build_inventory(self._native_loras(state_value, []), lora_dir)
        return inventory.get(normalize_id(lora_id)), lora_dir

    def _inspect_payload(self, entry) -> dict:
        """What the Inspect view renders for one LoRA.

        A LoRA with no catalogue at all still gets a payload rather than a bare
        error: the view is also where the catalogue is fetched from, so it has
        to be openable before there is anything to show.
        """
        detail = cat.read_detail(entry.path)
        return {
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
            "has_catalogue": cat.sidecar_dir(entry.path) is not None,
            # Named so the view can say what an existing catalogue still lacks
            # rather than looking finished because it has pictures.
            "missing": civitai.missing_parts(entry.path),
            "note": detail.error,
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
        }

    def _serve_inspect(self, instance, request, state_value) -> dict:
        entry, lora_dir = self._entry_for(state_value, request.get("id"))
        if entry is None or not entry.path:
            return {"kind": "inspect", "error": "That LoRA is not in the current inventory."}
        return self._inspect_payload(entry)

    def _civitai_key(self) -> str:
        """The environment wins: a key exported for the session is never
        written into the plugin's own JSON."""
        from_env = str(os.environ.get("CIVITAI_API_KEY", "") or "").strip()
        if from_env:
            return from_env
        return self._metadata.civitai_api_key if self._metadata else ""

    def _serve_fetch(self, instance, request, state_value) -> dict:
        """Build or top up the Civitai catalogue folder for one LoRA.

        Deliberately synchronous: it runs on an explicit click, the frontend
        shows it as in flight, and a background job would need progress
        plumbing that a one-LoRA fetch does not earn.
        """
        entry, lora_dir = self._entry_for(state_value, request.get("id"))
        if entry is None or not entry.path:
            return {"kind": "fetch", "error": "That LoRA is not in the current inventory."}

        report = civitai.fetch_sidecar(
            entry.path, api_key=self._civitai_key(), lora_root=lora_dir
        )
        # The index is cached against the sidecar folder's mtime; a fetch that
        # only rewrote documents inside it must not read as unchanged.
        instance.catalogue_cache.pop(entry.id, None)

        payload = self._inspect_payload(entry) if report.ok else {}
        payload.update({
            "kind": "fetch",
            "id": entry.id,
            "ok": report.ok,
            "message": report.message,
            "downloaded": report.downloaded,
        })
        instance.note(report.message, not report.ok)
        return payload

    def _bulk_payload(self, extra: dict | None = None) -> dict:
        payload = {"kind": "fetch_all", "running": False, "total": 0, "done": 0}
        if self._bulk is not None:
            payload.update(self._bulk.status())
            payload["running"] = self._bulk.running
        payload.update(extra or {})
        return payload

    def _serve_fetch_all(self, instance, request, state_value) -> dict:
        """Start, poll or stop a catalogue fetch across the whole inventory.

        The work runs on a worker thread and the frontend polls: a hundred
        LoRAs is minutes of hashing and downloading, which no single bridge
        call should be holding open.
        """
        action = str(request.get("action", "start"))

        if action == "stop":
            if self._bulk is not None:
                self._bulk.cancel()
            return self._bulk_payload()

        if action == "status":
            if self._bulk is not None and not self._bulk.running:
                instance.note(self._bulk.summary())
            return self._bulk_payload()

        if self._bulk is not None and self._bulk.running:
            return self._bulk_payload()      # already going; just report it

        model_type = self._model_type(state_value)
        lora_dir = self._lora_dir(model_type)
        inventory = build_inventory(self._native_loras(state_value, []), lora_dir)
        targets = [
            entry.path for entry in inventory.entries
            if entry.path and civitai.needs_fetch(entry.path)
        ]
        if not targets:
            instance.note("Every LoRA this model offers already has a catalogue.")
            return self._bulk_payload({"running": False, "total": 0, "finished": True})

        self._bulk = civitai.BulkFetch(
            targets, api_key=self._civitai_key(), lora_root=lora_dir
        ).start()
        return self._bulk_payload({"running": True})

    def _serve_catalogue_media(self, instance, request, state_value) -> dict:
        entry, _ = self._entry_for(state_value, request.get("id"))
        if entry is None or not entry.path:
            return {"kind": "media", "error": "unknown LoRA"}

        try:
            wanted = int(request.get("index", 0))
        except (TypeError, ValueError):
            return {"kind": "media", "error": "bad index"}

        detail = cat.read_detail(entry.path)
        item = next((media for media in detail.media if media.index == wanted), None)
        if item is None:
            return {"kind": "media", "error": "no such media"}

        data = (
            self._image_data_uri(item.path)
            if item.kind == "image"
            else self._file_data_uri(item.path)
        )
        if not data:
            return {"kind": "media", "id": entry.id, "index": wanted,
                    "error": "could not read media"}
        return {"kind": "media", "id": entry.id, "index": wanted,
                "media_kind": item.kind, "data": data}

    def _serve_preview_video(self, instance, request, state_value) -> dict:
        """The same-stem video beside the .safetensors, for tile playback."""
        entry, lora_dir = self._entry_for(state_value, request.get("id"))
        if entry is None or not entry.path:
            return {"kind": "video", "error": "unknown LoRA"}

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
            return {"kind": "video", "id": entry.id, "error": "no preview video"}

        data = self._file_data_uri(match)
        if not data:
            return {"kind": "video", "id": entry.id, "error": "video too large to preview"}
        return {"kind": "video", "id": entry.id, "data": data}

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
                raw = handle.read()
        except OSError:
            return ""
        payload = base64.b64encode(raw).decode("ascii")
        # Typed by what the bytes are, not the name: a WebM saved as .mp4 by an
        # older catalogue still plays. QuickTime is labelled MP4 because that is
        # what browsers will play an H.264 .mov as.
        _, sniffed = cat.sniff_media(raw[: cat.SNIFF_BYTES])
        mime = _SNIFFED_MIME.get(sniffed) or mimetypes.guess_type(path)[0] or "application/octet-stream"
        return f"data:{mime};base64,{payload}"

    def _serve_thumbnails(self, instance: InstanceState, request, state_value) -> dict:
        """Resolve previews for the tiles the browser can actually see.

        The frontend only ever sends stable LoRA IDs; paths are resolved here
        against the current model's LoRA directory, and nothing under it is
        exposed as a static route -- each tile gets a small derived WebP as a
        data URI instead.
        """
        requested = [normalize_id(value) for value in request.get("ids", []) or []]
        if not requested or self._thumbnails is None:
            return {"kind": "thumbs", "thumbs": {}}

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
        return {"kind": "thumbs", "thumbs": thumbs}
