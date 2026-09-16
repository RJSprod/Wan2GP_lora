"""End-to-end checks of the JSON bridge in plugin.py.

WanGP and Gradio are not importable in a test environment, so both are stubbed
with just enough surface for the payload/action functions.  The UI construction
in ``_build_panel`` is not exercised here -- that needs a real Gradio -- but
every function that moves state between the browser and WanGP is.
"""

import importlib.util
import json
import os
import sys
import types

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE = "wan2gp_lora_browser_under_test"


def _install_stubs():
    """Minimal stand-ins for the two imports plugin.py makes at module level."""
    if "gradio" not in sys.modules:
        gradio = types.ModuleType("gradio")

        class _Update(dict):
            pass

        gradio.update = lambda **kwargs: _Update(kwargs)
        gradio.Info = lambda *args, **kwargs: None
        gradio.Warning = lambda *args, **kwargs: None
        sys.modules["gradio"] = gradio

    if "shared.utils.plugins" not in sys.modules:
        shared = sys.modules.setdefault("shared", types.ModuleType("shared"))
        utils = sys.modules.setdefault("shared.utils", types.ModuleType("shared.utils"))
        plugins = types.ModuleType("shared.utils.plugins")

        class WAN2GPPlugin:
            def __init__(self):
                self.name = ""
                self.version = "1.0.0"
                self.description = ""
                self.type = ["app"]
                self._component_requests = []
                self._global_requests = []
                self._insert_after_requests = []

            def request_component(self, component_id):
                self._component_requests.append(component_id)

            def request_global(self, global_name):
                self._global_requests.append(global_name)

            def insert_after(self, target_component_id, new_component_constructor):
                self._insert_after_requests.append((target_component_id, new_component_constructor))

        plugins.WAN2GPPlugin = WAN2GPPlugin
        shared.utils = utils
        utils.plugins = plugins
        sys.modules["shared.utils.plugins"] = plugins


def _load_plugin_module():
    _install_stubs()
    if PACKAGE + ".plugin" in sys.modules:
        return sys.modules[PACKAGE + ".plugin"]
    spec = importlib.util.spec_from_file_location(
        PACKAGE, os.path.join(ROOT, "__init__.py"), submodule_search_locations=[ROOT]
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE] = package
    spec.loader.exec_module(package)
    return importlib.import_module(PACKAGE + ".plugin")


plugin_module = _load_plugin_module()

H3_MODEL_DEF = {
    "guidance_max_phases": 2,
    "lora_multiplier_phases": 2,
    "phase_2_spatial_tiling": True,
}
LORAS = ["a.safetensors", "b.safetensors", "sub/c.safetensors"]


@pytest.fixture
def plugin(tmp_path):
    """A plugin wired to a fake WanGP with a MiniMax-H3-shaped model."""
    instance = plugin_module.LoraBrowserPlugin()
    instance.server_config_filename = str(tmp_path / "wgp_config.json")
    instance.get_state_model_type = lambda state: state.get("model_type", "")
    instance.get_model_def = lambda model_type: H3_MODEL_DEF
    instance.get_lora_dir = lambda model_type: str(tmp_path / "loras")
    instance._init_stores()
    return instance


@pytest.fixture
def state():
    return {"model_type": "minimax_h3", "loras": list(LORAS)}


def payload_of(plugin, state, selected, multipliers, guidance=2, steps=None):
    instance = plugin._instance("test")
    return json.loads(plugin._build_payload(instance, state, selected, multipliers, guidance, steps))


def act(plugin, action, state, selected, multipliers, guidance=2, steps=None):
    instance = plugin._instance("test")
    choices, mults, payload = plugin._apply_action(
        instance, json.dumps(action), state, selected, multipliers, guidance, steps
    )
    return choices, mults, json.loads(payload)


def row_of(payload, lora_id):
    return next(row for row in payload["active"] if row["id"] == lora_id)


# Step schedules only exist in One Phase guidance, so the schedule suites drive
# the bridge there unless a test is specifically about another mode.
def sact(plugin, action, state, selected, multipliers, guidance=1, steps=None):
    return act(plugin, action, state, selected, multipliers, guidance, steps)


def spayload(plugin, state, selected, multipliers, guidance=1, steps=None):
    return payload_of(plugin, state, selected, multipliers, guidance, steps)


class _FakeCivitai:
    """Stands in for urlopen inside lora_browser.civitai."""

    VERSION = {
        "id": 9001, "modelId": 4242, "name": "v1", "baseModel": "LTXV 2.3",
        "trainedWords": ["trigger"],
        "images": [{
            "url": "https://image.civitai.com/x/1.jpeg", "type": "image",
            "meta": {"prompt": "a quiet street"},
        }],
    }
    MODEL = {"id": 4242, "name": "Live Wallpaper Style", "creator": {"username": "NRDX"}}

    class _Response:
        def __init__(self, body, content_type):
            self._body = body
            self.headers = {"Content-Type": content_type}

        def read(self, amount=None):
            body, self._body = self._body, b""
            return body

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def __call__(self, request, timeout=None):
        url = request.full_url
        if "by-hash" in url:
            return self._Response(json.dumps(self.VERSION).encode(), "application/json")
        if "/models/" in url:
            return self._Response(json.dumps(self.MODEL).encode(), "application/json")
        return self._Response(b"\xff\xd8\xff", "image/jpeg")


def serve(plugin, request, state, state_value=None):
    instance = plugin._instance("test")
    return json.loads(plugin._serve_media(instance, json.dumps(request), state))


class TestSetup:
    def test_requests_the_components_it_bridges(self):
        instance = plugin_module.LoraBrowserPlugin()
        instance.setup_ui()
        for name in ("loras_choices", "loras_multipliers", "guidance_phases", "state", "main"):
            assert name in instance._component_requests

    def test_declares_itself_an_extension(self):
        assert plugin_module.LoraBrowserPlugin().type == ["extension"]

    def test_injects_after_the_native_multiplier_box(self, plugin):
        components = {
            "loras_choices": types.SimpleNamespace(elem_id=None),
            "loras_multipliers": types.SimpleNamespace(elem_id=None, _id=7),
            "state": object(),
            "main": object(),
        }
        plugin.post_ui_setup(components)
        assert plugin._insert_after_requests[0][0] == "loras_multipliers"

    def test_missing_components_leave_the_native_ui_alone(self, plugin):
        plugin.post_ui_setup({"loras_choices": object()})
        assert plugin._insert_after_requests == []


class TestPayload:
    def test_inventory_comes_from_wangp_not_from_a_disk_scan(self, plugin, state):
        payload = payload_of(plugin, state, [], "")
        assert [item["id"] for item in payload["items"]] == LORAS

    def test_selection_and_strengths_are_reflected(self, plugin, state):
        payload = payload_of(plugin, state, ["a.safetensors"], "0.8;0.4")
        item = next(item for item in payload["items"] if item["id"] == "a.safetensors")
        assert item["active"] is True
        assert payload["active"][0]["phase_values"] == [0.8, 0.4]

    def test_one_phase_mode_shows_a_single_control(self, plugin, state):
        payload = payload_of(plugin, state, ["a.safetensors"], "0.8;0.4", guidance=1)
        assert payload["phases"]["effective"] == 1
        assert payload["active"][0]["phase_values"] == [0.8]

    def test_tiling_shows_two_controls_not_three(self, plugin, state):
        payload = payload_of(plugin, state, ["a.safetensors"], "1;1", guidance="2~")
        assert payload["phases"]["effective"] == 2
        assert len(payload["active"][0]["phase_values"]) == 2

    def test_a_missing_lora_is_surfaced_as_a_warning(self, plugin, state):
        payload = payload_of(plugin, state, ["ghost.safetensors"], "1;1")
        assert payload["active"][0]["missing"] is True
        assert payload["status_warn"] is True

    def test_a_render_failure_asks_the_frontend_to_restore_native_controls(self, plugin, state, monkeypatch):
        def boom(*args, **kwargs):
            raise RuntimeError("inventory exploded")

        monkeypatch.setattr(plugin, "_context", boom)
        payload = payload_of(plugin, state, [], "")
        assert "error" in payload

    def test_payload_contains_no_filesystem_paths(self, plugin, state, tmp_path):
        raw = plugin._build_payload(plugin._instance("test"), state, ["a.safetensors"], "1;1", 2)
        assert str(tmp_path) not in raw

    def test_revision_increases_so_stale_responses_can_be_ignored(self, plugin, state):
        first = payload_of(plugin, state, [], "")
        second = payload_of(plugin, state, [], "")
        assert second["revision"] > first["revision"]


class TestActions:
    def test_toggle_on_adds_at_one_per_phase(self, plugin, state):
        choices, mults, _ = act(plugin, {"type": "toggle", "id": "a.safetensors", "enabled": True}, state, [], "")
        assert choices["value"] == ["a.safetensors"]
        assert mults["value"] == "1;1"

    def test_toggle_off_removes(self, plugin, state):
        choices, mults, _ = act(
            plugin, {"type": "toggle", "id": "a.safetensors", "enabled": False},
            state, ["a.safetensors", "b.safetensors"], "0.5;0.5 0.9;0.9",
        )
        assert choices["value"] == ["b.safetensors"]
        assert mults["value"] == "0.9;0.9"

    def test_a_lora_wangp_does_not_offer_cannot_be_activated(self, plugin, state):
        choices, _, payload = act(
            plugin, {"type": "toggle", "id": "../../etc/passwd", "enabled": True}, state, [], ""
        )
        assert "value" not in choices
        assert payload["status_warn"] is True

    def test_removing_the_middle_lora_keeps_the_others_aligned(self, plugin, state):
        choices, mults, _ = act(
            plugin, {"type": "remove", "id": "b.safetensors"},
            state, LORAS, "0.1;0.1 0.2;0.2 0.3;0.3",
        )
        assert choices["value"] == ["a.safetensors", "sub/c.safetensors"]
        assert mults["value"] == "0.1;0.1 0.3;0.3"

    def test_a_batch_of_slider_moves_is_applied_in_one_write(self, plugin, state):
        action = {
            "type": "batch",
            "actions": [
                {"type": "set_strength", "id": "a.safetensors", "phase": 0, "value": 0.62},
                {"type": "set_strength", "id": "b.safetensors", "phase": 1, "value": 1.4},
            ],
        }
        _, mults, _ = act(plugin, action, state, ["a.safetensors", "b.safetensors"], "1;1 1;1")
        assert mults["value"] == "0.62;1 1;1.4"

    def test_the_accelerator_boundary_survives_a_round_trip(self, plugin, state):
        _, mults, _ = act(
            plugin, {"type": "set_strength", "id": "b.safetensors", "phase": 0, "value": 0.5},
            state, ["a.safetensors", "b.safetensors"], "1;1|0.9;0.9",
        )
        assert mults["value"] == "1;1|0.5;0.9"

    def test_an_untouched_advanced_schedule_is_not_rewritten(self, plugin, state):
        _, mults, _ = act(
            plugin, {"type": "set_strength", "id": "a.safetensors", "phase": 0, "value": 0.5},
            state, ["a.safetensors", "b.safetensors"], "1;1 0.5,0.9,1.2",
        )
        assert mults["value"] == "0.5;1 0.5,0.9,1.2"

    def test_metadata_actions_do_not_touch_native_state(self, plugin, state):
        choices, mults, payload = act(
            plugin, {"type": "favorite", "id": "a.safetensors", "value": True}, state, [], ""
        )
        assert "value" not in choices and "value" not in mults
        assert next(item for item in payload["items"] if item["id"] == "a.safetensors")["favorite"] is True

    def test_tags_are_stored_and_returned(self, plugin, state):
        _, _, payload = act(plugin, {"type": "tag", "id": "a.safetensors", "value": "portrait"}, state, [], "")
        assert next(item for item in payload["items"] if item["id"] == "a.safetensors")["tag"] == "portrait"

    def test_zoom_is_persisted_without_a_native_write(self, plugin, state):
        _, _, payload = act(plugin, {"type": "zoom", "value": 140}, state, [], "")
        assert payload["zoom_px"] == 140

    def test_malformed_action_json_is_ignored(self, plugin, state):
        instance = plugin._instance("test")
        choices, mults, _ = plugin._apply_action(instance, "{not json", state, [], "", 2)
        assert "value" not in choices and "value" not in mults


class TestDisableAllAndRestore:
    def test_disable_all_then_restore(self, plugin, state):
        choices, mults, payload = act(plugin, {"type": "disable_all"}, state, LORAS, "0.1;0.1 0.2;0.2 0.3;0.3")
        assert choices["value"] == []
        assert payload["can_restore"] is True

        choices, mults, _ = act(plugin, {"type": "restore"}, state, [], "")
        assert choices["value"] == LORAS
        assert mults["value"] == "0.1;0.1 0.2;0.2 0.3;0.3"

    def test_wangp_managed_loras_are_not_part_of_a_bulk_disable(self, plugin, state):
        choices, mults, _ = act(
            plugin, {"type": "disable_all"}, state, ["a.safetensors", "b.safetensors"], "1;1|0.9;0.9"
        )
        assert choices["value"] == ["a.safetensors"]

    def test_a_stale_snapshot_is_dropped_when_native_state_moves(self, plugin, state):
        act(plugin, {"type": "disable_all"}, state, LORAS, "0.1;0.1 0.2;0.2 0.3;0.3")
        # Something else (a preset, an .lset) changed the selection meanwhile.
        payload = payload_of(plugin, state, ["b.safetensors"], "0.7;0.7")
        assert payload["can_restore"] is False


class TestProfiles:
    def test_save_recall_round_trip(self, plugin, state):
        act(plugin, {"type": "profile_save", "name": "Mine"}, state, ["a.safetensors"], "0.8;0.4")
        choices, mults, payload = act(plugin, {"type": "profile_recall", "name": "Mine"}, state, [], "")
        assert choices["value"] == ["a.safetensors"]
        assert mults["value"] == "0.8;0.4"

    def test_the_dropdown_only_claims_an_exact_match(self, plugin, state):
        act(plugin, {"type": "profile_save", "name": "Mine"}, state, ["a.safetensors"], "0.8;0.4")
        assert payload_of(plugin, state, ["a.safetensors"], "0.8;0.4")["active_profile"] == "Mine"
        # External state diverged: the profile must no longer show as active.
        assert payload_of(plugin, state, ["a.safetensors"], "0.2;0.2")["active_profile"] == ""

    def test_a_profile_from_another_model_is_applied_when_its_loras_exist(self, plugin, state):
        """Sibling models (the LTX 2 family) share a LoRA folder -- never blocked."""
        act(plugin, {"type": "profile_save", "name": "Mine"}, state, ["a.safetensors"], "0.8;0.4")
        other = {"model_type": "wan22", "loras": list(LORAS)}
        choices, mults, payload = act(plugin, {"type": "profile_recall", "name": "Mine"}, other, [], "")
        assert choices["value"] == ["a.safetensors"]
        assert mults["value"] == "0.8;0.4"
        assert payload["status_warn"] is False

    def test_a_profile_whose_loras_are_all_absent_is_refused(self, plugin, state):
        act(plugin, {"type": "profile_save", "name": "Mine"}, state, ["a.safetensors"], "0.8;0.4")
        bare = {"model_type": "wan22", "loras": ["other.safetensors"]}
        choices, _, payload = act(plugin, {"type": "profile_recall", "name": "Mine"}, bare, [], "")
        assert "value" not in choices
        assert payload["status_warn"] is True
        assert "was not applied" in payload["status"]

    def test_a_partly_available_profile_applies_what_it_can(self, plugin, state):
        act(
            plugin, {"type": "profile_save", "name": "Mine"},
            state, ["a.safetensors", "b.safetensors"], "0.8;0.4 0.5;0.5",
        )
        partial = {"model_type": "wan22", "loras": ["a.safetensors"]}
        choices, mults, payload = act(plugin, {"type": "profile_recall", "name": "Mine"}, partial, [], "")
        assert choices["value"] == ["a.safetensors"]
        assert mults["value"] == "0.8;0.4"
        assert payload["status_warn"] is True

    def test_the_dropdown_flags_a_profile_this_model_cannot_fully_supply(self, plugin, state):
        act(
            plugin, {"type": "profile_save", "name": "Mine"},
            state, ["a.safetensors", "b.safetensors"], "0.8;0.4 0.5;0.5",
        )
        partial = {"model_type": "wan22", "loras": ["a.safetensors"]}
        payload = payload_of(plugin, partial, [], "")
        assert payload["profiles"] == ["Mine"]
        assert payload["profiles_incomplete"] == ["Mine"]
        assert payload_of(plugin, state, [], "")["profiles_incomplete"] == []

    def test_recall_preserves_wangp_managed_loras(self, plugin, state):
        act(plugin, {"type": "profile_save", "name": "Mine"}, state, ["b.safetensors"], "0.5;0.5")
        choices, mults, _ = act(
            plugin, {"type": "profile_recall", "name": "Mine"}, state, ["a.safetensors"], "1;1|"
        )
        assert choices["value"] == ["a.safetensors", "b.safetensors"]
        assert mults["value"] == "1;1|0.5;0.5"

    def test_saving_an_empty_stack_is_refused(self, plugin, state):
        _, _, payload = act(plugin, {"type": "profile_save", "name": "Empty"}, state, [], "")
        assert payload["status_warn"] is True
        assert payload["profiles"] == []


class TestCivitaiFetch:
    """The Inspect view can build a catalogue for a LoRA that has none."""

    @pytest.fixture
    def on_disk(self, tmp_path):
        """Make a.safetensors a real file so it can be hashed and enriched."""
        root = tmp_path / "loras"
        root.mkdir(exist_ok=True)
        (root / "a.safetensors").write_bytes(b"pretend weights")
        return root

    def test_inspect_opens_for_a_lora_with_no_catalogue(self, plugin, state, on_disk):
        payload = serve(plugin, {"kind": "inspect", "id": "a.safetensors"}, state)
        assert payload["has_catalogue"] is False
        assert payload["note"]              # explains there is nothing yet
        assert "error" not in payload       # so the fetch button still renders

    def test_a_fetch_builds_the_catalogue_and_returns_the_fresh_view(
        self, plugin, state, on_disk, monkeypatch
    ):
        monkeypatch.setattr(plugin_module.civitai, "urlopen", _FakeCivitai())
        payload = serve(plugin, {"kind": "fetch", "id": "a.safetensors"}, state)

        assert payload["ok"] is True
        assert payload["has_catalogue"] is True
        assert payload["civitai_name"] == "Live Wallpaper Style"
        assert [item["prompt"] for item in payload["media"]] == ["a quiet street"]
        assert (on_disk / "a" / "summary.txt").exists()

    def test_the_index_cache_does_not_hide_a_fresh_fetch(
        self, plugin, state, on_disk, monkeypatch
    ):
        assert payload_of(plugin, state, [], "")["items"][0]["has_catalogue"] is False
        monkeypatch.setattr(plugin_module.civitai, "urlopen", _FakeCivitai())
        serve(plugin, {"kind": "fetch", "id": "a.safetensors"}, state)
        item = payload_of(plugin, state, [], "")["items"][0]
        assert item["has_catalogue"] is True
        assert item["civitai_name"] == "Live Wallpaper Style"

    def test_a_lora_wangp_does_not_offer_cannot_be_fetched(self, plugin, state, on_disk):
        payload = serve(plugin, {"kind": "fetch", "id": "../secrets.safetensors"}, state)
        assert payload["error"]

    def test_a_failed_fetch_reports_without_claiming_success(
        self, plugin, state, on_disk, monkeypatch
    ):
        def offline(*args, **kwargs):
            raise OSError("no route to host")

        monkeypatch.setattr(plugin_module.civitai, "urlopen", offline)
        payload = serve(plugin, {"kind": "fetch", "id": "a.safetensors"}, state)
        assert payload["ok"] is False
        assert payload["message"]


class TestFetchAll:
    """One menu item, over everything the model offers that has no catalogue."""

    @pytest.fixture
    def library(self, tmp_path):
        root = tmp_path / "loras"
        (root / "sub").mkdir(parents=True, exist_ok=True)
        for name in LORAS:
            (root / name).write_bytes(b"weights " + name.encode())
        return root

    def _drain(self, plugin, state):
        """Run the job to completion the way the frontend's polling would."""
        started = serve(plugin, {"kind": "fetch_all", "action": "start"}, state)
        if plugin._bulk is not None and plugin._bulk._thread is not None:
            plugin._bulk._thread.join(timeout=30)
        return started, serve(plugin, {"kind": "fetch_all", "action": "status"}, state)

    def test_it_enriches_every_lora_that_had_nothing(
        self, plugin, state, library, monkeypatch
    ):
        monkeypatch.setattr(plugin_module.civitai, "urlopen", _FakeCivitai())
        started, done = self._drain(plugin, state)

        assert started["total"] == len(LORAS)
        assert done["finished"] is True
        assert done["fetched"] == len(LORAS)
        for item in payload_of(plugin, state, [], "")["items"]:
            assert item["has_catalogue"] is True
            assert item["civitai_name"] == "Live Wallpaper Style"

    def test_a_lora_that_already_has_a_catalogue_is_left_alone(
        self, plugin, state, library, monkeypatch
    ):
        monkeypatch.setattr(plugin_module.civitai, "urlopen", _FakeCivitai())
        serve(plugin, {"kind": "fetch", "id": "a.safetensors"}, state)

        started, _ = self._drain(plugin, state)
        assert started["total"] == len(LORAS) - 1

    def test_a_catalogue_with_images_but_no_prompts_is_picked_up(
        self, plugin, state, library, monkeypatch
    ):
        """Having pictures is not the same as being complete."""
        monkeypatch.setattr(plugin_module.civitai, "urlopen", _FakeCivitai())
        self._drain(plugin, state)
        os.remove(library / "a" / "media" / "001.json")

        started, done = self._drain(plugin, state)
        assert started["total"] == 1
        assert done["fetched"] == 1
        assert (library / "a" / "media" / "001.json").exists()

        payload = serve(plugin, {"kind": "inspect", "id": "a.safetensors"}, state)
        assert payload["media"][0]["prompt"] == "a quiet street"
        assert payload["missing"] == []

    def test_nothing_to_do_reports_instead_of_starting_a_run(
        self, plugin, state, library, monkeypatch
    ):
        monkeypatch.setattr(plugin_module.civitai, "urlopen", _FakeCivitai())
        self._drain(plugin, state)

        again = serve(plugin, {"kind": "fetch_all", "action": "start"}, state)
        assert again["total"] == 0
        assert again["running"] is False
        assert "already has a catalogue" in payload_of(plugin, state, [], "")["status"]

    def test_the_outcome_lands_on_the_status_line(
        self, plugin, state, library, monkeypatch
    ):
        monkeypatch.setattr(plugin_module.civitai, "urlopen", _FakeCivitai())
        self._drain(plugin, state)
        status = payload_of(plugin, state, [], "")["status"]
        assert status.startswith("Fetched 3 of 3 LoRA(s)")

    def test_a_second_start_does_not_launch_a_competing_run(
        self, plugin, state, library, monkeypatch
    ):
        import threading

        release = threading.Event()

        def blocking(path, **kwargs):
            release.wait(timeout=10)
            return plugin_module.civitai.FetchReport(ok=True, message="held")

        monkeypatch.setattr(plugin_module.civitai, "fetch_sidecar", blocking)
        serve(plugin, {"kind": "fetch_all", "action": "start"}, state)
        first = plugin._bulk

        try:
            payload = serve(plugin, {"kind": "fetch_all", "action": "start"}, state)
            assert payload["running"] is True
            assert plugin._bulk is first
        finally:
            release.set()
            first._thread.join(timeout=10)

    def test_stop_is_accepted_even_with_no_run_in_flight(self, plugin, state, library):
        payload = serve(plugin, {"kind": "fetch_all", "action": "stop"}, state)
        assert payload["running"] is False


class TestCivitaiKey:
    def test_a_stored_key_is_reported_but_never_sent_to_the_browser(self, plugin, state):
        act(plugin, {"type": "civitai_key", "value": "secret-key"}, state, [], "")
        payload = payload_of(plugin, state, [], "")
        assert payload["civitai_key_set"] is True
        assert "secret-key" not in json.dumps(payload)

    def test_clearing_the_key(self, plugin, state):
        act(plugin, {"type": "civitai_key", "value": "secret-key"}, state, [], "")
        act(plugin, {"type": "civitai_key", "value": ""}, state, [], "")
        assert payload_of(plugin, state, [], "")["civitai_key_set"] is False

    def test_the_environment_wins_and_is_not_written_to_the_store(
        self, plugin, state, monkeypatch
    ):
        monkeypatch.setenv("CIVITAI_API_KEY", "from-env")
        _, _, payload = act(plugin, {"type": "civitai_key", "value": "ignored"}, state, [], "")
        assert payload["status_warn"] is True
        assert plugin._metadata.civitai_api_key == ""
        assert plugin._civitai_key() == "from-env"


class TestMediaBridge:
    """One bridge serves grid thumbnails, catalogue detail and media bytes."""

    def _library(self, tmp_path, with_sidecar=True):
        pillow = pytest.importorskip("PIL.Image")
        lora_dir = tmp_path / "loras"
        (lora_dir / "sub").mkdir(parents=True, exist_ok=True)
        for name in ("a.safetensors", "b.safetensors"):
            (lora_dir / name).write_bytes(b"")
        (lora_dir / "sub" / "c.safetensors").write_bytes(b"")
        pillow.new("RGB", (64, 64), (200, 30, 30)).save(str(lora_dir / "a.png"))

        if with_sidecar:
            side = lora_dir / "a"
            (side / "media").mkdir(parents=True, exist_ok=True)
            (side / "summary.txt").write_text(
                "Civitai model name: Alpha Motion\n\nTrained words:\n- alphaword\n",
                encoding="utf-8",
            )
            (side / "a.json").write_text(json.dumps({
                "sha256": "deadbeef",
                "modelVersion": {"id": 2, "modelId": 1, "name": "v1",
                                 "trainedWords": ["alphaword"]},
                "model": {"id": 1, "name": "Alpha Motion",
                          "description": "<p>Alpha <b>does</b> things</p>",
                          "creator": {"username": "maker"}},
            }), encoding="utf-8")
            pillow.new("RGB", (900, 600), (20, 90, 160)).save(str(side / "media" / "001.jpg"))
            (side / "media" / "001.json").write_text(json.dumps({
                "index": 1, "source_url": "https://cdn/1.jpg",
                "civitai": {"width": 900, "height": 600,
                            "meta": {"prompt": "alpha prompt", "negativePrompt": "bad"}},
            }), encoding="utf-8")
        return lora_dir

    def ask(self, plugin, state, request):
        return json.loads(
            plugin._serve_media(plugin._instance("test"), json.dumps(request), state)
        )

    def test_thumbnails_are_addressed_by_stable_id(self, plugin, state, tmp_path):
        self._library(tmp_path)
        result = self.ask(plugin, state, {"kind": "thumbs", "ids": ["a.safetensors"]})
        assert result["thumbs"]["a.safetensors"].startswith("data:image/webp;base64,")

    def test_an_arbitrary_path_resolves_to_nothing(self, plugin, state, tmp_path):
        self._library(tmp_path)
        assert self.ask(plugin, state, {"kind": "thumbs", "ids": ["../../secret"]})["thumbs"] == {}

    def test_a_lora_without_a_preview_is_simply_absent(self, plugin, state, tmp_path):
        self._library(tmp_path)
        assert self.ask(plugin, state, {"kind": "thumbs", "ids": ["b.safetensors"]})["thumbs"] == {}

    def test_malformed_requests_are_ignored(self, plugin, state):
        result = json.loads(plugin._serve_media(plugin._instance("test"), "not json", state))
        assert result["kind"] == "none"

    def test_inspect_returns_the_catalogue(self, plugin, state, tmp_path):
        self._library(tmp_path)
        result = self.ask(plugin, state, {"kind": "inspect", "id": "a.safetensors"})
        assert result["civitai_name"] == "Alpha Motion"
        assert result["creator"] == "maker"
        assert result["trained_words"] == ["alphaword"]
        assert "does" in result["description"] and "<b>" not in result["description"]
        assert result["media"][0]["prompt"] == "alpha prompt"

    def test_inspect_never_leaks_filesystem_paths(self, plugin, state, tmp_path):
        self._library(tmp_path)
        raw = plugin._serve_media(
            plugin._instance("test"),
            json.dumps({"kind": "inspect", "id": "a.safetensors"}),
            state,
        )
        assert str(tmp_path) not in raw

    def test_inspect_without_a_sidecar_reports_an_error(self, plugin, state, tmp_path):
        self._library(tmp_path)
        result = self.ask(plugin, state, {"kind": "inspect", "id": "b.safetensors"})
        assert result["error"]

    def test_inspect_refuses_an_unknown_lora(self, plugin, state, tmp_path):
        self._library(tmp_path)
        assert self.ask(plugin, state, {"kind": "inspect", "id": "../escape"})["error"]

    def test_catalogue_media_is_fetched_by_index(self, plugin, state, tmp_path):
        self._library(tmp_path)
        result = self.ask(plugin, state, {"kind": "media", "id": "a.safetensors", "index": 1})
        assert result["media_kind"] == "image"
        assert result["data"].startswith("data:image/webp;base64,")

    def test_a_missing_media_index_is_reported(self, plugin, state, tmp_path):
        self._library(tmp_path)
        assert self.ask(plugin, state, {"kind": "media", "id": "a.safetensors", "index": 99})["error"]

    def test_preview_video_is_served_only_when_one_exists(self, plugin, state, tmp_path):
        lora_dir = self._library(tmp_path)
        assert self.ask(plugin, state, {"kind": "video", "id": "a.safetensors"})["error"]

        (lora_dir / "a.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64)
        result = self.ask(plugin, state, {"kind": "video", "id": "a.safetensors"})
        assert result["data"].startswith("data:video/mp4;base64,")

    def test_an_oversized_file_is_refused_rather_than_inlined(self, plugin, state, tmp_path, monkeypatch):
        lora_dir = self._library(tmp_path)
        (lora_dir / "a.mp4").write_bytes(b"\x00" * 4096)
        monkeypatch.setattr(plugin_module, "MEDIA_INLINE_LIMIT", 16)
        assert self.ask(plugin, state, {"kind": "video", "id": "a.safetensors"})["error"]


class TestCatalogueInPayload:
    def test_civitai_name_and_words_reach_the_browser(self, plugin, state, tmp_path):
        TestMediaBridge()._library(tmp_path)
        payload = payload_of(plugin, state, [], "")
        item = next(i for i in payload["items"] if i["id"] == "a.safetensors")
        assert item["civitai_name"] == "Alpha Motion"
        assert item["words"] == ["alphaword"]
        assert item["has_catalogue"] is True

    def test_a_lora_without_a_sidecar_is_marked_as_such(self, plugin, state, tmp_path):
        TestMediaBridge()._library(tmp_path)
        payload = payload_of(plugin, state, [], "")
        item = next(i for i in payload["items"] if i["id"] == "b.safetensors")
        assert item["civitai_name"] == ""
        assert item["has_catalogue"] is False

    def test_video_preview_presence_is_flagged(self, plugin, state, tmp_path):
        lora_dir = TestMediaBridge()._library(tmp_path)
        (lora_dir / "b.mp4").write_bytes(b"\x00")
        payload = payload_of(plugin, state, [], "")
        flags = {i["id"]: i["has_video"] for i in payload["items"]}
        assert flags["b.safetensors"] is True
        assert flags["a.safetensors"] is False

    def test_sort_and_name_preferences_round_trip(self, plugin, state):
        act(plugin, {"type": "sort", "value": "recent"}, state, [], "")
        act(plugin, {"type": "name_mode", "value": "file"}, state, [], "")
        payload = payload_of(plugin, state, [], "")
        assert payload["sort_mode"] == "recent"
        assert payload["name_mode"] == "file"

    def test_an_unknown_sort_mode_falls_back(self, plugin, state):
        act(plugin, {"type": "sort", "value": "nonsense"}, state, [], "")
        assert payload_of(plugin, state, [], "")["sort_mode"] == "name"


class TestDefaultProfile:
    def test_set_and_clear(self, plugin, state):
        act(plugin, {"type": "profile_save", "name": "Mine"}, state, ["a.safetensors"], "0.8;0.4")
        act(plugin, {"type": "profile_default", "name": "Mine"}, state, [], "")
        assert payload_of(plugin, state, [], "")["default_profile"] == "Mine"

        # Choosing the same profile again toggles it off.
        act(plugin, {"type": "profile_default", "name": "Mine"}, state, [], "")
        assert payload_of(plugin, state, [], "")["default_profile"] == ""

    def test_deleting_the_default_clears_it(self, plugin, state):
        act(plugin, {"type": "profile_save", "name": "Mine"}, state, ["a.safetensors"], "1;1")
        act(plugin, {"type": "profile_default", "name": "Mine"}, state, [], "")
        act(plugin, {"type": "profile_delete", "name": "Mine"}, state, [], "")
        assert payload_of(plugin, state, [], "")["default_profile"] == ""

    def test_renaming_the_default_follows_it(self, plugin, state):
        act(plugin, {"type": "profile_save", "name": "Old"}, state, ["a.safetensors"], "1;1")
        act(plugin, {"type": "profile_default", "name": "Old"}, state, [], "")
        act(plugin, {"type": "profile_rename", "name": "Old", "new_name": "New"}, state, [], "")
        assert payload_of(plugin, state, [], "")["default_profile"] == "New"

    def test_an_unknown_profile_cannot_become_default(self, plugin, state):
        _, _, payload = act(plugin, {"type": "profile_default", "name": "Ghost"}, state, [], "")
        assert payload["status_warn"] is True
        assert payload["default_profile"] == ""


class TestModelChange:
    def test_switching_model_drops_per_model_caches(self, plugin, state):
        instance = plugin._instance("test")
        act(plugin, {"type": "disable_all"}, state, LORAS, "1;1 1;1 1;1")
        assert instance.restore_snapshot is not None

        plugin.on_model_change({"model_type": "wan22"}, "wan22")
        assert instance.restore_snapshot is None
        assert instance.inventory_ids == []
        assert instance.phase_memory == {}


class TestScheduleActions:
    """The timeline's action contract, end to end through the bridge."""

    LORAS = ["a.safetensors", "b.safetensors"]

    def _schedule(self, payload, lora_id, phase=0):
        return row_of(payload, lora_id)["phase_schedules"][phase]

    def test_an_imported_schedule_is_offered_as_regions(self, plugin, state):
        payload = spayload(plugin, state, self.LORAS, "1,0.8,0.4,0;0.7 1;1")
        row = row_of(payload, "a.safetensors")
        assert row["multiplier_kind"] == "scheduled"
        # One Phase guidance, so one editable phase however many the token holds.
        assert row["phase_values"] == [1.0]
        first = row["phase_schedules"][0]
        # Every non-zero run is a region; the trailing 0 is a gap.
        assert [(r["start"], r["end"], r["strength"]) for r in first["regions"]] == [
            (1, 1, 1.0), (2, 2, 0.8), (3, 3, 0.4)
        ]
        assert first["gap_strength"] == 0

    def test_rendering_an_imported_schedule_never_rewrites_it(self, plugin, state):
        raw = "1.0,0.80,0.400;0.7,0.5"
        for _ in range(3):
            payload = spayload(plugin, state, self.LORAS, raw + " 1;1")
            assert row_of(payload, "a.safetensors")["multiplier_raw"] == raw
        # No native write happened at all: the action returns no component update.
        choices, mults, _ = sact(plugin, {"type": "ready"}, state, self.LORAS, raw + " 1;1")
        assert "value" not in choices and "value" not in mults

    def test_opening_a_schedule_does_not_change_the_native_token(self, plugin, state):
        choices, mults, payload = sact(
            plugin, {"type": "schedule_enable", "id": "a.safetensors", "phase": 0},
            state, self.LORAS, "0.71;1.23 1;1",
        )
        assert "value" not in mults
        schedule = self._schedule(payload, "a.safetensors", 0)
        assert schedule["base"] == 0.71
        assert schedule["regions"] == []
        # Not yet part of what WanGP is told.
        assert schedule["active"] is False

    def test_the_first_region_starts_at_the_strength_already_set(self, plugin, state):
        _, mults, _ = sact(
            plugin, {"type": "schedule_enable", "id": "a.safetensors", "phase": 0},
            state, self.LORAS, "0.71;1.23 1;1", steps=20,
        )
        # Opening the scheduler writes nothing at all.
        assert "value" not in mults

        _, mults, payload = sact(
            plugin, {"type": "schedule_add_region", "id": "a.safetensors", "phase": 0},
            state, self.LORAS, "0.71;1.23 1;1", steps=20,
        )
        schedule = self._schedule(payload, "a.safetensors", 0)
        region = schedule["regions"][0]
        assert region["strength"] == 0.71
        assert schedule["selected_region_id"] == region["id"]

        # Drawing it is already a schedule: covered slots carry the strength,
        # every other slot is zero.
        token = mults["value"].split()[0]
        assert token.startswith("0.71,0.71,0.71,0.71,0,0")
        assert token.endswith(";1.23")
        assert len(token.split(";")[0].split(",")) == 20

    def test_a_scheduled_phase_refuses_a_plain_strength_edit(self, plugin, state):
        _, mults, _ = sact(
            plugin, {"type": "set_strength", "id": "a.safetensors", "phase": 0, "value": 0.2},
            state, self.LORAS, "1,0.8,0.4,0;0.7,0.5,0.2,0 1;1",
        )
        assert "value" not in mults

        _, mults, payload = sact(
            plugin, {"type": "schedule_set_base", "id": "a.safetensors", "phase": 0, "value": 0.2},
            state, self.LORAS, "1,0.8,0.4,0;0.7,0.5,0.2,0 1;1",
        )
        assert "value" not in mults
        assert payload["status_warn"] is True
        assert "timeline" in payload["status"]

    def test_a_phase_the_mode_hides_keeps_its_value(self, plugin, state):
        """H3 carries two multiplier phases even in One Phase guidance, so the
        phase 2 someone tuned must survive editing phase 1's schedule."""
        payload = spayload(plugin, state, self.LORAS, "1,0.8;0.64 1;1")
        region = self._schedule(payload, "a.safetensors", 0)["regions"][0]
        _, mults, _ = sact(
            plugin,
            {"type": "schedule_set_region_strength", "id": "a.safetensors", "phase": 0,
             "region_id": region["id"], "value": 0.6},
            state, self.LORAS, "1,0.8;0.64 1;1",
        )
        assert mults["value"].split()[0] == "0.6,0.8;0.64"

    def test_clear_phase_returns_it_to_a_plain_strength(self, plugin, state):
        _, mults, payload = sact(
            plugin, {"type": "schedule_clear_phase", "id": "a.safetensors", "phase": 0},
            state, self.LORAS, "1,0.8,0.4;0.64 1;1",
        )
        # Back to a scalar, and the phase the mode hides is untouched.
        assert mults["value"].split()[0] == "1;0.64"
        assert self._schedule(payload, "a.safetensors", 0) is None

    def test_a_move_is_committed_with_collisions_resolved(self, plugin, state):
        payload = spayload(plugin, state, self.LORAS, "0,0,0.5,0.5,0.5,0,0.2,0.2 1;1")
        schedule = self._schedule(payload, "a.safetensors", 0)
        assert [(r["id"], r["start"], r["end"]) for r in schedule["regions"]] == [
            ("r1", 3, 5), ("r2", 7, 8)
        ]
        _, mults, payload = sact(
            plugin,
            {"type": "schedule_commit_region", "id": "a.safetensors", "phase": 0,
             "region_id": "r1", "start": 5, "end": 7, "keep_width": True},
            state, self.LORAS, "0,0,0.5,0.5,0.5,0,0.2,0.2 1;1",
        )
        assert mults["value"].split()[0] == "0,0,0,0,0.5,0.5,0.5,0.2"
        assert [(r["id"], r["start"], r["end"]) for r in
                self._schedule(payload, "a.safetensors", 0)["regions"]] == [("r1", 5, 7), ("r2", 8, 8)]

    def test_a_drag_beyond_the_end_is_clamped_not_rejected(self, plugin, state):
        _, mults, _ = sact(
            plugin,
            {"type": "schedule_commit_region", "id": "a.safetensors", "phase": 0,
             "region_id": "r1", "start": 7, "end": 12, "keep_width": True},
            state, self.LORAS, "0,0.4,0.4,0 1;1",
        )
        # Four slots, two-wide region: it can only reach 3-4.
        assert mults["value"].split()[0] == "0,0,0.4,0.4"

    def test_deleting_a_region_leaves_its_slots_at_zero(self, plugin, state):
        _, mults, payload = sact(
            plugin,
            {"type": "schedule_delete_region", "id": "a.safetensors", "phase": 0, "region_id": "r1"},
            state, self.LORAS, "0,0.5,0,0.2 1;1",
        )
        assert mults["value"].split()[0] == "0,0,0,0.2"
        regions = self._schedule(payload, "a.safetensors", 0)["regions"]
        assert [(r["start"], r["end"]) for r in regions] == [(4, 4)]

    def test_a_full_timeline_refuses_instead_of_overlapping(self, plugin, state):
        # One region across the whole timeline, then ask for another.
        _, mults, _ = sact(
            plugin,
            {"type": "schedule_commit_region", "id": "a.safetensors", "phase": 0,
             "region_id": "r1", "start": 1, "end": 3},
            state, self.LORAS, "0,0.4,0.4 1;1")
        assert mults["value"].split()[0] == "0.4,0.4,0.4"

        _, mults, payload = sact(
            plugin, {"type": "schedule_add_region", "id": "a.safetensors", "phase": 0},
            state, self.LORAS, "0.4,0.4,0.4 1;1",
        )
        assert "value" not in mults
        assert payload["status_warn"] is True
        assert "full" in payload["status"]
        schedule = self._schedule(payload, "a.safetensors", 0)
        assert [(r["start"], r["end"]) for r in schedule["regions"]] == [(1, 3)]

    def test_region_ids_survive_a_round_trip(self, plugin, state):
        multipliers = "1,0.5,1,0.2 1;1"
        first = spayload(plugin, state, self.LORAS, multipliers)
        ids = [r["id"] for r in self._schedule(first, "a.safetensors", 0)["regions"]]
        again = spayload(plugin, state, self.LORAS, multipliers)
        assert [r["id"] for r in self._schedule(again, "a.safetensors", 0)["regions"]] == ids

    def test_a_shared_schedule_may_claim_exact_global_steps(self, plugin, state):
        values = ",".join(["0.8"] * 29 + ["0.2"])
        payload = spayload(plugin, state, self.LORAS, values + " 1;1", steps=30)
        schedule = self._schedule(payload, "a.safetensors", 0)
        assert schedule["shared"] is True
        assert schedule["slots"] == 30
        assert schedule["coordinate_mode"] == "global_exact"
        assert payload["steps"] == 30

    def test_a_phase_specific_schedule_never_claims_global_steps(self, plugin, state):
        payload = spayload(plugin, state, self.LORAS, "1,0.8;0.7,0.5 1;1", steps=30)
        assert self._schedule(payload, "a.safetensors", 0)["coordinate_mode"] == "phase_relative"
        assert payload["phase_boundaries_known"] is False

    def test_a_shared_schedule_at_a_coarser_resolution_stays_relative(self, plugin, state):
        payload = spayload(plugin, state, self.LORAS, "1,0.8,0.4 1;1", steps=30)
        assert self._schedule(payload, "a.safetensors", 0)["coordinate_mode"] == "phase_relative"

    def test_a_new_shared_timeline_uses_the_step_count(self, plugin, state):
        # A one-phase model: a comma list can only mean the whole run, so the
        # timeline is one slot per inference step.
        plugin.get_model_def = lambda model_type: {"guidance_max_phases": 1}
        _, _, payload = sact(
            plugin, {"type": "schedule_enable", "id": "a.safetensors", "phase": 0},
            state, self.LORAS, "0.8 1", guidance=1, steps=24,
        )
        schedule = self._schedule(payload, "a.safetensors", 0)
        assert schedule["shared"] is True
        assert schedule["slots"] == 24 and schedule["coordinate_mode"] == "global_exact"

    def test_scheduling_is_refused_with_more_than_one_guidance_phase(self, plugin, state):
        """WanGP squeezes each phase's values into that phase, so a timeline
        there would not be showing what runs."""
        _, mults, payload = sact(
            plugin, {"type": "schedule_enable", "id": "a.safetensors", "phase": 0},
            state, self.LORAS, "0.8;0.5 1;1", guidance=2, steps=24,
        )
        assert "value" not in mults
        assert payload["scheduling_enabled"] is False
        assert payload["status_warn"] is True
        assert row_of(payload, "a.safetensors")["phase_schedules"] == []

    def test_an_empty_timeline_follows_the_step_counter(self, plugin, state):
        sact(plugin, {"type": "schedule_enable", "id": "a.safetensors", "phase": 0},
            state, self.LORAS, "0.8 1", guidance=1, steps=30)
        payload = spayload(plugin, state, self.LORAS, "0.8 1", guidance=1, steps=45)
        assert self._schedule(payload, "a.safetensors", 0)["slots"] == 45

    def test_a_drawn_timeline_keeps_its_own_length(self, plugin, state):
        """Its length is in the token; re-gridding it is an explicit choice."""
        payload = spayload(plugin, state, self.LORAS, "0,0,0.5,0.5;0.7 1;1", steps=30)
        schedule = self._schedule(payload, "a.safetensors", 0)
        assert schedule["slots"] == 4
        assert schedule["active"] is True

    def test_a_drawn_region_lands_where_it_was_drawn(self, plugin, state):
        sact(plugin, {"type": "schedule_enable", "id": "a.safetensors", "phase": 0},
            state, self.LORAS, "0.4;0.5 1;1", steps=10)
        _, _, payload = sact(
            plugin,
            {"type": "schedule_add_region", "id": "a.safetensors", "phase": 0,
             "start": 3, "end": 6},
            state, self.LORAS, "0.4;0.5 1;1", steps=10,
        )
        region = self._schedule(payload, "a.safetensors", 0)["regions"][0]
        assert (region["start"], region["end"]) == (3, 6)
        assert region["strength"] == 0.4          # the strength already set

    def test_a_tap_draws_a_region_of_the_default_width(self, plugin, state):
        sact(plugin, {"type": "schedule_enable", "id": "a.safetensors", "phase": 0},
            state, self.LORAS, "0.4;0.5 1;1", steps=20)
        _, _, payload = sact(
            plugin,
            {"type": "schedule_add_region", "id": "a.safetensors", "phase": 0, "start": 6},
            state, self.LORAS, "0.4;0.5 1;1", steps=20,
        )
        region = self._schedule(payload, "a.safetensors", 0)["regions"][0]
        assert region["start"] == 6
        assert 2 <= region["end"] - region["start"] + 1 <= 6

    def test_drawing_stops_at_the_neighbouring_region(self, plugin, state):
        # 0,0,0.5,0.5,0,0 -> one region at slots 3-4; drawing from slot 1 across
        # it must stop at slot 2 rather than swallowing it.
        _, mults, payload = sact(
            plugin,
            {"type": "schedule_add_region", "id": "a.safetensors", "phase": 0,
             "start": 1, "end": 5},
            state, self.LORAS, "0,0,0.5,0.5,0,0;0.7 1;1", steps=6,
        )
        regions = [(r["start"], r["end"]) for r in
                   self._schedule(payload, "a.safetensors", 0)["regions"]]
        assert (3, 4) in regions
        assert (1, 2) in regions

    def test_drawing_on_top_of_a_region_falls_back_to_free_space(self, plugin, state):
        _, _, payload = sact(
            plugin,
            {"type": "schedule_add_region", "id": "a.safetensors", "phase": 0,
             "start": 3, "end": 4},
            state, self.LORAS, "0,0,0.5,0.5,0,0;0.7 1;1", steps=6,
        )
        regions = [(r["start"], r["end"]) for r in
                   self._schedule(payload, "a.safetensors", 0)["regions"]]
        assert (3, 4) in regions                       # the existing one is intact
        assert len(regions) == 2
        assert all(left[1] < right[0] or right[1] < left[0]
                   for left in regions for right in regions if left != right)

    def test_an_ambiguous_multi_phase_schedule_is_read_only(self, plugin, state):
        three_phase = dict(H3_MODEL_DEF, guidance_max_phases=3, lora_multiplier_phases=3)
        plugin.get_model_def = lambda model_type: three_phase
        payload = spayload(plugin, state, self.LORAS, "0.9,0.8;1,1 1", guidance=3)
        row = row_of(payload, "a.safetensors")
        assert row["multiplier_kind"] == "advanced"
        assert row["advanced_preserved"] is True
        assert "3-phase" in row["schedule_reason"]

        # And an edit to it is refused with an explanation, not applied.
        _, mults, payload = sact(
            plugin, {"type": "schedule_add_region", "id": "a.safetensors", "phase": 0},
            state, self.LORAS, "0.9,0.8;1,1 1", guidance=3,
        )
        assert "value" not in mults
        assert payload["status_warn"] is True

    def test_normalising_is_explicit_and_rewrites_the_token(self, plugin, state):
        _, mults, payload = sact(
            plugin,
            {"type": "schedule_normalize", "id": "a.safetensors", "phase": 0, "slots": 8},
            state, self.LORAS, "1,0.5 1;1",
        )
        assert mults["value"].split()[0] == "1,1,1,1,0.5,0.5,0.5,0.5"
        assert self._schedule(payload, "a.safetensors", 0)["slots"] == 8

    def test_a_schedule_survives_a_model_switch_that_keeps_the_capacity(self, plugin, state):
        multipliers = "1,0.5,0.2 1;1"
        payload = spayload(plugin, state, self.LORAS, multipliers, guidance=1)
        row = row_of(payload, "a.safetensors")
        assert row["multiplier_kind"] == "scheduled"
        assert len(row["phase_schedules"]) == 1

    def test_schedules_are_dropped_when_the_lora_leaves_the_stack(self, plugin, state):
        instance = plugin._instance("test")
        spayload(plugin, state, self.LORAS, "1,0.5,0.2 1;1")
        assert instance.schedules
        spayload(plugin, state, ["b.safetensors"], "1;1")
        assert not [key for key in instance.schedules if key[0] == "a.safetensors"]

    def test_a_profile_round_trips_a_scheduled_token(self, plugin, state):
        sact(plugin, {"type": "profile_save", "name": "sched"}, state, self.LORAS,
            "1,0.8,0.4;0.7 1;1")
        _, mults, _ = sact(plugin, {"type": "profile_recall", "name": "sched"}, state, [], "")
        assert mults["value"].split()[0] == "1,0.8,0.4;0.7"

    def test_disable_all_and_restore_keep_a_schedule(self, plugin, state):
        sact(plugin, {"type": "disable_all"}, state, self.LORAS, "1,0.8,0.4;0.7 1;1")
        _, mults, _ = sact(plugin, {"type": "restore"}, state, [], "")
        assert mults["value"].split()[0] == "1,0.8,0.4;0.7"

    def test_the_slider_grid_and_precision_are_published(self, plugin, state):
        payload = spayload(plugin, state, [], "")
        assert payload["slider_step"] == 0.05
        assert payload["nudge_step"] == 0.01
        assert payload["value_decimals"] == 4
        assert payload["schedule_slot_limits"]["max"] >= 30


class TestImportSafety:
    """What an existing install's multipliers go through when this build reads them.

    Nothing here needs a converter: the panel derives regions from the token and
    those regions say the same thing back, so a schedule written by WanGP, a
    preset, an .lset file or a hand edit is rendered, never rewritten.
    """

    LORAS = ["a.safetensors", "b.safetensors"]

    #: Shapes a real install can hold, including ones the editor's own model
    #: would never produce.
    CORPUS = [
        "1;1",
        "0.71;1.23",
        "1,0.8,0.4,0;0.7,0.5,0.2,0",          # WanGP's documented form
        "0.9,0.8;1.2,1.1",
        "1;0.7,0.5,0.2",                       # scalar phase + scheduled phase
        "0,0,0.5,0.5,0",                       # off, then on, then off
        "0.5,0.5,0.5",                         # flat and non-zero
        "0,0,0",                               # flat and off
        "1.0,0.80,0.400;0.7",                  # trailing zeros the codec would trim
        "1.4,-0.5;0.125",                      # out of range and fine decimals
        "0.5:0.9;1",                           # branch syntax, never editable
        "1,0.5,0.25",                          # shared across phases
    ]

    @pytest.mark.parametrize("token", CORPUS)
    def test_rendering_never_touches_the_multiplier(self, plugin, state, token):
        multipliers = token + " 1;1"
        for _ in range(3):
            payload = payload_of(plugin, state, self.LORAS, multipliers, steps=30)
            assert row_of(payload, "a.safetensors")["multiplier_raw"] == token

        # And no action that only looks at it writes anything back.
        for action in (
            {"type": "ready"},
            {"type": "sort", "value": "recent"},
            {"type": "zoom", "value": 120},
        ):
            choices, mults, _ = act(plugin, action, state, self.LORAS, multipliers, steps=30)
            assert "value" not in choices and "value" not in mults

    @pytest.mark.parametrize("token", CORPUS)
    def test_what_the_editor_shows_is_what_the_token_says(self, plugin, state, token):
        """Regions compiled back must equal the values WanGP was given."""
        multipliers = token + " 1;1"
        payload = payload_of(plugin, state, self.LORAS, multipliers, steps=30)
        row = row_of(payload, "a.safetensors")
        if row["multiplier_kind"] != "scheduled":
            return

        declared = [part.split(",") for part in token.split(";")]
        for index, schedule in enumerate(row["phase_schedules"] or []):
            if schedule is None:
                continue
            source = declared[0] if row["schedule_shared"] else declared[index]
            if len(source) < 2:
                continue
            rebuilt = [schedule["gap_strength"]] * schedule["slots"]
            for region in schedule["regions"]:
                for slot in range(region["start"], region["end"] + 1):
                    rebuilt[slot - 1] = region["strength"]
            assert rebuilt == [float(value) for value in source]

    def test_opening_the_scheduler_on_an_imported_schedule_writes_nothing(self, plugin, state):
        multipliers = "1,0.8,0.4,0;0.7,0.5,0.2,0 1;1"
        for phase in (0, 1):
            choices, mults, payload = act(
                plugin, {"type": "schedule_enable", "id": "a.safetensors", "phase": phase},
                state, self.LORAS, multipliers, steps=30,
            )
            assert "value" not in choices and "value" not in mults
            assert row_of(payload, "a.safetensors")["multiplier_raw"] == "1,0.8,0.4,0;0.7,0.5,0.2,0"

    def test_an_accelerator_managed_stack_with_a_schedule_survives(self, plugin, state):
        multipliers = "1;1|1,0.8,0.4,0"
        payload = payload_of(plugin, state, self.LORAS, multipliers, steps=30)
        assert row_of(payload, "b.safetensors")["multiplier_kind"] == "scheduled"
        _, mults, _ = act(
            plugin, {"type": "set_strength", "id": "a.safetensors", "phase": 0, "value": 0.5},
            state, self.LORAS, multipliers, steps=30,
        )
        # The bar stayed put and the schedule to its right is untouched.
        assert mults["value"] == "0.5;1|1,0.8,0.4,0"

    def test_a_profile_and_a_disable_restore_cycle_preserve_zeros(self, plugin, state):
        multipliers = "0,0,0.5,0.5,0;0.7 1;1"
        act(plugin, {"type": "profile_save", "name": "zeros"}, state, self.LORAS, multipliers)
        _, mults, _ = act(plugin, {"type": "profile_recall", "name": "zeros"}, state, [], "")
        assert mults["value"].split()[0] == "0,0,0.5,0.5,0;0.7"

        act(plugin, {"type": "disable_all"}, state, self.LORAS, multipliers)
        _, mults, _ = act(plugin, {"type": "restore"}, state, [], "")
        assert mults["value"].split()[0] == "0,0,0.5,0.5,0;0.7"

    def test_editing_a_token_canonicalises_its_formatting_but_not_its_values(self, plugin, state):
        """The one rewrite there is, and only once an edit actually happens."""
        multipliers = "1.0,0.80,0.400;0.7 1;1"
        payload = payload_of(plugin, state, self.LORAS, multipliers, steps=30)
        assert row_of(payload, "a.safetensors")["multiplier_raw"] == "1.0,0.80,0.400;0.7"

        _, mults, _ = act(
            plugin, {"type": "set_strength", "id": "a.safetensors", "phase": 1, "value": 0.6},
            state, self.LORAS, multipliers, steps=30,
        )
        written = mults["value"].split()[0]
        assert written == "1,0.8,0.4;0.6"
        # Same numbers, spelled the way the serialiser spells them.
        assert [float(value) for value in written.split(";")[0].split(",")] == [1.0, 0.80, 0.400]


class TestStepCountSync:
    """Timelines follow num_inference_steps, live, without being asked.

    One Phase throughout: it is the only mode where a schedule is step-exact,
    and so the only mode the editor offers one in.
    """

    LORAS = ["a.safetensors", "b.safetensors"]

    def _drawn(self, plugin, state, multipliers, steps):
        """A schedule the panel authored, so it is step-aligned."""
        payload = payload_of(plugin, state, self.LORAS, multipliers, guidance=1, steps=steps)
        region = row_of(payload, "a.safetensors")["phase_schedules"][0]["regions"][0]
        _, mults, _ = act(
            plugin,
            {"type": "schedule_set_region_strength", "id": "a.safetensors", "phase": 0,
             "region_id": region["id"], "value": region["strength"]},
            state, self.LORAS, multipliers, guidance=1, steps=steps,
        )
        return mults.get("value", multipliers)

    def _resync(self, plugin, state, multipliers, steps):
        _, mults, payload = act(
            plugin, {"type": "schedule_resync"}, state, self.LORAS, multipliers,
            guidance=1, steps=steps,
        )
        return mults.get("value", multipliers), payload

    def test_a_longer_run_inherits_what_was_there(self, plugin, state):
        multipliers = self._drawn(plugin, state, "1,0.8,0.4,0.2 1;1", 4)
        written, _ = self._resync(plugin, state, multipliers, 5)
        assert written.split()[0] == "1,0.8,0.4,0.2,0"

    def test_a_shorter_run_drops_only_the_steps_that_are_gone(self, plugin, state):
        multipliers = self._drawn(plugin, state, "1,0.8,0.4,0.2 1;1", 4)
        written, _ = self._resync(plugin, state, multipliers, 3)
        assert written.split()[0] == "1,0.8,0.4"

    def test_a_region_straddling_the_new_end_is_clipped(self, plugin, state):
        multipliers = self._drawn(plugin, state, "0,0.5,0.5,0.5,0.5 1;1", 5)
        written, _ = self._resync(plugin, state, multipliers, 3)
        assert written.split()[0] == "0,0.5,0.5"

    def test_the_panel_asks_for_a_resync_and_then_stops(self, plugin, state):
        """The flag drives one round trip, not a loop."""
        multipliers = self._drawn(plugin, state, "1,0.8,0.4,0.2 1;1", 4)
        assert payload_of(plugin, state, self.LORAS, multipliers,
                          guidance=1, steps=9)["schedules_out_of_sync"]

        written, payload = self._resync(plugin, state, multipliers, 9)
        assert payload["schedules_out_of_sync"] is False
        assert len(written.split()[0].split(",")) == 9

        # Asking again changes nothing.
        again, _ = self._resync(plugin, state, written, 9)
        assert again.split()[0] == written.split()[0]

    def test_every_lora_and_phase_follows_at_once(self, plugin, state):
        multipliers = "1,0.8 0.2,0.2,0.2 "
        # Author both so they are step-aligned, then move the counter.
        for lora in self.LORAS:
            payload = payload_of(plugin, state, self.LORAS, multipliers, guidance=1, steps=2)
            row = row_of(payload, lora)
            for phase, schedule in enumerate(row["phase_schedules"] or []):
                if not schedule or not schedule["regions"]:
                    continue
                _, mults, _ = act(
                    plugin,
                    {"type": "schedule_set_region_strength", "id": lora, "phase": phase,
                     "region_id": schedule["regions"][0]["id"],
                     "value": schedule["regions"][0]["strength"]},
                    state, self.LORAS, multipliers, guidance=1, steps=2,
                )
                multipliers = mults.get("value", multipliers)

        written, payload = self._resync(plugin, state, multipliers, 4)
        first, second = written.split()[0], written.split()[1]
        assert len(first.split(";")[0].split(",")) == 4
        assert len(second.split(";")[0].split(",")) == 4
        assert payload["schedules_out_of_sync"] is False

    def test_an_untouched_import_is_resampled_not_truncated(self, plugin, state):
        """WanGP spreads it across the run; truncating would change the render."""
        written, _ = self._resync(plugin, state, "1,0.8,0.4,0 1;1", 12)
        assert written.split()[0] == "1,1,1,0.8,0.8,0.8,0.4,0.4,0.4,0,0,0"

    def test_a_read_only_multiplier_is_never_refitted(self, plugin, state):
        for token in ("0.5:0.9", "1"):
            _, mults, _ = act(
                plugin, {"type": "schedule_resync"}, state, self.LORAS,
                token + " 1", guidance=1, steps=12,
            )
            assert "value" not in mults

    def test_an_over_long_schedule_is_left_preserved(self, plugin, state):
        long_token = ",".join(["0.5"] * 200)
        _, mults, payload = act(
            plugin, {"type": "schedule_resync"}, state, self.LORAS,
            long_token + " 1", guidance=1, steps=12,
        )
        assert "value" not in mults
        assert row_of(payload, "a.safetensors")["phase_schedules"][0]["editable"] is False

    def test_a_step_count_beyond_the_ceiling_settles(self, plugin, state):
        """It must stop asking rather than never matching."""
        multipliers = self._drawn(plugin, state, "1,0.8,0.4,0.2 1;1", 4)
        written, payload = self._resync(plugin, state, multipliers, 10_000)
        assert payload["schedules_out_of_sync"] is False
        assert len(written.split()[0].split(",")) == 120

    def test_nothing_happens_when_the_step_count_is_unknown(self, plugin, state):
        multipliers = "1,0.8,0.4,0.2 1"
        payload = payload_of(plugin, state, self.LORAS, multipliers, guidance=1)
        assert payload["schedules_out_of_sync"] is False
        _, mults, _ = act(plugin, {"type": "schedule_resync"}, state, self.LORAS,
                          multipliers, guidance=1)
        assert "value" not in mults

    def test_an_empty_timeline_needs_no_write_to_follow(self, plugin, state):
        act(plugin, {"type": "schedule_enable", "id": "a.safetensors", "phase": 0},
            state, self.LORAS, "0.8 1", guidance=1, steps=10)
        payload = payload_of(plugin, state, self.LORAS, "0.8 1", guidance=1, steps=25)
        schedule = row_of(payload, "a.safetensors")["phase_schedules"][0]
        assert schedule["slots"] == 25
        assert payload["schedules_out_of_sync"] is False
