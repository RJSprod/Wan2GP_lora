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


def payload_of(plugin, state, selected, multipliers, guidance=2):
    instance = plugin._instance("test")
    return json.loads(plugin._build_payload(instance, state, selected, multipliers, guidance))


def act(plugin, action, state, selected, multipliers, guidance=2):
    instance = plugin._instance("test")
    choices, mults, payload = plugin._apply_action(
        instance, json.dumps(action), state, selected, multipliers, guidance
    )
    return choices, mults, json.loads(payload)


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
