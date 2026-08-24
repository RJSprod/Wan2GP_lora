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

    def test_a_profile_from_another_model_is_not_applied(self, plugin, state):
        act(plugin, {"type": "profile_save", "name": "Mine"}, state, ["a.safetensors"], "0.8;0.4")
        other = {"model_type": "wan22", "loras": list(LORAS)}
        choices, _, payload = act(plugin, {"type": "profile_recall", "name": "Mine"}, other, [], "")
        assert "value" not in choices
        assert payload["status_warn"] is True

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


class TestThumbnailBridge:
    def test_the_frontend_can_only_ask_by_stable_id(self, plugin, state, tmp_path):
        pillow = pytest.importorskip("PIL.Image")
        lora_dir = tmp_path / "loras"
        lora_dir.mkdir(parents=True, exist_ok=True)
        (lora_dir / "a.safetensors").write_bytes(b"")
        pillow.new("RGB", (64, 64), (200, 30, 30)).save(str(lora_dir / "a.png"))

        instance = plugin._instance("test")
        result = json.loads(
            plugin._serve_thumbnails(instance, json.dumps({"ids": ["a.safetensors"]}), state)
        )
        assert result["thumbs"]["a.safetensors"].startswith("data:image/webp;base64,")

        # An arbitrary path is not resolvable and yields nothing.
        escaped = json.loads(
            plugin._serve_thumbnails(instance, json.dumps({"ids": ["../../secret"]}), state)
        )
        assert escaped["thumbs"] == {}

    def test_a_lora_without_a_preview_is_simply_absent(self, plugin, state, tmp_path):
        lora_dir = tmp_path / "loras"
        lora_dir.mkdir(parents=True, exist_ok=True)
        (lora_dir / "b.safetensors").write_bytes(b"")
        result = json.loads(
            plugin._serve_thumbnails(plugin._instance("test"), json.dumps({"ids": ["b.safetensors"]}), state)
        )
        assert result["thumbs"] == {}

    def test_malformed_requests_are_ignored(self, plugin, state):
        result = json.loads(plugin._serve_thumbnails(plugin._instance("test"), "not json", state))
        assert result["thumbs"] == {}


class TestModelChange:
    def test_switching_model_drops_per_model_caches(self, plugin, state):
        instance = plugin._instance("test")
        act(plugin, {"type": "disable_all"}, state, LORAS, "1;1 1;1 1;1")
        assert instance.restore_snapshot is not None

        plugin.on_model_change({"model_type": "wan22"}, "wan22")
        assert instance.restore_snapshot is None
        assert instance.inventory_ids == []
        assert instance.phase_memory == {}
