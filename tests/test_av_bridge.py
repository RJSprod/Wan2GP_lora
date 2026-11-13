"""Visual-only/audio-only LoRAs, groups and bakes, end to end through the bridge.

An LTX-2 model with real (tiny) LoRA files on disk: one that changes picture
and sound, one that only changes the picture, one that only changes the sound.
Builds run on the test's thread, so each test decides when they happen.
"""

import json
import os

import numpy as np
import pytest

from lora_browser import builder as bd
from lora_browser import derived as dv
from tests.lora_fixtures import lora_pair, read_tensors, write_lora
from tests.test_plugin_bridge import H3_MODEL_DEF, plugin_module

LTX_MODEL_DEF = {"architecture": "ltx2_22B", "guidance_max_phases": 2, "ltx2_22B_class": True}
BOTH, PICTURE, SOUND = "both.safetensors", "picture.safetensors", "sound.safetensors"
LORAS = [BOTH, PICTURE, SOUND, "big.safetensors"]


class SyncBuilder(bd.Builder):
    """Queues like the real one; builds only when the test says so."""

    def submit(self, job, start=True):
        return super().submit(job, start=False)


def _write(path, modules, rank, seed):
    rng = np.random.default_rng(seed)
    tensors = []
    for module in modules:
        down, up = lora_pair(rng, 12, 10, rank)
        tensors += [(f"diffusion_model.transformer_blocks.0.{module}.lora_A.weight", "BF16", down),
                    (f"diffusion_model.transformer_blocks.0.{module}.lora_B.weight", "BF16", up)]
    write_lora(path, tensors, {"alpha": str(rank)})


@pytest.fixture
def loras(tmp_path):
    folder = tmp_path / "loras" / "ltx2"
    folder.mkdir(parents=True)
    _write(folder / BOTH, ["attn1.to_q", "audio_attn1.to_q", "video_to_audio_attn.to_k"], 4, 1)
    _write(folder / PICTURE, ["attn1.to_q", "ff.net.2"], 4, 2)
    _write(folder / SOUND, ["audio_attn1.to_q"], 2, 3)
    _write(folder / "big.safetensors", ["attn1.to_q", "audio_ff.net.2"], 8, 4)
    return folder


@pytest.fixture
def plugin(tmp_path, loras, monkeypatch):
    monkeypatch.setattr(plugin_module, "PLUGIN_DIR", str(tmp_path / "plugin"))
    instance = plugin_module.LoraBrowserPlugin()
    instance.server_config_filename = str(tmp_path / "wgp_config.json")
    instance.get_state_model_type = lambda state: state.get("model_type", "")
    instance.get_model_def = lambda model_type: H3_MODEL_DEF if model_type == "minimax_h3" else LTX_MODEL_DEF
    instance.get_lora_dir = lambda model_type: str(loras)
    instance._init_stores()
    instance._builder = SyncBuilder()
    return instance


@pytest.fixture
def state():
    return {"model_type": "ltx2_22B_distilled", "loras": list(LORAS)}


def payload_of(plugin, state, selected, multipliers):
    instance = plugin._instance("t")
    return json.loads(plugin._build_payload(instance, state, selected, multipliers, 2))


def act(plugin, action, state, selected, multipliers):
    instance = plugin._instance("t")
    choices, mults, payload = plugin._apply_action(instance, json.dumps(action), state, selected, multipliers, 2)
    return choices, mults, json.loads(payload)


def native(update, fallback):
    return update.get("value", fallback) if isinstance(update, dict) else fallback


def serve(plugin, request, state):
    return json.loads(plugin._serve_media(plugin._instance("t"), json.dumps(request), state))


def build_all(plugin):
    plugin._builder.run_pending()


V_BOTH = dv.split_id(BOTH, "visual")
A_BOTH = dv.split_id(BOTH, "audio")


# ---------------------------------------------------------------- payload


def test_rows_know_their_streams(plugin, state):
    payload = payload_of(plugin, state, [BOTH, PICTURE], "1 0.5")
    assert payload["av_supported"] is True and payload["stack_view"] == "list"
    row = next(row for row in payload["active"] if row["id"] == BOTH)
    assert row["source_id"] == BOTH and row["mode"] == "normal" and row["build"]["state"] == "original"
    assert row["streams"]["picture"]["modules"] == 1 and row["streams"]["sound"]["modules"] == 2
    assert row["streams"]["rank"] == 4
    picture = next(row for row in payload["active"] if row["id"] == PICTURE)
    assert picture["streams"]["sound"]["modules"] == 0


def test_no_link_is_made_until_something_is_built(plugin, state, loras):
    payload_of(plugin, state, [BOTH], "1")
    assert not os.path.lexists(loras / dv.LINK_NAME)


def test_other_models_offer_no_modes(plugin):
    h3 = {"model_type": "minimax_h3", "loras": list(LORAS)}
    payload = payload_of(plugin, h3, [BOTH], "1;1")
    assert payload["av_supported"] is False and "streams" not in payload["active"][0]
    choices, mults, payload = act(plugin, {"type": "set_mode", "id": BOTH, "mode": "visual"}, h3, [BOTH], "1;1")
    assert choices == {} and "LTX-2" in payload["status"]


# ------------------------------------------------------------------ modes


def test_visual_only_builds_a_copy_wangp_can_load(plugin, state, loras):
    choices, mults, payload = act(plugin, {"type": "set_mode", "id": BOTH, "mode": "visual"}, state, [BOTH, PICTURE], "0.8;0.8 1;1")
    selected = native(choices, None)
    assert selected == [V_BOTH, PICTURE] and native(mults, None) == "0.8;0.8 1;1"
    row = payload["active"][0]
    assert row["id"] == V_BOTH and row["source_id"] == BOTH and row["mode"] == "visual"
    assert row["name"] == BOTH and not row["missing"]
    assert row["build"]["state"] == "queued" and payload["builds"]["waiting"] == 1
    # The original's tile is still the one in use.
    tile = next(item for item in payload["items"] if item["id"] == BOTH)
    assert tile["active"] is True

    build_all(plugin)
    payload = payload_of(plugin, state, selected, "0.8;0.8 1;1")
    assert payload["active"][0]["build"]["state"] == "built" and payload["builds"]["waiting"] == 0
    assert payload["built"]["count"] == 1
    # WanGP opens it through the LoRA folder: only the picture side is there.
    _, metadata, raw = read_tensors(loras / V_BOTH)
    assert all("audio" not in name for name in raw) and metadata["lora_browser"] == "visual"

    # Back to normal: the original again, the copy kept for next time.
    choices, mults, payload = act(plugin, {"type": "set_mode", "id": V_BOTH, "mode": "normal"}, state, selected, "0.8;0.8 1;1")
    assert native(choices, None) == [BOTH, PICTURE]
    choices, _, payload = act(plugin, {"type": "set_mode", "id": BOTH, "mode": "visual"}, state, [BOTH, PICTURE], "0.8;0.8 1;1")
    assert payload["active"][0]["build"]["state"] == "built"


def test_a_mode_that_leaves_nothing_switches_the_lora_off(plugin, state):
    choices, mults, payload = act(plugin, {"type": "set_mode", "id": PICTURE, "mode": "audio"}, state, [PICTURE, BOTH], "0.6;0.6 1;1")
    a_picture = dv.split_id(PICTURE, "audio")
    assert native(choices, None) == [BOTH]
    assert native(mults, None) == f"1;1\n#lora_skip 0 0.6;0.6 {a_picture}"
    assert "switches it off" in payload["status"]
    row = next(row for row in payload["active"] if row["id"] == a_picture)
    assert row["enabled"] is False and row["skipped"] and row["empty"]

    choices, mults, payload = act(plugin, {"type": "set_mode", "id": a_picture, "mode": "normal"}, state, [BOTH], native(mults, None))
    assert native(choices, None) == [PICTURE, BOTH] and native(mults, None) == "0.6;0.6 1;1"


def test_switching_an_empty_lora_on_brings_it_back_as_normal(plugin, state):
    a_picture = dv.split_id(PICTURE, "audio")
    text = f"1;1\n#lora_skip 0 0.6;0.6 {a_picture}"
    choices, mults, payload = act(plugin, {"type": "set_enabled", "id": a_picture, "value": True}, state, [BOTH], text)
    assert native(choices, None) == [PICTURE, BOTH] and "back to normal" in payload["status"]


def test_every_lora_visual_only(plugin, state):
    choices, mults, payload = act(plugin, {"type": "set_mode_all", "mode": "visual"}, state, [BOTH, PICTURE, SOUND], "1;1 1;1 0.5;0.5")
    v_picture, v_sound = dv.split_id(PICTURE, "visual"), dv.split_id(SOUND, "visual")
    assert native(choices, None) == [V_BOTH, v_picture]
    assert native(mults, None) == f"1;1 1;1\n#lora_skip 2 0.5;0.5 {v_sound}"
    assert "Switched off 1 that only change the sound" in payload["status"]
    build_all(plugin)
    # The picture-only LoRA kept everything: its "copy" is the original, linked.
    assert os.path.samefile(plugin._store(plugin._instance("t").lora_dir).path(v_picture),
                            os.path.join(plugin._instance("t").lora_dir, PICTURE))

    choices, mults, payload = act(plugin, {"type": "set_mode_all", "mode": "normal"}, state, native(choices, None), native(mults, None))
    assert native(choices, None) == [BOTH, PICTURE, SOUND] and native(mults, None) == "1;1 1;1 0.5;0.5"


def test_a_lora_switched_off_by_hand_stays_off(plugin, state):
    choices, mults, _ = act(plugin, {"type": "set_mode_all", "mode": "audio"}, state, [BOTH], "1;1\n#lora_off 1 0.7 picture.safetensors")
    choices, mults, _ = act(plugin, {"type": "set_mode_all", "mode": "normal"}, state, native(choices, None), native(mults, None))
    assert native(choices, None) == [BOTH] and "#lora_off 1 0.7 picture.safetensors" in native(mults, None)


# ----------------------------------------------------------------- groups


def test_group_moves_as_one(plugin, state):
    choices, mults, payload = act(plugin, {"type": "group_create", "ids": [PICTURE, SOUND], "lead": SOUND},
                                  state, [BOTH, PICTURE, SOUND], "1;1 0.8;0.8 0.4;0.4")
    assert native(choices, None) == [BOTH, SOUND, PICTURE]
    assert native(mults, None) == f"1;1 0.4;0.4 0.4;0.4\n#lora_group A {SOUND}\n#lora_group A {PICTURE}"
    group = next(row for row in payload["active"] if row.get("kind") == "group")
    assert group["id"] == SOUND and group["name"] == "Group A"
    assert [member["id"] for member in group["members"]] == [SOUND, PICTURE]
    assert [row.get("kind", "lora") for row in payload["active"]] == ["lora", "group"]

    # Editing the group edits every member.
    choices, mults, payload = act(plugin, {"type": "set_strength", "id": SOUND, "phase": 0, "value": 0.9},
                                  state, native(choices, None), native(mults, None))
    assert native(mults, None).splitlines()[0] == "1;1 0.9;0.4 0.9;0.4"

    # Off and on as one.
    choices, mults, _ = act(plugin, {"type": "set_enabled", "id": SOUND, "value": False, "group": "A"},
                            state, [BOTH, SOUND, PICTURE], native(mults, None))
    assert native(choices, None) == [BOTH]
    choices, mults, _ = act(plugin, {"type": "set_enabled", "id": SOUND, "value": True, "group": "A"},
                            state, native(choices, None), native(mults, None))
    assert native(choices, None) == [BOTH, SOUND, PICTURE]

    choices, mults, payload = act(plugin, {"type": "ungroup", "group": "A"}, state, native(choices, None), native(mults, None))
    assert "#lora_group" not in native(mults, None) and all(row.get("kind") != "group" for row in payload["active"])


def test_removing_a_group_row_removes_the_group(plugin, state):
    text = f"1;1 0.4;0.4 0.4;0.4\n#lora_group A {SOUND}\n#lora_group A {PICTURE}"
    choices, mults, _ = act(plugin, {"type": "remove", "id": SOUND, "group": "A"}, state, [BOTH, SOUND, PICTURE], text)
    assert native(choices, None) == [BOTH] and native(mults, None) == "1;1"


def test_bake_and_unbake(plugin, state, loras):
    text = f"0.7;0.7 0.7;0.7 1;1\n#lora_group A {BOTH}\n#lora_group A big.safetensors"
    selected = [BOTH, "big.safetensors", PICTURE]
    plan = serve(plugin, {"kind": "bake_plan", "members": [BOTH, "big.safetensors"], "group": "A"}, state)
    assert plan["default_rank"] == 8 and plan["combined_rank"] == 12
    assert [option["rank"] for option in plan["options"]] == [12, 8, 4]
    assert plan["options"][0]["exact"] and plan["options"][1]["bytes"] < plan["options"][0]["bytes"]

    choices, mults, payload = act(plugin, {"type": "bake", "group": "A", "rank": 8}, state, selected, text)
    selected = native(choices, None)
    bake_id = selected[0]
    assert dv.parse(bake_id).rank == 8 and selected[1:] == [PICTURE]
    assert native(mults, None) == f"0.7;0.7 1;1\n#lora_group A {BOTH}\n#lora_group A big.safetensors\n#lora_group A {bake_id}"
    group = payload["active"][0]
    assert group["kind"] == "group" and group["baked"] and group["bake"]["state"] == "queued"
    assert [member["id"] for member in group["members"]] == [BOTH, "big.safetensors"]

    build_all(plugin)
    payload = payload_of(plugin, state, selected, native(mults, None))
    info = payload["active"][0]["bake"]
    assert info["state"] == "built" and info["rank"] == 8 and 0 < info["kept"] <= 1
    assert os.path.isfile(loras / bake_id)

    # A member's mode changes what the bake holds: it points at a new bake.
    choices, mults, payload = act(plugin, {"type": "set_mode", "id": BOTH, "mode": "visual"}, state, selected, native(mults, None))
    assert native(choices, None)[0] != bake_id and payload["active"][0]["bake"]["state"] == "queued"
    assert f"#lora_group A {V_BOTH}" in native(mults, None)

    choices, mults, payload = act(plugin, {"type": "unbake", "group": "A"}, state, native(choices, None), native(mults, None))
    assert native(choices, None) == [V_BOTH, "big.safetensors", PICTURE]
    assert not payload["active"][0]["baked"]


# --------------------------------------------------------- profiles, text


def test_profiles_keep_modes_groups_and_bakes(plugin, state):
    text = f"0.7;0.7 0.7;0.7 0.3;0.3\n#lora_group A {BOTH}\n#lora_group A big.safetensors"
    choices, mults, _ = act(plugin, {"type": "set_mode", "id": PICTURE, "mode": "visual"}, state, [BOTH, "big.safetensors", PICTURE], text)
    choices, mults, _ = act(plugin, {"type": "bake", "group": "A", "rank": 8}, state, native(choices, None), native(mults, None))
    saved_choices, saved_mults = native(choices, None), native(mults, None)
    _, _, payload = act(plugin, {"type": "profile_save", "name": "Concert"}, state, saved_choices, saved_mults)
    assert "1 visual or audio only" in payload["status"] and "1 baked" in payload["status"]
    assert payload["active_profile"] == "Concert"

    choices, mults, payload = act(plugin, {"type": "profile_recall", "name": "Concert"}, state, [SOUND], "1")
    assert native(choices, None) == saved_choices and native(mults, None) == saved_mults


def test_stack_text_carries_tags_and_pastes_back(plugin, state):
    text = f"0.7;0.7 0.7;0.7 1;1\n#lora_group A {BOTH}\n#lora_group A big.safetensors"
    choices, mults, _ = act(plugin, {"type": "bake", "group": "A", "rank": 8}, state, [BOTH, "big.safetensors", PICTURE], text)
    choices, mults, payload = act(plugin, {"type": "set_mode", "id": PICTURE, "mode": "visual"}, state,
                                  native(choices, None), native(mults, None))
    assert payload["stack_text"].splitlines() == [
        "both [group A, baked r8]: x0.7;0.7",
        "big [group A, baked r8]: x0.7;0.7",
        "picture [visual]: x1;1",
    ]
    pasted_choices, pasted_mults, _ = act(plugin, {"type": "stack_paste", "text": payload["stack_text"]}, state, [SOUND], "1;1")
    assert native(pasted_choices, None) == native(choices, None)
    assert native(pasted_mults, None) == native(mults, None)


def test_paste_refuses_a_group_with_two_multipliers(plugin, state):
    text = "both [group A]: x0.5\nbig [group A]: x0.8"
    choices, mults, payload = act(plugin, {"type": "stack_paste", "text": text}, state, [], "")
    assert native(choices, None) == [BOTH]
    assert "share one multiplier" in payload["paste_report"]["problems"][0]["reason"]


# ----------------------------------------------------------- built files


def test_built_files_and_deleting_unused(plugin, state, loras):
    choices, mults, _ = act(plugin, {"type": "set_mode_all", "mode": "audio"}, state, [BOTH, "big.safetensors"], "1;1 1;1")
    build_all(plugin)
    listed = serve(plugin, {"kind": "built", "in_use": native(choices, None)}, state)
    assert sorted(item["id"] for item in listed["files"]) == sorted(native(choices, None))
    assert all(item["used_by"] == ["this stack"] for item in listed["files"]) and listed["unused"] == 0

    # The stack no longer uses one of them: it is the only one deleted.
    listed = serve(plugin, {"kind": "built", "in_use": [A_BOTH], "action": "delete_unused"}, state)
    assert [item["id"] for item in listed["files"]] == [A_BOTH]
    assert not os.path.exists(loras / dv.split_id("big.safetensors", "audio"))


def test_build_status_is_polled(plugin, state):
    act(plugin, {"type": "set_mode", "id": BOTH, "mode": "audio"}, state, [BOTH], "1;1")
    status = serve(plugin, {"kind": "builds"}, state)
    assert status["running"] and status["queued"][0]["key"] == A_BOTH
    build_all(plugin)
    assert serve(plugin, {"kind": "builds"}, state)["running"] is False


def test_view_preference(plugin, state):
    act(plugin, {"type": "view", "value": "cards"}, state, [], "")
    assert payload_of(plugin, state, [], "")["stack_view"] == "cards"


# ---------------------------------------- separate picture and sound strengths

S_BOTH = dv.split_id(BOTH, "sound")


def separate(plugin, state, selected=(BOTH, PICTURE), multipliers="0.8;0.8 1;1"):
    choices, mults, payload = act(plugin, {"type": "set_mode", "id": BOTH, "mode": "separate"},
                                  state, list(selected), multipliers)
    return native(choices, list(selected)), native(mults, multipliers), payload


def test_separate_strengths_are_one_row_over_two_halves(plugin, state, loras):
    selected, text, payload = separate(plugin, state)
    assert selected == [V_BOTH, S_BOTH, PICTURE] and text == "0.8;0.8 0.8;0.8 1;1"
    assert "picture strength and a sound strength" in payload["status"]
    rows = payload["active"]
    assert [row["id"] for row in rows] == [V_BOTH, PICTURE]
    row = rows[0]
    assert row["mode"] == "separate" and row["source_id"] == BOTH and row["name"] == BOTH
    assert row["sound_track"]["id"] == S_BOTH and row["sound_track"]["track"] == "sound"
    assert row["sound_track"]["phase_summary"] == [{"kind": "value", "value": 0.8}] * 2
    assert row["build"]["state"] == "queued" and len(row["build"]["files"]) == 2
    assert payload["builds"]["waiting"] == 2
    assert "both [picture]: x0.8;0.8\nboth [sound]: x0.8;0.8" in payload["stack_text"]

    build_all(plugin)
    payload = payload_of(plugin, state, selected, text)
    assert payload["active"][0]["build"]["state"] == "built"
    # The halves hold every tensor once: the shared layers stay in the picture's.
    _, _, picture = read_tensors(loras / V_BOTH)
    _, _, sound = read_tensors(loras / S_BOTH)
    _, _, original = read_tensors(loras / BOTH)
    assert set(picture) | set(sound) == set(original) and not set(picture) & set(sound)
    assert all("audio" in name or "video_to_audio" in name for name in sound)


def test_each_side_is_edited_on_its_own(plugin, state):
    selected, text, _ = separate(plugin, state)
    choices, mults, payload = act(plugin, {"type": "set_strength", "id": S_BOTH, "phase": 0, "value": 0.3},
                                  state, selected, text)
    assert native(mults, None).split()[:2] == ["0.8;0.8", "0.3;0.8"]
    row = payload["active"][0]
    assert row["phase_summary"][0]["value"] == 0.8 and row["sound_track"]["phase_summary"][0]["value"] == 0.3


def test_two_halves_switch_off_and_go_as_one(plugin, state):
    selected, text, _ = separate(plugin, state)
    choices, mults, payload = act(plugin, {"type": "set_enabled", "id": V_BOTH, "value": False}, state, selected, text)
    assert native(choices, None) == [PICTURE]
    assert f"#lora_off 0 0.8;0.8 {V_BOTH}" in native(mults, None) and f"#lora_off 1 0.8;0.8 {S_BOTH}" in native(mults, None)
    row = payload["active"][0]
    assert row["mode"] == "separate" and row["enabled"] is False
    choices, mults, _ = act(plugin, {"type": "set_enabled", "id": V_BOTH, "value": True}, state, [PICTURE], native(mults, None))
    assert native(choices, None) == [V_BOTH, S_BOTH, PICTURE]

    choices, mults, _ = act(plugin, {"type": "remove", "id": V_BOTH}, state, selected, text)
    assert native(choices, None) == [PICTURE] and native(mults, None) == "1;1"
    # Its tile takes it out the same way.
    choices, _, _ = act(plugin, {"type": "toggle", "id": BOTH, "enabled": False}, state, selected, text)
    assert native(choices, None) == [PICTURE]


def test_leaving_separate_strengths(plugin, state):
    selected, text, _ = separate(plugin, state)
    text = "0.8;0.8 0.3;0.3 1;1"
    choices, mults, _ = act(plugin, {"type": "set_mode", "id": V_BOTH, "mode": "audio"}, state, selected, text)
    assert native(choices, None) == [dv.split_id(BOTH, "audio"), PICTURE] and native(mults, None) == "0.3;0.3 1;1"
    choices, mults, _ = act(plugin, {"type": "set_mode", "id": V_BOTH, "mode": "normal"}, state, selected, text)
    assert native(choices, None) == [BOTH, PICTURE] and native(mults, None) == "0.8;0.8 1;1"


def test_a_lora_with_one_side_has_no_second_strength(plugin, state):
    choices, _, payload = act(plugin, {"type": "set_mode", "id": PICTURE, "mode": "separate"}, state, [PICTURE], "1;1")
    assert choices == {} and "no second strength" in payload["status"]


def test_profiles_and_stack_text_keep_both_strengths(plugin, state):
    selected, text, _ = separate(plugin, state, multipliers="0.8;0.8 1;1")
    text = "0.8;0.8 0.3;0.2 1;1"
    instance = plugin._instance("t")
    _, _, payload = act(plugin, {"type": "profile_save", "name": "split"}, state, selected, text)
    choices, mults, payload = act(plugin, {"type": "profile_recall", "name": "split"}, state, [PICTURE], "1;1")
    assert native(choices, None) == selected and native(mults, None) == text
    assert payload["active"][0]["mode"] == "separate"
    assert instance is plugin._instance("t")

    paste = "both [picture]: x0.8;0.8\nboth [sound]: x0.3;0.2\npicture: x1;1"
    choices, mults, payload = act(plugin, {"type": "stack_paste", "text": paste}, state, [PICTURE], "1;1")
    assert native(choices, None) == selected and native(mults, None) == text
    # A sound line on its own is the LoRA's audio-only copy.
    choices, mults, _ = act(plugin, {"type": "stack_paste", "text": "both [sound]: x0.5;0.5"}, state, [PICTURE], "1;1")
    assert native(choices, None) == [dv.split_id(BOTH, "audio")]


def test_a_group_with_separate_strengths(plugin, state, loras):
    big = "big.safetensors"
    choices, mults, _ = act(plugin, {"type": "group_create", "ids": [BOTH, big], "lead": BOTH},
                            state, [BOTH, big, PICTURE], "0.8;0.8 0.6;0.6 1;1")
    selected, text = native(choices, None), native(mults, None)
    choices, mults, payload = act(plugin, {"type": "group_separate", "group": "A", "value": True}, state, selected, text)
    selected, text = native(choices, None), native(mults, None)
    s_big, v_big = dv.split_id(big, "sound"), dv.split_id(big, "visual")
    assert selected == [V_BOTH, S_BOTH, v_big, s_big, PICTURE]
    group = payload["active"][0]
    assert group["kind"] == "group" and group["separate"] and group["id"] == V_BOTH
    assert group["sound_track"]["id"] == S_BOTH
    assert [member["mode"] for member in group["members"]] == ["normal", "normal"]
    assert [member["entries"] for member in group["members"]] == [[V_BOTH, S_BOTH], [v_big, s_big]]

    # The sound side's lead moves every sound half, and only those.
    choices, mults, _ = act(plugin, {"type": "set_strength", "id": S_BOTH, "phase": 0, "value": 0.25}, state, selected, text)
    text = native(mults, None)
    tokens = dict(zip(selected, text.split("\n")[0].split()))
    assert tokens[S_BOTH] == tokens[s_big] == "0.25;0.8" and tokens[V_BOTH] == tokens[v_big] == "0.8;0.8"

    # Baked, each side becomes a file of its own, with its own strength.
    choices, mults, payload = act(plugin, {"type": "bake", "group": "A", "rank": 4}, state, selected, text)
    baked, baked_text = native(choices, None), native(mults, None)
    assert len([value for value in baked if "_bakes/" in value]) == 2
    assert any(value.endswith(".sound.safetensors") and "_bakes/" in value for value in baked)
    assert baked_text.split("\n")[0].split()[:2] == ["0.8;0.8", "0.25;0.8"]
    build_all(plugin)
    payload = payload_of(plugin, state, baked, baked_text)
    group = payload["active"][0]
    assert group["baked"] and group["bake"]["state"] == "built" and len(group["bake"]["files"]) == 2
    assert group["sound_track"]["id"].endswith(".sound.safetensors")

    # Joined again, the group keeps the picture's strength -- and stays baked.
    choices, mults, payload = act(plugin, {"type": "group_separate", "group": "A", "value": False}, state, baked, baked_text)
    joined = native(choices, None)
    assert len([value for value in joined if "_bakes/" in value]) == 1 and PICTURE in joined
    assert native(mults, None).split("\n")[0].split()[0] == "0.8;0.8"
    assert not payload["active"][0]["separate"]


def test_a_picture_only_member_brings_nothing_to_the_sound_side(plugin, state):
    choices, mults, _ = act(plugin, {"type": "group_create", "ids": [BOTH, PICTURE], "lead": BOTH},
                            state, [BOTH, PICTURE], "1;1 1;1")
    choices, mults, payload = act(plugin, {"type": "group_separate", "group": "A", "value": True},
                                  state, native(choices, None), native(mults, None))
    s_picture = dv.split_id(PICTURE, "sound")
    assert s_picture not in native(choices, None) and f"#lora_skip" in native(mults, None)
    # A group whose LoRAs only change the picture has nothing to separate.
    v_big = dv.split_id("big.safetensors", "visual")
    lines = f"\n#lora_group A {PICTURE}\n#lora_group A {v_big}"
    choices, _, payload = act(plugin, {"type": "group_separate", "group": "A", "value": True},
                              state, [PICTURE, v_big], "1;1 1;1" + lines)
    assert choices == {} and "no second strength" in payload["status"]
