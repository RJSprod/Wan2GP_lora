"""Shrink: a stack baked down into as few files as run the same, end to end
through the bridge -- an LTX-2 model with real (tiny) LoRA files, builds run
when the test says so -- and the recipes kept for every built file."""

import os

import numpy as np
import pytest

from lora_browser import bake as bk
from lora_browser import compare as cmp
from lora_browser import derived as dv
from lora_browser import ui_payloads as up
from tests.lora_fixtures import effective_delta
from tests.test_av_bridge import (  # noqa: F401 - fixtures
    BOTH, PICTURE, SOUND, act, build_all, loras, native, payload_of, plugin, serve, state,
)

BIG = "big.safetensors"
SCHEDULE = "1,1,0.5;1,1,0.5"
A_BOTH = dv.split_id(BOTH, "audio")
V_BIG, S_BIG = dv.split_id(BIG, "visual"), dv.split_id(BIG, "sound")


def plan_of(plugin, state, selected, text):
    choices, mults, payload = act(plugin, {"type": "shrink_plan"}, state, selected, text)
    # Asking changes nothing.
    assert "value" not in choices and "value" not in mults
    return payload["shrink_plan"]


def shrink(plugin, state, selected, text):
    choices, mults, payload = act(plugin, {"type": "shrink"}, state, selected, text)
    return native(choices, selected), native(mults, text), payload


def _weighted(*parts):
    total = {}
    for path, weight in parts:
        for key, value in effective_delta(str(path), canonical=bk.canonical).items():
            total[key] = total.get(key, 0) + weight * value
    return total


# -------------------------------------------------------------------- plan


def test_the_plan_bakes_one_strengths_together_and_each_shared_schedule_apart(plugin, state):
    plan = plan_of(plugin, state, [BOTH, PICTURE, SOUND, BIG], f"0.7;0.7 0.5 {SCHEDULE} {SCHEDULE}")
    flat, shaped = plan["bundles"]
    # One strength: baked in at it, the file at 1, the rank of the best LoRA in it.
    assert (flat["kind"], flat["token"], flat["rank"], flat["loras"]) == ("flat", "1", 4, 2)
    assert [(member["id"], member["weight"]) for member in flat["members"]] == [(BOTH, 0.7), (PICTURE, 0.5)]
    # A schedule two share: baked in at 1, the file runs the schedule.
    assert (shaped["kind"], shaped["token"], shaped["rank"]) == ("shaped", SCHEDULE, 8)
    assert [member["id"] for member in shaped["members"]] == [SOUND, BIG]
    assert (plan["files_before"], plan["files_after"]) == (4, 2)
    assert plan["kept"] == []
    assert 0 < flat["bytes_after"] <= flat["bytes_before"]


def test_what_cannot_go_in_stays_and_says_why(plugin, state):
    state = dict(state, loras=state["loras"] + ["ghost.safetensors"])
    selected = [BOTH, PICTURE, SOUND, BIG, "ghost.safetensors"]
    text = f"0.7;0.7 0.5;0.5 {SCHEDULE} 1:0.5 0.3;0.3"
    plan = plan_of(plugin, state, selected, text)
    assert [member["id"] for member in plan["bundles"][0]["members"]] == [BOTH, PICTURE]
    kept = {member["id"]: member["reason"] for member in plan["kept"]}
    assert kept[SOUND] == "No other LoRA runs this schedule."
    assert "syntax" in kept[BIG]
    assert kept["ghost.safetensors"] == "Its file is not on this machine."
    # Shrunk, they are where they were, as they were.
    selected, text, _ = shrink(plugin, state, selected, text)
    assert selected[1:] == [SOUND, BIG, "ghost.safetensors"]
    assert text.splitlines()[0] == f"1;1 {SCHEDULE} 1:0.5 0.3;0.3"


def test_nothing_to_shrink_says_so(plugin, state):
    choices, mults, payload = act(plugin, {"type": "shrink"}, state, [BOTH, SOUND], f"0.7;0.7 {SCHEDULE}")
    assert "value" not in mults and payload["status_warn"]
    assert payload["status"].startswith("Nothing to shrink")


# ---------------------------------------------------------- shrink, build


def test_shrinking_writes_shrunk_groups_that_build_and_unshrink(plugin, state, loras):
    selected, text, payload = shrink(plugin, state, [BOTH, PICTURE, SOUND, BIG],
                                     f"0.7;0.7 0.5;0.5 {SCHEDULE} {SCHEDULE}")
    flat, shaped = selected
    assert dv.parse(flat).rank == 4 and dv.parse(shaped).rank == 8
    lines = text.splitlines()
    assert lines[0] == f"1;1 {SCHEDULE}"
    assert lines[1:] == [
        f"#lora_group A {BOTH}", f"#lora_group A {PICTURE}", f"#lora_group A {flat}",
        f"#lora_group B {SOUND}", f"#lora_group B {BIG}", f"#lora_group B {shaped}",
        f"#lora_weight A 0.7 {BOTH}", f"#lora_weight A 0.5 {PICTURE}",
        f"#lora_weight B 1 {SOUND}", f"#lora_weight B 1 {BIG}",
    ]
    assert payload["status"].startswith("Shrinking 4 LoRAs into 2 files.")
    rows = payload["active"]
    assert [(row["name"], row["shrink"], row["bake"]["state"]) for row in rows] == [
        ("Shrunk A", True, "queued"), ("Shrunk B", True, "queued")]
    assert [member["weights"] for member in rows[0]["members"]] == [[0.7], [0.5]]

    build_all(plugin)
    assert os.path.isfile(loras / flat) and os.path.isfile(loras / shaped)
    rows = payload_of(plugin, state, selected, text)["active"]
    assert rows[0]["bake"]["state"] == "built" and 0 < rows[0]["bake"]["kept"] < 1
    # No module of the scheduled two needs more than rank 8: nothing lost.
    assert rows[1]["bake"]["exact"]

    # Unshrinking: each LoRA again, at the strength it had.
    choices, mults, payload = act(plugin, {"type": "unshrink", "group": "A"}, state, selected, text)
    assert native(choices, None) == [BOTH, PICTURE, shaped]
    assert native(mults, None).splitlines()[0] == f"0.7;0.7 0.5;0.5 {SCHEDULE}"
    assert payload["status"].startswith("Unshrunk")
    # Unbake on a shrunk group is Unshrink too.
    choices, mults, _ = act(plugin, {"type": "unbake", "group": "B"}, state, native(choices, None), native(mults, None))
    assert native(choices, None) == [BOTH, PICTURE, SOUND, BIG]
    assert native(mults, None) == f"0.7;0.7 0.5;0.5 {SCHEDULE} {SCHEDULE}"


def test_a_shrunk_file_is_the_loras_at_their_strengths(plugin, state, loras):
    selected, text, _ = shrink(plugin, state, [BOTH, PICTURE], "0.7;0.7 0.5;0.5")
    # At every rank they have -- Change size on the shrunk row -- nothing is lost.
    choices, mults, _ = act(plugin, {"type": "bake", "group": "A", "rank": 8}, state, selected, text)
    selected, text = native(choices, None), native(mults, None)
    assert dv.parse(selected[0]).rank == 8 and "#lora_weight A 0.7" in text
    build_all(plugin)
    got = effective_delta(str(loras / selected[0]), canonical=bk.canonical)
    expected = _weighted((loras / BOTH, 0.7), (loras / PICTURE, 0.5))
    assert set(got) == set(expected)
    for key, value in expected.items():
        np.testing.assert_allclose(got[key], value, rtol=2e-2, atol=2e-3)


def test_the_shrunk_file_s_strength_scales_every_lora(plugin, state):
    selected, text, _ = shrink(plugin, state, [BOTH, PICTURE], "0.7;0.7 0.5;0.5")
    text = text.replace("1;1", "0.5;0.5", 1)
    _, mults, _ = act(plugin, {"type": "unshrink", "group": "A"}, state, selected, text)
    assert native(mults, None).splitlines()[0] == "0.35;0.35 0.25;0.25"


def test_modes_go_in_as_they_run(plugin, state):
    """An audio-only LoRA brings its sound side; one with separate strengths
    its picture half at the one and its sound half at the other."""
    selected = [A_BOTH, V_BIG, S_BIG, PICTURE]
    text = "0.6;0.6 0.8;0.8 0.4;0.4 0.5;0.5"
    plan = plan_of(plugin, state, selected, text)
    bundle = plan["bundles"][0]
    assert [(member["id"], member["mode"], member["weight"]) for member in bundle["members"]] == [
        (A_BOTH, "audio", 0.6), (V_BIG, "visual", 0.8), (S_BIG, "sound", 0.4), (PICTURE, "normal", 0.5)]
    assert bundle["loras"] == 3
    selected, text, payload = shrink(plugin, state, selected, text)
    assert len(selected) == 1
    # One file, its sides in it at their own strengths: no picture and sound strengths of its own.
    row = payload["active"][0]
    assert not row["separate"] and [member["weights"] for member in row["members"]] == [[0.6], [0.8, 0.4], [0.5]]


def test_a_lora_goes_in_whole_or_not_at_all(plugin, state):
    selected = [V_BIG, S_BIG, BOTH, PICTURE]
    text = f"0.8;0.8 {SCHEDULE} 0.7;0.7 0.5;0.5"
    plan = plan_of(plugin, state, selected, text)
    assert [member["id"] for member in plan["bundles"][0]["members"]] == [BOTH, PICTURE]
    kept = {member["id"]: member["reason"] for member in plan["kept"]}
    assert kept[S_BIG] == "No other LoRA runs this schedule."
    assert kept[V_BIG] == "Its other side stays as it is."
    selected, text, _ = shrink(plugin, state, selected, text)
    assert selected[:2] == [V_BIG, S_BIG]


def test_shrinking_again_starts_from_the_originals(plugin, state):
    selected, text, _ = shrink(plugin, state, [BOTH, PICTURE], "0.7;0.7 0.5;0.5")
    lines = text.splitlines()
    lines[0] += " 0.9;0.9"
    selected, text, payload = shrink(plugin, state, selected + [SOUND], "\n".join(lines))
    assert len(selected) == 1
    row = payload["active"][0]
    assert [(member["id"], member["weights"]) for member in row["members"]] == [
        (BOTH, [0.7]), (PICTURE, [0.5]), (SOUND, [0.9])]


def test_a_group_s_members_go_in_with_its_strength(plugin, state):
    text = f"0.6;0.6 0.6;0.6 0.5;0.5\n#lora_group A {BOTH}\n#lora_group A {BIG}"
    selected, text, payload = shrink(plugin, state, [BOTH, BIG, PICTURE], text)
    assert len(selected) == 1
    assert [member["weights"] for member in payload["active"][0]["members"]] == [[0.6], [0.6], [0.5]]


# ----------------------------------------------- what keeps a shrunk stack


def test_profiles_and_stack_text_keep_what_each_brings(plugin, state):
    selected, text, payload = shrink(plugin, state, [BOTH, PICTURE, SOUND, BIG],
                                     f"0.7;0.7 0.5;0.5 {SCHEDULE} {SCHEDULE}")
    assert payload["stack_text"].splitlines() == [
        "both [group A, baked r4, weight 0.7]: x1;1",
        "picture [group A, baked r4, weight 0.5]: x1;1",
        f"sound [group B, baked r8, weight 1]: x{SCHEDULE}",
        f"big [group B, baked r8, weight 1]: x{SCHEDULE}",
    ]
    choices, mults, _ = act(plugin, {"type": "stack_paste", "text": payload["stack_text"]}, state, [SOUND], "1;1")
    assert native(choices, None) == selected and native(mults, None) == text

    act(plugin, {"type": "profile_save", "name": "Lean"}, state, selected, text)
    choices, mults, _ = act(plugin, {"type": "profile_recall", "name": "Lean"}, state, [SOUND], "1;1")
    assert native(choices, None) == selected and native(mults, None) == text


def test_a_shrink_pasted_without_its_file_is_its_loras(plugin, state):
    text = "both [group A, weight 0.7]: x1;1\npicture [group A, weight 0.5]: x1;1"
    choices, mults, _ = act(plugin, {"type": "stack_paste", "text": text}, state, [SOUND], "1;1")
    assert native(choices, None) == [BOTH, PICTURE]
    assert native(mults, None) == "0.7;0.7 0.5;0.5"


def test_compare_shows_a_shrunk_lora_at_what_it_brings_and_leaves_it_be():
    baked = "_lora_browser/_bakes/abcdef123456.r8.safetensors"
    text = (f"0.5;0.5\n#lora_group A a.safetensors\n#lora_group A b.safetensors\n#lora_group A {baked}\n"
            "#lora_weight A 0.7 a.safetensors\n#lora_weight A 0.4 b.safetensors")
    phases = up.resolve_phases({"guidance_max_phases": 2, "lora_multiplier_phases": 2}, 2)
    found = cmp.entries(up.Stack.from_native([baked], text), phases)
    assert [(entry["key"], entry["phases"], entry["shrunk"], entry["link"]) for entry in found] == [
        ("a.safetensors", [[0.35], [0.35]], True, ""), ("b.safetensors", [[0.2], [0.2]], True, "")]
    edited = cmp.apply_edits([baked], text, phases, {"a.safetensors": {"phases": [[1], [1]], "on": False}}, [])
    assert edited.to_native()[1] == text


# ----------------------------------------------------------------- recipes


def test_a_deleted_file_keeps_its_recipe_and_is_built_again(plugin, state, loras):
    selected, text, _ = shrink(plugin, state, [BOTH, PICTURE], "0.7;0.7 0.5;0.5")
    build_all(plugin)
    baked = selected[0]
    # This stack's own file is not offered for deleting.
    listed = serve(plugin, {"kind": "built", "in_use": [baked], "action": "delete", "ids": [baked]}, state)
    assert os.path.isfile(loras / baked) and not listed["files"][0]["deletable"]
    assert listed["files"][0]["shrink"]

    listed = serve(plugin, {"kind": "built", "in_use": [], "action": "delete", "ids": [baked]}, state)
    assert not os.path.exists(loras / baked) and listed["files"] == []
    recipe = listed["recipes"][0]
    assert (recipe["id"], recipe["shrink"], recipe["members"]) == (baked, True, [BOTH, PICTURE])

    # A video's settings that name it -- with every line saying what it holds
    # gone -- get its group back from the recipe, and the file built again.
    payload = payload_of(plugin, state, [baked], "1;1")
    row = payload["active"][0]
    assert (row["kind"], row["shrink"], row["bake"]["state"]) == ("group", True, "queued")
    assert [member["weights"] for member in row["members"]] == [[0.7], [0.5]]
    build_all(plugin)
    assert os.path.isfile(loras / baked)
    # It is a shrunk group again in every way: the next write carries its
    # lines, and Unshrink gives its LoRAs back at their strengths.
    _, mults, _ = act(plugin, {"type": "set_strength", "id": baked, "phase": 0, "value": 0.9}, state, [baked], "1;1")
    assert f"#lora_group A {baked}" in native(mults, "") and "#lora_weight A 0.7 " + BOTH in native(mults, "")
    _, mults, _ = act(plugin, {"type": "unshrink", "group": "A"}, state, [baked], "1;1")
    assert native(mults, None) == "0.7;0.7 0.5;0.5"


def test_a_copy_s_recipe_is_kept_too(plugin, state, loras):
    choices, mults, _ = act(plugin, {"type": "set_mode", "id": BOTH, "mode": "audio"}, state, [BOTH], "1;1")
    build_all(plugin)
    listed = serve(plugin, {"kind": "built", "in_use": [], "action": "delete_all"}, state)
    assert listed["files"] == [] and [recipe["id"] for recipe in listed["recipes"]] == [A_BOTH]
    assert listed["recipes"][0]["mode"] == "audio"
    payload = payload_of(plugin, state, native(choices, None), native(mults, None))
    assert payload["active"][0]["build"]["state"] == "queued"
    build_all(plugin)
    assert os.path.isfile(loras / A_BOTH)


def test_a_bake_with_only_a_recipe_still_builds(plugin, state, loras):
    """Lines that cannot come back -- a member named elsewhere in the stack --
    leave the recipe to say what to build."""
    selected, text, _ = shrink(plugin, state, [BOTH, PICTURE], "0.7;0.7 0.5;0.5")
    build_all(plugin)
    baked = selected[0]
    serve(plugin, {"kind": "built", "in_use": [], "action": "delete", "ids": [baked]}, state)
    payload_of(plugin, state, [baked, BOTH], "1;1 0.2;0.2")
    build_all(plugin)
    assert os.path.isfile(loras / baked)
    got = effective_delta(str(loras / baked), canonical=bk.canonical)
    assert got  # built from the recipe's members


def test_looking_for_a_recipe_makes_no_folder(plugin, state, loras):
    """A stack naming a baked file nothing was ever built for: looking up its
    recipe makes no store and no link in the LoRA folder."""
    payload = payload_of(plugin, state, ["_lora_browser/_bakes/abcdef123456.r4.safetensors"], "1;1")
    assert payload["active"][0].get("kind") != "group"
    assert not os.path.lexists(loras / dv.LINK_NAME)


# ------------------------------------------- LoRAs WanGP knows by name


@pytest.mark.parametrize("name,signature", [
    ("ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors", "ic-lora"),
    ("styles/LTX-2.3-22B-IC-LoRA-HDR-0.9.safetensors", "ic-lora"),
    ("ltx-2.3-22b-distilled-lora-384-1.1.safetensors", "distilled-lora"),
    ("ltx-2.3-22b-in-outpainting-0.9.safetensors", "in-outpainting"),
    ("ltx-2-id-lora-celebvhq.safetensors", "id-lora-celebvhq"),
    ("Wan2.2-I2V-A14B-lightx2v-high.safetensors", "lightx2v"),
    ("wan22_i2v_seko_low.safetensors", "seko"),
    # Not what WanGP looks for: left to be shrunk like any other.
    ("my_face_id-lora.safetensors", ""),
    ("lightx2v_t2v_style.safetensors", ""),
    ("detail_enhancer.safetensors", ""),
])
def test_what_wangp_recognises_by_name(name, signature):
    assert dv.recognised_by_name(name) == signature


IC = "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"


@pytest.fixture
def ic_state(state, loras):
    from tests.test_av_bridge import _write
    _write(loras / IC, ["attn1.to_q"], 4, 9)
    return dict(state, loras=state["loras"] + [IC])


def test_a_lora_wangp_knows_by_name_stays_out_of_a_shrink(plugin, ic_state):
    plan = plan_of(plugin, ic_state, [BOTH, PICTURE, IC], "0.7;0.7 0.5;0.5 0.9;0.9")
    assert [member["id"] for member in plan["bundles"][0]["members"]] == [BOTH, PICTURE]
    kept = {member["id"]: member["reason"] for member in plan["kept"]}
    assert kept[IC].startswith("WanGP recognises it by its file name ('ic-lora')")
    selected, text, _ = shrink(plugin, ic_state, [BOTH, PICTURE, IC], "0.7;0.7 0.5;0.5 0.9;0.9")
    # Still selected under its own name, at its own strength.
    assert selected[1] == IC and text.splitlines()[0] == "1;1 0.9;0.9"


def test_a_group_holding_one_is_not_baked(plugin, ic_state):
    text = f"0.7;0.7 0.7;0.7\n#lora_group A {BOTH}\n#lora_group A {IC}"
    plan = serve(plugin, {"kind": "bake_plan", "members": [BOTH, IC], "group": "A"}, ic_state)
    assert "recognises" in plan["error"] and "ic-lora" in plan["error"]
    choices, mults, payload = act(plugin, {"type": "bake", "group": "A", "rank": 4}, ic_state, [BOTH, IC], text)
    assert "value" not in mults and payload["status_warn"] and "recognises" in payload["status"]
