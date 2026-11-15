"""Shrink: a stack baked down into as few files as run the same, end to end
through the bridge -- an LTX-2 model with real (tiny) LoRA files, builds run
when the test says so -- and the recipes kept for every built file."""

import os

import numpy as np
import pytest

from lora_browser import bake as bk
from lora_browser import compare as cmp
from lora_browser import derived as dv
from lora_browser import lora_files as lf
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


def shrink(plugin, state, selected, text, **extra):
    choices, mults, payload = act(plugin, dict({"type": "shrink"}, **extra), state, selected, text)
    return native(choices, selected), native(mults, text), payload


def _files(bundle):
    """A bundle's files as phase and multiplier, without their measurement keys."""
    return [{"phase": item["phase"], "token": item["token"]} for item in bundle["files"]]


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
    assert (flat["kind"], flat["token"], flat["rank"], flat["loras"]) == ("flat", "1;1", 4, 2)
    assert not flat["phased"] and [file["phase"] for file in flat["files"]] == [0]
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
    # At the remembered compression level: low, keeping 99.9%.
    assert dv.parse(flat).quality == dv.parse(shaped).quality == 999
    lines = text.splitlines()
    assert lines[0] == f"1;1 {SCHEDULE}"
    assert lines[1:] == [
        f"#lora_group A {BOTH}", f"#lora_group A {PICTURE}", f"#lora_group A {flat}",
        f"#lora_group B {SOUND}", f"#lora_group B {BIG}", f"#lora_group B {shaped}",
        f"#lora_weight A 0.7 {BOTH}", f"#lora_weight A 0.5 {PICTURE}",
        f"#lora_weight B 1 {SOUND}", f"#lora_weight B 1 {BIG}",
    ]
    assert payload["status"].startswith("Shrinking 4 LoRAs into 2 files at low compression")
    rows = payload["active"]
    assert [(row["name"], row["shrink"], row["bake"]["state"]) for row in rows] == [
        ("Shrunk A", True, "queued"), ("Shrunk B", True, "queued")]
    assert [member["weights"] for member in rows[0]["members"]] == [[0.7], [0.5]]

    build_all(plugin)
    assert os.path.isfile(loras / flat) and os.path.isfile(loras / shaped)
    rows = payload_of(plugin, state, selected, text)["active"]
    assert rows[0]["bake"]["state"] == "built" and rows[0]["bake"]["kept"] >= 0.999
    assert rows[1]["bake"]["kept"] >= 0.999

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


def test_a_shrunk_stack_is_not_shrunk_again(plugin, state):
    selected, text, payload = shrink(plugin, state, [BOTH, PICTURE], "0.7;0.7 0.5;0.5")
    assert payload["can_shrink"] is False
    lines = text.splitlines()
    lines[0] += " 0.9;0.9"
    choices, mults, payload = act(plugin, {"type": "shrink"}, state, selected + [SOUND], "\n".join(lines))
    assert "value" not in choices and payload["status_warn"]
    assert payload["status"].startswith("The stack is already shrunk (Shrunk A): unshrink it")
    plan = plan_of(plugin, state, selected + [SOUND], "\n".join(lines))
    assert plan["error"].startswith("The stack is already shrunk")
    # Unshrunk, it shrinks afresh from the originals.
    choices, mults, _ = act(plugin, {"type": "unshrink", "group": "A"}, state, selected + [SOUND], "\n".join(lines))
    selected, text, payload = shrink(plugin, state, native(choices, None), native(mults, None))
    assert len(selected) == 1 and payload["can_shrink"] is False
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
        "both [group A, baked 99.9%, weight 0.7]: x1;1",
        "picture [group A, baked 99.9%, weight 0.5]: x1;1",
        f"sound [group B, baked 99.9%, weight 1]: x{SCHEDULE}",
        f"big [group B, baked 99.9%, weight 1]: x{SCHEDULE}",
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
    _, _, payload = act(plugin, {"type": "bake_plan", "group": "A"}, ic_state, [BOTH, IC], text)
    plan = payload["bake_plan"]
    assert "recognises" in plan["error"] and "ic-lora" in plan["error"]
    choices, mults, payload = act(plugin, {"type": "bake", "group": "A", "rank": 4}, ic_state, [BOTH, IC], text)
    assert "value" not in mults and payload["status_warn"] and "recognises" in payload["status"]


# ------------------------------------------------------- a file per phase

#: Three LoRAs whose phases differ: one file per phase, each holding all three.
PHASED = "0.8;0.5 0.6;0.6 0.7;0.2"


def _phase_files(selected):
    return [dv.parse(value).track for value in selected]


def test_loras_whose_phases_differ_get_a_file_per_phase(plugin, state):
    plan = plan_of(plugin, state, [BOTH, PICTURE, SOUND], PHASED)
    (bundle,) = plan["bundles"]
    assert (bundle["kind"], bundle["phased"], bundle["loras"], bundle["rank"]) == ("flat", True, 3, 4)
    # Phase 1's file runs in phase 1 only, phase 2's in phase 2 only.
    assert _files(bundle) == [{"phase": 1, "token": "1;0"}, {"phase": 2, "token": "0;1"}]
    assert [member["weights"] for member in bundle["members"]] == [[0.8, 0.5], [0.6, 0.6], [0.7, 0.2]]
    assert (plan["files_before"], plan["files_after"]) == (3, 2)

    selected, text, payload = shrink(plugin, state, [BOTH, PICTURE, SOUND], PHASED)
    assert _phase_files(selected) == ["p1", "p2"]
    lines = text.splitlines()
    assert lines[0] == "1;0 0;1"
    assert lines[1:4] == [f"#lora_group A {BOTH}", f"#lora_group A {PICTURE}", f"#lora_group A {SOUND}"]
    assert f"#lora_weight A 0.8;0.5 {BOTH}" in lines and f"#lora_weight A 0.7;0.2 {SOUND}" in lines
    assert payload["status"].startswith("Shrinking 3 LoRAs into 2 files (a file per phase")
    (row,) = payload["active"]
    assert (row["name"], row["shrink"]) == ("Shrunk A", True)
    assert [(track["pinned_phase"], track["multiplier_raw"]) for track in row["phase_tracks"]] == [(1, "1;0"), (2, "0;1")]
    assert row["group_token"] == "1;1"
    assert [member["weights"] for member in row["members"]] == [[[0.8, 0.5]], [[0.6, 0.6]], [[0.7, 0.2]]]


def test_each_phase_file_is_the_loras_at_that_phase_s_strengths(plugin, state, loras):
    selected, text, _ = shrink(plugin, state, [BOTH, PICTURE, SOUND], PHASED)
    # Every rank they have, so nothing is lost: each file exactly its phase.
    choices, mults, _ = act(plugin, {"type": "bake", "group": "A", "rank": 8}, state, selected, text)
    selected, text = native(choices, None), native(mults, None)
    assert _phase_files(selected) == ["p1", "p2"] and text.splitlines()[0] == "1;0 0;1"
    build_all(plugin)
    for value, phase in zip(selected, (0, 1)):
        got = effective_delta(str(loras / value), canonical=bk.canonical)
        weights = [[0.8, 0.5], [0.6, 0.6], [0.7, 0.2]]
        expected = _weighted(*[(loras / name, weights[index][phase])
                               for index, name in enumerate([BOTH, PICTURE, SOUND])])
        assert set(got) == set(expected)
        for key, delta in expected.items():
            np.testing.assert_allclose(got[key], delta, rtol=2e-2, atol=2e-3)


def test_a_phase_file_never_runs_in_another_phase(plugin, state):
    selected, text, _ = shrink(plugin, state, [BOTH, PICTURE, SOUND], PHASED)
    p1, p2 = selected
    # Its own phase moves...
    _, mults, _ = act(plugin, {"type": "set_strength", "id": p1, "phase": 0, "value": 0.9}, state, selected, text)
    assert native(mults, "").splitlines()[0] == "0.9;0 0;1"
    # ...the other stays 0, whatever is set, pasted or typed.
    _, mults, _ = act(plugin, {"type": "set_strength", "id": p1, "phase": 1, "value": 0.7}, state, selected, text)
    assert "value" not in mults or native(mults, "").splitlines()[0] == "1;0 0;1"
    _, mults, _ = act(plugin, {"type": "paste_schedule", "ids": [p2], "token": "0.7;0.3"}, state, selected, text)
    assert native(mults, "").splitlines()[0] == "1;0 0;0.3"
    typed = text.replace("1;0 0;1", "1;1 0.4;1", 1)
    _, mults, _ = act(plugin, {"type": "view", "value": "list"}, state, selected, typed)
    _, mults, _ = act(plugin, {"type": "set_strength", "id": p2, "phase": 1, "value": 0.8}, state, selected, typed)
    assert native(mults, "").splitlines()[0] == "1;0 0;0.8"


def test_unshrink_gives_each_lora_both_its_phases_back(plugin, state):
    selected, text, _ = shrink(plugin, state, [BOTH, PICTURE, SOUND], PHASED)
    choices, mults, _ = act(plugin, {"type": "unshrink", "group": "A"}, state, selected, text)
    assert native(choices, None) == [BOTH, PICTURE, SOUND] and native(mults, None) == PHASED
    # Phase 1's file at 0.5 halves what each brings to phase 1.
    halved = text.replace("1;0 0;1", "0.5;0 0;1", 1)
    _, mults, _ = act(plugin, {"type": "unshrink", "group": "A"}, state, selected, halved)
    assert native(mults, None) == "0.4;0.5 0.3;0.6 0.35;0.2"


def test_a_phase_nothing_brings_has_no_file(plugin, state):
    selected, text, _ = shrink(plugin, state, [BOTH, PICTURE], "0.8;0 0.6;0")
    assert _phase_files(selected) == ["p1"] and text.splitlines()[0] == "1;0"
    _, mults, _ = act(plugin, {"type": "unshrink", "group": "A"}, state, selected, text)
    assert native(mults, None) == "0.8;0 0.6;0"


def test_a_schedule_inside_a_phase(plugin, state):
    """The same steps in phase 1, different strengths in phase 2: a phase 1
    file running those steps, a phase 2 file holding each LoRA's strength."""
    tokens = "1,1,0.5;0.4 1,1,0.5;0.9 1,1,0.5;0.2"
    plan = plan_of(plugin, state, [BOTH, PICTURE, SOUND], tokens)
    (bundle,) = plan["bundles"]
    assert (bundle["kind"], bundle["phased"]) == ("shaped", True)
    assert _files(bundle) == [{"phase": 1, "token": "1,1,0.5;0"}, {"phase": 2, "token": "0;1"}]
    selected, text, _ = shrink(plugin, state, [BOTH, PICTURE, SOUND], tokens)
    assert text.splitlines()[0] == "1,1,0.5;0 0;1"
    _, mults, _ = act(plugin, {"type": "unshrink", "group": "A"}, state, selected, text)
    assert native(mults, None) == tokens


def test_a_file_per_phase_is_made_only_where_it_means_fewer_files(plugin, state):
    """Two LoRAs whose phases differ would be two files either way."""
    plan = plan_of(plugin, state, [BOTH, PICTURE], "0.8;0.5 0.6;0.7")
    assert plan["bundles"] == []
    assert [piece["reason"] for piece in plan["kept"]] == [
        "A file per phase would be 2 files for 2 LoRAs: no fewer than now."] * 2
    # One phase with nothing in it: one file for the two.
    plan = plan_of(plugin, state, [BOTH, PICTURE], "0.8;0 0.6;0")
    (bundle,) = plan["bundles"]
    assert _files(bundle) == [{"phase": 1, "token": "1;0"}]


def test_profiles_and_stack_text_keep_a_file_per_phase(plugin, state):
    selected, text, payload = shrink(plugin, state, [BOTH, PICTURE, SOUND], PHASED)
    assert payload["stack_text"].splitlines() == [
        "both [group A, baked 99.9%, weight 0.8;0.5]: x1;1",
        "picture [group A, baked 99.9%, weight 0.6;0.6]: x1;1",
        "sound [group A, baked 99.9%, weight 0.7;0.2]: x1;1",
    ]
    choices, mults, _ = act(plugin, {"type": "stack_paste", "text": payload["stack_text"]}, state, [SOUND], "1;1")
    assert native(choices, None) == selected and native(mults, None) == text
    act(plugin, {"type": "profile_save", "name": "Phased"}, state, selected, text)
    choices, mults, _ = act(plugin, {"type": "profile_recall", "name": "Phased"}, state, [SOUND], "1;1")
    assert native(choices, None) == selected and native(mults, None) == text


def test_compare_shows_each_lora_once_with_its_phases(plugin, state):
    selected, text, _ = shrink(plugin, state, [BOTH, PICTURE, SOUND], PHASED)
    phases = up.resolve_phases({"guidance_max_phases": 2, "lora_multiplier_phases": 2}, 2)
    found = cmp.entries(up.Stack.from_native(selected, text), phases)
    assert [(entry["key"], entry["phases"], entry["shrunk"]) for entry in found] == [
        (BOTH, [[0.8], [0.5]], True), (PICTURE, [[0.6], [0.6]], True), (SOUND, [[0.7], [0.2]], True)]


def test_deleted_phase_files_come_back_as_one_group(plugin, state, loras):
    selected, text, _ = shrink(plugin, state, [BOTH, PICTURE, SOUND], PHASED)
    build_all(plugin)
    listed = serve(plugin, {"kind": "built", "in_use": [], "action": "delete_all"}, state)
    assert listed["files"] == [] and sorted(recipe["id"] for recipe in listed["recipes"]) == sorted(selected)
    # Settings that name both files and nothing else.
    payload = payload_of(plugin, state, selected, "1;0 0;1")
    (row,) = payload["active"]
    assert row["shrink"] and [track["pinned_phase"] for track in row["phase_tracks"]] == [1, 2]
    assert [member["weights"] for member in row["members"]] == [[[0.8, 0.5]], [[0.6, 0.6]], [[0.7, 0.2]]]
    build_all(plugin)
    assert all(os.path.isfile(loras / value) for value in selected)


def test_one_phase_mode_has_one_file(plugin, state):
    """LTX-2 in One Phase runs phase 1 alone: there is one phase to bake."""
    instance = plugin._instance("t")
    import json
    choices, mults, _ = plugin._apply_action(instance, json.dumps({"type": "shrink"}), state,
                                             [BOTH, PICTURE], "0.8 0.6", 1)
    selected = native(choices, None)
    assert len(selected) == 1 and dv.parse(selected[0]).track == ""
    assert "#lora_weight A 0.8 " + BOTH in native(mults, "")


# ---------------------------------------------------- compression levels


def test_the_plan_measures_each_file_and_shrinks_at_the_level_picked(plugin, state, loras):
    plan = plan_of(plugin, state, [BOTH, PICTURE, SOUND], PHASED)
    assert plan["size"] == "q999"
    (bundle,) = plan["bundles"]
    keys = [item["measure"] for item in bundle["files"]]
    assert len(keys) == 2 and all(keys) and keys[0] != keys[1]
    found = serve(plugin, {"kind": "measure", "sets": {"all": keys, "p1": keys[:1]}}, state)["sets"]
    assert found["all"]["state"] == found["p1"]["state"] == "done"
    high = next(level for level in found["all"]["levels"] if level["key"] == "high")
    assert high["kept"] >= 0.95 and high["bytes"] < found["all"]["exact"]["bytes"] * 2

    selected, text, payload = shrink(plugin, state, [BOTH, PICTURE, SOUND], PHASED, quality=950)
    assert [dv.parse(value).quality for value in selected] == [950, 950]
    assert "high compression" in payload["status"]
    build_all(plugin)
    bake = payload_of(plugin, state, selected, text)["active"][0]["bake"]
    assert bake["state"] == "built" and bake["quality"] == 950
    # Each phase's file keeps at least the level's share of its own change,
    # and together they keep what was predicted.
    files = bake["files"]
    assert [item["id"] for item in files] == selected and all(item["kept"] >= 0.95 for item in files)
    assert bake["kept"] == pytest.approx(min(item["kept"] for item in files))


def test_a_level_survives_unshrinking_and_shrinking_again(plugin, state):
    selected, text, _ = shrink(plugin, state, [BOTH, PICTURE, SOUND], PHASED, quality=990)
    choices, mults, _ = act(plugin, {"type": "unshrink", "group": "A"}, state, selected, text)
    assert native(mults, None) == PHASED
    # Stack text and profiles carry the level.
    _, _, payload = act(plugin, {"type": "shrink", "quality": 990}, state, native(choices, None), native(mults, None))
    assert payload["stack_text"].splitlines()[0] == "both [group A, baked 99%, weight 0.8;0.5]: x1;1"


# ------------------------------------------- a LoRA beside its shrunk file


def _beside(plugin, state):
    """BOTH and PICTURE shrunk into Shrunk A, then BOTH added again on its own."""
    selected, text, payload = shrink(plugin, state, [BOTH, PICTURE], "0.7;0.7 0.5;0.5")
    tile = next(item for item in payload["items"] if item["id"] == BOTH)
    assert tile["active"] and tile["inside"] == "Shrunk A"
    choices, mults, payload = act(plugin, {"type": "toggle", "id": BOTH, "enabled": True}, state, selected, text)
    return native(choices, None), native(mults, None), payload


def test_a_lora_runs_beside_the_shrunk_file_that_holds_it(plugin, state):
    selected, text, payload = _beside(plugin, state)
    bake = selected[0]
    assert selected == [bake, BOTH]
    lines = text.splitlines()
    assert lines[0] == "1;1 1;1"
    assert f"#lora_group A {BOTH}" in lines and f"#lora_weight A 0.7 {BOTH}" in lines
    assert "applies twice now" in payload["status"]
    rows = payload["active"]
    assert [(row["kind"] if "kind" in row else "", row["id"]) for row in rows] == [("group", bake), ("", BOTH)]
    assert [member["id"] for member in rows[0]["members"]] == [BOTH, PICTURE]
    tile = next(item for item in payload["items"] if item["id"] == BOTH)
    assert tile["active"] and tile["inside"] == ""
    # Its own strength, apart from what the file brings.
    _, mults, _ = act(plugin, {"type": "set_strength", "id": BOTH, "phase": 0, "value": 0.3}, state, selected, text)
    assert native(mults, "").splitlines()[0] == "1;1 0.3;1"
    # Shrinking again waits for the shrunk stack to be taken apart.
    assert payload["can_shrink"] is False


def test_unshrinking_makes_the_copy_beside_it_a_double(plugin, state):
    selected, text, _ = _beside(plugin, state)
    text = text.replace("1;1 1;1", "1;1 0.3;0.3", 1)
    choices, mults, payload = act(plugin, {"type": "unshrink", "group": "A"}, state, selected, text)
    assert native(choices, None) == [BOTH, PICTURE]
    # Kept off, right after the copy that stays.
    assert native(mults, None).splitlines() == ["0.7;0.7 0.5;0.5", f"#lora_double 1 0.3;0.3 {BOTH}"]
    rows = payload["active"]
    assert [(row["id"], row["enabled"], row.get("kind", "")) for row in rows] == [
        (BOTH, True, ""), (BOTH, False, "double"), (PICTURE, True, "")]
    assert payload["can_shrink"] is True
    # Removed, it is gone; the LoRA it doubled stays.
    choices, mults, _ = act(plugin, {"type": "remove_double", "id": BOTH}, state, native(choices, None), native(mults, None))
    assert native(mults, None) == "0.7;0.7 0.5;0.5"


def test_a_double_is_free_again_once_its_other_copy_is_gone(plugin, state):
    selected, text, _ = _beside(plugin, state)
    text = text.replace("1;1 1;1", "1;1 0.3;0.3", 1)
    choices, mults, _ = act(plugin, {"type": "unshrink", "group": "A"}, state, selected, text)
    choices, mults, payload = act(plugin, {"type": "remove", "id": BOTH}, state, native(choices, None), native(mults, None))
    assert native(choices, None) == [PICTURE]
    # Off, where it was, with its own strength -- and free to come back on.
    assert native(mults, None).splitlines() == ["0.5;0.5", f"#lora_off 1 0.3;0.3 {BOTH}"]
    row = next(row for row in payload["active"] if row["id"] == BOTH)
    assert row.get("kind", "") != "double" and not row["enabled"]
    choices, mults, _ = act(plugin, {"type": "set_enabled", "id": BOTH, "value": True}, state,
                            native(choices, None), native(mults, None))
    assert native(choices, None) == [PICTURE, BOTH] and native(mults, None) == "0.5;0.5 0.3;0.3"


def test_taking_a_lora_out_of_its_shrunk_file_doubles_its_copy_beside_it(plugin, state):
    selected, text, _ = shrink(plugin, state, [BOTH, PICTURE, SOUND], "0.7;0.7 0.5;0.5 0.6;0.6")
    choices, mults, _ = act(plugin, {"type": "toggle", "id": BOTH, "enabled": True}, state, selected, text)
    choices, mults, payload = act(plugin, {"type": "group_leave", "group": "A", "id": BOTH}, state,
                                  native(choices, None), native(mults, None))
    selected, text = native(choices, None), native(mults, None)
    # Back on its own, at what it brought; the copy beside the file, a double.
    assert selected[-1] == BOTH and text.splitlines()[0] == "1;1 0.7;0.7"
    assert f"#lora_double 2 1;1 {BOTH}" in text.splitlines() and f"#lora_group A {BOTH}" not in text
    assert [row["id"] for row in payload["active"] if row.get("kind") == "double"] == [BOTH]


def test_stack_text_and_profiles_keep_a_lora_inside_and_beside(plugin, state):
    selected, text, payload = _beside(plugin, state)
    assert payload["stack_text"].splitlines() == [
        "both [group A, baked 99.9%, weight 0.7]: x1;1",
        "picture [group A, baked 99.9%, weight 0.5]: x1;1",
        "both: x1;1",
    ]
    choices, mults, _ = act(plugin, {"type": "stack_paste", "text": payload["stack_text"]}, state, [SOUND], "1;1")
    assert native(choices, None) == selected and native(mults, None) == text
    act(plugin, {"type": "profile_save", "name": "Twice"}, state, selected, text)
    choices, mults, _ = act(plugin, {"type": "profile_recall", "name": "Twice"}, state, [SOUND], "1;1")
    assert native(choices, None) == selected and native(mults, None) == text


def test_a_member_s_mode_names_its_group_beside_a_copy_of_it(plugin, state):
    """A group's member and its copy beside the baked file share an id: the
    group's row says which it means."""
    text = f"0.7;0.7 0.7;0.7 1;1\n#lora_group A {BOTH}\n#lora_group A {BIG}"
    choices, mults, _ = act(plugin, {"type": "bake", "group": "A", "rank": 8}, state, [BOTH, BIG, PICTURE], text)
    choices, mults, _ = act(plugin, {"type": "toggle", "id": BOTH, "enabled": True}, state,
                            native(choices, None), native(mults, None))
    selected, text = native(choices, None), native(mults, None)
    assert selected[-1] == BOTH
    # The group's member goes visual only; the copy beside it stays as it is.
    choices, mults, _ = act(plugin, {"type": "set_mode", "id": BOTH, "mode": "visual", "group": "A"}, state, selected, text)
    assert native(choices, None)[-1] == BOTH
    assert f"#lora_group A {dv.split_id(BOTH, 'visual')}" in native(mults, None)
    # Without the group, it is the copy beside the file.
    choices, mults, _ = act(plugin, {"type": "set_mode", "id": BOTH, "mode": "audio"}, state, selected, text)
    assert native(choices, None)[-1] == A_BOTH and f"#lora_group A {BOTH}" in native(mults, None)


# ------------------------------------- a shrunk file used visual or audio only


def test_a_phase_file_is_used_audio_only_as_a_copy_of_itself(plugin, state, loras):
    selected, text, _ = shrink(plugin, state, [BOTH, PICTURE, SOUND], PHASED)
    build_all(plugin)
    p1, p2 = selected
    built = os.path.getmtime(loras / p1)
    choices, mults, payload = act(plugin, {"type": "file_mode", "group": "A", "track": "p1", "mode": "audio"},
                                  state, selected, text)
    selected, text = native(choices, None), native(mults, None)
    audio = dv.bake_mode_id(p1, "audio")
    assert selected == [audio, p2] and audio.endswith(".q999.p1.audio.safetensors")
    assert text.splitlines()[0] == "1;0 0;1" and f"#lora_group A {audio}" in text
    assert "applies audio only now" in payload["status"]
    row = payload["active"][0]
    assert [(track["id"], track["file_mode"]) for track in row["phase_tracks"]] == [(audio, "audio"), (p2, "normal")]
    # Made from the built file: only the copy is built, nothing baked again.
    assert plugin._builder.status()["queued"] == [{"key": audio, "label": "Shrunk A, phase 1 file, audio only"}]
    build_all(plugin)
    assert os.path.getmtime(loras / p1) == built
    kept = {tensor.name for tensor in lf.read_header(str(loras / audio)).tensors}
    whole = {tensor.name for tensor in lf.read_header(str(loras / p1)).tensors}
    assert kept and kept < whole
    assert all(lf.keeps(lf.ltx2_side(lf.split_key(name)[0]), "audio") for name in kept)
    recipe = plugin._stack_files(plugin._instance("t"), _inventory(plugin, state)).store.recipe(audio)
    assert (recipe["kind"], recipe["mode"], recipe["source"]) == ("split", "audio", p1)
    # Pinned to its phase like the file it came from.
    _, mults, _ = act(plugin, {"type": "set_strength", "id": audio, "phase": 1, "value": 0.7}, state, selected, text)
    assert "value" not in mults or native(mults, "").splitlines()[0] == "1;0 0;1"
    # Normal brings the file back as it is.
    choices, mults, _ = act(plugin, {"type": "file_mode", "group": "A", "track": "p1", "mode": "normal"},
                            state, selected, text)
    assert native(choices, None) == [p1, p2]


def _inventory(plugin, state):
    from lora_browser.inventory import build_inventory
    model_type = state["model_type"]
    return build_inventory(plugin._native_loras(state, []), plugin._lora_dir(model_type))


def test_a_deleted_file_copy_is_built_again_from_its_recipes(plugin, state, loras):
    selected, text, _ = shrink(plugin, state, [BOTH, PICTURE, SOUND], PHASED)
    choices, mults, _ = act(plugin, {"type": "file_mode", "group": "A", "track": "p2", "mode": "visual"},
                            state, selected, text)
    selected, text = native(choices, None), native(mults, None)
    payload_of(plugin, state, selected, text)
    build_all(plugin)
    payload_of(plugin, state, selected, text)
    build_all(plugin)
    visual = selected[1]
    assert os.path.isfile(loras / visual)
    serve(plugin, {"kind": "built", "in_use": [], "action": "delete_all"}, state)
    assert not os.path.isfile(loras / visual) and not os.path.isfile(loras / dv.parse(visual).base)
    # A video's settings that name the copy and nothing else: its recipe
    # names the phase file, whose recipe names the LoRAs.
    bare = "1;0 0;1"
    payload = payload_of(plugin, state, selected, bare)
    row = payload["active"][0]
    assert row["shrink"] and [track["file_mode"] for track in row["phase_tracks"]] == ["normal", "visual"]
    for _ in range(2):                 # the phase file, then its copy
        build_all(plugin)
        payload_of(plugin, state, selected, bare)
    build_all(plugin)
    assert os.path.isfile(loras / visual) and os.path.isfile(loras / dv.parse(visual).base)


def test_change_size_and_text_and_profiles_keep_a_file_s_mode(plugin, state):
    selected, text, _ = shrink(plugin, state, [BOTH, PICTURE, SOUND], PHASED)
    choices, mults, payload = act(plugin, {"type": "file_mode", "group": "A", "track": "p1", "mode": "audio"},
                                  state, selected, text)
    selected, text = native(choices, None), native(mults, None)
    assert payload["stack_text"].splitlines()[0] == \
        "both [group A, baked 99.9%, file p1 audio, weight 0.8;0.5]: x1;1"
    choices, mults, _ = act(plugin, {"type": "stack_paste", "text": payload["stack_text"]}, state, [SOUND], "1;1")
    assert native(choices, None) == selected and native(mults, None) == text
    act(plugin, {"type": "profile_save", "name": "Sided"}, state, selected, text)
    choices, mults, _ = act(plugin, {"type": "profile_recall", "name": "Sided"}, state, [SOUND], "1;1")
    assert native(choices, None) == selected and native(mults, None) == text
    # Another size keeps the copy's mode.
    choices, _, _ = act(plugin, {"type": "bake", "group": "A", "quality": 990}, state, selected, text)
    assert native(choices, None)[0].endswith(".q990.p1.audio.safetensors")
    assert native(choices, None)[1].endswith(".q990.p2.safetensors")


def test_only_a_shrunk_group_s_files_take_a_mode(plugin, state):
    text = f"0.7;0.7 0.7;0.7 1;1\n#lora_group A {BOTH}\n#lora_group A {BIG}"
    choices, mults, _ = act(plugin, {"type": "bake", "group": "A", "rank": 8}, state, [BOTH, BIG, PICTURE], text)
    _, mults, payload = act(plugin, {"type": "file_mode", "group": "A", "track": "all", "mode": "audio"},
                            state, native(choices, None), native(mults, None))
    assert "value" not in mults


def test_shrinking_again_frees_a_double_beside_the_new_file(plugin, state):
    selected, text, _ = _beside(plugin, state)
    choices, mults, _ = act(plugin, {"type": "unshrink", "group": "A"}, state, selected, text)
    assert "#lora_double" in native(mults, None)
    selected, text, payload = shrink(plugin, state, native(choices, None), native(mults, None))
    # Its other copy is inside the new file: it is a LoRA beside it, off.
    assert "#lora_double" not in text and f"#lora_off 1 1;1 {BOTH}" in text.splitlines()
    choices, mults, _ = act(plugin, {"type": "set_enabled", "id": BOTH, "value": True}, state, selected, text)
    assert native(choices, None)[-1] == BOTH
