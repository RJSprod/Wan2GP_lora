"""Modes, groups and bakes, as they live in WanGP's own LoRA state."""

from lora_browser import multiplier_codec as codec
from lora_browser import stack_modes as sm
from lora_browser import ui_payloads as up

V = "_lora_browser/a.visual.safetensors"
A = "_lora_browser/a.audio.safetensors"
#: Another LoRA's visual-only copy: a group is two LoRAs or more.
VB = "_lora_browser/b.visual.safetensors"
BAKE = "_lora_browser/_bakes/abcdef123456.r8.safetensors"


def stack(selected, multipliers=""):
    return up.Stack.from_native(selected, multipliers)


def native(s):
    return s.to_native()


# ------------------------------------------------------------------ codec


def test_codec_reads_and_writes_skip_and_group_lines():
    text = "1 0.5\n#lora_off 1 0.7 b.safetensors\n#lora_skip 2 0.3 c.safetensors\n#lora_group A a.safetensors\n#lora_group A d.safetensors"
    entries = codec.parse_disabled(text)
    assert [(e.id, e.skip) for e in entries] == [("b.safetensors", False), ("c.safetensors", True)]
    assert codec.parse_groups(text) == {"A": ["a.safetensors", "d.safetensors"]}
    out = codec.serialize(["1", "0.5"], -1, entries, {"A": ["a.safetensors", "d.safetensors"]})
    assert out == text
    # WanGP reads only the first line.
    assert codec.strip_comments(out) == "1 0.5"


def test_stack_round_trips_groups_and_skips():
    text = "1 1\n#lora_skip 1 0.4 b.safetensors\n#lora_group A a.safetensors\n#lora_group A c.safetensors"
    s = stack(["a.safetensors", "c.safetensors"], text)
    assert s.skipped() == {"b.safetensors"}
    assert s.groups == {"A": ["a.safetensors", "c.safetensors"]}
    values, multipliers = native(s)
    assert values == ["a.safetensors", "c.safetensors"]
    assert multipliers == "1 1\n#lora_skip 1 0.4 b.safetensors\n#lora_group A a.safetensors\n#lora_group A c.safetensors"


def test_a_group_a_preset_took_apart_is_not_written():
    s = stack(["a.safetensors"], "1\n#lora_group A a.safetensors\n#lora_group A gone.safetensors")
    assert sm.groups(s) == {}
    assert native(s)[1] == "1"


# ------------------------------------------------------------------ modes


def test_source_of():
    assert sm.source_of(V, ["a.safetensors"]) == ("a.safetensors", "visual")
    assert sm.source_of("a.safetensors") == ("a.safetensors", "normal")
    assert sm.source_of(BAKE) == (BAKE, "normal")


def test_set_mode_renames_in_place():
    s = stack(["x.safetensors", "a.safetensors", "y.safetensors"], "1 0.5;0.2 1")
    assert sm.set_mode(s, "a.safetensors", "visual", ["a.safetensors"]) == V
    assert native(s) == (["x.safetensors", V, "y.safetensors"], "1 0.5;0.2 1")
    assert sm.set_mode(s, V, "audio", ["a.safetensors"]) == A
    assert sm.set_mode(s, A, "normal", ["a.safetensors"]) == "a.safetensors"
    assert native(s) == (["x.safetensors", "a.safetensors", "y.safetensors"], "1 0.5;0.2 1")
    assert sm.set_mode(s, "a.safetensors", "normal") == ""


def test_set_mode_on_an_off_lora():
    s = stack(["x.safetensors"], "1\n#lora_off 1 0.8 a.safetensors")
    assert sm.set_mode(s, "a.safetensors", "audio", ["a.safetensors"]) == A
    assert native(s)[1] == f"1\n#lora_off 1 0.8 {A}"


def test_set_mode_refuses_a_copy_already_in_the_stack():
    s = stack(["a.safetensors", V], "1 1")
    assert sm.set_mode(s, "a.safetensors", "visual", ["a.safetensors"]) == ""


def test_settle_skips_what_has_nothing_and_brings_it_back():
    s = stack(["p.safetensors", "a.safetensors"], "1 0.7")
    empty = {"_lora_browser/p.audio.safetensors"}
    sm.set_mode(s, "p.safetensors", "audio", ["p.safetensors"])
    moved = sm.settle(s, lambda lora_id: lora_id in empty)
    assert moved == {"off": ["_lora_browser/p.audio.safetensors"], "on": []}
    assert native(s) == (["a.safetensors"], "0.7\n#lora_skip 0 1 _lora_browser/p.audio.safetensors")

    sm.set_mode(s, "_lora_browser/p.audio.safetensors", "normal", ["p.safetensors"])
    moved = sm.settle(s, lambda lora_id: lora_id in empty)
    assert moved == {"off": [], "on": ["p.safetensors"]}
    assert native(s) == (["p.safetensors", "a.safetensors"], "1 0.7")


def test_settle_leaves_what_was_switched_off_by_hand():
    s = stack(["a.safetensors"], "0.7\n#lora_off 0 1 p.safetensors")
    sm.settle(s, lambda lora_id: False)
    assert native(s) == (["a.safetensors"], "0.7\n#lora_off 0 1 p.safetensors")


# ----------------------------------------------------------------- groups


def test_create_group_brings_members_together_with_one_multiplier():
    s = stack(["a.safetensors", "x.safetensors", "b.safetensors"], "1;0.25 0.3 0.8\n#lora_off 3 0.5 c.safetensors")
    letter = sm.create_group(s, ["b.safetensors", "a.safetensors", "c.safetensors"], lead="b.safetensors")
    assert letter == "A"
    values, text = native(s)
    # Together where the first of them was, the lead first, all on, its multiplier.
    assert values == ["b.safetensors", "a.safetensors", "c.safetensors", "x.safetensors"]
    assert text == "0.8 0.8 0.8 0.3\n#lora_group A b.safetensors\n#lora_group A a.safetensors\n#lora_group A c.safetensors"
    assert sm.create_group(s, ["a.safetensors", "x.safetensors"]) == ""      # already grouped
    assert sm.create_group(s, ["x.safetensors"]) == ""                         # one is not a group


def test_group_tokens_follow_the_lead():
    s = stack(["a.safetensors", "b.safetensors"], "1 1\n#lora_group A a.safetensors\n#lora_group A b.safetensors")
    s.tokens[0] = "0.4;0.2"
    assert sm.sync_tokens(s)
    assert native(s)[1].startswith("0.4;0.2 0.4;0.2")


def test_lead_is_the_first_member_that_is_on():
    s = stack(["b.safetensors"], "0.6\n#lora_skip 0 0.6 a.safetensors\n#lora_group A a.safetensors\n#lora_group A b.safetensors")
    group = sm.groups(s)["A"]
    assert sm.lead_of(s, group) == "b.safetensors" and sm.group_on(s, group)


def test_add_and_leave():
    s = stack(["a.safetensors", "b.safetensors", "x.safetensors", "c.safetensors"], "0.5 0.5 1 0.9\n#lora_group A a.safetensors\n#lora_group A b.safetensors")
    assert sm.add_to_group(s, "A", "c.safetensors")
    assert native(s) == (
        ["a.safetensors", "b.safetensors", "c.safetensors", "x.safetensors"],
        "0.5 0.5 0.5 1\n#lora_group A a.safetensors\n#lora_group A b.safetensors\n#lora_group A c.safetensors",
    )
    assert sm.leave_group(s, "A", "a.safetensors")
    assert s.groups["A"] == ["b.safetensors", "c.safetensors"]
    # Two left; one more leaving dissolves the group.
    assert sm.leave_group(s, "A", "b.safetensors")
    assert sm.groups(s) == {} and "A" not in s.groups
    assert native(s)[1] == "0.5 0.5 0.5 1"


def test_group_on_and_off_as_one():
    s = stack(["a.safetensors", "b.safetensors"], "1 1\n#lora_group A a.safetensors\n#lora_group A b.safetensors")
    assert sm.set_group_enabled(s, "A", False)
    assert native(s)[0] == [] and not sm.group_on(s, sm.groups(s)["A"])
    assert sm.set_group_enabled(s, "A", True)
    assert native(s)[0] == ["a.safetensors", "b.safetensors"]


def test_remove_group():
    s = stack(["a.safetensors", "x.safetensors", "b.safetensors"], "1 0.5 1\n#lora_group A a.safetensors\n#lora_group A b.safetensors")
    # Not adjacent in a hand edit: still one group, removed as one.
    assert sm.remove_group(s, "A")
    assert native(s) == (["x.safetensors"], "0.5")


def test_settle_in_a_group():
    s = stack(["a.safetensors", "p.safetensors"], "1 1\n#lora_group A a.safetensors\n#lora_group A p.safetensors")
    sm.set_mode(s, "p.safetensors", "audio", ["p.safetensors"])
    P = "_lora_browser/p.audio.safetensors"
    sm.settle(s, lambda lora_id: lora_id == P)
    values, text = native(s)
    assert values == ["a.safetensors"]
    assert f"#lora_skip 1 1 {P}" in text and sm.groups(s)["A"].members == ["a.safetensors", P]

    # Every member empty: the whole group is skipped, and comes back as one.
    sm.set_mode(s, "a.safetensors", "audio", ["a.safetensors"])
    sm.settle(s, lambda lora_id: lora_id.endswith(".audio.safetensors"))
    assert native(s)[0] == []
    sm.set_mode(s, A, "normal", ["a.safetensors"])
    sm.set_mode(s, P, "normal", ["p.safetensors"])
    sm.settle(s, lambda lora_id: False)
    assert native(s)[0] == ["a.safetensors", "p.safetensors"]


def test_a_group_switched_off_by_hand_stays_off():
    s = stack([], "#lora_off 0 1 a.safetensors\n#lora_skip 1 1 p.safetensors\n#lora_group A a.safetensors\n#lora_group A p.safetensors")
    sm.settle(s, lambda lora_id: False)
    assert native(s)[0] == []


# ------------------------------------------------------------------- bakes


def test_bake_and_unbake():
    s = stack(["x.safetensors", "a.safetensors", VB, "y.safetensors"],
              f"1 0.5 0.5 1\n#lora_group A a.safetensors\n#lora_group A {VB}")
    assert sm.bake(s, "A", BAKE)
    values, text = native(s)
    assert values == ["x.safetensors", BAKE, "y.safetensors"]
    assert text == f"1 0.5 1\n#lora_group A a.safetensors\n#lora_group A {VB}\n#lora_group A {BAKE}"
    group = sm.groups(s)["A"]
    assert group.baked and group.members == ["a.safetensors", VB] and sm.lead_of(s, group) == BAKE

    assert sm.unbake(s, "A")
    assert native(s) == (["x.safetensors", "a.safetensors", VB, "y.safetensors"],
                         f"1 0.5 0.5 1\n#lora_group A a.safetensors\n#lora_group A {VB}")


def test_a_mode_change_in_a_baked_group_changes_only_its_line():
    s = stack([BAKE], f"0.5\n#lora_group A a.safetensors\n#lora_group A b.safetensors\n#lora_group A {BAKE}")
    assert sm.set_mode(s, "a.safetensors", "visual", ["a.safetensors"]) == V
    assert native(s) == ([BAKE], f"0.5\n#lora_group A {V}\n#lora_group A b.safetensors\n#lora_group A {BAKE}")
    other = "_lora_browser/_bakes/0123456789ab.r8.safetensors"
    assert sm.rebake(s, "A", other)
    assert native(s)[0] == [other] and sm.groups(s)["A"].bake == other


def test_leaving_a_baked_group_selects_the_member_again():
    s = stack([BAKE], f"0.5\n#lora_group A a.safetensors\n#lora_group A b.safetensors\n#lora_group A c.safetensors\n#lora_group A {BAKE}")
    assert sm.leave_group(s, "A", "b.safetensors")
    assert native(s)[0] == [BAKE, "b.safetensors"]
    assert sm.groups(s)["A"].members == ["a.safetensors", "c.safetensors"]
    assert sm.add_to_group(s, "A", "b.safetensors")
    assert native(s)[0] == [BAKE] and sm.groups(s)["A"].members == ["a.safetensors", "c.safetensors", "b.safetensors"]


def test_ungrouping_a_baked_group_unbakes_it():
    s = stack([BAKE], f"0.5\n#lora_group A a.safetensors\n#lora_group A b.safetensors\n#lora_group A {BAKE}")
    assert sm.ungroup(s, "A")
    assert native(s) == (["a.safetensors", "b.safetensors"], "0.5 0.5")


def test_a_baked_group_whose_members_bring_nothing_is_skipped():
    s = stack([BAKE], f"0.5\n#lora_group A a.safetensors\n#lora_group A b.safetensors\n#lora_group A {BAKE}")
    sm.settle(s, lambda lora_id: True)
    assert native(s) == ([], f"#lora_skip 0 0.5 {BAKE}\n#lora_group A a.safetensors\n#lora_group A b.safetensors\n#lora_group A {BAKE}")
    sm.settle(s, lambda lora_id: False)
    assert native(s)[0] == [BAKE]


# -------------------------------------------------------------------- rows


def test_fold_rows():
    s = stack(["x.safetensors", "a.safetensors", "b.safetensors"],
              "1 0.5 0.5\n#lora_group A a.safetensors\n#lora_group A b.safetensors")
    rows = [{"id": lora_id, "name": lora_id, "enabled": True, "phase_values": [token]}
            for lora_id, token in (("x.safetensors", 1), ("a.safetensors", 0.5), ("b.safetensors", 0.5))]
    folded = sm.fold_rows(rows, s, describe=lambda lora_id: {"name": lora_id})
    assert [row["id"] for row in folded] == ["x.safetensors", "a.safetensors"]
    group = folded[1]
    assert group["kind"] == "group" and group["group"] == "A" and group["name"] == "Group A"
    assert [member["id"] for member in group["members"]] == ["a.safetensors", "b.safetensors"]
    assert group["phase_values"] == [0.5]


def test_fold_rows_of_a_baked_group_describes_its_members():
    s = stack([BAKE], f"0.5\n#lora_group A a.safetensors\n#lora_group A {VB}\n#lora_group A {BAKE}")
    rows = [{"id": BAKE, "name": "bake", "enabled": True}]
    folded = sm.fold_rows(rows, s, describe=lambda lora_id: {"name": "described " + lora_id})
    assert folded[0]["baked"] and folded[0]["bake_id"] == BAKE
    assert [member["name"] for member in folded[0]["members"]] == ["described a.safetensors", "described " + VB]


def test_phase_summary_counts_steps_not_strengths():
    two = up.PhaseConfig(capacity=2, effective=2, staged=True)
    assert up.phase_summary("0.8;0.4", two) == [{"kind": "value", "value": 0.8}, {"kind": "value", "value": 0.4}]
    assert up.phase_summary("0,0,0,0,0.325,0.325,0,0;0", two) == [
        {"kind": "steps", "steps": 2, "of": 8, "spans": [[5, 6]], "shared": False}, {"kind": "value", "value": 0}]
    assert up.phase_summary("1;0.5,0.25,0", two)[1] == {
        "kind": "steps", "steps": 2, "of": 3, "spans": [[1, 2]], "shared": False}
    assert up.phase_summary("0.5:0.9", two) == [{"kind": "raw", "text": "0.5:0.9"}]
    one = up.PhaseConfig(capacity=1, effective=1)
    assert up.phase_summary("1,1,0.5,0", one) == [
        {"kind": "steps", "steps": 3, "of": 4, "spans": [[1, 3]], "shared": True}]
    assert up.phase_summary("1,0,1,1,0,2", one)[0]["spans"] == [[1, 1], [3, 4], [6, 6]]


# ----------------------------------------- separate picture and sound strengths

KNOWN = ["p.safetensors", "q.safetensors", "r.safetensors"]
PV, PS, PA = (f"_lora_browser/p.{mode}.safetensors" for mode in ("visual", "sound", "audio"))
QV, QS = (f"_lora_browser/q.{mode}.safetensors" for mode in ("visual", "sound"))
RS, RA = (f"_lora_browser/r.{mode}.safetensors" for mode in ("sound", "audio"))
PICTURE_BAKE = "_lora_browser/_bakes/111111111111.r8.safetensors"
SOUND_BAKE = "_lora_browser/_bakes/222222222222.r8.sound.safetensors"


def separate_group():
    """p (Normal, both halves), q (Normal), r (audio only): picture 0.8, sound 0.3."""
    lines = "".join(f"\n#lora_group A {value}" for value in (PV, PS, QV, QS, RS))
    return stack([PV, PS, QV, QS, RS], "0.8 0.3 0.8 0.3 0.3" + lines)


def test_separate_selects_both_halves_at_the_strength_it_had():
    s = stack(["x.safetensors", "p.safetensors", "y.safetensors"], "1 0.8;0.5 1")
    assert sm.set_mode_entries(s, "p.safetensors", "separate", KNOWN) == [PV, PS]
    assert native(s) == (["x.safetensors", PV, PS, "y.safetensors"], "1 0.8;0.5 0.8;0.5 1")
    assert sm.mode_of(s, PS, KNOWN) == "separate" and sm.unit_of(s, PS, KNOWN) == [PV, PS]
    assert sm.partner(s, PV, KNOWN) == PS
    # Each half has a multiplier of its own; audio only keeps the sound's.
    s.set_token(PS, "0.3;0")
    assert sm.set_mode(s, PV, "audio", KNOWN) == PA
    assert native(s) == (["x.safetensors", PA, "y.safetensors"], "1 0.3;0 1")


def test_leaving_separate_for_normal_or_visual_keeps_the_picture_strength():
    s = stack([PV, PS], "0.8 0.3")
    assert sm.set_mode(s, PS, "normal", KNOWN) == "p.safetensors"
    assert native(s) == (["p.safetensors"], "0.8")
    s = stack([PV, PS], "0.8 0.3")
    assert sm.set_mode_entries(s, PV, "separate", KNOWN) == []   # separate already
    assert sm.set_mode(s, PS, "visual", KNOWN) == PV             # its sound half goes
    assert native(s) == ([PV], "0.8")


def test_separate_halves_switched_off_stay_one_lora():
    s = stack([PV], f"0.8\n#lora_off 1 0.3 {PS}")
    assert sm.mode_of(s, PV, KNOWN) == "separate" and sm.unit_of(s, PS, KNOWN) == [PV, PS]


def test_a_sound_half_on_its_own_is_repaired_to_audio_only():
    s = stack([PS, "x.safetensors"], "0.4 1")
    assert sm.repair(s, KNOWN)
    assert native(s) == ([PA, "x.safetensors"], "0.4 1")
    assert not sm.repair(s, KNOWN)


def test_two_halves_of_one_lora_are_no_group():
    s = stack([PV, PS], f"1 1\n#lora_group A {PV}\n#lora_group A {PS}")
    assert sm.groups(s) == {} and native(s)[1] == "1 1"


def test_grouping_with_a_separate_lead_gives_the_group_both_strengths():
    s = stack([PV, PS, "q.safetensors", RA], "0.8 0.3 1 0.5")
    assert sm.create_group(s, [PV, "q.safetensors", RA], lead=PV, known_ids=KNOWN) == "A"
    values, text = native(s)
    assert values == [PV, PS, QV, QS, RS]
    assert text.split("\n")[0] == "0.8 0.3 0.8 0.3 0.3"
    group = sm.groups(s)["A"]
    assert group.separate and group.tracks == ["picture", "sound"]
    assert group.track_members("picture") == [PV, QV] and group.track_members("sound") == [PS, QS, RS]
    assert sm.lead_of(s, group) == PV and sm.lead_of(s, group, "sound") == PS
    # The sound side's lead moves the sound side only.
    s.set_token(PS, "0.1")
    assert sm.sync_tokens(s)
    assert native(s)[1].split("\n")[0] == "0.8 0.1 0.8 0.1 0.1"


def test_grouping_with_a_plain_lead_joins_a_separate_lora_back_into_one():
    s = stack([PV, PS, "q.safetensors"], "0.8 0.3 1")
    assert sm.create_group(s, [PV, "q.safetensors"], lead="q.safetensors", known_ids=KNOWN) == "A"
    assert native(s) == (["q.safetensors", "p.safetensors"],
                         "1 1\n#lora_group A q.safetensors\n#lora_group A p.safetensors")


def test_separating_a_group_and_joining_it_again():
    lines = "".join(f"\n#lora_group A {value}" for value in ("p.safetensors", QV, RA))
    s = stack(["p.safetensors", QV, RA], "0.7 0.7 0.7" + lines)
    assert sm.set_separate(s, "A", True, KNOWN)
    assert native(s)[0] == [PV, PS, QV, RS]
    assert sm.groups(s)["A"].separate
    s.set_token(PS, "0.2")
    sm.sync_tokens(s)
    assert native(s)[1].split("\n")[0] == "0.7 0.2 0.7 0.2"
    assert not sm.set_separate(s, "A", True, KNOWN)
    # Joined, the group keeps the picture's strength.
    assert sm.set_separate(s, "A", False, KNOWN)
    assert native(s)[0] == ["p.safetensors", QV, RA]
    assert native(s)[1].split("\n")[0] == "0.7 0.7 0.7"


def test_a_members_mode_in_a_separate_group():
    s = separate_group()
    assert sm.set_mode_entries(s, QV, "audio", KNOWN) == [QS]
    assert native(s)[0] == [PV, PS, QS, RS]
    assert sm.set_mode_entries(s, QS, "normal", KNOWN) == [QV, QS]
    values, text = native(s)
    assert values == [PV, PS, QV, QS, RS] and text.split("\n")[0] == "0.8 0.3 0.8 0.3 0.3"


def test_baking_a_separate_group_bakes_each_side_into_a_file():
    s = separate_group()
    assert sm.bake(s, "A", {"picture": PICTURE_BAKE, "sound": SOUND_BAKE})
    values, text = native(s)
    assert values == [PICTURE_BAKE, SOUND_BAKE] and text.split("\n")[0] == "0.8 0.3"
    group = sm.groups(s)["A"]
    assert group.bakes == {"picture": PICTURE_BAKE, "sound": SOUND_BAKE}
    assert group.members == [PV, PS, QV, QS, RS] and group.files == [PICTURE_BAKE, SOUND_BAKE]
    assert sm.lead_of(s, group, "sound") == SOUND_BAKE
    assert sm.group_of(s, SOUND_BAKE).letter == "A"
    assert sm.unbake(s, "A")
    values, text = native(s)
    assert sorted(values) == sorted([PV, PS, QV, QS, RS])
    assert dict(zip(values, text.split("\n")[0].split())) == {PV: "0.8", QV: "0.8", PS: "0.3", QS: "0.3", RS: "0.3"}


def test_a_side_with_one_member_is_not_baked():
    lines = "".join(f"\n#lora_group A {value}" for value in (PV, PS, QV))
    s = stack([PV, PS, QV], "0.8 0.3 0.8" + lines)
    assert sm.bake(s, "A", {"picture": PICTURE_BAKE, "sound": SOUND_BAKE})
    assert native(s)[0] == [PICTURE_BAKE, PS]
    assert sm.groups(s)["A"].bakes == {"picture": PICTURE_BAKE}


def test_a_baked_side_whose_members_bring_nothing_is_skipped():
    s = separate_group()
    sm.bake(s, "A", {"picture": PICTURE_BAKE, "sound": SOUND_BAKE})
    sm.settle(s, lambda lora_id: lora_id in (PS, QS, RS))
    assert native(s)[0] == [PICTURE_BAKE] and SOUND_BAKE in s.skipped()


def test_leaving_a_separate_group():
    s = separate_group()
    # r was audio only: out of the group it is its audio-only copy again.
    assert sm.leave_group(s, "A", RS, KNOWN)
    assert native(s)[0] == [PV, PS, QV, QS, RA]
    # Two LoRAs left: the group dissolves, each keeping both strengths.
    assert sm.leave_group(s, "A", QV, KNOWN)
    assert sm.groups(s) == {}
    assert sm.mode_of(s, PV, KNOWN) == "separate" and sm.mode_of(s, QS, KNOWN) == "separate"


def test_joining_a_separate_group_selects_the_halves_it_needs():
    lines = "".join(f"\n#lora_group A {value}" for value in (PV, PS, QV, QS))
    s = stack([PV, PS, QV, QS, "r.safetensors"], "0.8 0.3 0.8 0.3 1" + lines)
    assert sm.add_to_group(s, "A", "r.safetensors", KNOWN)
    values, text = native(s)
    assert values == [PV, PS, QV, QS, _rv := "_lora_browser/r.visual.safetensors", RS]
    assert text.split("\n")[0] == "0.8 0.3 0.8 0.3 0.8 0.3"


def test_a_normal_member_in_a_separate_group_is_repaired_into_halves():
    lines = "".join(f"\n#lora_group A {value}" for value in (PV, PS, "q.safetensors"))
    s = stack([PV, PS, "q.safetensors"], "0.8 0.3 1" + lines)
    assert sm.repair(s, KNOWN)
    assert native(s)[0] == [PV, PS, QV, QS]
    assert native(s)[1].split("\n")[0] == "0.8 0.3 0.8 0.3"


def test_fold_rows_of_separate_strengths():
    s = stack([PV, PS, "x.safetensors"], "0.8 0.3 1")
    rows = [{"id": value, "name": value, "enabled": True} for value in (PV, PS, "x.safetensors")]
    folded = sm.fold_rows(rows, s, describe=lambda lora_id: {"name": lora_id}, known_ids=KNOWN)
    assert [row["id"] for row in folded] == [PV, "x.safetensors"]
    assert folded[0]["mode"] == "separate" and folded[0]["sound_track"]["id"] == PS
    assert folded[0]["sound_track"]["track"] == "sound"

    s = separate_group()
    rows = [{"id": value, "name": value, "enabled": True} for value in (PV, PS, QV, QS, RS)]
    folded = sm.fold_rows(rows, s, describe=lambda lora_id: {"name": lora_id}, known_ids=KNOWN)
    assert len(folded) == 1
    group = folded[0]
    assert group["separate"] and group["id"] == PV and group["sound_track"]["id"] == PS
    assert [member["entries"] for member in group["members"]] == [[PV, PS], [QV, QS], [RS]]
    assert [member["member_mode"] for member in group["members"]] == ["normal", "normal", "audio"]
