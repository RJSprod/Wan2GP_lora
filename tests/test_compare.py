"""Two recipes side by side: entries, phase values, and edits written back."""

import pytest

from lora_browser import compare as cmp
from lora_browser import ui_payloads as up

ONE_PHASE = up.PhaseConfig(capacity=1, effective=1)
#: MiniMax H3 in One Phase: two multiplier phases, one of them running.
H3_ONE = up.PhaseConfig(capacity=2, effective=1, staged=True)
#: LTX-2 in Two Phases: two stages, eight steps then three.
LTX_TWO = up.PhaseConfig(capacity=2, effective=2, staged=True)
BAKE = "_lora_browser/_bakes/abcdef123456.r8.safetensors"
V = "_lora_browser/c.visual.safetensors"
SND = "_lora_browser/c.sound.safetensors"
AUD = "_lora_browser/d.audio.safetensors"


def entries(selected, multipliers, phases=ONE_PHASE, steps=()):
    return cmp.entries(up.Stack.from_native(selected, multipliers), phases, steps)


class TestReferences:
    def test_a_downloaded_lora_is_the_file_it_became(self):
        assert cmp.local_id("https://hf.co/u/r/resolve/main/detail.safetensors|style") == "style/detail.safetensors"
        assert cmp.local_id("https://hf.co/u/r/resolve/main/detail.safetensors") == "detail.safetensors"
        assert cmp.local_id("sub\\\\x.safetensors") == "sub/x.safetensors"


class TestPhaseLists:
    def test_one_value_runs_in_every_phase_that_runs(self):
        assert cmp.phase_lists("0.8", LTX_TWO) == [[0.8], [0.8]]
        assert cmp.phase_lists("0.8;0.5", LTX_TWO) == [[0.8], [0.5]]
        # A phase this mode does not run is not shown.
        assert cmp.phase_lists("0.8;0.5", H3_ONE) == [[0.8]]

    def test_a_schedule_per_phase_and_one_for_the_run(self):
        assert cmp.phase_lists("1,0.5;0,1,1", LTX_TWO) == [[1, 0.5], [0, 1, 1]]
        assert cmp.phase_lists("1,0.5,0", ONE_PHASE) == [[1, 0.5, 0]]
        # One list for the whole run, two stages running: each its stretch.
        assert cmp.phase_lists("1,0", LTX_TWO, [8, 3]) == [[1, 1, 1, 1, 1, 1, 0, 0], [0, 0, 0]]

    def test_branches_are_not_compared(self):
        assert cmp.phase_lists("0.5:0.9", ONE_PHASE) is None


class TestTokens:
    def test_a_phase_this_mode_does_not_run_keeps_its_value(self):
        assert cmp.token_for("0.85", [[0.6]], H3_ONE) == "0.6;0.85"
        assert cmp.token_for("0.7;0.4", [[1, 0.9]], H3_ONE) == "1,0.9;0.4"

    def test_a_list_for_the_run_stays_one(self):
        assert cmp.token_for("1,1,0.8", [[1, 0.5, 0.8]], H3_ONE) == "1,0.5,0.8"

    def test_tokens_are_as_short_as_they_can_be(self):
        assert cmp.token_for("0.8;0.5", [[0.5], [0.5]], LTX_TWO) == "0.5"
        assert cmp.token_for("1", [[1, 0], [0.3]], LTX_TWO) == "1,0;0.3"
        assert cmp.token_for("1", [[0.25]], ONE_PHASE) == "0.25"

    @pytest.mark.parametrize("bad", [None, [], [[]], [[True]], [["x"]], [[11]], [[float("nan")]]])
    def test_values_wangp_would_not_take_are_refused(self, bad):
        assert cmp.clean_lists(bad) is None

    def test_values_are_rounded_to_what_the_panel_writes(self):
        assert cmp.clean_lists([[0.123456, -2]]) == [[0.1235, -2]]


class TestEntries:
    def test_one_entry_per_lora_and_strength(self):
        found = entries(
            ["a.safetensors", V, SND, AUD],
            "0.7 0.9 0.3 0.5\n#lora_off 4 0.4 e.safetensors",
        )
        assert [(e["key"], e["mode"], e["on"]) for e in found] == [
            ("a.safetensors", "", True),
            ("c.safetensors", "Picture", True),
            ("c.safetensors|sound", "Sound", True),
            ("d.safetensors", "Audio only", True),
            ("e.safetensors", "", False),
        ]
        assert found[4]["phases"] == [[0.4]]

    def test_a_visual_copy_alone_is_visual_only(self):
        found = entries([V], "0.9")
        assert (found[0]["key"], found[0]["mode"], found[0]["add_id"]) == ("c.safetensors", "Visual only", V)

    def test_group_members_move_as_one(self):
        found = entries(
            ["a.safetensors", "b.safetensors"],
            "0.5 0.5\n#lora_group A a.safetensors\n#lora_group A b.safetensors",
        )
        assert [(e["key"], e["group"], e["link"]) for e in found] == [
            ("a.safetensors", "A", "A|all"), ("b.safetensors", "A", "A|all"),
        ]

    def test_a_baked_group_is_its_members(self):
        found = entries(
            [BAKE], f"0.5\n#lora_group A a.safetensors\n#lora_group A b.safetensors\n#lora_group A {BAKE}",
        )
        assert [(e["key"], e["id"], e["baked"], e["phases"]) for e in found] == [
            ("a.safetensors", BAKE, True, [[0.5]]), ("b.safetensors", BAKE, True, [[0.5]]),
        ]
        assert [e["add_id"] for e in found] == ["a.safetensors", "b.safetensors"]

    def test_wangp_s_own_loras_are_listed_and_marked(self):
        found = entries(["accel.safetensors", "a.safetensors"], "1|0.7")
        assert [(e["key"], e["system"]) for e in found] == [("accel.safetensors", True), ("a.safetensors", False)]


class TestFromSettings:
    def test_a_video_s_recipe(self):
        found = cmp.from_settings({
            "activated_loras": ["https://hf.co/x/detail.safetensors|sub", "b.safetensors"],
            "loras_multipliers": "1,1,0.8,0.8 0.7", "num_inference_steps": 4, "guidance_phases": 1,
            "model_type": "wan",
        }, {})
        assert found["model_type"] == "wan" and found["steps"] == [4] and found["phases"] == 1
        assert [(e["key"], e["phases"]) for e in found["entries"]] == [
            ("sub/detail.safetensors", [[1, 1, 0.8, 0.8]]), ("b.safetensors", [[0.7]]),
        ]

    def test_no_settings_is_no_recipe(self):
        assert cmp.from_settings(None, {}) is None
        assert cmp.from_settings({}, {})["entries"] == []


class TestApply:
    def test_edits_on_and_off_and_added(self):
        stack = cmp.apply_edits(
            ["a.safetensors", "b.safetensors"], "0.85 0.5", H3_ONE,
            {"a.safetensors": {"phases": [[0.6]]}, "b.safetensors": {"on": False}},
            [{"id": "c.safetensors", "phases": [[1, 1, 0.5, 0.5]]}],
        )
        assert stack.to_native() == (
            ["a.safetensors", "c.safetensors"], "0.6;0.85 1,1,0.5,0.5;1\n#lora_off 1 0.5 b.safetensors",
        )

    def test_a_switched_off_lora_takes_a_new_value_too(self):
        stack = cmp.apply_edits(
            ["a.safetensors"], "0.7\n#lora_off 1 0.4 b.safetensors", ONE_PHASE,
            {"b.safetensors": {"phases": [[0.9]]}}, [],
        )
        assert stack.to_native()[1] == "0.7\n#lora_off 1 0.9 b.safetensors"

    def test_what_cannot_be_written_is_left_as_it_was(self):
        before = (["accel.safetensors", "a.safetensors", "b.safetensors"], "1|0.7 0.5:0.9")
        stack = cmp.apply_edits(
            *before, ONE_PHASE,
            {"accel.safetensors": {"phases": [[0.1]]}, "b.safetensors": {"phases": [[0.2]]},
             "a.safetensors": {"phases": [[42]]}, "nope.safetensors": {"phases": [[0.3]]}},
            [{"id": "a.safetensors", "phases": [[0.1]]}, {"id": "", "phases": [[1]]}, "junk"],
        )
        assert stack.to_native() == before

    def test_a_baked_group_takes_one_value(self):
        text = f"0.5\n#lora_group A a.safetensors\n#lora_group A b.safetensors\n#lora_group A {BAKE}"
        stack = cmp.apply_edits(
            [BAKE], text, ONE_PHASE,
            {"a.safetensors": {"phases": [[0.8]]}, "b.safetensors": {"phases": [[0.8]]}}, [],
        )
        assert stack.tokens == ["0.8"]
