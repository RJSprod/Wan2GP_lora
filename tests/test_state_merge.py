"""Reconciliation between the panel and WanGP's native LoRA state."""

import pytest

from lora_browser import multiplier_codec as codec
from lora_browser import ui_payloads as up
from lora_browser.inventory import build_inventory

H3_MODEL_DEF = {
    "guidance_max_phases": 2,
    "lora_multiplier_phases": 2,
    "phase_2_spatial_tiling": True,
}
SINGLE_PHASE_MODEL_DEF = {"guidance_max_phases": 1}


@pytest.fixture
def inventory():
    return build_inventory(
        ["a.safetensors", "b.safetensors", "c.safetensors", "sub/d.safetensors"],
        "/loras",
    )


class TestPhaseResolution:
    def test_one_phase_shows_one_slider(self):
        phases = up.resolve_phases(H3_MODEL_DEF, 1)
        assert phases.effective == 1
        # H3 still declares two multiplier phases, so the token may carry two.
        assert phases.capacity == 2

    def test_two_phases(self):
        assert up.resolve_phases(H3_MODEL_DEF, 2).effective == 2

    def test_tiling_is_still_two_phases_not_three(self):
        phases = up.resolve_phases(H3_MODEL_DEF, up.PHASE_2_TILING_GUIDANCE_VALUE)
        assert phases.effective == 2
        assert phases.capacity == 2

    def test_single_phase_model(self):
        phases = up.resolve_phases(SINGLE_PHASE_MODEL_DEF, 1)
        assert (phases.effective, phases.capacity) == (1, 1)

    def test_guidance_is_clamped_to_the_model_maximum(self):
        assert up.resolve_phases(SINGLE_PHASE_MODEL_DEF, 3).effective == 1

    def test_locked_guidance_uses_the_model_maximum(self):
        model_def = {"guidance_max_phases": 2, "lock_guidance_phases": True}
        assert up.resolve_phases(model_def, 1).effective == 2

    @pytest.mark.parametrize("value", [None, "", "none", 0])
    def test_unusable_guidance_values_fall_back_to_one_phase(self, value):
        assert up.resolve_phases({}, value).effective == 1

    def test_labels_are_generic(self):
        assert up.phase_labels(1) == ["Strength"]
        assert up.phase_labels(2) == ["Phase 1", "Phase 2"]


class TestStackBasics:
    def test_new_lora_defaults_to_one_per_phase(self, inventory):
        phases = up.resolve_phases(H3_MODEL_DEF, 2)
        stack = up.Stack.from_native([], "")
        stack.add("a.safetensors", phases.capacity)
        assert stack.tokens == ["1;1"]

    def test_new_lora_in_single_phase_model_defaults_to_one(self, inventory):
        phases = up.resolve_phases(SINGLE_PHASE_MODEL_DEF, 1)
        stack = up.Stack.from_native([], "")
        stack.add("a.safetensors", phases.capacity)
        assert stack.tokens == ["1"]

    def test_a_new_lora_does_not_inherit_the_previous_row_at_that_index(self, inventory):
        phases = up.resolve_phases(H3_MODEL_DEF, 2)
        stack = up.Stack.from_native(["a.safetensors"], "0.25;0.25")
        stack.remove("a.safetensors")
        stack.add("b.safetensors", phases.capacity)
        assert stack.tokens == ["1;1"]

    def test_duplicate_add_is_ignored(self):
        stack = up.Stack.from_native(["a.safetensors"], "1")
        assert stack.add("a.safetensors", 1) is False

    def test_unknown_ids_are_dropped_when_mapping_to_native(self, inventory):
        assert inventory.to_native(["a.safetensors", "not-installed.safetensors"]) == ["a.safetensors"]


class TestRemovalDoesNotShiftMultipliers:
    """Removing the middle LoRA must not hand its multiplier to the next one."""

    def test_removing_b_from_abc(self, inventory):
        stack = up.Stack.from_native(
            ["a.safetensors", "b.safetensors", "c.safetensors"], "0.1 0.2 0.3"
        )
        stack.remove("b.safetensors")
        values, text = stack.to_native(inventory)
        assert values == ["a.safetensors", "c.safetensors"]
        assert text == "0.1 0.3"

    def test_removal_keeps_the_accelerator_boundary_aligned(self, inventory):
        # a is accelerator-managed; b and c are the user's.
        stack = up.Stack.from_native(
            ["a.safetensors", "b.safetensors", "c.safetensors"], "1|0.2 0.3"
        )
        assert stack.is_system(0) and not stack.is_system(1)
        stack.remove("b.safetensors")
        _, text = stack.to_native(inventory)
        assert text == "1|0.3"

    def test_removing_an_accelerator_lora_moves_the_boundary_left(self, inventory):
        stack = up.Stack.from_native(
            ["a.safetensors", "b.safetensors", "c.safetensors"], "1 0.5|0.3"
        )
        stack.remove("a.safetensors")
        _, text = stack.to_native(inventory)
        assert text == "0.5|0.3"

    def test_missing_lora_keeps_its_place_and_token(self, inventory):
        stack = up.Stack.from_native(
            ["a.safetensors", "gone.safetensors", "c.safetensors"], "0.1 0.2 0.3"
        )
        values, text = stack.to_native(inventory)
        assert values == ["a.safetensors", "gone.safetensors", "c.safetensors"]
        assert text == "0.1 0.2 0.3"


class TestAdvancedPreservation:
    def test_editing_one_lora_leaves_another_advanced_token_untouched(self, inventory):
        phases = up.resolve_phases(H3_MODEL_DEF, 2)
        memory = {}
        stack = up.Stack.from_native(
            ["a.safetensors", "b.safetensors"], "1;1 0.5,0.9,1.2"
        )
        up.set_phase_value(stack, "a.safetensors", 1, 0.6, phases, memory)
        up.normalize_stack_tokens(stack, phases, memory)
        assert stack.tokens == ["1;0.6", "0.5,0.9,1.2"]

    def test_normalising_never_rewrites_an_advanced_token(self, inventory):
        phases = up.resolve_phases(H3_MODEL_DEF, 2)
        stack = up.Stack.from_native(["b.safetensors"], "0.5,0.9")
        up.normalize_stack_tokens(stack, phases, {})
        assert stack.tokens == ["0.5,0.9"]

    def test_sliders_refuse_to_edit_an_advanced_token(self, inventory):
        phases = up.resolve_phases(H3_MODEL_DEF, 2)
        stack = up.Stack.from_native(["b.safetensors"], "0.5,0.9")
        assert up.set_phase_value(stack, "b.safetensors", 0, 0.7, phases, {}) is False
        assert stack.tokens == ["0.5,0.9"]

    def test_explicit_conversion_replaces_the_schedule(self, inventory):
        phases = up.resolve_phases(H3_MODEL_DEF, 2)
        stack = up.Stack.from_native(["b.safetensors"], "0.5,0.9")
        assert up.convert_to_simple(stack, "b.safetensors", phases) is True
        assert stack.tokens == ["1;1"]

    def test_phases_are_never_auto_linked(self, inventory):
        """Equal values are a coincidence, not a request to couple the sliders."""
        phases = up.resolve_phases(H3_MODEL_DEF, 2)
        assert up.multiplier_fields("1;1", phases, "a.safetensors")["linked"] is False
        assert up.multiplier_fields("0.5;0.5", phases, "a.safetensors")["linked"] is False

    def test_unlinked_edit_leaves_the_other_phase_alone(self, inventory):
        phases = up.resolve_phases(H3_MODEL_DEF, 2)
        stack = up.Stack.from_native(["a.safetensors"], "1;1")
        up.set_phase_value(stack, "a.safetensors", 0, 0.45, phases, {}, linked=False)
        assert stack.tokens == ["0.45;1"]

    def test_advanced_rows_are_flagged_for_the_editor(self, inventory):
        phases = up.resolve_phases(H3_MODEL_DEF, 2)
        fields = up.multiplier_fields("0.5,0.9", phases, "b.safetensors")
        assert fields["multiplier_kind"] == codec.ADVANCED
        assert fields["phase_values"] == []
        assert fields["multiplier_raw"] == "0.5,0.9"


class TestPhaseModeSwitching:
    def test_phase_two_survives_a_trip_through_one_phase(self, inventory):
        two = up.resolve_phases(H3_MODEL_DEF, 2)
        one = up.resolve_phases(H3_MODEL_DEF, 1)
        memory = {}
        stack = up.Stack.from_native(["a.safetensors"], "0.8;0.4")

        # Switch to One Phase: only one slider is shown.
        fields = up.multiplier_fields(stack.tokens[0], one, "a.safetensors", memory)
        assert fields["phase_values"] == [0.8]
        assert fields["hidden_values"] == [0.4]

        # Edit phase 1 while phase 2 is hidden.
        up.set_phase_value(stack, "a.safetensors", 0, 0.5, one, memory)
        assert stack.tokens == ["0.5;0.4"]

        # Back to Two Phases: the tuned phase 2 is still there.
        fields = up.multiplier_fields(stack.tokens[0], two, "a.safetensors", memory)
        assert fields["phase_values"] == [0.5, 0.4]

    def test_narrow_capacity_remembers_the_hidden_phase_out_of_band(self):
        """A model whose capacity follows the guidance mode must drop the extra
        value from the token, so the plugin holds onto it instead."""
        wide = up.resolve_phases({"guidance_max_phases": 2}, 2)
        narrow = up.resolve_phases({"guidance_max_phases": 2}, 1)
        memory = {}

        stack = up.Stack.from_native(["a.safetensors"], "0.8;0.4")
        up.remember_hidden_phases(stack, wide, memory)
        assert memory["a.safetensors"] == [0.8, 0.4]

        up.normalize_stack_tokens(stack, narrow, memory)
        assert stack.tokens == ["0.8"]

        up.normalize_stack_tokens(stack, wide, memory)
        assert stack.tokens == ["0.8;0.4"]

    def test_growing_without_memory_defaults_the_new_phase(self):
        wide = up.resolve_phases({"guidance_max_phases": 2}, 2)
        stack = up.Stack.from_native(["a.safetensors"], "0.8")
        up.normalize_stack_tokens(stack, wide, {})
        assert stack.tokens == ["0.8;0.8"]


class TestStrengthEditing:
    def test_values_above_one_are_kept(self, inventory):
        phases = up.resolve_phases(H3_MODEL_DEF, 2)
        stack = up.Stack.from_native(["a.safetensors"], "1;1")
        up.set_phase_value(stack, "a.safetensors", 0, 1.75, phases, {})
        assert stack.tokens == ["1.75;1"]

    def test_linked_phases_move_together(self, inventory):
        phases = up.resolve_phases(H3_MODEL_DEF, 2)
        stack = up.Stack.from_native(["a.safetensors"], "0.2;0.9")
        up.set_phase_value(stack, "a.safetensors", 0, 0.6, phases, {}, linked=True)
        assert stack.tokens == ["0.6;0.6"]

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), "abc", None])
    def test_bad_values_fall_back_instead_of_corrupting_the_token(self, bad, inventory):
        phases = up.resolve_phases(H3_MODEL_DEF, 2)
        stack = up.Stack.from_native(["a.safetensors"], "0.4;0.9")
        up.set_phase_value(stack, "a.safetensors", 0, bad, phases, {})
        assert stack.tokens == ["0.4;0.9"]

    def test_out_of_range_reverts_instead_of_clamping(self, inventory):
        """A typo must not quietly become the boundary value."""
        phases = up.resolve_phases(SINGLE_PHASE_MODEL_DEF, 1)
        stack = up.Stack.from_native(["a.safetensors"], "0.7")
        up.set_phase_value(stack, "a.safetensors", 0, 1e12, phases, {})
        assert stack.tokens == ["0.7"]
        up.set_phase_value(stack, "a.safetensors", 0, -50, phases, {})
        assert stack.tokens == ["0.7"]

    def test_negative_multipliers_are_supported(self, inventory):
        phases = up.resolve_phases(H3_MODEL_DEF, 2)
        stack = up.Stack.from_native(["a.safetensors"], "1;1")
        up.set_phase_value(stack, "a.safetensors", 0, -0.75, phases, {})
        assert stack.tokens == ["-0.75;1"]

    def test_the_full_supported_range_is_accepted(self, inventory):
        phases = up.resolve_phases(SINGLE_PHASE_MODEL_DEF, 1)
        stack = up.Stack.from_native(["a.safetensors"], "1")
        for value in (up.VALUE_MIN, up.VALUE_MAX, 3.5, -2.25):
            up.set_phase_value(stack, "a.safetensors", 0, value, phases, {})
            assert float(stack.tokens[0]) == value

    def test_values_are_rounded_to_two_decimals(self, inventory):
        phases = up.resolve_phases(SINGLE_PHASE_MODEL_DEF, 1)
        stack = up.Stack.from_native(["a.safetensors"], "1")
        up.set_phase_value(stack, "a.safetensors", 0, 0.123456, phases, {})
        assert stack.tokens == ["0.12"]


class TestRowsAndItems:
    def test_items_mark_active_and_system_managed(self, inventory):
        phases = up.resolve_phases(H3_MODEL_DEF, 2)
        stack = up.Stack.from_native(["a.safetensors", "b.safetensors"], "1;1|0.5;0.5")
        items = {item["id"]: item for item in up.build_items(inventory, stack, phases)}
        assert items["a.safetensors"]["active"] and items["a.safetensors"]["system_managed"]
        assert items["b.safetensors"]["active"] and not items["b.safetensors"]["system_managed"]
        assert not items["c.safetensors"]["active"]

    def test_active_rows_follow_native_order(self, inventory):
        phases = up.resolve_phases(H3_MODEL_DEF, 2)
        stack = up.Stack.from_native(["c.safetensors", "a.safetensors"], "0.3;0.3 0.1;0.1")
        rows = up.build_active_rows(inventory, stack, phases)
        assert [row["id"] for row in rows] == ["c.safetensors", "a.safetensors"]

    def test_a_missing_lora_is_shown_not_discarded(self, inventory):
        phases = up.resolve_phases(H3_MODEL_DEF, 2)
        stack = up.Stack.from_native(["gone.safetensors"], "0.5;0.5")
        rows = up.build_active_rows(inventory, stack, phases)
        assert rows[0]["missing"] is True
        assert rows[0]["name"] == "gone.safetensors"


class TestSignature:
    def test_equivalent_strings_share_a_signature(self):
        assert up.stack_signature(["a"], "1 1") == up.stack_signature(["a"], "1 1")

    def test_different_multipliers_differ(self):
        assert up.stack_signature(["a"], "1") != up.stack_signature(["a"], "0.5")

    def test_comments_do_not_change_the_signature(self):
        assert up.stack_signature(["a"], "# note\n1") == up.stack_signature(["a"], "1")
