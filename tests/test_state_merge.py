"""Reconciliation between the panel and WanGP's native LoRA state."""

import pytest

from lora_browser import multiplier_codec as codec
from lora_browser import schedule as sch
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
        stack = up.Stack.from_native(["b.safetensors"], "0.5:0.9")
        up.normalize_stack_tokens(stack, phases, {})
        assert stack.tokens == ["0.5:0.9"]

    def test_normalising_never_rewrites_a_scheduled_token(self, inventory):
        """An imported schedule is rendered, not re-serialised."""
        phases = up.resolve_phases(H3_MODEL_DEF, 2)
        stack = up.Stack.from_native(["b.safetensors"], "0.50,0.90")
        up.normalize_stack_tokens(stack, phases, {})
        assert stack.tokens == ["0.50,0.90"]

    def test_sliders_refuse_to_edit_an_advanced_token(self, inventory):
        phases = up.resolve_phases(H3_MODEL_DEF, 2)
        stack = up.Stack.from_native(["b.safetensors"], "0.5:0.9")
        assert up.set_phase_value(stack, "b.safetensors", 0, 0.7, phases, {}) is False
        assert stack.tokens == ["0.5:0.9"]

    def test_explicit_conversion_replaces_the_schedule(self, inventory):
        phases = up.resolve_phases(H3_MODEL_DEF, 2)
        stack = up.Stack.from_native(["b.safetensors"], "0.5:0.9")
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
        fields = up.multiplier_fields("0.5:0.9", phases, "b.safetensors")
        assert fields["multiplier_kind"] == codec.ADVANCED
        assert fields["phase_values"] == []
        assert fields["multiplier_raw"] == "0.5:0.9"
        assert fields["advanced_preserved"] is True
        assert fields["phase_schedules"] == []
        assert fields["schedule_reason"]


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

    def test_direct_entry_keeps_four_decimals(self, inventory):
        """Two decimals would quietly round a deliberate 0.125 to 0.13."""
        phases = up.resolve_phases(SINGLE_PHASE_MODEL_DEF, 1)
        stack = up.Stack.from_native(["a.safetensors"], "1")
        up.set_phase_value(stack, "a.safetensors", 0, 0.123456, phases, {})
        assert stack.tokens == ["0.1235"]
        up.set_phase_value(stack, "a.safetensors", 0, 0.125, phases, {})
        assert stack.tokens == ["0.125"]


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


class TestScheduleStateModel:
    """Schedule state, and the rules that keep it honest.

    One Phase guidance throughout: it is the only mode where a comma list runs
    one value per step, so the only mode the editor schedules in.
    """

    def setup_method(self):
        self.phases = up.resolve_phases(H3_MODEL_DEF, 1)
        self.context = up.ScheduleContext(steps=30)
        self.schedules = {}
        self.memory = {}

    def _stack(self, multipliers, ids=("a.safetensors",)):
        stack = up.Stack.from_native(list(ids), multipliers)
        up.sync_schedules(stack, self.phases, self.schedules, self.context)
        return stack

    def test_deleting_a_region_leaves_zero_and_spares_the_hidden_phase(self):
        stack = self._stack("1,0.8,0.4;0.64")
        first = self.schedules[("a.safetensors", 0)]
        assert [(r.start, r.end) for r in first.regions] == [(1, 1), (2, 2), (3, 3)]

        up.schedule_delete_region(
            stack, "a.safetensors", 0, first.regions[0].id,
            self.phases, self.memory, self.schedules, self.context,
        )
        assert stack.tokens[0] == "0,0.8,0.4;0.64"

    def test_scheduling_and_linking_never_coexist(self):
        """Linking needs two visible phases; scheduling needs exactly one."""
        two = up.resolve_phases(H3_MODEL_DEF, 2)
        assert up.scheduling_allowed(self.phases) is True
        assert up.scheduling_allowed(two) is False

    def test_reopening_a_schedule_does_not_reset_it(self):
        stack = self._stack("0.6;0.9")
        up.schedule_enable(stack, "a.safetensors", 0, self.phases, self.memory, self.schedules, self.context)
        up.schedule_add_region(stack, "a.safetensors", 0, self.phases, self.memory, self.schedules, self.context)
        before = [(r.id, r.start, r.end) for r in self.schedules[("a.safetensors", 0)].regions]

        # Collapsing is a view-only act; the panel re-opens onto the same data.
        up.sync_schedules(stack, self.phases, self.schedules, self.context)
        up.schedule_enable(stack, "a.safetensors", 0, self.phases, self.memory, self.schedules, self.context)
        assert [(r.id, r.start, r.end) for r in self.schedules[("a.safetensors", 0)].regions] == before

    def test_a_scheduled_phase_has_no_plain_strength_to_edit(self):
        """The timeline owns every slot, so a stray strength edit does nothing."""
        stack = self._stack("1,0.8,0.4;0.5")
        with pytest.raises(sch.ScheduleError):
            up.schedule_set_base(
                stack, "a.safetensors", 0, 0.9, self.phases, self.memory,
                self.schedules, self.context,
            )
        assert up.set_phase_value(
            stack, "a.safetensors", 0, 0.9, self.phases, self.memory, schedules=self.schedules
        ) is False
        assert stack.tokens[0] == "1,0.8,0.4;0.5"

    def test_leaving_one_phase_resets_a_scheduled_lora_to_zero(self):
        """No single number carries a schedule over, so it switches off."""
        stack = self._stack("1,0.8,0.4;0.64 0.75;0.75", ids=("a.safetensors", "b.safetensors"))
        two = up.resolve_phases(H3_MODEL_DEF, 2)
        assert up.schedules_need_resync(stack, two, self.schedules, self.context) is True

        up.refit_schedules(stack, two, self.schedules, self.memory, self.context)
        assert stack.tokens[0] == "0;0"          # was scheduled
        assert stack.tokens[1] == "0.75;0.75"    # a plain strength means the same
        assert self.schedules == {}

    def _load(self, multipliers, steps, ids=("a.safetensors",)):
        """A token as it comes back after a reload: nothing but the token.

        This is the state the bug lived in. The editor keeps its region
        objects in memory only, so after a WanGP restart or a page reload the
        schedule is rebuilt from the token and has never been edited *in this
        process* -- which is a fact about the process, not about the schedule.
        """
        stack = up.Stack.from_native(list(ids), multipliers)
        up.sync_schedules(stack, self.phases, self.schedules, up.ScheduleContext(steps=steps))
        return stack

    def _steps(self, stack, steps):
        """One payload round at a new step count, in plugin.py's own order:
        the payload build syncs, reports what is out of step, and the browser
        answers with a resync."""
        context = up.ScheduleContext(steps=steps)
        up.sync_schedules(stack, self.phases, self.schedules, context)
        if up.schedules_need_resync(stack, self.phases, self.schedules, context):
            up.refit_schedules(stack, self.phases, self.schedules, self.memory, context)
            up.sync_schedules(stack, self.phases, self.schedules, context)
        # What the timeline shows: the run's end cuts it, whatever is kept past.
        return [
            (r.start, r.end, r.strength)
            for r in sch.visible_regions(self.schedules[("a.safetensors", 0)])
        ]

    def test_more_steps_adds_undefined_slots_and_moves_nothing(self):
        """The reported bug: 4 -> 5 stretched the regions, 5 -> 6 did not.

        Both went through the same branch and disagreed because the *first*
        change ran against a schedule rebuilt from the token, which is not
        marked dirty, and resampling set the flag the second change then read.
        A schedule drawn one slot per step grows at its end whichever change
        it is.
        """
        stack = self._load("0.5,0.5,0.3,0.3;0.64", 4)
        drawn = [(1, 2, 0.5), (3, 4, 0.3)]
        assert self._steps(stack, 4) == drawn
        assert self.schedules[("a.safetensors", 0)].dirty is False

        assert self._steps(stack, 5) == drawn
        assert stack.tokens[0] == "0.5,0.5,0.3,0.3,0;0.64"
        assert self._steps(stack, 6) == drawn
        assert stack.tokens[0] == "0.5,0.5,0.3,0.3,0,0;0.64"

    def test_fewer_steps_cuts_into_the_last_region(self):
        """Down is the same rule read the other way: the end of the timeline
        moves, and a region straddling it is clipped rather than slid."""
        stack = self._load("0.5,0.5,0.3,0.3;0.64", 4)
        assert self._steps(stack, 3) == [(1, 2, 0.5), (3, 3, 0.3)]
        assert stack.tokens[0] == "0.5,0.5,0.3;0.64"
        assert self._steps(stack, 2) == [(1, 2, 0.5)]
        assert stack.tokens[0] == "0.5,0.5;0.64"

        # The cut was a window, not a deletion: back up, and it is all there.
        assert self._steps(stack, 4) == [(1, 2, 0.5), (3, 4, 0.3)]
        assert stack.tokens[0] == "0.5,0.5,0.3,0.3;0.64"

    # -- events as Gradio actually delivers them --------------------------
    #
    # Every event carries the multiplier text as it stood when it was *sent*.
    # The step slider fires for every value it passes, so a step change is a
    # burst of events, and the token one of them carries is routinely a step or
    # a write behind the one WanGP holds by the time it is handled.

    KEY = ("a.safetensors", 0)

    def _event(self, multipliers, steps):
        """A payload build for an event carrying ``multipliers``, current or not.

        Returns whether the payload asks the browser for a resync.
        """
        stack = up.Stack.from_native(["a.safetensors"], multipliers)
        context = up.ScheduleContext(steps=steps)
        up.sync_schedules(stack, self.phases, self.schedules, context)
        return up.schedules_need_resync(stack, self.phases, self.schedules, context)

    def _resync(self, multipliers, steps):
        """The browser's answer: one resync action, built from ``multipliers``
        the way plugin.py builds it. Returns the multiplier text it writes."""
        stack = up.Stack.from_native(["a.safetensors"], multipliers)
        context = up.ScheduleContext(steps=steps)
        up.sync_schedules(stack, self.phases, self.schedules, context)
        up.refit_schedules(stack, self.phases, self.schedules, self.memory, context)
        return codec.serialize(stack.tokens, stack.separator_index)

    def _edit(self, multipliers, steps, apply):
        """A panel edit made at ``steps``; ``apply`` gets the stack, the visible
        regions and the context. Returns the multiplier text it writes."""
        stack = up.Stack.from_native(["a.safetensors"], multipliers)
        context = up.ScheduleContext(steps=steps)
        up.sync_schedules(stack, self.phases, self.schedules, context)
        apply(stack, sch.visible_regions(self.schedules[self.KEY]), context)
        return codec.serialize(stack.tokens, stack.separator_index)

    def test_a_drag_through_several_counts_never_stretches(self):
        """The reported bug, as it actually arrives.

        The drag reaches 6 before the resync for 5 has landed, so the event for
        6 carries the 4-step token. That token used to be rebuilt from, judged
        an import because four values disagree with six steps, and resampled
        by the next resync: 0.5,0.5,0.5,0.3,0.3,0 -- and back at 4 the stretched
        copy was cut, 0.5,0.5,0.5,0.3, never the schedule that was drawn.
        """
        stack = self._load("0.5,0.5,0.3,0.3;0.64", 4)
        drawn = stack.tokens[0]
        assert self._event(drawn, 5)
        five = self._resync(drawn, 5)
        assert five == "0.5,0.5,0.3,0.3,0;0.64"

        self._event(drawn, 6)              # sent before `five` reached WanGP
        assert self._event(five, 6)
        six = self._resync(five, 6)
        assert six == "0.5,0.5,0.3,0.3,0,0;0.64"

        # And dragging back down lands exactly where it started.
        self._event(six, 5)
        assert self._event(six, 4)
        assert self._resync(six, 4) == drawn

    def test_no_order_of_late_events_changes_where_a_drag_lands(self):
        """Whatever order a burst of step changes lands in, and however late
        the token each event carries, the run ends on the schedule as drawn,
        cut or extended at the final count -- never resampled."""
        import random

        rng = random.Random(20261001)
        drawn = [0.5, 0.5, 0, 0.3, 0.8, 0.8]
        start = codec.build_schedule([drawn, [0.64]])
        for _ in range(300):
            self.schedules.clear()
            self._load(start, 6)
            held = [start]          # every token WanGP has held, oldest first
            for _ in range(rng.randint(1, 14)):
                steps = rng.randint(0, 14)
                if self._event(rng.choice(held[-4:]), steps):
                    # The resync is built from whatever was current when it
                    # was sent, which may itself be a write behind.
                    held.append(self._resync(rng.choice(held[-3:]), steps))

            final = rng.randint(2, 14)
            self._event(held[-1], final)
            landed = self._resync(held[-1], final)
            assert landed == codec.build_schedule([(drawn + [0] * final)[:final], [0.64]])

    def test_typing_a_two_digit_count_loses_nothing(self):
        """Typing 12 over 4 passes through 1, and 1 used to cut it to one step."""
        stack = self._load("0.5,0.5,0.3,0.3;0.64", 4)
        drawn = stack.tokens[0]
        assert self._event(drawn, 1) is False
        # Even asked outright, a one-step count is not followed.
        assert self._resync(drawn, 1) == drawn
        assert self._event(drawn, 12)
        assert self._resync(drawn, 12) == "0.5,0.5,0.3,0.3,0,0,0,0,0,0,0,0;0.64"

    def test_lowering_then_raising_the_count_restores_what_was_cut(self):
        stack = self._load("0.5,0.5,0.3,0.3,0.8;0.64", 5)
        drawn = stack.tokens[0]
        four = self._resync(drawn, 4)
        assert four == "0.5,0.5,0.3,0.3;0.64"
        assert self._resync(four, 3) == "0.5,0.5,0.3;0.64"
        assert self._resync("0.5,0.5,0.3;0.64", 5) == drawn

    def test_an_edit_at_the_lower_count_keeps_what_is_past_the_end(self):
        """What comes back is what was not changed -- next to what was."""
        stack = self._load("0.5,0.5,0,0.3,0.3,0.3;0.64", 6)
        four = self._resync(stack.tokens[0], 4)
        assert four == "0.5,0.5,0,0.3;0.64"

        def strengthen_first(stack, shown, context):
            up.schedule_set_region_strength(
                stack, "a.safetensors", 0, shown[0].id, 0.9,
                self.phases, self.memory, self.schedules, context,
            )

        edited = self._edit(four, 4, strengthen_first)
        assert edited == "0.9,0.9,0,0.3;0.64"
        assert self._resync(edited, 6) == "0.9,0.9,0,0.3,0.3,0.3;0.64"

    def test_a_region_crossing_the_end_is_still_one_region(self):
        """A new strength reaches the part past the end; a deletion takes it."""
        stack = self._load("0.9,0,0.5,0.5,0.5;0.64", 5)
        four = self._resync(stack.tokens[0], 4)
        assert four == "0.9,0,0.5,0.5;0.64"

        def soften_last(stack, shown, context):
            up.schedule_set_region_strength(
                stack, "a.safetensors", 0, shown[-1].id, 0.7,
                self.phases, self.memory, self.schedules, context,
            )

        softer = self._edit(four, 4, soften_last)
        assert self._resync(softer, 5) == "0.9,0,0.7,0.7,0.7;0.64"

        def delete_last(stack, shown, context):
            up.schedule_delete_region(
                stack, "a.safetensors", 0, shown[-1].id,
                self.phases, self.memory, self.schedules, context,
            )

        four = self._resync("0.9,0,0.7,0.7,0.7;0.64", 4)
        gone = self._edit(four, 4, delete_last)
        assert gone == "0.9,0,0,0;0.64"
        assert self._resync(gone, 5) == "0.9,0,0,0,0;0.64"

    def _commit_last(self, start, end, keep_width):
        def commit(stack, shown, context):
            up.schedule_commit_region(
                stack, "a.safetensors", 0, shown[-1].id, start, end,
                self.phases, self.memory, self.schedules, context, keep_width=keep_width,
            )
        return commit

    def test_moving_a_region_off_the_end_lets_the_rest_go(self):
        """Moved, it is the region the user placed -- what the run did not
        reach did not come with it."""
        stack = self._load("0.9,0,0,0.5,0.5,0.5;0.64", 6)
        four = self._resync(stack.tokens[0], 4)
        moved = self._edit(four, 4, self._commit_last(3, 3, keep_width=True))
        assert moved == "0.9,0,0.5,0;0.64"
        assert self._resync(moved, 6) == "0.9,0,0.5,0,0,0;0.64"

    def test_a_region_left_reaching_the_end_keeps_the_part_past_it(self):
        """Its left edge moved or it was put back: the right was never touched."""
        stack = self._load("0.9,0,0,0.5,0.5,0.5;0.64", 6)
        four = self._resync(stack.tokens[0], 4)

        widened = self._edit(four, 4, self._commit_last(3, 4, keep_width=False))
        assert widened == "0.9,0,0.5,0.5;0.64"
        assert self._resync(widened, 6) == "0.9,0,0.5,0.5,0.5,0.5;0.64"

        four = self._resync("0.9,0,0.5,0.5,0.5,0.5;0.64", 4)
        unmoved = self._edit(four, 4, self._commit_last(3, 4, keep_width=True))
        assert unmoved == four
        assert self._resync(unmoved, 6) == "0.9,0,0.5,0.5,0.5,0.5;0.64"

    def test_a_late_token_from_before_an_edit_keeps_what_is_past_the_end(self):
        """The edit's own write and an older event land in either order.

        The older token is not what the schedule says any more, so without a
        memory of what the panel wrote it was rebuilt from -- and a token only
        carries what the run reaches.
        """
        stack = self._load("0.5,0.5,0,0.3,0.3,0.3;0.64", 6)
        four = self._resync(stack.tokens[0], 4)

        def strengthen_first(stack, shown, context):
            up.schedule_set_region_strength(
                stack, "a.safetensors", 0, shown[0].id, 0.9,
                self.phases, self.memory, self.schedules, context,
            )

        edited = self._edit(four, 4, strengthen_first)
        self._event(four, 4)               # sent before the edit landed
        self._event(edited, 4)
        assert self._resync(edited, 6) == "0.9,0.9,0,0.3,0.3,0.3;0.64"

    def test_a_plain_strength_is_not_taken_for_a_schedule_cut_short(self):
        """A preset setting 0.5 is a plain 0.5, even where step 1 says 0.5."""
        stack = self._load("0.5,0.5,0.3,0.3;0.64", 4)
        self._resync(self._resync(stack.tokens[0], 5), 4)    # some history
        assert self._event("0.5;0.64", 4) is False
        assert self.schedules == {}

    def test_a_list_that_comes_to_fit_the_run_is_step_aligned_from_then_on(self):
        """A token that lands before its step count does is not an import."""
        stack = self._load("0.5,0.5,0.3,0.3;0.64", 30)
        assert self.schedules[self.KEY].step_aligned is False
        assert self._event(stack.tokens[0], 4) is False
        assert self.schedules[self.KEY].step_aligned is True
        assert self._resync(stack.tokens[0], 5) == "0.5,0.5,0.3,0.3,0;0.64"

    def test_a_list_authored_at_another_resolution_is_resampled_once(self):
        """The case the other branch is for, and it still works.

        Eight values against a four-step run were never one slot per step, so
        truncating them would change what WanGP renders. They are resampled
        into alignment instead -- once. From then on the timeline is
        step-aligned like any other and grows at its end.
        """
        stack = self._load("1,1,0.8,0.8,0.5,0.5,0.2,0.2;0.64", 4)
        assert self.schedules[("a.safetensors", 0)].step_aligned is False
        assert self._steps(stack, 4) == [(1, 1, 1.0), (2, 2, 0.8), (3, 3, 0.5), (4, 4, 0.2)]
        assert stack.tokens[0] == "1,0.8,0.5,0.2;0.64"

        assert self._steps(stack, 5) == [(1, 1, 1.0), (2, 2, 0.8), (3, 3, 0.5), (4, 4, 0.2)]
        assert stack.tokens[0] == "1,0.8,0.5,0.2,0;0.64"

    def test_a_token_that_already_fits_the_run_is_left_alone(self):
        """One value per step is step-aligned on arrival, whoever wrote it."""
        stack = self._load("0.5,0.5,0.3,0.3;0.64", 4)
        assert self.schedules[("a.safetensors", 0)].step_aligned is True
        assert up.schedules_need_resync(
            stack, self.phases, self.schedules, up.ScheduleContext(steps=4)
        ) is False
        assert stack.tokens[0] == "0.5,0.5,0.3,0.3;0.64"

    def test_an_empty_timeline_still_takes_a_strength(self):
        """Nothing is drawn yet, so the phase is worth one value."""
        stack = self._stack("0.4;0.5")
        up.schedule_enable(stack, "a.safetensors", 0, self.phases, self.memory, self.schedules, self.context)
        up.schedule_set_base(
            stack, "a.safetensors", 0, 0.8, self.phases, self.memory, self.schedules, self.context
        )
        assert stack.tokens[0] == "0.8;0.5"

    def test_an_edit_leaves_the_hidden_phase_alone(self):
        stack = self._stack("1,0.8,0.4;0.65")
        first = self.schedules[("a.safetensors", 0)]
        up.schedule_set_region_strength(
            stack, "a.safetensors", 0, first.regions[0].id, 0.5,
            self.phases, self.memory, self.schedules, self.context,
        )
        assert stack.tokens[0] == "0.5,0.8,0.4;0.65"

    def test_switching_phase_chips_rewrites_nothing(self):
        stack = self._stack("1.0,0.80;0.7")
        for phase in (0, 1, 0):
            up.multiplier_fields(
                stack.tokens[0], self.phases, "a.safetensors", self.memory,
                self.schedules, self.context,
            )
        assert stack.tokens[0] == "1.0,0.80;0.7"

    def test_an_imported_schedule_starts_clean_and_remembers_its_source(self):
        self._stack("1,0.5,0.2;0.4")
        schedule = self.schedules[("a.safetensors", 0)]
        assert schedule.dirty is False
        assert schedule.source_values == [1, 0.5, 0.2]
        assert schedule.source_raw == "1,0.5,0.2;0.4"

    def test_a_material_edit_marks_the_schedule_dirty(self):
        stack = self._stack("1,0.5,0.2;0.4")
        schedule = self.schedules[("a.safetensors", 0)]
        up.schedule_set_region_strength(
            stack, "a.safetensors", 0, schedule.regions[0].id, 0.9,
            self.phases, self.memory, self.schedules, self.context,
        )
        assert self.schedules[("a.safetensors", 0)].dirty is True

    def test_a_gap_is_zero_not_a_hidden_strength(self):
        """Boxes over 1-2 and 4-6 mean slot 3 is off."""
        stack = self._stack("0.9;0.5")
        up.schedule_enable(stack, "a.safetensors", 0, self.phases, self.memory, self.schedules, self.context)
        self.schedules[("a.safetensors", 0)].slots = 6
        up.schedule_add_region(
            stack, "a.safetensors", 0, self.phases, self.memory, self.schedules, self.context,
            start=1, end=2,
        )
        up.schedule_add_region(
            stack, "a.safetensors", 0, self.phases, self.memory, self.schedules, self.context,
            start=4, end=6,
        )
        assert stack.tokens[0] == "0.9,0.9,0,0.9,0.9,0.9;0.5"

    def test_editing_a_scheduled_lora_leaves_its_neighbours_alone(self):
        stack = self._stack("1,0.5 0.5:0.9 0.8;0.8",
                            ids=("a.safetensors", "b.safetensors", "c.safetensors"))
        up.set_phase_value(
            stack, "c.safetensors", 0, 0.3, self.phases, self.memory, schedules=self.schedules
        )
        assert stack.tokens == ["1,0.5", "0.5:0.9", "0.3;0.8"]

    def test_a_read_only_token_refuses_schedule_actions(self):
        stack = self._stack("0.5:0.9")
        for call in (
            lambda: up.schedule_enable(
                stack, "a.safetensors", 0, self.phases, self.memory, self.schedules, self.context
            ),
            lambda: up.schedule_add_region(
                stack, "a.safetensors", 0, self.phases, self.memory, self.schedules, self.context
            ),
        ):
            with pytest.raises(sch.ScheduleError):
                call()
        assert stack.tokens == ["0.5:0.9"]

    def test_a_missing_region_is_reported_not_guessed(self):
        stack = self._stack("1,0.5,0.2")
        with pytest.raises(sch.ScheduleError):
            up.schedule_commit_region(
                stack, "a.safetensors", 0, "nope", 1, 2,
                self.phases, self.memory, self.schedules, self.context,
            )

    def test_coordinate_mode_says_phase_relative_without_a_step_count(self):
        stack = self._stack("1,0.5,0.2")
        fields = up.multiplier_fields(
            stack.tokens[0], self.phases, "a.safetensors", self.memory,
            self.schedules, up.ScheduleContext(steps=0),
        )
        assert fields["phase_schedules"][0]["coordinate_mode"] == up.COORD_PHASE_RELATIVE

    def test_browser_tiles_do_not_carry_region_data(self):
        stack = self._stack("1,0.5,0.2")
        fields = up.multiplier_fields(
            stack.tokens[0], self.phases, "a.safetensors", self.memory,
            self.schedules, self.context, with_schedules=False,
        )
        assert fields["phase_schedules"] == []


# ---------------------------------------------------------------- LTX-2
#
# Model definitions as WanGP's get_model_def returns them once the LTX-2
# handler's query_model_def has been merged in: two phases at most, the
# distilled checkpoints locking the step counter, and no lora_multiplier_phases
# -- so the token's phase count follows the guidance mode.

LTX23_DISTILLED = {
    "architecture": "ltx2_22B", "ltx2_pipeline": "distilled", "guidance_max_phases": 2,
    "visible_phases": 0, "lock_inference_steps": True, "ltx2_22B_class": True,
}
LTX25_DEV = {
    "architecture": "ltx2_25_22B", "guidance_max_phases": 2, "visible_phases": 1,
    "ltx2_22B_class": True,
}
JOYAI_ECHO = {"architecture": "joyai_echo", "guidance_max_phases": 2, "ltx2_22B_class": False}


class TestLtxStages:
    """LTX-2's phases are pipeline stages of known length, so they schedule."""

    def setup_method(self):
        self.schedules = {}
        self.memory = {}

    @pytest.mark.parametrize("model_def", [LTX23_DISTILLED, LTX25_DEV, JOYAI_ECHO])
    def test_ltx2_is_recognised_and_staged(self, model_def):
        two = up.resolve_phases(model_def, 2)
        assert (two.capacity, two.effective, two.staged) == (2, 2, True)
        one = up.resolve_phases(model_def, 1)
        assert (one.capacity, one.effective, one.staged) == (1, 1, True)
        assert up.scheduling_allowed(two) is True

    def test_other_two_phase_models_still_do_not_schedule(self):
        """Their phase 2 starts at a switch step nobody knows while editing."""
        assert up.resolve_phases(H3_MODEL_DEF, 2).staged is False
        assert up.scheduling_allowed(up.resolve_phases(H3_MODEL_DEF, 2)) is False
        assert up.resolve_phases({"architecture": "ltxv_13B", "guidance_max_phases": 2}, 2).staged is False

    @pytest.mark.parametrize("model_def, guidance, steps, sampler, expected", [
        (LTX23_DISTILLED, 2, 8, "", (8, 3)),
        (LTX23_DISTILLED, 2, 30, "", (8, 3)),        # the counter is locked, and ignored
        (LTX23_DISTILLED, 1, 8, "", (8,)),           # One Phase skips the refinement
        (LTX25_DEV, 2, 30, "euler", (30, 3)),
        (LTX25_DEV, 2, 40, "res2s", (40, 3)),
        (LTX25_DEV, 2, 30, "distilled_8_steps_ancestral", (8, 3)),
        (LTX25_DEV, 1, 30, "distilled_8_steps", (8,)),
        (LTX25_DEV, 2, 0, "euler", ()),              # no counter, nothing to follow
        (JOYAI_ECHO, 2, 20, "", (8, 3)),
        (H3_MODEL_DEF, 2, 30, "", ()),
    ])
    def test_each_stage_runs_a_known_number_of_steps(self, model_def, guidance, steps, sampler, expected):
        phases = up.resolve_phases(model_def, guidance)
        context = up.ScheduleContext.from_native(steps, model_def, phases, sampler)
        assert context.phase_steps == expected
        assert context.boundaries_known is bool(expected)

    def _at(self, model_def, guidance, steps, sampler=""):
        phases = up.resolve_phases(model_def, guidance)
        return phases, up.ScheduleContext.from_native(steps, model_def, phases, sampler)

    def _settle(self, stack, phases, context):
        """A payload round in plugin.py's order, resync included."""
        up.sync_schedules(stack, phases, self.schedules, context)
        if up.schedules_need_resync(stack, phases, self.schedules, context):
            up.refit_schedules(stack, phases, self.schedules, self.memory, context)
            up.sync_schedules(stack, phases, self.schedules, context)

    def test_each_stage_gets_a_timeline_of_its_own_length(self):
        phases, context = self._at(LTX23_DISTILLED, 2, 8)
        stack = up.Stack.from_native(["a.safetensors"], "1;1")
        self._settle(stack, phases, context)
        for phase, start, end in ((0, 1, 4), (1, 1, 2)):
            up.schedule_enable(stack, "a.safetensors", phase, phases, self.memory, self.schedules, context)
            up.schedule_add_region(
                stack, "a.safetensors", phase, phases, self.memory, self.schedules, context,
                start=start, end=end,
            )
        assert self.schedules[("a.safetensors", 0)].slots == 8
        assert self.schedules[("a.safetensors", 1)].slots == 3
        assert stack.tokens[0] == "1,1,1,1,0,0,0,0;1,1,0"

        fields = up.multiplier_fields(stack.tokens[0], phases, "a.safetensors", self.memory, self.schedules, context)
        first, second = fields["phase_schedules"]
        assert (first["coordinate_mode"], first["stage"], first["stage_steps"]) == (up.COORD_STAGE_EXACT, 1, 8)
        assert (second["coordinate_mode"], second["stage"], second["stage_steps"]) == (up.COORD_STAGE_EXACT, 2, 3)

    def test_two_phases_keep_a_schedule_instead_of_zeroing_it(self):
        """What leaving One Phase still does to every other model."""
        phases, context = self._at(LTX23_DISTILLED, 2, 8)
        stack = up.Stack.from_native(["a.safetensors"], "1,1,1,1,0,0,0,0;1,1,0")
        self._settle(stack, phases, context)
        assert stack.tokens[0] == "1,1,1,1,0,0,0,0;1,1,0"

    def test_the_step_counter_moves_stage_1_only(self):
        phases, context = self._at(LTX25_DEV, 2, 30, "euler")
        stack = up.Stack.from_native(["a.safetensors"], "1;1")
        self._settle(stack, phases, context)
        up.schedule_add_region(
            stack, "a.safetensors", 0, phases, self.memory, self.schedules, context, start=1, end=10,
        )
        up.schedule_add_region(
            stack, "a.safetensors", 1, phases, self.memory, self.schedules, context, start=3, end=3,
        )
        assert stack.tokens[0].split(";")[1] == "0,0,1"

        phases, context = self._at(LTX25_DEV, 2, 20, "euler")
        self._settle(stack, phases, context)
        stage_1, stage_2 = stack.tokens[0].split(";")
        assert stage_1 == ",".join(["1"] * 10 + ["0"] * 10)
        assert stage_2 == "0,0,1"

    def test_a_distilled_sampler_runs_eight_first_stage_steps_whatever_the_counter(self):
        phases, context = self._at(LTX25_DEV, 2, 30, "distilled_8_steps")
        stack = up.Stack.from_native(["a.safetensors"], "1;1")
        self._settle(stack, phases, context)
        up.schedule_enable(stack, "a.safetensors", 0, phases, self.memory, self.schedules, context)
        assert self.schedules[("a.safetensors", 0)].slots == 8

    def test_an_imported_stage_2_list_is_resampled_the_way_wangp_plays_it(self):
        """Six values over a three-step stage: WanGP plays values 1, 3 and 5."""
        phases, context = self._at(LTX23_DISTILLED, 2, 8)
        stack = up.Stack.from_native(["a.safetensors"], "1,1,1,1,1,1,1,1;1,1,0.5,0.5,0,0")
        self._settle(stack, phases, context)
        assert stack.tokens[0] == "1,1,1,1,1,1,1,1;1,0.5,0"

    def test_a_shared_list_follows_stage_1(self):
        """No ';' means both stages, each stretched to its own length; the
        timeline is where it runs one value per step -- stage 1."""
        phases, context = self._at(LTX23_DISTILLED, 2, 8)
        stack = up.Stack.from_native(["a.safetensors"], "1,1,1,1,0,0,0,0")
        self._settle(stack, phases, context)
        schedule = self.schedules[("a.safetensors", up.SHARED_PHASE)]
        assert schedule.slots == 8
        fields = up.multiplier_fields(stack.tokens[0], phases, "a.safetensors", self.memory, self.schedules, context)
        assert fields["phase_schedules"][0]["coordinate_mode"] == up.COORD_STAGE_EXACT
        assert stack.tokens[0] == "1,1,1,1,0,0,0,0"


class TestPhaseStructureFit:
    """Switching to fewer phases must leave every token one WanGP accepts."""

    def setup_method(self):
        self.hidden = {}

    def _fit(self, multipliers, model_def, guidance, ids=("a.safetensors",)):
        stack = up.Stack.from_native(list(ids), multipliers)
        phases = up.resolve_phases(model_def, guidance)
        up.forget_superseded_phases(stack, self.hidden)
        needed = up.phase_structure_needs_fit(stack, phases, self.hidden)
        changed = up.fit_phase_structure(stack, phases, self.hidden)
        assert needed is changed
        assert up.phase_structure_needs_fit(stack, phases, self.hidden) is False
        return codec.serialize(stack.tokens, stack.separator_index)

    def test_a_stage_2_schedule_survives_a_trip_through_one_phase(self):
        one = self._fit("1,1,1,1,0,0,0,0;1,1,0 0.7;0.3", LTX23_DISTILLED, 1, ids=("a.safetensors", "b.safetensors"))
        # WanGP refuses "x;y" in One Phase; these it accepts.
        assert one == "1,1,1,1,0,0,0,0 0.7"
        two = self._fit(one, LTX23_DISTILLED, 2, ids=("a.safetensors", "b.safetensors"))
        assert two == "1,1,1,1,0,0,0,0;1,1,0 0.7;0.3"

    def test_an_edit_in_one_phase_keeps_what_two_phases_held(self):
        assert self._fit("0.8;0.4", LTX23_DISTILLED, 1) == "0.8"
        # Edited while One Phase hid phase 2: still the shape the cut left.
        assert self._fit("0.5", LTX23_DISTILLED, 2) == "0.5;0.4"

    def test_a_token_that_already_fits_is_not_touched(self):
        for token in ("1;1", "0.8", "1,0.5;0.2", "1,0.5"):
            assert self._fit(token, LTX23_DISTILLED, 2) == token
        assert self._fit("1,0.5", LTX23_DISTILLED, 1) == "1,0.5"

    def test_branch_syntax_and_malformed_tokens_are_never_recut(self):
        for token in ("0.5:0.9;1", "abc;1"):
            assert self._fit(token, LTX23_DISTILLED, 1) == token

    def test_a_rewritten_token_forgets_what_was_cut_from_the_old_one(self):
        """A preset arriving in between must not get somebody else's phase 2."""
        three = {"guidance_max_phases": 3}
        assert self._fit("0.9;0.8;0.7", three, 2) == "0.9;0.8"
        assert self._fit("0.4", three, 2) == "0.4"          # a different token now
        assert self._fit("0.4", three, 3) == "0.4"          # nothing appended

    def test_narrowing_twice_remembers_everything(self):
        three = {"guidance_max_phases": 3}
        assert self._fit("0.9;0.8;0.7", three, 2) == "0.9;0.8"
        assert self._fit("0.9;0.8", three, 1) == "0.9"
        assert self._fit("0.9", three, 3) == "0.9;0.8;0.7"

    def test_a_late_event_with_the_old_token_does_not_erase_the_memory(self):
        """The token the cut was made from, arriving after it, is no rewrite."""
        assert self._fit("1,1,0;0,0.5,1", LTX23_DISTILLED, 1) == "1,1,0"
        late = up.Stack.from_native(["a.safetensors"], "1,1,0;0,0.5,1")
        up.forget_superseded_phases(late, self.hidden)
        assert self._fit("1,1,0", LTX23_DISTILLED, 2) == "1,1,0;0,0.5,1"

    def test_a_late_token_cut_again_keeps_everything_it_stood_for(self):
        three = {"guidance_max_phases": 3}
        assert self._fit("0.9;0.8;0.7", three, 2) == "0.9;0.8"
        assert self._fit("0.9;0.8", three, 1) == "0.9"
        # The two-phase token arriving late and being cut once more.
        assert self._fit("0.9;0.8", three, 1) == "0.9"
        assert self._fit("0.9", three, 3) == "0.9;0.8;0.7"
