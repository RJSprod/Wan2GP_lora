"""The region timeline, tested without a browser.

Collision resolution and placement are where a mis-drag would silently change
what WanGP renders, so they live in pure functions and are exercised here
exhaustively rather than through pointer events.
"""

import pytest

from lora_browser import schedule as sch


def region(rid, start, end, strength=0.5):
    return sch.Region(id=rid, start=start, end=end, strength=strength)


def bounds(regions):
    return [(item.id, item.start, item.end) for item in regions]


class TestCompile:
    def test_an_empty_timeline_is_worth_its_base(self):
        """Nothing drawn is not a schedule of zeros -- it is not a schedule."""
        schedule = sch.PhaseSchedule(base=0.8, slots=5, regions=[])
        assert sch.compile_schedule(schedule) == [0.8] * 5

    def test_slots_no_region_covers_are_zero(self):
        schedule = sch.PhaseSchedule(base=1.0, slots=6, regions=[region("r1", 2, 4, 0.3)])
        assert sch.compile_schedule(schedule) == [0, 0.3, 0.3, 0.3, 0, 0]

    def test_the_gap_between_two_regions_is_zero(self):
        # Boxes over 1-2 and 4-6: slot 3 is off until something covers it.
        schedule = sch.PhaseSchedule(
            base=0.9, slots=6,
            regions=[region("r1", 1, 2, 0.9), region("r2", 4, 6, 0.5)],
        )
        assert sch.compile_schedule(schedule) == [0.9, 0.9, 0, 0.5, 0.5, 0.5]

    def test_multiple_regions(self):
        schedule = sch.PhaseSchedule(
            base=1.0, slots=8,
            regions=[region("r1", 1, 2, 0.5), region("r2", 5, 6, 0.25)],
        )
        assert sch.compile_schedule(schedule) == [0.5, 0.5, 0, 0, 0.25, 0.25, 0, 0]

    def test_exact_strengths_are_not_rounded(self):
        schedule = sch.PhaseSchedule(base=1.23, slots=3, regions=[region("r1", 2, 2, 0.0625)])
        assert sch.compile_schedule(schedule) == [0, 0.0625, 0]

    def test_an_empty_timeline_writes_nothing(self):
        """Opening the scheduler must not change what WanGP renders."""
        empty = sch.PhaseSchedule(base=0.71, slots=20, regions=[])
        assert sch.native_values(empty) == [0.71]
        assert sch.is_materialized(empty) is False

    def test_one_region_is_already_a_schedule(self):
        """Even at the strength the phase already had: the rest is now zero."""
        drawn = sch.PhaseSchedule(base=0.71, slots=4, regions=[region("r1", 1, 2, 0.71)])
        assert sch.native_values(drawn) == [0.71, 0.71, 0, 0]
        assert sch.is_materialized(drawn) is True

    def test_a_region_covering_everything_is_still_a_schedule(self):
        schedule = sch.PhaseSchedule(base=1.0, slots=3, regions=[region("r1", 1, 3, 0.4)])
        assert sch.native_values(schedule) == [0.4, 0.4, 0.4]
        assert sch.is_materialized(schedule) is True


class TestReconstruct:
    def test_every_non_zero_run_becomes_a_region(self):
        schedule = sch.reconstruct_schedule([0.6, 0.6, 0.6])
        assert bounds(schedule.regions) == [("r1", 1, 3)]
        assert schedule.slots == 3

    def test_zeros_stay_gaps(self):
        schedule = sch.reconstruct_schedule([0, 0, 0.4, 0.4, 0])
        assert bounds(schedule.regions) == [("r1", 3, 4)]
        assert schedule.regions[0].strength == 0.4

    def test_several_runs(self):
        schedule = sch.reconstruct_schedule([1, 0.8, 0.8, 0, 0.2])
        assert bounds(schedule.regions) == [("r1", 1, 1), ("r2", 2, 3), ("r3", 5, 5)]

    def test_a_run_of_a_different_strength_is_its_own_region(self):
        schedule = sch.reconstruct_schedule([0.5, 0.9, 0.5, 0.9])
        assert bounds(schedule.regions) == [("r1", 1, 1), ("r2", 2, 2), ("r3", 3, 3), ("r4", 4, 4)]

    def test_the_base_is_the_first_real_strength(self):
        """It seeds new regions and survives Clear phase; it never fills a gap."""
        schedule = sch.reconstruct_schedule([0, 0, 0.45, 0.45])
        assert schedule.base == 0.45
        assert sch.compile_schedule(schedule) == [0, 0, 0.45, 0.45]

    def test_an_all_zero_list_is_an_empty_timeline(self):
        schedule = sch.reconstruct_schedule([0, 0, 0])
        assert schedule.regions == []
        assert schedule.base == 0
        assert sch.compile_schedule(schedule) == [0, 0, 0]

    def test_values_over_one_and_negative(self):
        schedule = sch.reconstruct_schedule([1.4, 1.4, -0.3])
        assert schedule.base == 1.4
        assert bounds(schedule.regions) == [("r1", 1, 2), ("r2", 3, 3)]
        assert schedule.regions[1].strength == -0.3

    @pytest.mark.parametrize(
        "values",
        [[1, 0.8, 0.4, 0], [0.7, 0.7, 0.2, 0.7], [1.23, 0.0625, 1.23], [0.5],
         [0, 0, 0], [0, 1, 0, 1], [0.5, 0, 0, 0.5]],
    )
    def test_reconstruction_is_exactly_reversible(self, values):
        assert sch.compile_schedule(sch.reconstruct_schedule(values)) == values

    def test_an_imported_list_is_remembered_verbatim(self):
        schedule = sch.reconstruct_schedule([1, 0.5], raw="1,0.5")
        assert schedule.source_values == [1, 0.5]
        assert schedule.source_raw == "1,0.5"
        assert schedule.dirty is False

    def test_an_absurdly_long_list_is_preserved_but_not_offered(self):
        schedule = sch.reconstruct_schedule([0.5] * (sch.MAX_SLOTS + 1))
        assert schedule.editable is False
        assert schedule.normalization_required is True
        assert str(sch.MAX_SLOTS) in schedule.reason

    def test_a_list_too_long_to_edit_is_still_held_whole(self):
        """Its length is what may be dragged, not what may be kept."""
        values = [1.0] * 150 + [0.25] * 150
        schedule = sch.reconstruct_schedule(values)
        assert schedule.slots == 300
        assert sch.compile_schedule(schedule) == values


class TestPlacement:
    def test_empty_timeline_starts_at_the_beginning(self):
        assert sch.find_region_slot([], 20) == (1, sch.preferred_width(20))

    def test_preferred_width_is_about_a_fifth(self):
        assert 0.18 <= sch.preferred_width(20) / 20 <= 0.26
        assert sch.preferred_width(3) >= 1

    def test_free_space_after_the_last_region_wins(self):
        regions = [region("r1", 1, 4)]
        assert sch.find_region_slot(regions, 20) == (5, 5 + sch.preferred_width(20) - 1)

    def test_a_hole_before_the_last_region_is_used_when_the_tail_is_short(self):
        # Tail is one slot; the interior hole 3-8 fits the preferred width.
        regions = [region("r1", 1, 2), region("r2", 9, 19)]
        assert sch.find_region_slot(regions, 20, width=4) == (3, 6)

    def test_largest_interval_is_the_fallback(self):
        regions = [region("r1", 1, 3), region("r2", 6, 12)]
        # Neither gap (4-5, 13-14) fits a width of 6; the first largest wins.
        assert sch.find_region_slot(regions, 14, width=6) == (4, 5)

    def test_a_full_timeline_refuses(self):
        assert sch.find_region_slot([region("r1", 1, 10)], 10) is None

    def test_a_drawn_span_is_taken_as_drawn(self):
        assert sch.fit_region([], 20, 4, 9) == (4, 9)

    def test_a_drawn_span_stops_at_the_next_region(self):
        regions = [region("r1", 10, 14)]
        assert sch.fit_region(regions, 20, 6, 18) == (6, 9)

    def test_a_drawn_span_stops_at_the_previous_region(self):
        regions = [region("r1", 1, 5)]
        assert sch.fit_region(regions, 20, 12, 8) == (8, 12)

    def test_a_tap_gets_the_default_width_inside_its_gap(self):
        assert sch.fit_region([], 20, 3) == (3, 3 + sch.preferred_width(20) - 1)
        # Clipped when the gap is shorter than the default width.
        assert sch.fit_region([region("r1", 6, 20)], 20, 3) == (3, 5)

    def test_drawing_inside_an_existing_region_is_refused(self):
        assert sch.fit_region([region("r1", 4, 9)], 20, 6, 8) is None

    def test_a_drawn_span_is_clamped_to_the_timeline(self):
        assert sch.fit_region([], 10, 0, 99) == (1, 10)

    def test_placement_never_overlaps(self):
        regions = [region("r1", 3, 6), region("r2", 10, 12)]
        span = sch.find_region_slot(regions, 20)
        candidate = region("new", span[0], span[1])
        assert not any(sch.overlaps(candidate, existing) for existing in regions)


class TestCollisions:
    def test_partial_overlap_on_the_left_shrinks_the_loser(self):
        regions = [region("A", 2, 8), region("B", 10, 15)]
        result = sch.resolve_collision(regions, region("A", 7, 12))
        assert bounds(result) == [("A", 7, 12), ("B", 13, 15)]

    def test_partial_overlap_on_the_right_shrinks_the_loser(self):
        regions = [region("A", 2, 8), region("B", 10, 15)]
        result = sch.resolve_collision(regions, region("B", 6, 15))
        assert bounds(result) == [("A", 2, 5), ("B", 6, 15)]

    def test_winner_inside_loser_splits_it(self):
        regions = [region("A", 2, 15, 0.4), region("B", 5, 8, 0.9)]
        result = sch.resolve_collision(regions, region("B", 6, 9, 0.9))
        assert bounds(result) == [("A", 2, 5), ("B", 6, 9), ("r1", 10, 15)]
        # Both fragments keep the loser's strength.
        assert [item.strength for item in result if item.id != "B"] == [0.4, 0.4]

    def test_a_split_fragment_gets_a_fresh_id(self):
        regions = [region("r1", 1, 10), region("r2", 12, 14)]
        result = sch.resolve_collision(regions, region("r2", 4, 6))
        ids = [item.id for item in result]
        assert ids.count("r1") == 1
        assert len(set(ids)) == len(ids)
        assert "r3" in ids

    def test_winner_covering_the_loser_deletes_it(self):
        regions = [region("A", 4, 6), region("B", 1, 2)]
        result = sch.resolve_collision(regions, region("B", 1, 8))
        assert bounds(result) == [("B", 1, 8)]

    def test_equal_bounds_delete_the_loser(self):
        regions = [region("A", 3, 5), region("B", 8, 9)]
        result = sch.resolve_collision(regions, region("B", 3, 5))
        assert bounds(result) == [("B", 3, 5)]

    def test_several_losers_are_all_resolved(self):
        regions = [region("A", 1, 3), region("B", 5, 7), region("C", 9, 14), region("W", 20, 20)]
        result = sch.resolve_collision(regions, region("W", 2, 10))
        assert bounds(result) == [("A", 1, 1), ("W", 2, 10), ("C", 11, 14)]

    def test_adjacent_regions_do_not_collide(self):
        regions = [region("A", 1, 4), region("B", 8, 10)]
        result = sch.resolve_collision(regions, region("B", 5, 7))
        assert bounds(result) == [("A", 1, 4), ("B", 5, 7)]

    def test_the_winner_is_never_changed(self):
        regions = [region("A", 1, 12), region("B", 14, 15)]
        result = sch.resolve_collision(regions, region("B", 3, 5))
        winner = next(item for item in result if item.id == "B")
        assert (winner.start, winner.end) == (3, 5)

    @pytest.mark.parametrize("start,end", [(1, 1), (2, 6), (5, 20), (10, 11)])
    def test_the_result_is_always_sorted_and_disjoint(self, start, end):
        regions = [region("A", 1, 4), region("B", 6, 9), region("C", 12, 20)]
        result = sch.resolve_collision(regions, region("B", start, end))
        sch.validate_regions(result, 20)
        assert bounds(result) == sorted(bounds(result), key=lambda item: (item[1], item[2]))
        assert all(item.start <= item.end for item in result)


class TestClampAndValidate:
    def test_a_body_drag_keeps_its_duration(self):
        assert sch.clamp_region(18, 22, 20, keep_width=True) == (16, 20)
        assert sch.clamp_region(-3, 1, 20, keep_width=True) == (1, 5)

    def test_a_resize_clamps_only_the_moved_edge(self):
        assert sch.clamp_region(5, 40, 20) == (5, 20)
        assert sch.clamp_region(0, 8, 20) == (1, 8)

    def test_a_region_can_never_be_zero_length(self):
        assert sch.clamp_region(7, 7, 20) == (7, 7)
        first, last = sch.clamp_region(9, 4, 20)
        assert first <= last

    def test_bounds_are_whole_slots(self):
        assert sch.clamp_region(3.4, 7.8, 20) == (3, 8)

    def test_validation_rejects_overlap_and_escapes(self):
        with pytest.raises(sch.ScheduleError):
            sch.validate_regions([region("A", 1, 5), region("B", 4, 8)], 20)
        with pytest.raises(sch.ScheduleError):
            sch.validate_regions([region("A", 1, 25)], 20)
        with pytest.raises(sch.ScheduleError):
            sch.validate_regions([region("A", 0, 3)], 20)

    def test_validation_accepts_an_adjacent_pair(self):
        sch.validate_regions([region("A", 1, 4), region("B", 5, 9)], 20)


class TestNormalisation:
    def test_same_length_is_a_no_op(self):
        assert sch.normalize_values([1, 0.5, 0.2], 3) == [1, 0.5, 0.2]

    def test_stretching_repeats_nearest_values(self):
        assert sch.normalize_values([1, 0], 4) == [1, 1, 0, 0]

    def test_shrinking_samples(self):
        assert sch.normalize_values([1, 1, 0.5, 0.5], 2) == [1, 0.5]

    def test_never_interpolates_a_value_the_user_did_not_choose(self):
        result = sch.normalize_values([1, 0.25], 5)
        assert set(result) <= {1, 0.25}

    def test_normalising_a_schedule_keeps_its_shape_and_marks_it_dirty(self):
        schedule = sch.PhaseSchedule(base=1.0, slots=4, regions=[region("r1", 3, 4, 0.2)])
        rebuilt = sch.normalize_schedule(schedule, 8)
        assert rebuilt.slots == 8
        # Gaps re-grid as gaps, not as a base that was never there.
        assert sch.compile_schedule(rebuilt) == [0, 0, 0, 0, 0.2, 0.2, 0.2, 0.2]
        assert rebuilt.dirty is True

    def test_normalising_makes_an_over_long_schedule_editable(self):
        schedule = sch.reconstruct_schedule([0.5] * (sch.MAX_SLOTS + 10))
        assert schedule.editable is False
        rebuilt = sch.normalize_schedule(schedule, 20)
        assert rebuilt.editable is True
        assert rebuilt.normalization_required is False

    def test_regridding_a_long_schedule_keeps_its_whole_shape(self):
        """Nothing beyond what a timeline could draw may be quietly dropped."""
        schedule = sch.reconstruct_schedule([1.0] * 150 + [0.25] * 150)
        rebuilt = sch.normalize_schedule(schedule, 20)
        assert sch.compile_schedule(rebuilt) == [1.0] * 10 + [0.25] * 10

    def test_slot_counts_are_clamped(self):
        assert sch.clamp_slots(0) == sch.MIN_SLOTS
        assert sch.clamp_slots(10_000) == sch.MAX_SLOTS
        assert sch.clamp_slots("nonsense") == sch.DEFAULT_SLOTS


class TestRegionIds:
    def test_ids_do_not_reuse_a_live_number(self):
        assert sch.next_region_id([region("r1", 1, 2), region("r3", 4, 5)]) == "r4"

    def test_ids_survive_unrelated_deletions(self):
        regions = [region("r1", 1, 2), region("r2", 4, 5)]
        remaining = [item for item in regions if item.id != "r1"]
        assert sch.next_region_id(remaining) == "r3"
