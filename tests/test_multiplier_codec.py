"""Round-trip guarantees for WanGP's positional multiplier string."""

import pytest

from lora_browser import multiplier_codec as codec


class TestParsing:
    def test_plain_tokens(self):
        parsed = codec.parse_multipliers("1 0.8 1.2")
        assert parsed.tokens == ["1", "0.8", "1.2"]
        assert parsed.separator_index == -1

    def test_accelerator_bar_records_boundary(self):
        parsed = codec.parse_multipliers("1 1|0.8 0.5")
        assert parsed.tokens == ["1", "1", "0.8", "0.5"]
        assert parsed.separator_index == 2

    def test_bar_at_start_and_end(self):
        assert codec.parse_multipliers("|0.8").separator_index == 0
        assert codec.parse_multipliers("0.8|").separator_index == 1

    def test_comment_lines_are_dropped(self):
        parsed = codec.parse_multipliers("# leading note\n1 0.5\n#trailing")
        assert parsed.tokens == ["1", "0.5"]

    def test_newlines_separate_tokens(self):
        assert codec.parse_multipliers("1\n0.5\n0.25").tokens == ["1", "0.5", "0.25"]

    def test_empty_input(self):
        parsed = codec.parse_multipliers("")
        assert parsed.tokens == []
        assert parsed.separator_index == -1


class TestAlignment:
    def test_missing_tokens_default_to_one(self):
        assert codec.align_tokens(["0.5"], 3) == ["0.5", "1", "1"]

    def test_surplus_tokens_are_dropped(self):
        assert codec.align_tokens(["0.5", "1", "1"], 1) == ["0.5"]


class TestClassification:
    def test_single_value_pads_to_phase_count(self):
        info = codec.classify("0.8", 2)
        assert info.kind == codec.SIMPLE
        # WanGP pads a short token by repeating its last value.
        assert info.values == [0.8, 0.8]

    def test_phase_syntax(self):
        info = codec.classify("0.4;0.9", 2)
        assert info.kind == codec.SIMPLE
        assert info.values == [0.4, 0.9]

    def test_extra_phases_become_overflow(self):
        info = codec.classify("0.4;0.9", 1)
        assert info.values == [0.4]
        assert info.overflow == [0.9]

    @pytest.mark.parametrize("token", ["0.5:0.9", "abc", "1;xyz", "1,0.5:0.9", "1,abc"])
    def test_advanced_and_invalid_tokens_are_preserved(self, token):
        info = codec.classify(token, 2)
        assert info.kind == codec.ADVANCED
        assert info.raw == token
        assert info.values is None
        assert info.reason

    @pytest.mark.parametrize("token", ["nan", "inf", "-inf", "Infinity"])
    def test_non_finite_values_are_rejected(self, token):
        assert codec.classify(token, 1).kind == codec.ADVANCED


class TestScheduledClassification:
    """Comma tokens are WanGP's time-varying multipliers; v2 edits them."""

    def test_comma_only_token_is_one_shared_schedule(self):
        info = codec.classify("1,0.8,0.4,0", 2)
        assert info.kind == codec.SCHEDULED
        assert info.schedule.shared is True
        assert info.schedule.phase_values == [[1, 0.8, 0.4, 0]]
        assert info.schedule.declared_phase_count == 1

    def test_phase_major_orientation(self):
        # ';' separates phases, ',' schedules values inside one phase.
        info = codec.classify("1,0.8;0.7,0.5", 2)
        assert info.kind == codec.SCHEDULED
        assert info.schedule.shared is False
        assert info.schedule.phase_values == [[1, 0.8], [0.7, 0.5]]

    def test_three_phase_schedule(self):
        info = codec.classify("1,0.9;0.8,0.7;0.6,0.5", 3)
        assert info.kind == codec.SCHEDULED
        assert info.schedule.phase_values == [[1, 0.9], [0.8, 0.7], [0.6, 0.5]]

    def test_mixed_scalar_and_scheduled_phases(self):
        info = codec.classify("1;0.7,0.5,0.2", 2)
        assert info.kind == codec.SCHEDULED
        assert info.schedule.phase_values == [[1], [0.7, 0.5, 0.2]]
        assert info.schedule.values_for(0) == [1]
        assert info.schedule.values_for(1) == [0.7, 0.5, 0.2]

    def test_negative_and_over_one_values_are_valid(self):
        info = codec.classify("1.4,-0.5", 1)
        assert info.kind == codec.SCHEDULED
        assert info.schedule.phase_values == [[1.4, -0.5]]

    def test_decimal_precision_survives_parsing(self):
        info = codec.classify("0.125,0.0625", 1)
        assert info.schedule.phase_values == [[0.125, 0.0625]]

    def test_under_declared_multi_phase_schedule_is_not_guessed(self):
        """WanGP expands 2-of-3 using model_switch_phase; we do not replicate it."""
        info = codec.classify("0.9,0.8;1,1", 3)
        assert info.kind == codec.ADVANCED
        assert info.raw == "0.9,0.8;1,1"
        assert "3-phase" in info.reason

    def test_over_declared_schedule_is_preserved_read_only(self):
        info = codec.classify("0.9,0.8;1.2,1.1", 1)
        assert info.kind == codec.ADVANCED
        assert "accepts 1" in info.reason

    def test_parse_schedule_returns_none_for_other_kinds(self):
        assert codec.parse_schedule("0.8;0.9", 2) is None
        assert codec.parse_schedule("0.5:0.9", 2) is None
        assert codec.parse_schedule("1,0.5", 2) is not None

    def test_parsing_never_mutates_the_raw_token(self):
        raw = "1.0,0.80,0.400"
        token = codec.parse_schedule(raw, 1)
        assert token.raw == raw


class TestScheduleSerialization:
    def test_phase_major_serialisation(self):
        assert codec.build_schedule([[1, 0.8], [0.7, 0.5]]) == "1,0.8;0.7,0.5"

    def test_shared_schedule_emits_one_comma_list(self):
        assert codec.build_schedule([[1, 0.8], [0.7]], shared=True) == "1,0.8"

    def test_mixed_scalar_and_schedule(self):
        assert codec.build_schedule([[1], [0.7, 0.5, 0.2]]) == "1;0.7,0.5,0.2"

    def test_numbers_are_formatted_without_junk(self):
        assert codec.build_schedule([[0.1 + 0.2, 1.0]]) == "0.3,1"

    def test_round_trip_of_an_unedited_schedule(self):
        raw = "1,0.8,0.4,0;0.7,0.5,0.2,0"
        token = codec.parse_schedule(raw, 2)
        assert codec.build_schedule(token.phase_values, shared=token.shared) == raw


class TestSerialization:
    def test_no_spaces_around_the_bar(self):
        # WanGP replaces '|' with a space and splits on single spaces, so
        # "1 | 1" would tokenise into empty strings and fail validation.
        assert codec.serialize(["1", "0.8"], 1) == "1|0.8"

    def test_without_separator(self):
        assert codec.serialize(["1", "0.8"], -1) == "1 0.8"

    def test_build_token_emits_exactly_the_phase_count(self):
        assert codec.build_token([0.8, 1.0], 2) == "0.8;1"
        assert codec.build_token([0.8, 1.0], 1) == "0.8"
        assert codec.build_token([0.8], 2) == "0.8;0.8"

    def test_default_token(self):
        assert codec.default_token(1) == "1"
        assert codec.default_token(2) == "1;1"

    def test_numbers_have_no_floating_point_junk(self):
        assert codec.build_token([0.1 + 0.2], 1) == "0.3"

    def test_round_trip_preserves_advanced_tokens_verbatim(self):
        original = "1 0.5|0.8;1 1,0.5,0.25"
        parsed = codec.parse_multipliers(original)
        assert codec.serialize(parsed.tokens, parsed.separator_index) == original


class TestSeparatorMaintenance:
    def test_removing_before_the_bar_moves_it_left(self):
        assert codec.separator_after_removal(2, [0]) == 1

    def test_removing_after_the_bar_leaves_it_alone(self):
        assert codec.separator_after_removal(2, [3]) == 2

    def test_no_bar_stays_absent(self):
        assert codec.separator_after_removal(-1, [0]) == -1


class TestRescale:
    def test_growing_prefers_remembered_values(self):
        assert codec.rescale_values([0.5], 2, remembered=[0.5, 0.9]) == [0.5, 0.9]

    def test_growing_without_memory_repeats_the_last_value(self):
        assert codec.rescale_values([0.5], 2) == [0.5, 0.5]

    def test_shrinking_truncates(self):
        assert codec.rescale_values([0.5, 0.9], 1) == [0.5]
