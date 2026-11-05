"""A LoRA stack pasted as text: reading the lines, and finding the LoRAs."""

import pytest

from lora_browser import stack_text as st
from lora_browser.inventory import build_inventory

#: The stack the feature was asked for with: two-phase LTX, some of it
#: scheduled.
EXAMPLE = """Vocal_Enhance_1: x0,0,0,0,0.325,0.325,0,0;0
Vocal_Percussion_1: x0.8;0.375,0.25,0.25
Body_Structure_1: x1;0.25
Camera_Movement_1: x1,1,0.875,0.875,0.875,0.875,0.25,0.25;0
"""


def pairs(text):
    return [(line.name, line.token) for line in st.parse(text)]


class TestReadingLines:
    def test_the_example_reads_as_four_names_and_their_tokens(self):
        assert pairs(EXAMPLE) == [
            ("Vocal_Enhance_1", "0,0,0,0,0.325,0.325,0,0;0"),
            ("Vocal_Percussion_1", "0.8;0.375,0.25,0.25"),
            ("Body_Structure_1", "1;0.25"),
            ("Camera_Movement_1", "1,1,0.875,0.875,0.875,0.875,0.25,0.25;0"),
        ]

    @pytest.mark.parametrize("line", [
        "Detail: x0.8;0.5", "Detail: 0.8;0.5", "Detail:x0.8;0.5", "Detail : X 0.8;0.5",
        "Detail: ×0.8;0.5", "Detail: *0.8;0.5", "Detail: x 0.8 ; 0.5", "- Detail: x0.8;0.5",
        "• Detail: x0.8;0.5", "  Detail: x0.8;0.5  ",
    ])
    def test_the_multiplication_mark_and_spacing_are_optional(self, line):
        assert pairs(line) == [("Detail", "0.8;0.5")]

    def test_a_name_with_no_multiplier_loads_at_the_default(self):
        assert pairs("Detail\nOther:\nThird: x") == [("Detail", ""), ("Other", ""), ("Third", "")]

    def test_blank_lines_comments_and_fences_are_skipped_and_lines_keep_their_numbers(self):
        lines = st.parse("```\n# a note\n\n// another\nA: x1\r\n---\nB: x0.5\n```")
        assert [(line.number, line.name, line.token) for line in lines] == [(5, "A", "1"), (7, "B", "0.5")]

    def test_a_colon_in_a_name_does_not_split_it(self):
        # A Civitai name can have one; the token is what follows the last.
        assert pairs("Detail Tweaker: XL: x0.8") == [("Detail Tweaker: XL", "0.8")]
        assert pairs("Detail Tweaker: XL") == [("Detail Tweaker: XL", "")]

    def test_branch_syntax_stays_in_the_token(self):
        assert pairs("Detail: x0.5:0.9") == [("Detail", "0.5:0.9")]

    def test_scientific_notation_and_signs_are_numbers(self):
        assert pairs("A: x-0.5;1e-1\nB: x+.5") == [("A", "-0.5;1e-1"), ("B", "+.5")]

    def test_trailing_text_is_not_a_multiplier(self):
        # Not read as "A" at 0.5: the whole line is a name nothing answers to.
        assert pairs("A: x0.5 please") == [("A: x0.5 please", "")]


class TestTokens:
    @pytest.mark.parametrize("token", [
        "", "1", "0.8;0.5", "1,0.5,0", "0,0,0.325;0", "0.5:0.9", "-1e-1", "1;0.375,0.25,0.25",
    ])
    def test_tokens_wangp_can_read(self, token):
        assert st.token_problem(token) == ""

    @pytest.mark.parametrize("token", ["1,,2", "1;", ";1", "1:", ".", "-", "1e", "1..2"])
    def test_tokens_it_cannot(self, token):
        assert "not a multiplier" in st.token_problem(token)

    def test_phase_count_is_one_more_than_the_semicolons(self):
        assert [st.phase_count(token) for token in ("", "1", "1;0.5", "1,1;0;0")] == [1, 1, 2, 3]


LIBRARY = [
    "Vocal_Enhance_1.safetensors", "ltx/Body_Structure_1.safetensors",
    "Petite.LD.LTX2.3.safetensors", "styles/c.safetensors", "other/c.safetensors",
    "c.safetensors", "pack/d.safetensors", "more/d.safetensors", "e.safetensors", "e.ckpt",
    "Detail-Slider_v2.safetensors",
]


@pytest.fixture
def index():
    return st.NameIndex(
        build_inventory(LIBRARY), {"Detail-Slider_v2.safetensors": "Detail Slider"}
    )


class TestFindingLoras:
    @pytest.mark.parametrize("name, expected", [
        ("Vocal_Enhance_1", "Vocal_Enhance_1.safetensors"),            # stem
        ("Vocal_Enhance_1.safetensors", "Vocal_Enhance_1.safetensors"),  # filename
        ("vocal_enhance_1", "Vocal_Enhance_1.safetensors"),             # any case
        ("Vocal Enhance 1", "Vocal_Enhance_1.safetensors"),             # any punctuation
        ("Body_Structure_1", "ltx/Body_Structure_1.safetensors"),       # in a folder
        ("ltx/Body_Structure_1.safetensors", "ltx/Body_Structure_1.safetensors"),
        ("ltx\\Body_Structure_1.safetensors", "ltx/Body_Structure_1.safetensors"),
        ("Petite.LD.LTX2.3", "Petite.LD.LTX2.3.safetensors"),         # dots in the stem
        ("Detail Slider", "Detail-Slider_v2.safetensors"),              # Civitai name
        ("detail slider", "Detail-Slider_v2.safetensors"),
        ("c.safetensors", "c.safetensors"),          # its exact path, though others share the name
        ("styles/c", "styles/c.safetensors"),
        ("e.ckpt", "e.ckpt"),
    ])
    def test_names_resolve(self, index, name, expected):
        assert index.resolve(name) == (expected, [])

    def test_a_name_two_loras_answer_to_is_not_guessed(self, index):
        lora_id, candidates = index.resolve("d")
        assert lora_id == ""
        assert sorted(candidates) == ["more/d.safetensors", "pack/d.safetensors"]
        # Two files with one stem, too.
        assert index.resolve("e")[1] == ["e.safetensors", "e.ckpt"]

    def test_a_name_nothing_answers_to(self, index):
        assert index.resolve("Nope") == ("", [])
        assert index.resolve("") == ("", [])

    def test_punctuation_alone_never_matches(self, index):
        assert index.resolve("---") == ("", [])


class TestWritingTheStackOut:
    def test_each_lora_gets_the_shortest_name_that_reads_back(self, index):
        names = {lora_id: index.name_for(lora_id) for lora_id in LIBRARY}
        assert names["Vocal_Enhance_1.safetensors"] == "Vocal_Enhance_1"
        assert names["ltx/Body_Structure_1.safetensors"] == "Body_Structure_1"
        assert names["Petite.LD.LTX2.3.safetensors"] == "Petite.LD.LTX2.3"
        # The stem is shared, the filename is not.
        assert names["e.safetensors"] == "e.safetensors"
        # Neither is unique: the path is.
        assert names["pack/d.safetensors"] == "pack/d.safetensors"
        assert names["styles/c.safetensors"] == "styles/c.safetensors"
        # "c" is shared by three folders, but it is also this one's own path.
        assert names["c.safetensors"] == "c"

    def test_what_is_written_reads_back_as_the_same_stack(self, index):
        stack = [(lora_id, f"{position % 3 * 0.25:g};1,0.5") for position, lora_id in enumerate(LIBRARY)]
        text = st.format_stack(stack, index)
        read = [(index.resolve(line.name)[0], line.token) for line in st.parse(text)]
        assert read == stack

    def test_the_example_round_trips(self):
        library = [f"{name}.safetensors" for name, _ in pairs(EXAMPLE)]
        index = st.NameIndex(build_inventory(library))
        stack = [(f"{name}.safetensors", token) for name, token in pairs(EXAMPLE)]
        assert st.format_stack(stack, index) == EXAMPLE.strip()
