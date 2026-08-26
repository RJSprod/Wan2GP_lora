"""Reading the Civitai sidecar folder beside a LoRA, complete or partial."""

import json
import os

import pytest

from lora_browser import catalogue as cat


SUMMARY = """Local file: cool_lora.safetensors
SHA256: abc123
MiniMax H3 detection: explicit metadata: ss_base_model_version=minimax_h3

Civitai model name: Cool Motion LoRA
Civitai model ID: 4242
Civitai version name: v2 Final
Civitai version ID: 9001
Base model: MiniMax H3
Creator: somebody

Trained words:
- c00lmotion
- dramatic zoom

Version media entries: 2
Media available locally after this run: 2

Civitai page: https://civitai.com/models/4242?modelVersionId=9001
"""


@pytest.fixture
def library(tmp_path):
    """A LoRA with a full sidecar, and one with nothing beside it."""
    root = tmp_path / "loras"
    (root / "cool_lora" / "media").mkdir(parents=True)
    (root / "cool_lora.safetensors").write_bytes(b"")
    (root / "bare.safetensors").write_bytes(b"")

    side = root / "cool_lora"
    (side / "summary.txt").write_text(SUMMARY, encoding="utf-8")
    (side / "cool_lora.txt").write_text("c00lmotion\ndramatic zoom\n", encoding="utf-8")
    (side / "cool_lora.json").write_text(json.dumps({
        "sourceFile": "cool_lora.safetensors",
        "sha256": "abc123",
        "minimaxH3Detection": "explicit metadata",
        "modelVersion": {
            "id": 9001, "modelId": 4242, "name": "v2 Final",
            "baseModel": "MiniMax H3",
            "trainedWords": ["c00lmotion", "dramatic zoom"],
            "description": "<p>Version notes</p>",
        },
        "model": {
            "id": 4242, "name": "Cool Motion LoRA",
            "description": "<p>Does <b>cool</b> things.</p><ul><li>one</li></ul>",
            "creator": {"username": "somebody"},
        },
    }), encoding="utf-8")

    media = side / "media"
    (media / "001.jpg").write_bytes(b"\xff\xd8\xff")
    (media / "001.json").write_text(json.dumps({
        "index": 1, "source_url": "https://cdn/1.jpg",
        "civitai": {"width": 832, "height": 1216, "type": "image",
                    "meta": {"prompt": "a cat", "negativePrompt": "blurry"}},
    }), encoding="utf-8")
    (media / "002.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42")
    (media / "002.json").write_text(json.dumps({
        "index": 2, "source_url": "https://cdn/2.mp4",
        "civitai": {"width": 720, "height": 1280, "type": "video",
                    "meta": {"prompt": "a dog running"}},
    }), encoding="utf-8")
    return root


class TestSidecarDiscovery:
    def test_finds_the_folder_named_after_the_stem(self, library):
        found = cat.sidecar_dir(str(library / "cool_lora.safetensors"))
        assert found == str(library / "cool_lora")

    def test_none_when_there_is_no_sidecar(self, library):
        assert cat.sidecar_dir(str(library / "bare.safetensors")) is None

    def test_empty_path(self):
        assert cat.sidecar_dir("") is None


class TestIndex:
    def test_reads_name_and_words_from_summary(self, library):
        index = cat.read_index(str(library / "cool_lora.safetensors"))
        assert index.civitai_name == "Cool Motion LoRA"
        assert index.version_name == "v2 Final"
        assert index.creator == "somebody"
        assert index.trained_words == ["c00lmotion", "dramatic zoom"]
        assert index.has_sidecar is True

    def test_searchable_text_covers_name_and_triggers(self, library):
        index = cat.read_index(str(library / "cool_lora.safetensors"))
        assert "cool motion" in index.searchable
        assert "c00lmotion" in index.searchable

    def test_missing_sidecar_is_empty_not_an_error(self, library):
        index = cat.read_index(str(library / "bare.safetensors"))
        assert index.has_sidecar is False
        assert index.civitai_name == ""
        assert index.trained_words == []

    def test_falls_back_to_the_words_file_when_summary_has_none(self, library):
        (library / "cool_lora" / "summary.txt").write_text(
            "Civitai model name: Cool Motion LoRA\n", encoding="utf-8"
        )
        index = cat.read_index(str(library / "cool_lora.safetensors"))
        assert index.trained_words == ["c00lmotion", "dramatic zoom"]

    def test_falls_back_to_the_combined_json_when_summary_is_absent(self, library):
        """The common shape of a hand-made or half-built sidecar."""
        os.remove(library / "cool_lora" / "summary.txt")
        index = cat.read_index(str(library / "cool_lora.safetensors"))
        assert index.civitai_name == "Cool Motion LoRA"
        assert index.version_name == "v2 Final"
        assert index.creator == "somebody"
        assert index.base_model == "MiniMax H3"
        assert index.trained_words == ["c00lmotion", "dramatic zoom"]

    def test_falls_back_to_model_json_when_only_the_raw_halves_exist(self, library):
        os.remove(library / "cool_lora" / "summary.txt")
        os.remove(library / "cool_lora" / "cool_lora.json")
        (library / "cool_lora" / "model.json").write_text(
            json.dumps({"name": "From model.json"}), encoding="utf-8"
        )
        assert cat.read_index(str(library / "cool_lora.safetensors")).civitai_name == "From model.json"

    def test_the_version_model_stub_names_a_lora_with_no_model_record(self, library):
        os.remove(library / "cool_lora" / "summary.txt")
        (library / "cool_lora" / "cool_lora.json").write_text(json.dumps({
            "modelVersion": {"name": "v1", "model": {"name": "Stub Name"}},
        }), encoding="utf-8")
        assert cat.read_index(str(library / "cool_lora.safetensors")).civitai_name == "Stub Name"

    def test_a_good_summary_is_not_paid_for_twice(self, library, monkeypatch):
        """The cheap path must stay cheap: no record parsing when summary.txt answers."""
        monkeypatch.setattr(cat, "load_records", _explode)
        assert cat.read_index(str(library / "cool_lora.safetensors")).civitai_name == "Cool Motion LoRA"


def _explode(*args, **kwargs):
    raise AssertionError("the expensive record read should not have happened")


class TestPartialSidecars:
    """Folders in the wild are rarely complete; none of these may go blank."""

    def test_detail_reads_a_sidecar_that_has_only_the_combined_json(self, library):
        for name in ("summary.txt", "cool_lora.txt"):
            os.remove(library / "cool_lora" / name)
        detail = cat.read_detail(str(library / "cool_lora.safetensors"))
        assert detail.civitai_name == "Cool Motion LoRA"
        assert detail.base_model == "MiniMax H3"
        assert detail.trained_words == ["c00lmotion", "dramatic zoom"]
        assert [item.index for item in detail.media] == [1, 2]

    def test_media_dropped_straight_into_the_folder_still_renders(self, library):
        side = library / "cool_lora"
        for name in os.listdir(side / "media"):
            os.remove(side / "media" / name)
        (side / "preview.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        (side / "clip.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42")

        detail = cat.read_detail(str(library / "cool_lora.safetensors"))
        assert [item.kind for item in detail.media] == ["video", "image"]
        assert all(item.prompt == "" for item in detail.media)

    def test_the_media_folder_wins_over_loose_files(self, library):
        (library / "cool_lora" / "stray.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        detail = cat.read_detail(str(library / "cool_lora.safetensors"))
        assert [item.index for item in detail.media] == [1, 2]

    def test_a_generic_detection_key_is_read(self, library):
        combined = json.loads((library / "cool_lora" / "cool_lora.json").read_text())
        combined.pop("minimaxH3Detection")
        combined["detection"] = "some other family"
        (library / "cool_lora" / "cool_lora.json").write_text(json.dumps(combined), encoding="utf-8")
        detail = cat.read_detail(str(library / "cool_lora.safetensors"))
        assert detail.detection == "some other family"


class TestDetail:
    def test_reads_the_full_record(self, library):
        detail = cat.read_detail(str(library / "cool_lora.safetensors"))
        assert detail.civitai_name == "Cool Motion LoRA"
        assert detail.creator == "somebody"
        assert detail.base_model == "MiniMax H3"
        assert detail.trained_words == ["c00lmotion", "dramatic zoom"]
        assert detail.sha256 == "abc123"
        assert detail.civitai_url == "https://civitai.com/models/4242?modelVersionId=9001"

    def test_description_is_flattened_to_text(self, library):
        detail = cat.read_detail(str(library / "cool_lora.safetensors"))
        assert "<b>" not in detail.description
        assert "Does cool things." in detail.description
        assert "one" in detail.description

    def test_media_is_ordered_and_typed_with_prompts(self, library):
        detail = cat.read_detail(str(library / "cool_lora.safetensors"))
        assert [item.index for item in detail.media] == [1, 2]
        assert detail.media[0].kind == "image"
        assert detail.media[0].prompt == "a cat"
        assert detail.media[0].negative_prompt == "blurry"
        assert detail.media[1].kind == "video"
        assert detail.media[1].prompt == "a dog running"

    def test_json_sidecars_are_not_treated_as_media(self, library):
        detail = cat.read_detail(str(library / "cool_lora.safetensors"))
        assert all(not item.path.endswith(".json") for item in detail.media)

    def test_missing_sidecar_reports_an_error_instead_of_raising(self, library):
        detail = cat.read_detail(str(library / "bare.safetensors"))
        assert detail.error
        assert detail.media == []

    def test_corrupt_json_degrades_gracefully(self, library):
        (library / "cool_lora" / "cool_lora.json").write_text("{ broken", encoding="utf-8")
        detail = cat.read_detail(str(library / "cool_lora.safetensors"))
        # Falls back to summary.txt rather than blowing up.
        assert detail.civitai_name == "Cool Motion LoRA"


class TestHtmlFlattening:
    def test_block_tags_become_line_breaks(self):
        assert cat.strip_html("<p>a</p><p>b</p>") == "a\n\nb"

    def test_list_items_do_not_run_together(self):
        assert cat.strip_html("Intro<ul><li>a</li><li>b</li></ul>") == "Intro\n\na\n\nb"

    def test_entities_are_unescaped(self):
        assert cat.strip_html("a &amp; b") == "a & b"

    def test_script_bodies_are_dropped(self):
        assert cat.strip_html("<script>alert(1)</script>Safe") == "Safe"

    def test_empty_input(self):
        assert cat.strip_html(None) == ""
