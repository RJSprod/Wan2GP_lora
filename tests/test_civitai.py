"""Fetching a Civitai catalogue for one LoRA, without touching the network."""

import json
import os

import pytest

from lora_browser import catalogue as cat
from lora_browser import civitai


VERSION = {
    "id": 9001,
    "modelId": 4242,
    "name": "LTX 2.3 I2V v1.0",
    "baseModel": "LTXV 2.3",
    "trainedWords": ["l1v3w4llp4p3r", "l1v3w4llp4p3r [Motion Description]"],
    "description": "<p>Version notes</p>",
    "model": {"name": "Live Wallpaper Style"},
    "images": [
        {
            "url": "https://image.civitai.com/abc/original=true/1.mp4",
            "type": "video",
            "width": 1024,
            "height": 576,
            "meta": {"prompt": "a quiet street at night"},
        },
        {
            "url": "https://image.civitai.com/abc/original=true/2.jpeg",
            "type": "image",
            "width": 832,
            "height": 1216,
            "meta": {"prompt": "a lake", "negativePrompt": "blurry"},
        },
    ],
}

MODEL = {
    "id": 4242,
    "name": "Live Wallpaper Style",
    "description": "<p>Loops nicely.</p>",
    "creator": {"username": "NRDX"},
}


class _Response:
    """The slice of an ``http.client.HTTPResponse`` the module actually uses."""

    def __init__(self, body: bytes, content_type: str = "application/json"):
        self._body = body
        self._offset = 0
        self.headers = {"Content-Type": content_type}

    def read(self, amount=None):
        if amount is None:
            chunk, self._offset = self._body[self._offset:], len(self._body)
            return chunk
        chunk = self._body[self._offset:self._offset + amount]
        self._offset += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeCivitai:
    """Serves the two API documents and the media bytes; records every URL."""

    def __init__(self, version=None, model=None):
        self.version = VERSION if version is None else version
        self.model = MODEL if model is None else model
        self.urls: list[str] = []

    def __call__(self, request, timeout=None):
        url = request.full_url
        self.urls.append(url)
        if "/model-versions/by-hash/" in url:
            return _Response(json.dumps(self.version).encode("utf-8"))
        if "/models/" in url:
            return _Response(json.dumps(self.model).encode("utf-8"))
        if url.endswith(".mp4"):
            return _Response(b"\x00\x00\x00\x18ftypmp42", "video/mp4")
        return _Response(b"\xff\xd8\xff", "image/jpeg")


@pytest.fixture
def lora(tmp_path):
    root = tmp_path / "loras"
    root.mkdir()
    path = root / "livewallpaper_ltx23.safetensors"
    path.write_bytes(b"pretend weights")
    return path


@pytest.fixture
def civitai_api(monkeypatch):
    fake = FakeCivitai()
    monkeypatch.setattr(civitai, "urlopen", fake)
    return fake


class TestUrlAllowList:
    @pytest.mark.parametrize("url", [
        "https://civitai.com/api/v1/models/1",
        "https://image.civitai.com/abc/1.jpg",
    ])
    def test_civitai_hosts_are_allowed(self, url):
        assert civitai.is_civitai_url(url) is True

    @pytest.mark.parametrize("url", [
        "http://civitai.com/api/v1/models/1",        # plaintext
        "https://civitai.com.evil.test/1.jpg",       # suffix lookalike
        "https://evil.test/1.jpg",
        "file:///etc/passwd",
        "",
    ])
    def test_everything_else_is_refused(self, url):
        assert civitai.is_civitai_url(url) is False

    def test_a_refused_url_never_reaches_the_network(self, monkeypatch):
        def explode(*args, **kwargs):
            raise AssertionError("should not have been requested")

        monkeypatch.setattr(civitai, "urlopen", explode)
        payload, error = civitai.get_json("https://evil.test/api")
        assert payload is None
        assert "non-Civitai" in error


class TestFetch:
    def test_builds_the_documents_the_panel_reads(self, lora, civitai_api):
        report = civitai.fetch_sidecar(str(lora))
        assert report.ok is True

        detail = cat.read_detail(str(lora))
        assert detail.civitai_name == "Live Wallpaper Style"
        assert detail.version_name == "LTX 2.3 I2V v1.0"
        assert detail.base_model == "LTXV 2.3"
        assert detail.creator == "NRDX"
        assert detail.trained_words == VERSION["trainedWords"]
        assert "Loops nicely." in detail.description
        assert detail.civitai_url == "https://civitai.com/models/4242?modelVersionId=9001"

    def test_media_arrives_with_its_prompts(self, lora, civitai_api):
        civitai.fetch_sidecar(str(lora))
        detail = cat.read_detail(str(lora))
        assert [item.kind for item in detail.media] == ["video", "image"]
        assert detail.media[0].prompt == "a quiet street at night"
        assert detail.media[1].negative_prompt == "blurry"

    def test_the_cheap_index_reads_the_summary_it_wrote(self, lora, civitai_api):
        civitai.fetch_sidecar(str(lora))
        index = cat.read_index(str(lora))
        assert index.civitai_name == "Live Wallpaper Style"
        assert index.version_name == "LTX 2.3 I2V v1.0"
        assert index.creator == "NRDX"
        assert index.base_model == "LTXV 2.3"
        assert index.trained_words == VERSION["trainedWords"]

    def test_a_second_run_downloads_nothing_again(self, lora, civitai_api):
        first = civitai.fetch_sidecar(str(lora))
        assert (first.downloaded, first.skipped) == (2, 0)

        civitai_api.urls.clear()
        second = civitai.fetch_sidecar(str(lora))
        assert (second.downloaded, second.skipped) == (0, 2)
        assert not [url for url in civitai_api.urls if "image.civitai.com" in url]

    def test_missing_media_is_topped_up_without_touching_the_rest(self, lora, civitai_api):
        civitai.fetch_sidecar(str(lora))
        media_dir = lora.parent / lora.stem / "media"
        os.remove(media_dir / "002.jpeg")

        report = civitai.fetch_sidecar(str(lora))
        assert (report.downloaded, report.skipped) == (1, 1)
        assert (media_dir / "002.jpeg").exists()

    def test_a_partial_sidecar_is_rebuilt_into_the_expected_shape(self, lora, civitai_api):
        """The reported case: a folder holding only the combined JSON and words."""
        side = lora.parent / lora.stem
        side.mkdir()
        (side / f"{lora.stem}.json").write_text(json.dumps({"modelVersion": {}}), encoding="utf-8")

        assert civitai.fetch_sidecar(str(lora)).ok is True
        for name in ("summary.txt", "model.json", "modelVersion.json", f"{lora.stem}.txt"):
            assert (side / name).exists()
        combined = json.loads((side / f"{lora.stem}.json").read_text())
        assert combined["modelVersion"]["baseModel"] == "LTXV 2.3"
        assert combined["sha256"]

    def test_a_preview_is_promoted_so_the_tile_gets_a_thumbnail(self, lora, civitai_api):
        civitai.fetch_sidecar(str(lora))
        assert (lora.parent / f"{lora.stem}.jpeg").exists()
        assert (lora.parent / f"{lora.stem}.mp4").exists()

    def test_an_existing_preview_is_never_overwritten(self, lora, civitai_api):
        mine = lora.parent / f"{lora.stem}.png"
        mine.write_bytes(b"my own preview")
        civitai.fetch_sidecar(str(lora))
        assert mine.read_bytes() == b"my own preview"
        assert not (lora.parent / f"{lora.stem}.jpeg").exists()

    def test_media_hosted_off_civitai_is_skipped_not_fetched(self, lora, monkeypatch):
        fake = FakeCivitai(version=dict(VERSION, images=[
            {"url": "https://evil.test/payload.mp4", "type": "video"},
        ]))
        monkeypatch.setattr(civitai, "urlopen", fake)

        report = civitai.fetch_sidecar(str(lora))
        assert report.ok is True
        assert report.failed == 1
        assert not [url for url in fake.urls if "evil.test" in url]

    def test_oversized_media_is_abandoned_rather_than_retried(
        self, lora, civitai_api, monkeypatch
    ):
        monkeypatch.setattr(civitai, "MEDIA_SIZE_LIMIT", 2)
        report = civitai.fetch_sidecar(str(lora))
        assert report.failed == 2
        media_dir = lora.parent / lora.stem / "media"
        assert not [name for name in os.listdir(media_dir) if name.endswith(".part")]
        # Two entries, one attempt each -- a size failure must not burn retries.
        assert len([url for url in civitai_api.urls if "image.civitai.com" in url]) == 2

    def test_an_unknown_file_reports_instead_of_writing_anything(self, lora, monkeypatch):
        from urllib.error import HTTPError

        def not_found(request, timeout=None):
            raise HTTPError(request.full_url, 404, "Not Found", {}, None)

        monkeypatch.setattr(civitai, "urlopen", not_found)
        report = civitai.fetch_sidecar(str(lora))
        assert report.ok is False
        assert "does not have this exact file" in report.message
        assert not (lora.parent / lora.stem).exists()

    def test_a_lora_outside_the_lora_root_is_refused(self, lora, civitai_api, tmp_path):
        report = civitai.fetch_sidecar(str(lora), lora_root=str(tmp_path / "elsewhere"))
        assert report.ok is False
        assert not (lora.parent / lora.stem).exists()

    def test_a_missing_file_is_reported_not_raised(self, tmp_path):
        report = civitai.fetch_sidecar(str(tmp_path / "ghost.safetensors"))
        assert report.ok is False
        assert "not on disk" in report.message


class TestSummaryRoundTrip:
    def test_what_is_written_is_what_the_index_parses(self):
        text = civitai.build_summary(
            source_name="x.safetensors",
            file_hash="abc123",
            version=VERSION,
            model=MODEL,
            media_total=2,
            media_local=2,
        )
        fields, words = cat.parse_summary(text)
        assert fields["civitai_name"] == "Live Wallpaper Style"
        assert fields["version_name"] == "LTX 2.3 I2V v1.0"
        assert fields["base_model"] == "LTXV 2.3"
        assert fields["creator"] == "NRDX"
        assert words == VERSION["trainedWords"]

    def test_it_survives_a_lora_with_no_trained_words(self):
        text = civitai.build_summary(
            source_name="x.safetensors",
            file_hash="abc123",
            version=dict(VERSION, trainedWords=[]),
            model=MODEL,
            media_total=0,
            media_local=0,
        )
        fields, words = cat.parse_summary(text)
        assert words == []
        assert fields["civitai_name"] == "Live Wallpaper Style"
