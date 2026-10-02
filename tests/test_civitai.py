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


class TestMissingParts:
    """What counts as missing, judged against the record already on disk."""

    def test_a_lora_with_no_folder_needs_everything(self, lora):
        assert civitai.missing_parts(str(lora)) == ["catalogue"]

    def test_a_freshly_fetched_lora_is_complete(self, lora, civitai_api):
        civitai.fetch_sidecar(str(lora))
        assert civitai.missing_parts(str(lora)) == []
        assert civitai.needs_fetch(str(lora)) is False

    def test_a_folder_without_a_summary(self, lora, civitai_api):
        civitai.fetch_sidecar(str(lora))
        os.remove(lora.parent / lora.stem / "summary.txt")
        assert civitai.missing_parts(str(lora)) == ["summary"]

    def test_media_present_but_prompts_missing(self, lora, civitai_api):
        """The reported case: images on disk, structure built without prompts."""
        civitai.fetch_sidecar(str(lora))
        media_dir = lora.parent / lora.stem / "media"
        for name in ("001.json", "002.json"):
            os.remove(media_dir / name)

        assert civitai.missing_parts(str(lora)) == ["prompts"]
        assert civitai.needs_fetch(str(lora)) is True

    def test_a_prompt_sidecar_without_its_record_counts_as_missing(self, lora, civitai_api):
        civitai.fetch_sidecar(str(lora))
        (lora.parent / lora.stem / "media" / "001.json").write_text(
            json.dumps({"index": 1, "source_url": "https://image.civitai.com/x"}),
            encoding="utf-8",
        )
        assert civitai.missing_parts(str(lora)) == ["prompts"]

    def test_an_unreadable_prompt_sidecar_counts_as_missing(self, lora, civitai_api):
        civitai.fetch_sidecar(str(lora))
        (lora.parent / lora.stem / "media" / "002.json").write_text("{ broken", encoding="utf-8")
        assert civitai.missing_parts(str(lora)) == ["prompts"]

    def test_one_absent_media_file_out_of_several(self, lora, civitai_api):
        civitai.fetch_sidecar(str(lora))
        os.remove(lora.parent / lora.stem / "media" / "002.jpeg")
        assert civitai.missing_parts(str(lora)) == ["media"]

    def test_a_folder_with_no_record_at_all(self, lora):
        side = lora.parent / lora.stem
        (side / "media").mkdir(parents=True)
        (side / "media" / "001.jpg").write_bytes(b"\xff\xd8\xff")
        assert civitai.missing_parts(str(lora)) == ["summary", "records"]

    def test_a_missing_trigger_words_file(self, lora, civitai_api):
        civitai.fetch_sidecar(str(lora))
        os.remove(lora.parent / lora.stem / f"{lora.stem}.txt")
        assert civitai.missing_parts(str(lora)) == ["trigger words"]

    def test_a_lora_civitai_has_no_media_for_is_complete_once_fetched(
        self, lora, monkeypatch
    ):
        """Otherwise it would be re-hashed by every single Fetch all."""
        monkeypatch.setattr(civitai, "urlopen", FakeCivitai(version=dict(VERSION, images=[])))
        civitai.fetch_sidecar(str(lora))
        assert civitai.missing_parts(str(lora)) == []

    def test_everything_at_once_is_reported_together(self, lora, civitai_api):
        civitai.fetch_sidecar(str(lora))
        side = lora.parent / lora.stem
        os.remove(side / "summary.txt")
        os.remove(side / f"{lora.stem}.txt")
        for name in os.listdir(side / "media"):
            os.remove(side / "media" / name)
        assert civitai.missing_parts(str(lora)) == [
            "summary", "trigger words", "media", "prompts",
        ]


class TestToppingUpAnIncompleteCatalogue:
    """A fetch over an existing folder must repair it, not just skip it."""

    def test_prompts_are_rewritten_for_media_already_on_disk(self, lora, civitai_api):
        civitai.fetch_sidecar(str(lora))
        media_dir = lora.parent / lora.stem / "media"
        for name in ("001.json", "002.json"):
            os.remove(media_dir / name)

        civitai_api.urls.clear()
        report = civitai.fetch_sidecar(str(lora))

        assert report.ok is True
        assert report.downloaded == 0          # the images were already there
        assert report.skipped == 2
        assert not [url for url in civitai_api.urls if "image.civitai.com" in url]

        detail = cat.read_detail(str(lora))
        assert detail.media[0].prompt == "a quiet street at night"
        assert detail.media[1].negative_prompt == "blurry"
        assert civitai.missing_parts(str(lora)) == []

    def test_a_thinner_older_prompt_record_is_replaced(self, lora, civitai_api):
        civitai.fetch_sidecar(str(lora))
        (lora.parent / lora.stem / "media" / "001.json").write_text(
            json.dumps({"index": 1, "civitai": {"type": "video"}}), encoding="utf-8"
        )
        assert cat.read_detail(str(lora)).media[0].prompt == ""

        civitai.fetch_sidecar(str(lora))
        assert cat.read_detail(str(lora)).media[0].prompt == "a quiet street at night"


class TestBulkFetch:
    @pytest.fixture
    def library(self, tmp_path):
        root = tmp_path / "loras"
        root.mkdir()
        paths = []
        for name in ("one", "two", "three"):
            path = root / f"{name}.safetensors"
            path.write_bytes(b"weights for " + name.encode())
            paths.append(str(path))
        return root, paths

    def test_every_lora_gets_a_catalogue(self, library, civitai_api):
        root, paths = library
        job = civitai.BulkFetch(paths, lora_root=str(root))
        job.run()

        state = job.status()
        assert (state["done"], state["fetched"], state["finished"]) == (3, 3, True)
        assert state["media"] == 6
        assert all(not civitai.needs_fetch(path) for path in paths)

    def test_a_lora_civitai_does_not_know_is_counted_apart_from_a_failure(
        self, library, monkeypatch
    ):
        from urllib.error import HTTPError

        root, paths = library
        calls = {"n": 0}

        def sometimes_missing(request, timeout=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise HTTPError(request.full_url, 404, "Not Found", {}, None)
            if calls["n"] == 2:
                raise OSError("no route to host")
            return FakeCivitai()(request, timeout)

        monkeypatch.setattr(civitai, "urlopen", sometimes_missing)
        monkeypatch.setattr(civitai, "DEFAULT_RETRIES", 0)
        job = civitai.BulkFetch(paths, lora_root=str(root), retries=0)
        job.run()

        state = job.status()
        assert state["done"] == 3
        assert state["missing"] == 1
        assert state["failed"] == 1
        assert state["fetched"] == 1
        assert "not on Civitai" in job.summary()

    def test_one_bad_lora_does_not_end_the_run(self, library, civitai_api, monkeypatch):
        root, paths = library
        real = civitai.fetch_sidecar

        def explode(path, **kwargs):
            if path.endswith("two.safetensors"):
                raise RuntimeError("disk on fire")
            return real(path, **kwargs)

        monkeypatch.setattr(civitai, "fetch_sidecar", explode)
        job = civitai.BulkFetch(paths, lora_root=str(root))
        job.run()

        state = job.status()
        assert (state["done"], state["fetched"], state["failed"]) == (3, 2, 1)

    def test_cancelling_stops_the_loop_and_says_so(self, library, civitai_api):
        root, paths = library
        job = civitai.BulkFetch(paths, lora_root=str(root))
        job.cancel()
        job.run()

        state = job.status()
        assert (state["done"], state["cancelled"], state["finished"]) == (0, True, True)
        assert "stopped early" in job.summary()

    def test_an_empty_run_is_finished_before_it_starts(self):
        job = civitai.BulkFetch([])
        assert job.running is False
        assert job.status()["total"] == 0

    def test_the_thread_reports_the_same_result(self, library, civitai_api):
        root, paths = library
        job = civitai.BulkFetch(paths, lora_root=str(root)).start()
        job._thread.join(timeout=30)
        assert job.running is False
        assert job.status()["fetched"] == 3


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


# --------------------------------------------------------------------- LTX-2
#
# Shaped on what Civitai's API actually returns for a version looked up by
# hash (src/pages/api/v1/model-versions/[id].ts, prepareModelVersionResponse):
# up to ten media records, each with its type, size and generation ``meta``,
# and a URL built by getEdgeUrl with ``original=true`` and a name of
# ``<image id>`` plus ``.mp4`` for a video or ``.jpeg`` for an image -- whatever
# the uploaded file was. LTX-2 LoRAs are video LoRAs, so their media is often
# nothing but videos, and a prompt is only there when the uploader's file
# carried one.

EDGE = "https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA"

LTX_VERSION = {
    "id": 2300001,
    "modelId": 1900001,
    "name": "v1.0 LTX-2.3",
    "baseModel": "LTXV 2.3",
    "trainedWords": ["dolly zoom", "vertigo effect"],
    "description": "<p>Trained on <b>LTX-2.3</b> 22B. Use <i>1.0</i> in stage 1.</p>",
    "model": {"name": "Vertigo Dolly Zoom", "type": "LORA"},
    "images": [
        {
            "url": f"{EDGE}/0b7d1c3e-aaaa-4c1f-9a5b-1111/original=true/88001.mp4",
            "type": "video", "width": 1280, "height": 704, "nsfwLevel": 1,
            "hasMeta": True,
            "meta": {"prompt": "dolly zoom on a man in a corridor", "negativePrompt": "static"},
        },
        {
            "url": f"{EDGE}/0b7d1c3e-bbbb-4c1f-9a5b-2222/original=true/88002.mp4",
            "type": "video", "width": 1280, "height": 704, "nsfwLevel": 1,
            "hasMeta": True,
            "meta": {"prompt": "vertigo effect, a lighthouse at dusk"},
        },
        {
            "url": f"{EDGE}/0b7d1c3e-cccc-4c1f-9a5b-3333/original=true/88003.mp4",
            "type": "video", "width": 704, "height": 1280, "nsfwLevel": 1,
            "hasMeta": False,
            "meta": None,
        },
    ],
}

LTX_MODEL = {
    "id": 1900001,
    "name": "Vertigo Dolly Zoom",
    "description": "<h3>Vertigo</h3><p>Camera effect LoRA for LTX-2.3.</p><ul><li>stage 1</li></ul>",
    "creator": {"username": "frames"},
}

MP4 = b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isomiso2avc1mp41" + b"\x00" * 32
WEBM = b"\x1a\x45\xdf\xa3\x9f\x42\x86\x81\x01\x42\xf7\x81\x01\x42\xf2\x81\x04\x42\xf3\x81\x08\x42\x82\x84webm"


class LtxCivitai(FakeCivitai):
    """LTX-shaped API documents; media bytes chosen per URL."""

    def __init__(self, bodies=None, **kwargs):
        super().__init__(version=kwargs.get("version", LTX_VERSION), model=kwargs.get("model", LTX_MODEL))
        self.bodies = bodies or {}

    def __call__(self, request, timeout=None):
        url = request.full_url
        for marker, (body, content_type) in self.bodies.items():
            if marker in url:
                self.urls.append(url)
                return _Response(body, content_type)
        if "image.civitai.com" in url:
            self.urls.append(url)
            return _Response(MP4, "video/mp4")
        return super().__call__(request, timeout)


@pytest.fixture
def ltx_lora(tmp_path):
    root = tmp_path / "loras" / "ltx2"
    root.mkdir(parents=True)
    path = root / "vertigo_dolly_zoom_ltx23.safetensors"
    path.write_bytes(b"pretend LTX-2.3 weights")
    return path


class TestLtxCatalogue:
    def test_videos_prompts_and_description_all_arrive(self, ltx_lora, monkeypatch):
        monkeypatch.setattr(civitai, "urlopen", LtxCivitai())
        report = civitai.fetch_sidecar(str(ltx_lora))
        assert report.ok is True
        assert (report.downloaded, report.failed) == (3, 0)

        detail = cat.read_detail(str(ltx_lora))
        assert [item.kind for item in detail.media] == ["video", "video", "video"]
        assert [item.prompt for item in detail.media] == [
            "dolly zoom on a man in a corridor",
            "vertigo effect, a lighthouse at dusk",
            "",
        ]
        assert detail.media[0].negative_prompt == "static"
        assert (detail.media[2].width, detail.media[2].height) == (704, 1280)
        assert detail.civitai_name == "Vertigo Dolly Zoom"
        assert detail.base_model == "LTXV 2.3"
        assert detail.creator == "frames"
        assert detail.trained_words == ["dolly zoom", "vertigo effect"]
        assert detail.description == "Vertigo\n\nCamera effect LoRA for LTX-2.3.\n\nstage 1"
        assert detail.version_description == "Trained on LTX-2.3 22B. Use 1.0 in stage 1."
        assert detail.civitai_url == "https://civitai.com/models/1900001?modelVersionId=2300001"

    def test_a_video_only_lora_still_gets_a_tile_preview(self, ltx_lora, monkeypatch):
        from lora_browser.thumbnails import find_preview

        monkeypatch.setattr(civitai, "urlopen", LtxCivitai())
        civitai.fetch_sidecar(str(ltx_lora))
        assert (ltx_lora.parent / f"{ltx_lora.stem}.mp4").exists()
        preview = find_preview(str(ltx_lora), str(ltx_lora.parent))
        assert preview is not None and preview.kind == "video"

    def test_a_webm_that_civitai_names_mp4_is_saved_as_webm(self, ltx_lora, monkeypatch):
        """getEdgeUrl names every video .mp4; the bytes say what it is."""
        monkeypatch.setattr(civitai, "urlopen", LtxCivitai(bodies={"88002": (WEBM, "video/webm")}))
        civitai.fetch_sidecar(str(ltx_lora))
        media = sorted(os.listdir(ltx_lora.parent / ltx_lora.stem / "media"))
        assert "002.webm" in media and "002.mp4" not in media
        assert cat.read_detail(str(ltx_lora)).media[1].kind == "video"

    def test_a_web_page_is_never_saved_as_media(self, ltx_lora, monkeypatch):
        """Saved, it would count as downloaded and never be fetched again."""
        page = b"<!DOCTYPE html><html><body>Just a moment...</body></html>"
        monkeypatch.setattr(civitai, "urlopen", LtxCivitai(bodies={"88001": (page, "text/html")}))
        report = civitai.fetch_sidecar(str(ltx_lora))
        assert report.failed == 1
        media_dir = ltx_lora.parent / ltx_lora.stem / "media"
        assert not [name for name in os.listdir(media_dir) if name.startswith("001.") and not name.endswith(".json")]
        assert "media" in civitai.missing_parts(str(ltx_lora))

    def test_an_empty_download_is_not_saved(self, ltx_lora, monkeypatch):
        monkeypatch.setattr(civitai, "urlopen", LtxCivitai(bodies={"88003": (b"", "video/mp4")}))
        report = civitai.fetch_sidecar(str(ltx_lora))
        assert report.failed == 1
        assert not civitai.existing_media(str(ltx_lora.parent / ltx_lora.stem / "media"), 3)


class TestSniffing:
    @pytest.mark.parametrize("head, expected", [
        (b"\xff\xd8\xff\xe0", ("image", ".jpg")),
        (b"\x89PNG\r\n\x1a\n", ("image", ".png")),
        (b"GIF89a", ("image", ".gif")),
        (b"RIFF\x00\x00\x00\x00WEBPVP8 ", ("image", ".webp")),
        (b"\x00\x00\x00\x1cftypavif", ("image", ".avif")),
        (MP4, ("video", ".mp4")),
        (b"\x00\x00\x00\x14ftypqt  ", ("video", ".mov")),
        (WEBM, ("video", ".webm")),
        (b"\x1a\x45\xdf\xa3\x93\x42\x82\x88matroska", ("video", ".mkv")),
        (b"\x00\x00\x00\x18ftypheic", ("", "")),     # a HEIF still, not a video
        (b"<!DOCTYPE html>", ("", "")),
        (b"", ("", "")),
    ])
    def test_the_bytes_say_what_a_file_is(self, head, expected):
        assert cat.sniff_media(head) == expected

    def test_a_jpeg_url_serving_jpeg_keeps_its_name(self):
        assert civitai._extension_for(f"{EDGE}/x/original=true/1.jpeg", "image/jpeg", "image", b"\xff\xd8\xff") == ".jpeg"

    def test_a_video_kept_under_an_image_name_still_reads_as_a_video(self, ltx_lora):
        """A catalogue an older script built may have done exactly this."""
        media_dir = ltx_lora.parent / ltx_lora.stem / "media"
        media_dir.mkdir(parents=True)
        (media_dir / "001.jpg").write_bytes(MP4)
        assert cat.read_detail(str(ltx_lora)).media[0].kind == "video"
