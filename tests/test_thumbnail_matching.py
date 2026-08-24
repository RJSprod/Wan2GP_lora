"""Same-stem preview discovery, image precedence, and cache invalidation."""

import os
import time

import pytest

from lora_browser.thumbnails import (
    Preview,
    ThumbnailCache,
    cache_key,
    find_preview,
)


def touch(directory, name, content=b""):
    path = os.path.join(directory, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(content)
    return path


@pytest.fixture
def library(tmp_path):
    return str(tmp_path)


class TestMatching:
    def test_image_preview_is_found(self, library):
        lora = touch(library, "foo.safetensors")
        touch(library, "foo.png")
        preview = find_preview(lora, library)
        assert preview.kind == "image"
        assert preview.path.endswith("foo.png")

    def test_image_wins_over_video(self, library):
        lora = touch(library, "foo.safetensors")
        touch(library, "foo.mp4")
        # Written after the video, but precedence is by type, not by mtime.
        time.sleep(0.01)
        touch(library, "foo.png")
        assert find_preview(lora, library).kind == "image"

    def test_video_only_falls_back_to_video(self, library):
        lora = touch(library, "bar.safetensors")
        touch(library, "bar.mp4")
        preview = find_preview(lora, library)
        assert preview.kind == "video"
        assert preview.path.endswith("bar.mp4")

    def test_extension_precedence_order(self, library):
        lora = touch(library, "foo.safetensors")
        for name in ("foo.jpeg", "foo.jpg", "foo.webp", "foo.png"):
            touch(library, name)
        assert find_preview(lora, library).path.endswith("foo.png")

    def test_extension_comparison_is_case_insensitive(self, library):
        lora = touch(library, "foo.safetensors")
        touch(library, "foo.PNG")
        assert find_preview(lora, library).path.endswith("foo.PNG")

    def test_no_preview_returns_none(self, library):
        lora = touch(library, "foo.safetensors")
        assert find_preview(lora, library) is None

    def test_a_different_stem_is_not_used(self, library):
        lora = touch(library, "foo.safetensors")
        touch(library, "unrelated.png")
        assert find_preview(lora, library) is None

    def test_preview_must_be_in_the_same_directory(self, library):
        lora = touch(library, "sub/foo.safetensors")
        touch(library, "foo.png")
        assert find_preview(lora, library) is None

    def test_subdirectory_preview_is_found(self, library):
        lora = touch(library, "sub/foo.safetensors")
        touch(library, "sub/foo.webp")
        assert find_preview(lora, library).path.endswith(os.path.join("sub", "foo.webp"))

    def test_lookup_outside_the_lora_root_is_refused(self, tmp_path):
        root = tmp_path / "loras"
        outside = tmp_path / "elsewhere"
        root.mkdir()
        outside.mkdir()
        lora = touch(str(outside), "secret.safetensors")
        touch(str(outside), "secret.png")
        # A LoRA value that escapes the model's root must not read from disk.
        assert find_preview(lora, str(root)) is None

    def test_missing_directory_is_handled(self, library):
        assert find_preview(os.path.join(library, "gone", "x.safetensors"), library) is None

    def test_empty_path(self):
        assert find_preview("", "") is None

    def test_supplied_listing_avoids_a_rescan(self, library):
        lora = touch(library, "foo.safetensors")
        touch(library, "foo.png")
        assert find_preview(lora, library, listing=["foo.png"]).path.endswith("foo.png")
        assert find_preview(lora, library, listing=[]) is None


class TestCacheKey:
    def test_key_changes_when_the_preview_file_is_replaced(self, library):
        path = touch(library, "foo.png", b"first")
        preview = Preview(path, "image")
        before = cache_key(preview)

        time.sleep(0.01)
        touch(library, "foo.png", b"second and longer")
        assert cache_key(preview) != before

    def test_key_is_stable_for_an_unchanged_file(self, library):
        preview = Preview(touch(library, "foo.png", b"data"), "image")
        assert cache_key(preview) == cache_key(preview)

    def test_target_size_is_part_of_the_key(self, library):
        preview = Preview(touch(library, "foo.png", b"data"), "image")
        assert cache_key(preview, 320) != cache_key(preview, 192)

    def test_a_missing_source_still_produces_a_key(self, library):
        assert cache_key(Preview(os.path.join(library, "gone.png"), "image"))


class TestCacheBehaviour:
    def test_undecodable_preview_returns_none_instead_of_raising(self, tmp_path, library):
        broken = touch(library, "foo.png", b"not really a png")
        cache = ThumbnailCache(str(tmp_path / "cache"))
        assert cache.get(Preview(broken, "image")) is None

    def test_a_failed_preview_is_not_retried_endlessly(self, tmp_path, library):
        broken = touch(library, "foo.png", b"not really a png")
        cache = ThumbnailCache(str(tmp_path / "cache"))
        preview = Preview(broken, "image")
        cache.get(preview)
        assert cache_key(preview) in cache._failures

    def test_generated_thumbnail_round_trips_as_a_data_uri(self, tmp_path, library):
        pillow = pytest.importorskip("PIL.Image")
        source = os.path.join(library, "foo.png")
        pillow.new("RGB", (640, 480), (10, 120, 200)).save(source)

        cache = ThumbnailCache(str(tmp_path / "cache"))
        uri = cache.get(Preview(source, "image"))
        assert uri and uri.startswith("data:image/webp;base64,")

        # Second read comes from the cache file rather than re-encoding.
        cached = cache.cached_path(cache_key(Preview(source, "image")))
        assert os.path.isfile(cached)
        with pillow.open(cached) as image:
            assert max(image.size) <= 320

    def test_prune_keeps_requested_keys(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        for index in range(5):
            (cache_dir / f"key{index}.webp").write_bytes(b"x")
        cache = ThumbnailCache(str(cache_dir))
        removed = cache.prune({"key0"}, limit=2)
        assert removed == 4
        assert (cache_dir / "key0.webp").exists()
