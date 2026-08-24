"""Stack profiles, favourites, tags and preference persistence."""

import json
import os

import pytest

from lora_browser.metadata_store import (
    DEFAULT_ZOOM_PX,
    MAX_ZOOM_PX,
    MIN_ZOOM_PX,
    MetadataStore,
    resolve_store_path,
)
from lora_browser.profile_store import (
    ProfileEntry,
    ProfileError,
    ProfileStore,
    sanitize_name,
)


@pytest.fixture
def store(tmp_path):
    return MetadataStore(str(tmp_path / "wan2gp_lora_browser.json"))


@pytest.fixture
def profiles(store):
    return ProfileStore(store)


STACK = [
    ProfileEntry(id="foo.safetensors", multiplier="0.8;1"),
    ProfileEntry(id="sub/bar.safetensors", multiplier="1;1"),
]


class TestStoreLocation:
    def test_prefers_the_wangp_config_directory(self):
        path = resolve_store_path("/opt/wangp/wgp_config.json")
        assert path == os.path.join("/opt/wangp", "wan2gp_lora_browser.json")

    def test_falls_back_when_the_config_path_is_unknown(self, tmp_path):
        path = resolve_store_path("", str(tmp_path))
        assert path == os.path.join(str(tmp_path), "wan2gp_lora_browser.json")


class TestPersistence:
    def test_favourites_and_tags_survive_a_reload(self, store):
        store.set_favorite("minimax_h3", "foo.safetensors", True)
        store.set_tag("minimax_h3", "foo.safetensors", "portrait")

        reloaded = MetadataStore(store.path)
        assert reloaded.is_favorite("minimax_h3", "foo.safetensors")
        assert reloaded.tag("minimax_h3", "foo.safetensors") == "portrait"

    def test_metadata_is_scoped_per_model(self, store):
        store.set_favorite("minimax_h3", "foo.safetensors", True)
        assert store.is_favorite("wan22", "foo.safetensors") is False

    def test_unfavouriting_removes_the_entry(self, store):
        store.set_favorite("h3", "foo.safetensors", True)
        store.set_favorite("h3", "foo.safetensors", False)
        assert store.data["favorites"] == {}

    def test_an_empty_tag_clears_it(self, store):
        store.set_tag("h3", "foo.safetensors", "portrait")
        store.set_tag("h3", "foo.safetensors", "   ")
        assert store.data["tags"] == {}

    def test_zoom_persists_and_is_bounded(self, store):
        assert store.set_zoom(9999) == MAX_ZOOM_PX
        assert store.set_zoom(1) == MIN_ZOOM_PX
        store.set_zoom(120)
        assert MetadataStore(store.path).zoom_px == 120

    def test_unparseable_zoom_falls_back_to_the_default(self, store):
        assert store.set_zoom("not a number") == DEFAULT_ZOOM_PX

    def test_writes_are_atomic_and_leave_no_temp_files(self, store, tmp_path):
        store.set_zoom(140)
        leftovers = [name for name in os.listdir(str(tmp_path)) if name.endswith(".tmp")]
        assert leftovers == []

    def test_a_corrupt_file_is_backed_up_rather_than_crashing(self, store, tmp_path):
        store.set_favorite("h3", "foo.safetensors", True)
        with open(store.path, "w", encoding="utf-8") as handle:
            handle.write("{ this is not json")

        recovered = MetadataStore(store.path)
        assert recovered.data["favorites"] == {}
        backups = [name for name in os.listdir(str(tmp_path)) if ".corrupt-" in name]
        assert len(backups) == 1

    def test_a_newer_schema_is_not_truncated(self, store):
        with open(store.path, "w", encoding="utf-8") as handle:
            json.dump({"schema_version": 99, "future_field": "keep me"}, handle)
        assert MetadataStore(store.path).data["future_field"] == "keep me"


class TestNames:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("  My Stack  ", "My Stack"),
            ("../../etc/passwd", "etcpasswd"),
            ("a/b\\c", "abc"),
            ("Porträt Stapel", "Porträt Stapel"),
        ],
    )
    def test_sanitisation(self, raw, expected):
        assert sanitize_name(raw) == expected

    @pytest.mark.parametrize("raw", ["", "   ", "...", "/", None])
    def test_empty_names_are_rejected(self, raw):
        with pytest.raises(ProfileError):
            sanitize_name(raw)

    def test_names_are_length_capped(self):
        assert len(sanitize_name("x" * 500)) == 64


class TestProfileLifecycle:
    def test_save_and_recall(self, profiles):
        profiles.save("H3 Portrait", "minimax_h3", STACK)
        recalled = profiles.get("H3 Portrait")
        assert recalled.model_key == "minimax_h3"
        assert [entry.id for entry in recalled.loras] == [entry.id for entry in STACK]
        assert recalled.loras[0].multiplier == "0.8;1"

    def test_duplicate_names_are_refused_unless_overwriting(self, profiles):
        profiles.save("Stack", "h3", STACK)
        with pytest.raises(ProfileError):
            profiles.save("Stack", "h3", STACK)
        updated = profiles.save("Stack", "h3", STACK[:1], overwrite=True)
        assert len(updated.loras) == 1

    def test_rename(self, profiles):
        profiles.save("Old", "h3", STACK)
        profiles.rename("Old", "New")
        assert profiles.get("Old") is None
        assert profiles.get("New") is not None

    def test_rename_onto_an_existing_name_is_refused(self, profiles):
        profiles.save("One", "h3", STACK)
        profiles.save("Two", "h3", STACK)
        with pytest.raises(ProfileError):
            profiles.rename("One", "Two")

    def test_rename_of_a_missing_profile_is_refused(self, profiles):
        with pytest.raises(ProfileError):
            profiles.rename("Nope", "Something")

    def test_delete(self, profiles):
        profiles.save("Gone", "h3", STACK)
        profiles.delete("Gone")
        assert profiles.names() == []
        with pytest.raises(ProfileError):
            profiles.delete("Gone")

    def test_profiles_survive_a_reload(self, profiles, store):
        profiles.save("Kept", "h3", STACK)
        reloaded = ProfileStore(MetadataStore(store.path))
        assert reloaded.get("Kept").loras[1].id == "sub/bar.safetensors"


class TestModelScope:
    def test_compatible_models(self, profiles):
        profiles.save("H3", "minimax_h3", STACK)
        assert profiles.is_compatible(profiles.get("H3"), "minimax_h3") is True

    def test_incompatible_model_is_flagged(self, profiles):
        profiles.save("H3", "minimax_h3", STACK)
        assert profiles.is_compatible(profiles.get("H3"), "wan22") is False

    def test_an_unscoped_profile_stays_usable(self, profiles):
        profiles.save("Legacy", "", STACK)
        assert profiles.is_compatible(profiles.get("Legacy"), "minimax_h3") is True

    def test_compatible_profiles_are_listed_first(self, profiles):
        profiles.save("zzz other", "wan22", STACK)
        profiles.save("aaa mine", "minimax_h3", STACK)
        assert profiles.names("minimax_h3")[0] == "aaa mine"


class TestExactMatchDetection:
    def test_the_current_stack_matches_its_profile(self, profiles):
        profiles.save("Exact", "h3", STACK)
        assert profiles.match("h3", STACK) == "Exact"

    def test_a_changed_multiplier_no_longer_matches(self, profiles):
        profiles.save("Exact", "h3", STACK)
        diverged = [ProfileEntry(id=STACK[0].id, multiplier="0.4;1"), STACK[1]]
        assert profiles.match("h3", diverged) == ""

    def test_order_matters_because_tokens_are_positional(self, profiles):
        profiles.save("Exact", "h3", STACK)
        assert profiles.match("h3", list(reversed(STACK))) == ""

    def test_a_profile_for_another_model_never_matches(self, profiles):
        profiles.save("Other", "wan22", STACK)
        assert profiles.match("minimax_h3", STACK) == ""
