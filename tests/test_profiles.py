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


ALL_IDS = [entry.id for entry in STACK]


class TestAvailability:
    """The model a profile was saved under never blocks it; only its LoRAs do."""

    def test_the_saving_model_is_kept_only_as_provenance(self, profiles):
        saved = profiles.save("H3", "minimax_h3", STACK)
        assert saved.model_key == "minimax_h3"
        # Recorded, never consulted: availability alone decides.
        assert profiles.partition(ALL_IDS) == (["H3"], [])

    def test_a_profile_from_another_model_is_available_when_its_loras_are(self, profiles):
        # A sibling model sharing the same LoRA folder sees the same files.
        profiles.save("H3", "minimax_h3", STACK)
        assert profiles.missing_ids(profiles.get("H3"), ALL_IDS) == []

    def test_absent_loras_are_reported(self, profiles):
        profiles.save("H3", "minimax_h3", STACK)
        assert profiles.missing_ids(profiles.get("H3"), ["foo.safetensors"]) == [
            "sub/bar.safetensors"
        ]

    def test_fully_available_profiles_are_listed_first(self, profiles):
        profiles.save("zzz complete", "wan22", STACK)
        profiles.save("aaa partial", "minimax_h3", STACK + [ProfileEntry(id="ghost.safetensors")])
        assert profiles.names(ALL_IDS) == ["zzz complete", "aaa partial"]

    def test_partition_names_the_incomplete_profiles(self, profiles):
        profiles.save("Complete", "wan22", STACK)
        profiles.save("Partial", "wan22", [ProfileEntry(id="ghost.safetensors")])
        assert profiles.partition(ALL_IDS) == (["Complete"], ["Partial"])

    def test_an_unknown_inventory_calls_nothing_incomplete(self, profiles):
        profiles.save("Whatever", "wan22", STACK)
        assert profiles.partition() == (["Whatever"], [])


class TestExactMatchDetection:
    def test_the_current_stack_matches_its_profile(self, profiles):
        profiles.save("Exact", "h3", STACK)
        assert profiles.match(STACK) == "Exact"

    def test_a_changed_multiplier_no_longer_matches(self, profiles):
        profiles.save("Exact", "h3", STACK)
        diverged = [ProfileEntry(id=STACK[0].id, multiplier="0.4;1"), STACK[1]]
        assert profiles.match(diverged) == ""

    def test_order_matters_because_tokens_are_positional(self, profiles):
        profiles.save("Exact", "h3", STACK)
        assert profiles.match(list(reversed(STACK))) == ""

    def test_a_profile_saved_under_another_model_still_matches(self, profiles):
        # The stack is on screen, so its LoRAs are available by definition.
        profiles.save("Other", "wan22", STACK)
        assert profiles.match(STACK) == "Other"
