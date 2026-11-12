"""Built-file ids, the store linked into the LoRA folder, and the build queue."""

import os
import threading

import numpy as np
import pytest

from lora_browser import builder as bd
from lora_browser import derived as dv
from lora_browser import lora_files as lf
from tests.lora_fixtures import lora_pair, read_tensors, write_lora


# -------------------------------------------------------------------- ids


def test_split_ids_round_trip():
    value = dv.split_id("styles/detail.safetensors", "visual")
    assert value == "_lora_browser/styles/detail.visual.safetensors"
    parsed = dv.parse(value)
    assert parsed.kind == "split" and parsed.stem == "styles/detail" and parsed.mode == "visual"
    assert dv.original_id(parsed, ["styles/detail.sft"]) == "styles/detail.sft"
    assert dv.original_id(parsed, []) == "styles/detail.safetensors"
    assert dv.split_id("a.safetensors", "normal") == "a.safetensors"


def test_ids_keep_an_ic_lora_recognisable():
    assert "ic-lora" in dv.split_id("ltx-2-ic-lora-canny.safetensors", "audio")


def test_bake_ids():
    members = [("a.safetensors", "ps"), ("b.safetensors", "p")]
    value = dv.bake_id(members, 32)
    parsed = dv.parse(value)
    assert parsed.kind == "bake" and parsed.rank == 32 and parsed.hash == dv.bake_hash(members)
    assert dv.bake_hash(members) != dv.bake_hash([("a.safetensors", "s"), ("b.safetensors", "p")])
    assert dv.bake_id(members, 16) != value


@pytest.mark.parametrize("value", [
    "a.safetensors", "_lora_browser/a.safetensors", "_lora_browser/_bakes/zz.r3.safetensors",
    "_lora_browser/_bakes/abcdef.r0.safetensors", "other/_lora_browser/a.visual.safetensors",
])
def test_parse_refuses_what_is_not_a_built_file(value):
    assert dv.parse(value) is None


def test_folder_key_is_unique_per_path(tmp_path):
    one, two = tmp_path / "x" / "ltx2", tmp_path / "y" / "ltx2"
    assert dv.folder_key(str(one)).startswith("ltx2-")
    assert dv.folder_key(str(one)) != dv.folder_key(str(two))


# ------------------------------------------------------------------ store


@pytest.fixture
def dirs(tmp_path):
    plugin, loras = tmp_path / "plugin", tmp_path / "loras" / "ltx2"
    plugin.mkdir()
    loras.mkdir(parents=True)
    return str(plugin), str(loras)


def test_store_is_linked_into_the_lora_folder(dirs):
    plugin, loras = dirs
    store = dv.Store(plugin, loras)
    base = store.base()
    assert os.path.realpath(base) == os.path.realpath(store.root)
    assert os.path.islink(os.path.join(loras, dv.LINK_NAME))
    assert store.linked
    target = store.write_path("_lora_browser/x.visual.safetensors")
    os.makedirs(os.path.dirname(target), exist_ok=True)
    open(target, "wb").close()
    # WanGP opens it through the LoRA folder.
    assert store.exists("_lora_browser/x.visual.safetensors")
    assert os.path.isfile(os.path.join(loras, "_lora_browser", "x.visual.safetensors"))


def test_store_falls_back_to_a_real_folder(dirs, monkeypatch):
    plugin, loras = dirs
    monkeypatch.setattr(dv, "_make_link", lambda target, link: False)
    store = dv.Store(plugin, loras)
    assert store.base() == os.path.join(loras, dv.LINK_NAME)
    assert not os.path.islink(store.base()) and not store.linked


def test_store_reuses_a_real_folder_and_replaces_a_dangling_link(dirs, tmp_path):
    plugin, loras = dirs
    link = os.path.join(loras, dv.LINK_NAME)
    os.symlink(str(tmp_path / "gone"), link)
    store = dv.Store(plugin, loras)
    assert os.path.realpath(store.base()) == os.path.realpath(store.root)

    other = os.path.join(str(tmp_path), "other_loras")
    os.makedirs(os.path.join(other, dv.LINK_NAME))
    assert dv.Store(plugin, other).base() == os.path.join(other, dv.LINK_NAME)


def test_store_paths_cannot_escape(dirs):
    plugin, loras = dirs
    store = dv.Store(plugin, loras)
    with pytest.raises(ValueError):
        store.path("styles/a.safetensors")
    with pytest.raises(ValueError):
        store.write_path("_lora_browser/../../escape.visual.safetensors")


def _lora(path, sides=("attn1", "audio_attn1"), rank=2, seed=0):
    rng = np.random.default_rng(seed)
    tensors = []
    for module in sides:
        down, up = lora_pair(rng, 6, 5, rank)
        tensors += [(f"transformer_blocks.0.{module}.to_q.lora_A.weight", "BF16", down),
                    (f"transformer_blocks.0.{module}.to_q.lora_B.weight", "BF16", up)]
    write_lora(path, tensors, {"alpha": str(rank)})


def test_manifest_notices_a_replaced_original(dirs):
    plugin, loras = dirs
    source = os.path.join(loras, "a.safetensors")
    _lora(source)
    store = dv.Store(plugin, loras)
    derived_id = dv.split_id("a.safetensors", "visual")
    job = bd.split_job(store, derived_id, "a.safetensors", source, "visual", "a")
    job.work(lambda fraction: None)
    assert store.ready(derived_id)
    entry = store.entry(derived_id)
    assert entry["kind"] == "split" and entry["sources"][0]["id"] == "a.safetensors" and len(entry["sources"][0]["sha256"]) == 64

    _lora(source, seed=1, rank=3)
    os.utime(source, (12345, 12345))
    assert store.exists(derived_id) and store.stale(derived_id) and not store.ready(derived_id)


def test_files_and_delete(dirs):
    plugin, loras = dirs
    source = os.path.join(loras, "a.safetensors")
    _lora(source)
    store = dv.Store(plugin, loras)
    for mode in ("visual", "audio"):
        bd.split_job(store, dv.split_id("a.safetensors", mode), "a.safetensors", source, mode, "a").work(lambda f: None)
    listed = store.files()
    assert [item["id"] for item in listed] == ["_lora_browser/a.audio.safetensors", "_lora_browser/a.visual.safetensors"]
    assert all(item["bytes"] > 0 and item["kind"] == "split" for item in listed)
    assert store.delete(["_lora_browser/a.audio.safetensors", "not/a/built/file"]) == 1
    assert [item["id"] for item in store.files()] == ["_lora_browser/a.visual.safetensors"]
    assert "_lora_browser/a.audio.safetensors" not in store.manifest()


# ------------------------------------------------------------------- jobs


def test_split_job_keeps_exactly_one_side(dirs):
    plugin, loras = dirs
    source = os.path.join(loras, "sub", "a.safetensors")
    os.makedirs(os.path.dirname(source))
    _lora(source)
    store = dv.Store(plugin, loras)
    derived_id = dv.split_id("sub/a.safetensors", "audio")
    seen = []
    bd.split_job(store, derived_id, "sub/a.safetensors", source, "audio", "a").work(seen.append)
    assert seen[-1] == 1.0
    arrays, metadata, raw = read_tensors(store.path(derived_id))
    assert all("audio_attn1" in name for name in raw)
    assert metadata["lora_browser"] == "audio" and metadata["alpha"] == "2"
    _, _, source_raw = read_tensors(source)
    assert all(raw[name] == source_raw[name] for name in raw)


def test_split_job_refuses_an_empty_side(dirs):
    plugin, loras = dirs
    source = os.path.join(loras, "v.safetensors")
    _lora(source, sides=("attn1",))
    store = dv.Store(plugin, loras)
    job = bd.split_job(store, dv.split_id("v.safetensors", "audio"), "v.safetensors", source, "audio", "v")
    with pytest.raises(ValueError):
        job.work(lambda f: None)


def test_bake_job_records_what_it_kept(dirs):
    plugin, loras = dirs
    a, b = os.path.join(loras, "a.safetensors"), os.path.join(loras, "b.safetensors")
    _lora(a, seed=1)
    _lora(b, seed=2)
    store = dv.Store(plugin, loras)
    derived_id = dv.bake_id([("a.safetensors", "ps"), ("b.safetensors", "ps")], 2)
    members = [("a.safetensors", a, "normal", "A"), ("b.safetensors", b, "normal", "B")]
    bd.bake_job(store, derived_id, members, 2, True, "Group A").work(lambda f: None)
    entry = store.entry(derived_id)
    assert entry["kind"] == "bake" and entry["rank"] == 2 and 0 < entry["kept"] <= 1
    assert [member["id"] for member in entry["members"]] == ["a.safetensors", "b.safetensors"]
    assert lf.read_header(store.path(derived_id)).tensors


# ------------------------------------------------------------------ queue


def test_builder_runs_jobs_in_order_and_reports():
    builder = bd.Builder()
    order = []

    def job(key, fail=False):
        def work(report):
            report(0.5)
            if fail:
                raise RuntimeError(f"{key} broke")
            order.append(key)
        return bd.Job(key=key, label=key, work=work)

    assert builder.submit(job("a"), start=False)
    assert not builder.submit(job("a"), start=False)          # already queued
    assert builder.submit(job("b", fail=True), start=False)
    assert builder.submit(job("c"), start=False)
    assert builder.busy("a") == "queued"
    status = builder.status()
    assert status["running"] and [item["key"] for item in status["queued"]] == ["a", "b", "c"]

    builder.run_pending()
    assert order == ["a", "c"]
    status = builder.status()
    assert not status["running"] and status["finished"] == 2 and status["failed"] == {"b": "b broke"}
    assert builder.busy("b") == "failed" and builder.failure("b") == "b broke"
    # A failure is not retried by asking again, only by retry().
    assert not builder.submit(job("b"), start=False)
    builder.retry("b")
    assert builder.submit(job("b"), start=False)
    builder.cancel("b")
    assert builder.busy("b") == ""


def test_builder_thread_finishes_the_queue():
    builder = bd.Builder()
    done = threading.Event()
    builder.submit(bd.Job(key="x", label="x", work=lambda report: done.set()))
    assert done.wait(5)
    for _ in range(100):
        if not builder.status()["running"]:
            break
        threading.Event().wait(0.02)
    assert builder.status()["finished"] == 1
