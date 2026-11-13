"""LoRA file reading, LTX-2 side sorting and exact partial copies."""

import hashlib
import os

import numpy as np
import pytest

from lora_browser import lora_files as lf
from tests.lora_fixtures import read_tensors, write_lora


def _ltx_lora(path, prefix="diffusion_model.", rank=4, dtype="BF16", metadata=None):
    """Two blocks' worth of an LTX-2 LoRA in one naming style, picture and sound."""
    rng = np.random.default_rng(7)
    tensors = []
    for block in range(2):
        for module in ("attn1.to_q", "ff.net.2", "audio_to_video_attn.to_k",
                       "audio_attn1.to_v", "audio_ff.net.0.proj", "video_to_audio_attn.to_q"):
            name = f"{prefix}transformer_blocks.{block}.{module}"
            tensors.append((f"{name}.lora_A.weight", dtype, rng.standard_normal((rank, 8))))
            tensors.append((f"{name}.lora_B.weight", dtype, rng.standard_normal((8, rank))))
    tensors.append((f"{prefix}audio_patchify_proj.lora_A.weight", dtype, rng.standard_normal((rank, 8))))
    tensors.append((f"{prefix}audio_patchify_proj.lora_B.weight", dtype, rng.standard_normal((8, rank))))
    tensors.append((f"{prefix}patchify_proj.alpha", "F32", np.array(4.0)))
    write_lora(path, tensors, metadata or {"ss_network_dim": str(rank)})
    return [name for name, _, _ in tensors]


# ------------------------------------------------------------------ header


def test_reads_tensor_table_and_metadata(tmp_path):
    path = tmp_path / "a.safetensors"
    names = _ltx_lora(path, metadata={"alpha": "8", "note": "x"})
    lora = lf.read_header(str(path))
    assert [tensor.name for tensor in lora.tensors] == names
    assert lora.metadata == {"alpha": "8", "note": "x"}
    first = lora.tensors[0]
    assert first.dtype == "BF16" and first.shape == (4, 8) and first.nbytes == 4 * 8 * 2
    assert lora.data_start % 8 == 0
    assert lora.tensors[-1].end == os.path.getsize(path)


@pytest.mark.parametrize("content", [b"", b"short", (10**12).to_bytes(8, "little") + b"{}", (4).to_bytes(8, "little") + b"nope"])
def test_refuses_what_is_not_safetensors(tmp_path, content):
    path = tmp_path / "bad.safetensors"
    path.write_bytes(content)
    with pytest.raises(lf.LoraFileError):
        lf.read_header(str(path))


def test_refuses_a_tensor_past_the_end(tmp_path):
    path = tmp_path / "cut.safetensors"
    write_lora(path, [("x.lora_A.weight", "F32", np.zeros((2, 2)))])
    data = path.read_bytes()
    path.write_bytes(data[:-4])
    with pytest.raises(lf.LoraFileError):
        lf.read_header(str(path))


# ------------------------------------------------------------------- keys


@pytest.mark.parametrize("name,expected", [
    ("diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight", ("diffusion_model.transformer_blocks.0.attn1.to_q", "down")),
    ("transformer_blocks.0.attn1.to_q.lora_B.weight", ("transformer_blocks.0.attn1.to_q", "up")),
    ("lora_unet_transformer_blocks_0_attn1_to_q.lora_down.weight", ("lora_unet_transformer_blocks_0_attn1_to_q", "down")),
    ("lora_unet_transformer_blocks_0_attn1_to_q.lora_up.weight", ("lora_unet_transformer_blocks_0_attn1_to_q", "up")),
    ("base_model.model.blocks.0.attn.lora_A.default.weight", ("base_model.model.blocks.0.attn", "down")),
    ("blocks.0.attn.alpha", ("blocks.0.attn", "alpha")),
    ("blocks.0.attn.diff_b", ("blocks.0.attn", "diff_b")),
    ("blocks.0.attn.diff", ("blocks.0.attn", "diff")),
    ("blocks.0.attn.dora_scale", ("blocks.0.attn", "dora")),
    ("blocks.0.attn.lokr_w1", ("blocks.0.attn", "lokr")),
    ("__wangp_h3_lora_source__", ("__wangp_h3_lora_source__", "other")),
])
def test_split_key(name, expected):
    assert lf.split_key(name) == expected


@pytest.mark.parametrize("name,side", [
    ("diffusion_model.transformer_blocks.3.attn1.to_q", lf.PICTURE),
    ("transformer_blocks.3.attn2.to_k", lf.PICTURE),
    ("transformer_blocks.3.ff.net.2", lf.PICTURE),
    ("transformer_blocks.3.audio_to_video_attn.to_v", lf.PICTURE),
    ("transformer_blocks.3.audio_attn1.to_q", lf.SOUND),
    ("transformer_blocks.3.audio_attn2.to_out.0", lf.SOUND),
    ("transformer_blocks.3.audio_ff.net.0.proj", lf.SOUND),
    ("transformer_blocks.3.video_to_audio_attn.to_q", lf.SOUND),
    ("lora_unet_transformer_blocks_3_audio_to_video_attn_to_q", lf.PICTURE),
    ("lora_unet_transformer_blocks_3_video_to_audio_attn_to_q", lf.SOUND),
    ("lora_unet_transformer_blocks_3_audio_attn1_to_q", lf.SOUND),
    ("lora_unet_transformer_blocks_3_attn1_to_q", lf.PICTURE),
    ("patchify_proj", lf.PICTURE),
    ("audio_patchify_proj", lf.SOUND),
    ("proj_out", lf.PICTURE),
    ("audio_proj_out", lf.SOUND),
    ("adaln_single.linear", lf.PICTURE),
    ("audio_adaln_single.linear", lf.SOUND),
    ("av_ca_video_scale_shift_adaln_single.linear", lf.PICTURE),
    ("av_ca_a2v_gate_adaln_single.linear", lf.PICTURE),
    ("av_ca_audio_scale_shift_adaln_single.linear", lf.SOUND),
    ("av_ca_v2a_gate_adaln_single.linear", lf.SOUND),
    ("video_embeddings_connector.transformer_1d_blocks.0.attn1.to_q", lf.PICTURE),
    ("audio_embeddings_connector.transformer_1d_blocks.0.attn1.to_q", lf.SOUND),
    ("feature_extractor_linear.aggregate_embed", lf.SHARED),
])
def test_ltx2_sides(name, side):
    assert lf.ltx2_side(name) == side


def test_keeps():
    assert lf.keeps(lf.PICTURE, "visual") and not lf.keeps(lf.SOUND, "visual")
    assert lf.keeps(lf.SOUND, "audio") and not lf.keeps(lf.PICTURE, "audio")
    assert lf.keeps(lf.SHARED, "visual") and lf.keeps(lf.SHARED, "audio")
    assert all(lf.keeps(side, "normal") for side in lf.SIDES)
    # The sound half sits beside the visual copy, which has the shared layers.
    assert lf.keeps(lf.SOUND, "sound") and not lf.keeps(lf.SHARED, "sound")
    assert not lf.keeps(lf.PICTURE, "sound")


# ---------------------------------------------------------------- summary


def test_summary_counts_each_side(tmp_path):
    path = tmp_path / "a.safetensors"
    _ltx_lora(path)
    summary = lf.summarize(lf.read_header(str(path)))
    # Per block: attn1, ff, audio_to_video on the picture side; audio_attn1,
    # audio_ff, video_to_audio on the sound side.  Plus the patchify alpha.
    assert summary.picture.modules == 2 * 3 + 1
    assert summary.sound.modules == 2 * 3 + 1
    assert summary.picture.tensors == 2 * 3 * 2 + 1
    assert summary.rank == 4
    assert summary.both and summary.has("visual") and summary.has("audio")


def test_summary_of_a_picture_only_lora(tmp_path):
    path = tmp_path / "v.safetensors"
    write_lora(path, [
        ("transformer_blocks.0.attn1.to_q.lora_A.weight", "F16", np.ones((2, 4))),
        ("transformer_blocks.0.attn1.to_q.lora_B.weight", "F16", np.ones((4, 2))),
    ])
    summary = lf.summarize(lf.read_header(str(path)))
    assert summary.has("visual") and not summary.has("audio") and not summary.both


def test_summary_cache_follows_the_file(tmp_path):
    path = tmp_path / "v.safetensors"
    write_lora(path, [("attn1.lora_A.weight", "F32", np.ones((2, 4))), ("attn1.lora_B.weight", "F32", np.ones((4, 2)))])
    cache = lf.SummaryCache()
    first = cache.get(str(path))
    assert first is cache.get(str(path))
    write_lora(path, [("audio_attn1.lora_A.weight", "F32", np.ones((3, 4))), ("audio_attn1.lora_B.weight", "F32", np.ones((4, 3)))])
    os.utime(path, (1, 1))
    second = cache.get(str(path))
    assert second.rank == 3 and second.has("audio") and not second.has("visual")
    assert cache.get(str(tmp_path / "gone.safetensors")) is None


# -------------------------------------------------------------- partial copies


@pytest.mark.parametrize("mode,side", [("visual", lf.PICTURE), ("audio", lf.SOUND)])
def test_copy_keeps_one_side_bit_for_bit(tmp_path, mode, side):
    source = tmp_path / "both.safetensors"
    _ltx_lora(source, metadata={"alpha": "8", "trained": "yes"})
    lora = lf.read_header(str(source))
    keep = [t.name for t in lora.tensors if lf.keeps(lf.ltx2_side(lf.split_key(t.name)[0]), mode)]
    seen = []
    target = tmp_path / "out" / f"both.{mode}.safetensors"
    digest = lf.copy_subset(lora, keep, str(target), dict(lora.metadata, lora_browser=mode),
                            progress=lambda done, total: seen.append((done, total)))

    assert digest == hashlib.sha256(source.read_bytes()).hexdigest()
    assert seen and seen[-1] == (lora.size, lora.size)
    assert not (tmp_path / "out" / f"both.{mode}.safetensors.part").exists()

    _, source_meta, source_raw = read_tensors(source)
    arrays, metadata, raw = read_tensors(target)
    assert list(raw) == keep
    assert all(raw[name] == source_raw[name] for name in keep)
    assert all(lf.ltx2_side(lf.split_key(name)[0]) == side for name in keep)
    assert metadata == dict(source_meta, lora_browser=mode)
    copied = lf.read_header(str(target))
    assert copied.data_start % 8 == 0


def test_copy_of_nothing_is_still_a_valid_file(tmp_path):
    source = tmp_path / "v.safetensors"
    write_lora(source, [("attn1.lora_A.weight", "F32", np.ones((2, 4)))])
    target = tmp_path / "empty.safetensors"
    lf.copy_subset(lf.read_header(str(source)), [], str(target), {})
    assert lf.read_header(str(target)).tensors == []


def test_stream_writer_refuses_a_tensor_out_of_layout(tmp_path):
    path = tmp_path / "w.safetensors"
    writer = lf.StreamWriter(str(path), [("a", "F32", (2,)), ("b", "F32", (1,))], {})
    writer.write("a", np.zeros(2, "<f4").tobytes())
    with pytest.raises(lf.LoraFileError):
        writer.write("c", np.zeros(1, "<f4").tobytes())
    assert not path.exists() and not (tmp_path / "w.safetensors.part").exists()


def test_stream_writer_refuses_a_missing_tensor(tmp_path):
    path = tmp_path / "w.safetensors"
    writer = lf.StreamWriter(str(path), [("a", "F32", (2,)), ("b", "F32", (1,))], {})
    writer.write("a", np.zeros(2, "<f4").tobytes())
    with pytest.raises(lf.LoraFileError):
        writer.close()
    assert not path.exists()
