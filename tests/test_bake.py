"""Baking a group of LoRAs: exact stacking and optimal low-rank re-fits."""

import numpy as np
import pytest

from lora_browser import bake as bk
from lora_browser import lora_files as lf
from tests.lora_fixtures import effective_delta, lora_pair, read_tensors, write_lora

OUT, IN = 24, 16


def _module_tensors(name, down, up, dtype="F32", alpha=None, style="AB"):
    a, b = ("lora_A", "lora_B") if style == "AB" else ("lora_down", "lora_up")
    tensors = [(f"{name}.{a}.weight", dtype, down), (f"{name}.{b}.weight", dtype, up)]
    if alpha is not None:
        tensors.append((f"{name}.alpha", "F32", np.array(alpha)))
    return tensors


def _dense(path):
    return effective_delta(str(path), canonical=bk.canonical)


def _sum(*deltas):
    total = {}
    for delta in deltas:
        for key, value in delta.items():
            total[key] = total.get(key, 0) + value
    return total


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


def _two_members(tmp_path, rng, alpha_a=8.0, alpha_b=None, header_b=None, dtype="F32"):
    a, b = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    tensors_a, tensors_b = [], []
    for module in ("blocks.0.attn1.to_q", "blocks.0.ff.net.2"):
        tensors_a += _module_tensors(f"diffusion_model.{module}", *lora_pair(rng, OUT, IN, 4), dtype, alpha_a)
        tensors_b += _module_tensors(f"diffusion_model.{module}", *lora_pair(rng, OUT, IN, 8), dtype, alpha_b)
    # A module only the second LoRA changes.
    tensors_b += _module_tensors("diffusion_model.blocks.1.attn2.to_k", *lora_pair(rng, OUT, IN, 8), dtype, alpha_b)
    write_lora(a, tensors_a)
    write_lora(b, tensors_b, {"alpha": header_b} if header_b is not None else None)
    return a, b


def test_canonical_names_meet_across_styles():
    dotted = bk.canonical("diffusion_model.transformer_blocks.0.attn1.to_q")
    assert dotted == bk.canonical("lora_unet_transformer_blocks_0_attn1_to_q")
    assert dotted == bk.canonical("transformer.transformer_blocks.0.attn1.to_q")
    assert dotted == bk.canonical("model.diffusion_model.transformer_blocks.0.attn1.to_q")
    assert bk.canonical("blocks.0.attn1.to_out.0") == bk.canonical("blocks.0.attn1.to_out")


def test_plan_ranks(tmp_path, rng):
    a, b = _two_members(tmp_path, rng)
    plan = bk.plan([bk.Member(str(a)), bk.Member(str(b))], sided=False)
    assert plan.member_ranks == [4, 8]
    assert plan.default_rank == 8
    assert plan.combined_rank == 12
    assert len(plan.modules) == 3


def test_full_rank_bake_is_the_exact_sum(tmp_path, rng):
    a, b = _two_members(tmp_path, rng, alpha_a=8.0, alpha_b=None, header_b="4")
    plan = bk.plan([bk.Member(str(a)), bk.Member(str(b))], sided=False)
    out = tmp_path / "baked.safetensors"
    result = bk.bake(plan, plan.combined_rank, str(out), {"lora_browser_group": "A"})
    assert result.exact and result.kept == 1.0

    expected = _sum(_dense(a), _dense(b))
    got = _dense(out)
    assert set(got) == set(expected)
    for key in expected:
        np.testing.assert_allclose(got[key], expected[key], rtol=1e-5, atol=1e-6)
    _, metadata, _ = read_tensors(out)
    assert metadata["lora_browser"] == "bake" and metadata["lora_browser_group"] == "A"


def test_full_rank_with_one_scale_copies_the_factors_untouched(tmp_path, rng):
    a, b = _two_members(tmp_path, rng, alpha_a=4.0, alpha_b=8.0)     # both scale 1
    plan = bk.plan([bk.Member(str(a)), bk.Member(str(b))], sided=False)
    out = tmp_path / "baked.safetensors"
    bk.bake(plan, 64, str(out))
    arrays_a, _, _ = read_tensors(a)
    arrays, _, _ = read_tensors(out)
    name = "diffusion_model.blocks.0.attn1.to_q"
    assert np.array_equal(arrays[name + ".lora_A.weight"][:4], arrays_a[name + ".lora_A.weight"])
    assert np.array_equal(arrays[name + ".lora_B.weight"][:, :4], arrays_a[name + ".lora_B.weight"])


def test_reduced_rank_is_the_best_approximation(tmp_path, rng):
    a, b = _two_members(tmp_path, rng)
    plan = bk.plan([bk.Member(str(a)), bk.Member(str(b))], sided=False)
    out = tmp_path / "baked.safetensors"
    rank = 6
    result = bk.bake(plan, rank, str(out))
    assert not result.exact

    expected = _sum(_dense(a), _dense(b))
    got = _dense(out)
    total = kept = 0.0
    for key, dense in expected.items():
        singular = np.linalg.svd(dense, compute_uv=False)
        total += float(np.sum(singular ** 2))
        kept += float(np.sum(singular[:rank] ** 2))
        # Eckart-Young: the error of the best rank-k fit is the tail of the spectrum.
        error = np.linalg.norm(got[key] - dense)
        assert error == pytest.approx(np.sqrt(np.sum(singular[rank:] ** 2)), rel=1e-4, abs=1e-6)
        assert np.linalg.matrix_rank(got[key], tol=1e-6) <= rank
    assert result.kept == pytest.approx(kept / total, rel=1e-6)


def test_members_in_different_styles_are_combined(tmp_path, rng):
    a, b = tmp_path / "comfy.safetensors", tmp_path / "kohya.safetensors"
    down_a, up_a = lora_pair(rng, OUT, IN, 4)
    down_b, up_b = lora_pair(rng, OUT, IN, 4)
    write_lora(a, _module_tensors("diffusion_model.transformer_blocks.0.attn1.to_q", down_a, up_a))
    write_lora(b, _module_tensors("lora_unet_transformer_blocks_0_attn1_to_q", down_b, up_b, style="downup"))
    plan = bk.plan([bk.Member(str(a)), bk.Member(str(b))], sided=False)
    assert len(plan.modules) == 1
    out = tmp_path / "baked.safetensors"
    bk.bake(plan, 8, str(out))
    arrays, _, _ = read_tensors(out)
    # Written the way the first LoRA spells it.
    assert "diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight" in arrays
    np.testing.assert_allclose(
        list(_dense(out).values())[0], up_a @ down_a + up_b @ down_b, rtol=1e-5, atol=1e-6
    )


def test_a_member_set_to_one_side_brings_only_that_side(tmp_path, rng):
    a, b = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    tensors = []
    for module in ("transformer_blocks.0.attn1.to_q", "transformer_blocks.0.audio_attn1.to_q"):
        tensors += _module_tensors(module, *lora_pair(rng, OUT, IN, 4))
    write_lora(a, tensors)
    write_lora(b, _module_tensors("transformer_blocks.0.attn1.to_q", *lora_pair(rng, OUT, IN, 4)))
    plan = bk.plan([bk.Member(str(a), mode="visual"), bk.Member(str(b))], sided=True)
    assert [module.name for module in plan.modules] == ["transformer_blocks.0.attn1.to_q"]

    # Two picture-only LoRAs set to audio only bring nothing to bake.
    with pytest.raises(bk.BakeError, match="change nothing"):
        bk.plan([bk.Member(str(b), mode="audio"), bk.Member(str(b), mode="audio")], sided=True)


def test_bias_deltas_ride_on_the_same_scale(tmp_path, rng):
    a, b = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    name = "blocks.0.attn1.to_q"
    bias_a, bias_b = rng.standard_normal(OUT).astype(np.float32), rng.standard_normal(OUT).astype(np.float32)
    write_lora(a, _module_tensors(name, *lora_pair(rng, OUT, IN, 4), alpha=8.0) + [(f"{name}.diff_b", "F32", bias_a)])
    write_lora(b, _module_tensors(name, *lora_pair(rng, OUT, IN, 4)) + [(f"{name}.diff_b", "F32", bias_b)])
    plan = bk.plan([bk.Member(str(a)), bk.Member(str(b))], sided=False)
    out = tmp_path / "baked.safetensors"
    bk.bake(plan, 3, str(out))
    arrays, _, _ = read_tensors(out)
    rank = arrays[f"{name}.lora_A.weight"].shape[0]
    scale = float(arrays[f"{name}.alpha"]) / rank
    # mmgp adds diff_b at the module's scale: 2 * bias_a (alpha 8 / rank 4) + bias_b.
    np.testing.assert_allclose(arrays[f"{name}.diff_b"] * scale, 2 * bias_a + bias_b, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dtype", ["BF16", "F16"])
def test_keeps_the_members_precision(tmp_path, rng, dtype):
    a, b = _two_members(tmp_path, rng, dtype=dtype)
    plan = bk.plan([bk.Member(str(a)), bk.Member(str(b))], sided=False)
    assert plan.dtype == dtype
    out = tmp_path / "baked.safetensors"
    bk.bake(plan, plan.combined_rank, str(out))
    lora = lf.read_header(str(out))
    factors = [t for t in lora.tensors if t.name.endswith(".weight")]
    assert factors and all(t.dtype == dtype for t in factors)
    expected = _sum(_dense(a), _dense(b))
    got = _dense(out)
    for key in expected:
        np.testing.assert_allclose(got[key], expected[key], rtol=2e-2, atol=2e-3)


@pytest.mark.parametrize("extra,message", [
    (("blocks.0.attn1.to_q.dora_scale", np.ones((1, OUT))), "DORA"),
    (("blocks.0.attn1.to_q.diff", np.ones((OUT, IN))), "directly"),
    (("__wangp_h3_lora_source__", np.ones(2)), "not part of a LoRA"),
])
def test_refuses_what_it_cannot_combine(tmp_path, rng, extra, message):
    a, b = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    write_lora(a, _module_tensors("blocks.0.attn1.to_q", *lora_pair(rng, OUT, IN, 4)) + [(extra[0], "F32", extra[1])])
    write_lora(b, _module_tensors("blocks.0.attn1.to_q", *lora_pair(rng, OUT, IN, 4)))
    with pytest.raises(bk.BakeError, match=message):
        bk.plan([bk.Member(str(a), label="A"), bk.Member(str(b))], sided=False)


def test_refuses_a_group_of_one(tmp_path, rng):
    a = tmp_path / "a.safetensors"
    write_lora(a, _module_tensors("blocks.0.attn1.to_q", *lora_pair(rng, OUT, IN, 4)))
    with pytest.raises(bk.BakeError):
        bk.plan([bk.Member(str(a))], sided=False)


def test_size_estimate_matches_the_file(tmp_path, rng):
    a, b = _two_members(tmp_path, rng, dtype="BF16")
    plan = bk.plan([bk.Member(str(a)), bk.Member(str(b))], sided=False)
    out = tmp_path / "baked.safetensors"
    result = bk.bake(plan, 6, str(out))
    lora = lf.read_header(str(out))
    assert result.bytes == sum(tensor.nbytes for tensor in lora.tensors)
    assert plan.size_for(6) < plan.size_for(plan.combined_rank)


def test_a_failed_bake_leaves_nothing(tmp_path, rng, monkeypatch):
    a, b = _two_members(tmp_path, rng)
    plan = bk.plan([bk.Member(str(a)), bk.Member(str(b))], sided=False)
    out = tmp_path / "baked.safetensors"

    def boom(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(bk, "_fit", boom)
    with pytest.raises(RuntimeError):
        bk.bake(plan, 4, str(out))
    assert not out.exists() and not (tmp_path / "baked.safetensors.part").exists()
