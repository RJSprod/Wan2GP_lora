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


# ---------------------------------------------------------------- weights


def _scaled(delta, factor):
    return {key: value * factor for key, value in delta.items()}


def test_weighted_members_bake_in_at_their_strength(tmp_path, rng):
    """A shrink bakes each LoRA in at the strength it ran at: at full rank the
    file is exactly their weighted sum -- a negative weight included."""
    a, b = _two_members(tmp_path, rng, alpha_a=8.0, alpha_b=None, header_b="4")
    plan = bk.plan([bk.Member(str(a), weight=0.7), bk.Member(str(b), weight=-0.5)], sided=False)
    out = tmp_path / "shrunk.safetensors"
    result = bk.bake(plan, plan.combined_rank, str(out))
    assert result.exact
    expected = _sum(_scaled(_dense(a), 0.7), _scaled(_dense(b), -0.5))
    got = _dense(out)
    assert set(got) == set(expected)
    for key in expected:
        np.testing.assert_allclose(got[key], expected[key], rtol=1e-5, atol=1e-6)


def test_weighted_members_at_a_lower_rank_fit_their_weighted_sum(tmp_path, rng):
    a, b = _two_members(tmp_path, rng)
    plan = bk.plan([bk.Member(str(a), weight=0.25), bk.Member(str(b), weight=1.5)], sided=False)
    out = tmp_path / "shrunk.safetensors"
    rank = 6
    bk.bake(plan, rank, str(out))
    expected = _sum(_scaled(_dense(a), 0.25), _scaled(_dense(b), 1.5))
    got = _dense(out)
    for key, dense in expected.items():
        singular = np.linalg.svd(dense, compute_uv=False)
        error = np.linalg.norm(got[key] - dense)
        assert error == pytest.approx(np.sqrt(np.sum(singular[rank:] ** 2)), rel=1e-4, abs=1e-6)


def test_a_weighted_member_brings_its_bias_at_its_weight(tmp_path, rng):
    a, b = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    name = "blocks.0.attn1.to_q"
    bias_a, bias_b = rng.standard_normal(OUT).astype(np.float32), rng.standard_normal(OUT).astype(np.float32)
    write_lora(a, _module_tensors(name, *lora_pair(rng, OUT, IN, 4), alpha=8.0) + [(f"{name}.diff_b", "F32", bias_a)])
    write_lora(b, _module_tensors(name, *lora_pair(rng, OUT, IN, 4)) + [(f"{name}.diff_b", "F32", bias_b)])
    plan = bk.plan([bk.Member(str(a), weight=0.5), bk.Member(str(b), weight=0.2)], sided=False)
    out = tmp_path / "shrunk.safetensors"
    bk.bake(plan, 3, str(out))
    arrays, _, _ = read_tensors(out)
    rank = arrays[f"{name}.lora_A.weight"].shape[0]
    scale = float(arrays[f"{name}.alpha"]) / rank
    np.testing.assert_allclose(arrays[f"{name}.diff_b"] * scale, 0.5 * 2 * bias_a + 0.2 * bias_b, rtol=1e-5, atol=1e-5)


def test_check_says_why_one_lora_cannot_be_baked(tmp_path, rng):
    good, dora = tmp_path / "good.safetensors", tmp_path / "dora.safetensors"
    write_lora(good, _module_tensors("blocks.0.attn1.to_q", *lora_pair(rng, OUT, IN, 4)))
    write_lora(dora, _module_tensors("blocks.0.attn1.to_q", *lora_pair(rng, OUT, IN, 4))
               + [("blocks.0.attn1.to_q.dora_scale", "F32", np.ones((1, OUT)))])
    assert bk.check(bk.Member(str(good)), sided=False) == ""
    assert "DORA" in bk.check(bk.Member(str(dora), label="Dora"), sided=False)
    assert "Dora" in bk.check(bk.Member(str(dora), label="Dora"), sided=False)
    assert bk.check(bk.Member(str(tmp_path / "gone.safetensors"), label="Gone"), sided=False).startswith("Gone")


# ---------------------------------------------------- compression levels


def _uneven_members(tmp_path, rng):
    """Two LoRAs whose change is strong in one layer and faint in another,
    with a third layer only one of them changes."""
    a, b = tmp_path / "ua.safetensors", tmp_path / "ub.safetensors"
    tensors_a, tensors_b = [], []
    for module, scale in (("blocks.0.attn1.to_q", 1.0), ("blocks.0.ff.net.2", 0.02)):
        tensors_a += _module_tensors(f"diffusion_model.{module}", *lora_pair(rng, OUT, IN, 8, scale), "F32", 8.0)
        tensors_b += _module_tensors(f"diffusion_model.{module}", *lora_pair(rng, OUT, IN, 8, scale), "F32", 8.0)
    tensors_b += _module_tensors("diffusion_model.blocks.1.attn2.to_k", *lora_pair(rng, OUT, IN, 4, 0.3), "F32", 4.0)
    write_lora(a, tensors_a)
    write_lora(b, tensors_b)
    return a, b


def _layer_kept(spectrum, ranks):
    return [float(np.sum(values[:spectrum.capped(index, k)])) / float(np.sum(values))
            for index, (values, k) in enumerate(zip(spectrum.energies, ranks)) if np.sum(values) > 0]


def test_measuring_finds_each_layers_singular_values(tmp_path, rng):
    a, b = _two_members(tmp_path, rng, alpha_a=8.0, alpha_b=None, header_b="4")
    plan = bk.plan([bk.Member(str(a), weight=0.7), bk.Member(str(b), weight=-1.5)], sided=False)
    spectrum = bk.analyze(plan)
    expected = _sum(_scaled(_dense(a), 0.7), _scaled(_dense(b), -1.5))
    assert set(spectrum.keys) == set(expected)
    for key, values in zip(spectrum.keys, spectrum.energies):
        singular = np.linalg.svd(expected[key], compute_uv=False)
        np.testing.assert_allclose(values, (singular ** 2)[:len(values)], rtol=1e-4, atol=1e-9)
        assert np.sum(values) == pytest.approx(np.sum(singular ** 2), rel=1e-5)


def test_a_level_keeps_its_share_and_no_layer_less_than_its_floor(tmp_path, rng):
    a, b = _uneven_members(tmp_path, rng)
    spectrum = bk.analyze(bk.plan([bk.Member(str(a)), bk.Member(str(b))], sided=False))
    for _, _, quality in bk.LEVELS:
        kept = quality / 1000
        ranks = bk.allocate(spectrum, kept)
        assert spectrum.kept(ranks) >= kept - 1e-9
        assert min(_layer_kept(spectrum, ranks)) >= bk.floor_for(kept) - 1e-9
        # The fewest directions that do it: one less anywhere falls short of
        # the share, or of that layer's floor.
        for index, k in enumerate(ranks):
            if k == 0:
                continue
            fewer = list(ranks)
            fewer[index] -= 1
            short = spectrum.kept(fewer) < kept - 1e-12
            values = spectrum.energies[index]
            starved = np.sum(values[:fewer[index]]) < bk.floor_for(kept) * np.sum(values) * (1 - 1e-9)
            assert short or starved


def test_a_rank_per_layer_is_smaller_than_one_rank_for_the_same_share(tmp_path, rng):
    a, b = _uneven_members(tmp_path, rng)
    spectrum = bk.analyze(bk.plan([bk.Member(str(a)), bk.Member(str(b))], sided=False))
    for kept in (0.95, 0.99, 0.999):
        chosen = bk.allocate(spectrum, kept)
        # The smallest single rank that keeps as much.
        uniform = next(spectrum.uniform(rank) for rank in range(1, 33)
                       if spectrum.kept(spectrum.uniform(rank)) >= spectrum.kept(chosen) - 1e-12)
        assert spectrum.size(chosen) < spectrum.size(uniform)
    # The faint layer is the one that gives way.
    ranks = dict(zip(spectrum.keys, bk.allocate(spectrum, 0.99)))
    strong, faint = (key for key in spectrum.keys if key.endswith(("attn1_to_q", "ff_net_2")))
    assert ranks[faint] < ranks[strong]


def test_a_rank_per_layer_bakes_exactly_what_was_predicted(tmp_path, rng):
    a, b = _uneven_members(tmp_path, rng)
    plan = bk.plan([bk.Member(str(a), weight=0.8), bk.Member(str(b), weight=0.6)], sided=False)
    spectrum = bk.analyze(plan)
    ranks = bk.allocate(spectrum, 0.99)
    out = tmp_path / "levelled.safetensors"
    result = bk.bake(plan, 0, str(out), ranks=spectrum.ranks_by_key(ranks))
    assert result.kept == pytest.approx(spectrum.kept(ranks), rel=1e-6)
    assert result.bytes == spectrum.size(ranks) == sum(tensor.nbytes for tensor in lf.read_header(str(out)).tensors)
    assert result.rank == max(spectrum.capped(index, k) for index, k in enumerate(ranks))
    expected = _sum(_scaled(_dense(a), 0.8), _scaled(_dense(b), 0.6))
    got = _dense(out)
    for key, k in spectrum.ranks_by_key(ranks).items():
        singular = np.linalg.svd(expected[key], compute_uv=False)
        error = np.linalg.norm(got[key] - expected[key])
        assert error == pytest.approx(np.sqrt(np.sum(singular[k:] ** 2)), rel=1e-3, abs=1e-6)
        # Each layer has its own rank in the file.
        assert np.linalg.matrix_rank(got[key], tol=1e-6) <= k


def test_predictions_give_each_level_beside_the_old_size_and_now(tmp_path, rng):
    a, b = _uneven_members(tmp_path, rng)
    plan = bk.plan([bk.Member(str(a)), bk.Member(str(b))], sided=False)
    spectrum = bk.analyze(plan)
    found = bk.predictions([spectrum, spectrum])
    assert [level["key"] for level in found["levels"]] == ["low", "medium", "high"]
    sizes = [level["bytes"] for level in found["levels"]]
    assert sizes == sorted(sizes, reverse=True)
    for level in found["levels"]:
        assert level["kept"] >= level["quality"] / 1000 - 1e-9
    # Matching the highest-rank member, as a bake was sized before.
    assert found["matched"]["rank"] == plan.default_rank == 8
    assert found["matched"]["bytes"] == 2 * plan.size_for(8)
    assert found["matched"]["kept"] == pytest.approx(spectrum.kept(spectrum.uniform(8)))
    assert found["exact"]["bytes"] == 2 * plan.size_for(plan.combined_rank)


def test_a_measurement_is_reused_only_for_the_same_files_and_strengths(tmp_path, rng):
    a, b = _two_members(tmp_path, rng)
    key = bk.measure_key([bk.Member(str(a)), bk.Member(str(b))], sided=False)
    assert key == bk.measure_key([bk.Member(str(a)), bk.Member(str(b))], sided=False)
    others = [
        bk.measure_key([bk.Member(str(a), weight=0.5), bk.Member(str(b))], sided=False),
        bk.measure_key([bk.Member(str(a), mode="visual"), bk.Member(str(b))], sided=False),
        bk.measure_key([bk.Member(str(b)), bk.Member(str(a))], sided=False),
        bk.measure_key([bk.Member(str(a)), bk.Member(str(b))], sided=True),
    ]
    assert key not in others and len(set(others)) == len(others)
    spectrum = bk.analyze(bk.plan([bk.Member(str(a)), bk.Member(str(b))], sided=False))
    bk.remember(key, spectrum)
    assert bk.recall(key) is spectrum
    # The file changing on disk is a new measurement.
    write_lora(a, _module_tensors("diffusion_model.blocks.0.attn1.to_q", *lora_pair(rng, OUT, IN, 4)))
    assert bk.measure_key([bk.Member(str(a)), bk.Member(str(b))], sided=False) != key
