"""Bake a group of LoRAs into one LoRA, re-fitted to a chosen rank.

WanGP already runs every LoRA active on a layer as one product -- mmgp joins
their factors at load time (``_prepare_joined_loras``) -- so merging LoRAs
without loss saves no memory and no time.  What does save memory is fitting
their combined change into fewer ranks, and that is what a bake is.

For each module the members change, mmgp applies

    dW = sum_i s_i * B_i @ A_i,   s_i = alpha_i / rank_i  (1 with no alpha)

which is the product of U = [s_1 B_1 | s_2 B_2 | ...] (out x R) and
V = [A_1; A_2; ...] (R x in).  Two thin QR factorisations reduce that to an
R x R core whose SVD gives dW's singular values without ever forming dW:
U = Qu Ru, V^T = Qv Rv, Ru Rv^T = P S Q^T, so dW = (Qu P) S (Qv Q)^T.  The k
largest singular values give the best rank-k approximation there is
(Eckart-Young), and the share of sum(S^2) they hold -- over every module -- is
reported as how much of the group's combined change the bake keeps.

A rank at or above what the members add up to keeps everything, and is built
by stacking the factors as they are.  The members' strength and schedule are
not baked in: the group's multiplier applies to the baked file at run time,
exactly as it applied to each member.

numpy is the only dependency, and WanGP already ships it.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from . import lora_files as lf

#: The prefixes WanGP's loaders strip before matching a key to a module
#: (LTX-2's ``preprocess_loras`` strips the first three).  Members written in
#: different styles name the same module once these are gone and ``.`` and
#: ``_`` are treated alike -- kohya's names are dotted names with every dot
#: replaced by an underscore.
_PREFIXES = ("model.diffusion_model.", "diffusion_model.", "model.", "transformer.", "base_model.model.", "lora_unet_")

#: Readable dtypes.  BF16 is decoded by hand: numpy has no bfloat16.
_READ = {"F32": "<f4", "F16": "<f2", "F64": "<f8"}
_WRITE = ("F32", "F16", "BF16")


class BakeError(ValueError):
    """The group cannot be baked; the message says which LoRA and why."""


def canonical(module: str) -> str:
    """One name per module whatever style a file spells it in."""
    name = str(module)
    stripped = True
    while stripped:
        stripped = False
        for prefix in _PREFIXES:
            if name.startswith(prefix):
                name = name[len(prefix):]
                stripped = True
    name = name.replace(".", "_")
    # Some exports name the attention output projection's Linear, some its
    # Sequential; WanGP resolves both to the same layer.
    return re.sub(r"_to_out_0$", "_to_out", name)


@dataclass
class Member:
    """One LoRA in the group, and which of its sides the bake takes."""

    path: str
    #: ``normal`` takes everything; ``visual``/``audio`` take one LTX-2 side.
    mode: str = "normal"
    label: str = ""


@dataclass
class _Part:
    """One member's contribution to one module."""

    member: int
    down: lf.TensorInfo | None = None
    up: lf.TensorInfo | None = None
    diff_b: lf.TensorInfo | None = None
    alpha: float | None = None

    @property
    def rank(self) -> int:
        return int(self.down.shape[0]) if self.down is not None else 0


@dataclass
class _Module:
    key: str                      # canonical name
    name: str                     # the first member's own spelling
    style: str                    # "AB" or "downup", from that member
    parts: list[_Part] = field(default_factory=list)
    out_dim: int = 0
    in_dim: int = 0

    @property
    def total_rank(self) -> int:
        return sum(part.rank for part in self.parts)

    @property
    def factored(self) -> list[_Part]:
        return [part for part in self.parts if part.down is not None]


@dataclass
class Plan:
    """What a bake will combine, read from the members' headers only."""

    members: list[Member]
    files: list[lf.LoraFile]
    modules: list[_Module]
    dtype: str
    #: Each member's own rank (its most common one), in member order.
    member_ranks: list[int]

    @property
    def combined_rank(self) -> int:
        """The rank that keeps everything: the most any module adds up to."""
        return max((module.total_rank for module in self.modules), default=0)

    @property
    def default_rank(self) -> int:
        """The rank of the highest-rank LoRA in the group."""
        return max(self.member_ranks, default=0)

    def output_rank(self, module: _Module, rank: int) -> int:
        total = module.total_rank
        if not total:
            return 0
        if rank >= total:
            return total
        return max(1, min(rank, module.out_dim, module.in_dim))

    def size_for(self, rank: int) -> int:
        """Bytes of tensor data a bake at ``rank`` writes."""
        width = 4 if self.dtype == "F32" else 2
        size = 0
        for module in self.modules:
            k = self.output_rank(module, rank)
            size += k * (module.out_dim + module.in_dim) * width + (4 if k else 0)
            if any(part.diff_b is not None for part in module.parts):
                size += module.out_dim * width
        return size


@dataclass
class Result:
    rank: int
    #: Share of the combined change kept, 0..1; 1 when nothing was cut.
    kept: float
    exact: bool
    modules: int
    bytes: int


def plan(members: list[Member], *, sided: bool) -> Plan:
    """Read every member's header and line their modules up.

    ``sided`` is LTX-2: a member set to visual or audio only brings only that
    side.  Anything a bake cannot combine is refused here, before a byte is
    written: DoRA and LoKr adapters, dense weight deltas, and tensors that are
    not LoRA factors at all (WanGP's own conversion data, for one).
    """
    if len(members) < 2:
        raise BakeError("A bake needs at least two LoRAs.")
    files, modules, by_key, ranks = [], [], {}, []
    dtypes: Counter = Counter()
    for index, member in enumerate(members):
        label = member.label or member.path
        try:
            lora = lf.read_header(member.path)
        except lf.LoraFileError as error:
            raise BakeError(f"{label}: {error}") from error
        files.append(lora)
        header_alpha = _header_alpha(lora.metadata)
        alphas: dict[str, float] = {}
        parts: dict[str, _Part] = {}
        member_ranks: Counter = Counter()
        with open(lora.path, "rb") as handle:
            alphas = _alphas(lora, handle, label, sided, member.mode)
        for tensor in lora.tensors:
            module_name, role = lf.split_key(tensor.name)
            if sided and not lf.keeps(lf.ltx2_side(module_name), member.mode):
                continue
            if role in ("dora", "lokr"):
                raise BakeError(f"{label} is a {role.upper() if role == 'dora' else 'LoKr'} adapter; only plain LoRAs can be baked.")
            if role == "diff":
                raise BakeError(f"{label} changes '{module_name}' directly rather than as a LoRA, which a bake cannot combine.")
            if role == "other":
                raise BakeError(f"{label} carries '{tensor.name}', which is not part of a LoRA; it cannot be baked.")
            if role == "alpha":
                continue
            if tensor.dtype not in _READ and tensor.dtype != "BF16":
                raise BakeError(f"{label} stores '{tensor.name}' as {tensor.dtype}, which a bake cannot read.")
            part = parts.setdefault(module_name, _Part(member=index))
            if role == "down":
                if len(tensor.shape) != 2:
                    raise BakeError(f"{label}: '{tensor.name}' is not a 2-D factor; only linear layers can be baked.")
                part.down = tensor
                member_ranks[int(tensor.shape[0])] += 1
                dtypes[tensor.dtype] += 1
            elif role == "up":
                if len(tensor.shape) != 2:
                    raise BakeError(f"{label}: '{tensor.name}' is not a 2-D factor; only linear layers can be baked.")
                part.up = tensor
            elif role == "diff_b":
                part.diff_b = tensor
        ranks.append(member_ranks.most_common(1)[0][0] if member_ranks else 0)

        for module_name, part in parts.items():
            if (part.down is None) != (part.up is None):
                raise BakeError(f"{label}: '{module_name}' has only one of its two factors.")
            if part.down is not None:
                if part.up.shape[1] != part.down.shape[0]:
                    raise BakeError(f"{label}: '{module_name}' has factors of different ranks.")
                part.alpha = alphas.get(module_name, header_alpha)
            key = canonical(module_name)
            module = by_key.get(key)
            if module is None:
                style = "downup" if part.down is not None and "lora_down" in part.down.name else "AB"
                module = _Module(key=key, name=module_name, style=style)
                by_key[key] = module
                modules.append(module)
            if any(existing.member == index for existing in module.parts):
                raise BakeError(f"{label} names '{module_name}' twice.")
            if part.down is not None:
                out_dim, in_dim = int(part.up.shape[0]), int(part.down.shape[1])
                if module.out_dim and (module.out_dim, module.in_dim) != (out_dim, in_dim):
                    raise BakeError(f"{label}: '{module_name}' does not fit the layer the other LoRAs change.")
                module.out_dim, module.in_dim = out_dim, in_dim
            elif part.diff_b is not None and not module.out_dim:
                module.out_dim = int(part.diff_b.shape[0]) if part.diff_b.shape else 0
            module.parts.append(part)

    if not any(module.factored for module in modules):
        raise BakeError("These LoRAs change nothing a bake could combine in their current modes.")
    dtype = dtypes.most_common(1)[0][0] if dtypes else "BF16"
    if dtype not in _WRITE:
        dtype = "F32"
    return Plan(members=members, files=files, modules=modules, dtype=dtype, member_ranks=ranks)


def bake(plan_: Plan, rank: int, path: str, metadata: dict[str, str] | None = None, progress=None) -> Result:
    """Write the group, re-fitted to ``rank``, to ``path``.

    ``progress(done, total)`` counts modules.  The file appears whole or not at
    all.
    """
    rank = int(rank)
    if rank < 1:
        raise BakeError("The rank must be at least 1.")
    layout, jobs = [], []
    for module in plan_.modules:
        k = plan_.output_rank(module, rank)
        down_name, up_name = _names(module)
        if k:
            layout.append((down_name, plan_.dtype, (k, module.in_dim)))
            layout.append((up_name, plan_.dtype, (module.out_dim, k)))
            layout.append((module.name + ".alpha", "F32", ()))
        has_bias = any(part.diff_b is not None for part in module.parts)
        if has_bias:
            layout.append((module.name + ".diff_b", plan_.dtype, (module.out_dim,)))
        jobs.append((module, k, has_bias))

    meta = {"lora_browser": "bake", "lora_browser_rank": str(rank)}
    meta.update(metadata or {})
    writer = lf.StreamWriter(path, layout, meta)
    handles = [open(lora.path, "rb") for lora in plan_.files]
    total_energy = kept_energy = 0.0
    exact = True
    try:
        for done, (module, k, has_bias) in enumerate(jobs, start=1):
            scales = {}
            if k:
                down, up, alpha, energy, kept = _fit(module, k, handles, plan_)
                total_energy += energy
                kept_energy += kept
                if kept < energy * (1 - 1e-6):
                    exact = False
                down_name, up_name = _names(module)
                writer.write(down_name, _encode(down, plan_.dtype))
                writer.write(up_name, _encode(up, plan_.dtype))
                writer.write(module.name + ".alpha", np.asarray(alpha, dtype="<f4").tobytes())
                scales = {part.member: _scale(part) for part in module.factored}
                # The new file applies its factors at alpha / k; a bias delta
                # rides on the same scale, so it is divided back out below.
                bias_scale = float(alpha) / k
            else:
                bias_scale = 1.0
            if has_bias:
                total = np.zeros(module.out_dim, dtype=np.float32)
                for part in module.parts:
                    if part.diff_b is not None:
                        delta = _load(part.diff_b, handles[part.member]).reshape(-1)
                        total += delta * scales.get(part.member, 1.0)
                writer.write(module.name + ".diff_b", _encode(total / bias_scale, plan_.dtype))
            if progress:
                progress(done, len(jobs))
        writer.close()
    except BaseException:
        writer.abort()
        raise
    finally:
        for handle in handles:
            handle.close()
    kept = 1.0 if total_energy <= 0 else min(1.0, kept_energy / total_energy)
    return Result(rank=rank, kept=1.0 if exact else kept, exact=exact,
                  modules=len(jobs), bytes=plan_.size_for(rank))


# ------------------------------------------------------------------ maths


def _fit(module: _Module, k: int, handles, plan_: Plan):
    """``(down, up, alpha, energy, kept energy)`` for one module at rank ``k``."""
    parts = module.factored
    downs = [_load(part.down, handles[part.member]) for part in parts]
    ups = [_load(part.up, handles[part.member]) for part in parts]
    scales = [_scale(part) for part in parts]
    total = module.total_rank

    if k >= total:
        # Everything fits: stack the factors.  A shared scale stays an alpha,
        # so the factors are copied untouched; different ones are folded in.
        down = np.concatenate(downs, axis=0)
        if len(set(scales)) == 1:
            up = np.concatenate(ups, axis=1)
            alpha = scales[0] * total
        else:
            up = np.concatenate([u * s for u, s in zip(ups, scales)], axis=1)
            alpha = float(total)
        energy = _energy(np.concatenate([u * s for u, s in zip(ups, scales)], axis=1), down)
        return down, up, alpha, energy, energy

    stacked_up = np.concatenate([u * s for u, s in zip(ups, scales)], axis=1)      # out x R
    stacked_down = np.concatenate(downs, axis=0)                                   # R x in
    q_up, r_up = np.linalg.qr(stacked_up.astype(np.float64))
    q_down, r_down = np.linalg.qr(stacked_down.T.astype(np.float64))
    left, singular, right = np.linalg.svd(r_up @ r_down.T)
    root = np.sqrt(singular[:k])
    up = (q_up @ left[:, :k]) * root                                               # out x k
    down = (root[:, None] * right[:k]) @ q_down.T                                  # k x in
    energy = float(np.sum(singular ** 2))
    kept = float(np.sum(singular[:k] ** 2))
    return down.astype(np.float32), up.astype(np.float32), float(k), energy, kept


def _energy(up: np.ndarray, down: np.ndarray) -> float:
    """||up @ down||_F^2 from two small Gram matrices."""
    up = up.astype(np.float64)
    down = down.astype(np.float64)
    return float(np.sum((up.T @ up) * (down @ down.T)))


def _scale(part: _Part) -> float:
    return float(part.alpha) / part.rank if part.alpha is not None and part.rank else 1.0


def _names(module: _Module) -> tuple[str, str]:
    if module.style == "downup":
        return module.name + ".lora_down.weight", module.name + ".lora_up.weight"
    return module.name + ".lora_A.weight", module.name + ".lora_B.weight"


# --------------------------------------------------------------- bytes in/out


def _header_alpha(metadata: dict[str, str]) -> float | None:
    """mmgp's fallback: an ``alpha`` in the header applies to every module
    that has none of its own."""
    try:
        value = float(metadata.get("alpha", ""))
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _alphas(lora: lf.LoraFile, handle, label: str, sided: bool, mode: str) -> dict[str, float]:
    """Every per-module alpha the member keeps, read through one handle."""
    alphas = {}
    for tensor in lora.tensors:
        module_name, role = lf.split_key(tensor.name)
        if role != "alpha" or (sided and not lf.keeps(lf.ltx2_side(module_name), mode)):
            continue
        values = _load(tensor, handle).reshape(-1)
        if values.size != 1 or not np.isfinite(values[0]):
            raise BakeError(f"{label}: '{tensor.name}' is not a single number.")
        alphas[module_name] = float(values[0])
    return alphas


def _load(tensor: lf.TensorInfo, handle) -> np.ndarray:
    handle.seek(tensor.start)
    raw = handle.read(tensor.nbytes)
    if len(raw) != tensor.nbytes:
        raise BakeError(f"'{tensor.name}' is cut short.")
    if tensor.dtype == "BF16":
        bits = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
        values = bits.view(np.float32)
    elif tensor.dtype in _READ:
        values = np.frombuffer(raw, dtype=_READ[tensor.dtype]).astype(np.float32)
    else:
        raise BakeError(f"'{tensor.name}' is stored as {tensor.dtype}, which a bake cannot read.")
    return values.reshape(tensor.shape) if tensor.shape else values.reshape(())


def _encode(values: np.ndarray, dtype: str) -> bytes:
    data = np.ascontiguousarray(values, dtype=np.float32)
    if dtype == "F32":
        return data.astype("<f4").tobytes()
    if dtype == "F16":
        return data.astype("<f2").tobytes()
    # BF16, rounded to nearest even from the float32 bits.
    bits = data.view(np.uint32)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    return rounded.astype("<u2").tobytes()
