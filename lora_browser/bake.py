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

A file need not use one rank everywhere: :func:`analyze` measures every
layer's directions before anything is built (the same singular values, from
two small Gram matrices), and :func:`allocate` gives each layer the rank that
buys the most of the combined change per byte -- the strongest directions
across every layer together, until a share of the whole is kept -- with no
layer keeping less than :func:`floor_for` of its own.  mmgp reads each layer's
rank from that layer's own factors, so a file whose layers differ in rank
loads like any other.

A rank at or above what the members add up to keeps everything, and is built
by stacking the factors as they are.  A group's strength and schedule are not
baked in: the group's multiplier applies to the baked file at run time,
exactly as it applied to each member.  A member's ``weight`` is: a shrunk
stack bakes each LoRA in at the strength it ran at, s_i above times w_i.

numpy is the only dependency, and WanGP already ships it.
"""

from __future__ import annotations

import hashlib
import os
import re
import threading
from collections import Counter, OrderedDict
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
    #: ``normal`` takes everything; ``visual``/``audio`` take one LTX-2 side,
    #: ``sound`` the sound side without the layers both sides share.
    mode: str = "normal"
    label: str = ""
    #: What the member brings: its change times this.
    weight: float = 1.0


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

    @property
    def width(self) -> int:
        return 4 if self.dtype == "F32" else 2

    def size_for(self, rank: int) -> int:
        """Bytes of tensor data a bake at ``rank`` writes."""
        return sum(_module_bytes(module.out_dim, module.in_dim, self.output_rank(module, rank), self.width,
                                 _has_bias(module))
                   for module in self.modules)

    def layer_rank(self, module: _Module, rank: int) -> int:
        """A layer's own rank, as a per-layer allocation asks for it: all of
        it stacked at or above what its members add up to, else at most its
        shape allows."""
        total = module.total_rank
        rank = int(rank)
        if not total or rank <= 0:
            return 0
        if rank >= total:
            return total
        return min(rank, module.out_dim, module.in_dim)


def _module_bytes(out_dim: int, in_dim: int, k: int, width: int, bias: bool) -> int:
    return k * (out_dim + in_dim) * width + (4 if k else 0) + (out_dim * width if bias else 0)


def _has_bias(module: _Module) -> bool:
    return any(part.diff_b is not None for part in module.parts)


@dataclass
class Result:
    #: The highest rank any layer has.
    rank: int
    #: Share of the combined change kept, 0..1; 1 when nothing was cut.
    kept: float
    exact: bool
    modules: int
    bytes: int


def check(member: Member, *, sided: bool) -> str:
    """Why this LoRA could not go into a bake, or ``""`` when it can: the
    checks :func:`plan` makes, for one member, from its header alone."""
    label = member.label or member.path
    try:
        lora = lf.read_header(member.path)
    except lf.LoraFileError as error:
        return f"{label}: {error}"
    found = False
    for tensor in lora.tensors:
        module_name, role = lf.split_key(tensor.name)
        if sided and not lf.keeps(lf.ltx2_side(module_name), member.mode):
            continue
        if role in ("dora", "lokr"):
            return f"{label} is a {role.upper() if role == 'dora' else 'LoKr'} adapter; only plain LoRAs can be baked."
        if role == "diff":
            return f"{label} changes '{module_name}' directly rather than as a LoRA."
        if role == "other":
            return f"{label} carries '{tensor.name}', which is not part of a LoRA."
        if role in ("down", "up"):
            if len(tensor.shape) != 2:
                return f"{label}: '{tensor.name}' is not a 2-D factor; only linear layers can be baked."
            if tensor.dtype not in _READ and tensor.dtype != "BF16":
                return f"{label} stores '{tensor.name}' as {tensor.dtype}, which a bake cannot read."
            found = True
    return "" if found else f"{label} changes nothing a bake could combine in its mode."


def plan(members: list[Member], *, sided: bool) -> Plan:
    """Read every member's header and line their modules up.

    ``sided`` is LTX-2: a member set to visual or audio only brings only that
    side.  Anything a bake cannot combine is refused here, before a byte is
    written: DoRA and LoKr adapters, dense weight deltas, and tensors that are
    not LoRA factors at all (WanGP's own conversion data, for one).
    """
    if len(members) < 2:
        raise BakeError("A bake needs at least two LoRAs.")
    for member in members:
        if not np.isfinite(float(member.weight)):
            raise BakeError(f"{member.label or member.path} has no usable strength.")
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


def bake(plan_: Plan, rank: int, path: str, metadata: dict[str, str] | None = None, progress=None,
         ranks: dict[str, int] | None = None) -> Result:
    """Write the group, re-fitted to ``rank``, to ``path`` -- or, with
    ``ranks``, each layer (by its canonical name) to a rank of its own, as
    :func:`allocate` chose.

    ``progress(done, total)`` counts modules.  The file appears whole or not at
    all.
    """
    rank = int(rank)
    if rank < 1 and not ranks:
        raise BakeError("The rank must be at least 1.")
    layout, jobs = [], []
    for module in plan_.modules:
        if ranks is not None:
            k = plan_.layer_rank(module, ranks.get(module.key, 0))
        else:
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
                scales = {part.member: _scale(part) * _weight(plan_, part.member) for part in module.factored}
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
                        total += delta * scales.get(part.member, _weight(plan_, part.member))
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
    if ranks is not None:
        size = sum(_module_bytes(module.out_dim, module.in_dim, k, plan_.width, has_bias)
                   for module, k, has_bias in jobs)
        rank = max((k for _, k, _ in jobs), default=0)
    else:
        size = plan_.size_for(rank)
    return Result(rank=rank, kept=1.0 if exact else kept, exact=exact, modules=len(jobs), bytes=size)


# ------------------------------------------------------------ compression

#: The compression levels a bake or a shrink is built at: a key, a name, and
#: the share of the combined change kept, per mille.  Each is the smallest
#: file that keeps its share, layer ranks chosen by :func:`allocate`.
LEVELS = (
    ("low", "Low compression", 999),
    ("medium", "Medium compression", 990),
    ("high", "High compression", 950),
)
DEFAULT_LEVEL = "low"


def level_quality(key) -> int:
    """A level's share kept, per mille; 0 for no such level."""
    return next((quality for name, _, quality in LEVELS if name == key), 0)


def level_of(quality: int) -> tuple[str, str]:
    """``(key, name)`` of the level keeping ``quality`` per mille, or of the
    share itself where no level does."""
    for key, name, value in LEVELS:
        if value == int(quality):
            return key, name
    return "", f"{int(quality) / 10:g}% kept"


def floor_for(kept: float) -> float:
    """The least any one layer keeps of its own change when the whole keeps
    ``kept``: no layer loses more than five times the overall loss, so none
    is gutted to save bytes another layer would have cost."""
    return max(0.0, 1.0 - 5.0 * (1.0 - float(kept)))


@dataclass
class Spectrum:
    """A group's combined change, layer by layer, as the directions it is
    made of: what a file of any size keeps, known before one is built."""

    #: Canonical layer names, in plan order.
    keys: list[str]
    #: Each layer's directions' energies (squared singular values),
    #: strongest first.
    energies: list[np.ndarray]
    #: Each layer's ``(out, in)``, the rank its members add up to, and
    #: whether it carries a bias change.
    dims: list[tuple[int, int]]
    totals: list[int]
    bias: list[bool]
    #: Bytes per stored value.
    width: int
    #: The rank of the highest-rank member: the size a bake matched before.
    default_rank: int

    @property
    def total(self) -> float:
        return float(sum(float(np.sum(values)) for values in self.energies))

    def capped(self, index: int, k: int) -> int:
        total = self.totals[index]
        if not total or k <= 0:
            return 0
        if k >= total:
            return total
        return min(int(k), *self.dims[index])

    def size(self, ranks: list[int]) -> int:
        """Bytes a file with these layer ranks writes."""
        return sum(_module_bytes(out_dim, in_dim, self.capped(index, k), self.width, self.bias[index])
                   for index, ((out_dim, in_dim), k) in enumerate(zip(self.dims, ranks)))

    def kept_energy(self, ranks: list[int]) -> float:
        kept = 0.0
        for index, (values, k) in enumerate(zip(self.energies, ranks)):
            kept += float(np.sum(values[:self.capped(index, k)]))
        return kept

    def kept(self, ranks: list[int]) -> float:
        """The share of the combined change these layer ranks keep."""
        total = self.total
        return 1.0 if total <= 0 else min(1.0, self.kept_energy(ranks) / total)

    def uniform(self, rank: int) -> list[int]:
        """The layer ranks one rank for every layer gives."""
        return [self.capped(index, rank) for index in range(len(self.keys))]

    def exact(self) -> list[int]:
        return list(self.totals)

    def ranks_by_key(self, ranks: list[int]) -> dict[str, int]:
        return {key: self.capped(index, k) for index, (key, k) in enumerate(zip(self.keys, ranks))}


def analyze(plan_: Plan, progress=None) -> Spectrum:
    """Measure every layer's directions, reading the members' factors once.
    ``progress(done, total)`` counts layers."""
    handles = [open(lora.path, "rb") for lora in plan_.files]
    energies = []
    try:
        for done, module in enumerate(plan_.modules, start=1):
            energies.append(_directions(module, handles, plan_))
            if progress:
                progress(done, len(plan_.modules))
    finally:
        for handle in handles:
            handle.close()
    return Spectrum(
        keys=[module.key for module in plan_.modules],
        energies=energies,
        dims=[(module.out_dim, module.in_dim) for module in plan_.modules],
        totals=[module.total_rank for module in plan_.modules],
        bias=[_has_bias(module) for module in plan_.modules],
        width=plan_.width,
        default_rank=min(plan_.default_rank or plan_.combined_rank, plan_.combined_rank),
    )


def _directions(module: _Module, handles, plan_: Plan) -> np.ndarray:
    """The energies of one layer's combined change, strongest first.

    dW = U V with U = [s_i B_i] (out x R) and V = [A_i] (R x in); its squared
    singular values are the eigenvalues of (U^T U)(V V^T), two R x R Gram
    matrices -- dW itself is never formed."""
    parts = module.factored
    if not parts:
        return np.zeros(0)
    ups = [_load(part.up, handles[part.member]) * np.float32(_scale(part) * _weight(plan_, part.member))
           for part in parts]
    downs = [_load(part.down, handles[part.member]) for part in parts]
    up = np.concatenate(ups, axis=1)
    down = np.concatenate(downs, axis=0)
    gram_up = (up.T @ up).astype(np.float64)
    gram_down = (down @ down.T).astype(np.float64)
    values, vectors = np.linalg.eigh((gram_down + gram_down.T) * 0.5)
    root = np.sqrt(np.clip(values, 0.0, None))
    core = (vectors.T @ gram_up @ vectors) * np.outer(root, root)
    found = np.clip(np.linalg.eigvalsh((core + core.T) * 0.5)[::-1], 0.0, None)
    return np.ascontiguousarray(found[:min(len(found), module.out_dim, module.in_dim)])


def allocate(spectrum: Spectrum, kept: float) -> list[int]:
    """Each layer's rank for a file keeping ``kept`` of the combined change
    in the fewest bytes: every layer first keeps :func:`floor_for` of its own
    change, then the directions that hold the most energy per byte, across
    every layer, go in until the whole keeps ``kept``."""
    kept = min(1.0, max(0.0, float(kept)))
    floor = floor_for(kept)
    ranks = []
    for values in spectrum.energies:
        ranks.append(_enough(values, floor))
    total = spectrum.total
    need = kept * total - spectrum.kept_energy(ranks)
    if total > 0 and need > total * 1e-12:
        ratios, energies, owners = [], [], []
        for index, (values, k) in enumerate(zip(spectrum.energies, ranks)):
            tail = values[k:]
            if not len(tail):
                continue
            out_dim, in_dim = spectrum.dims[index]
            ratios.append(tail / float((out_dim + in_dim) * spectrum.width))
            energies.append(tail)
            owners.append(np.full(len(tail), index))
        if ratios:
            ratio = np.concatenate(ratios)
            # Stable, so a layer's equal directions go in its own order.
            order = np.argsort(-ratio, kind="stable")
            gained = np.cumsum(np.concatenate(energies)[order])
            take = min(int(np.searchsorted(gained, need * (1 - 1e-12))) + 1, len(order))
            counts = np.bincount(np.concatenate(owners)[order[:take]], minlength=len(ranks))
            ranks = [k + int(count) for k, count in zip(ranks, counts)]
    return ranks


def _enough(values: np.ndarray, share: float) -> int:
    """The fewest of a layer's directions that keep ``share`` of its energy."""
    if share <= 0 or not len(values):
        return 0
    gained = np.cumsum(values)
    if gained[-1] <= 0:
        return 0
    return min(int(np.searchsorted(gained, share * gained[-1] * (1 - 1e-12))) + 1, len(values))


def predictions(spectra: list[Spectrum]) -> dict:
    """What each level would keep and weigh for these files together, beside
    one rank matching the highest-rank member (how a bake was sized before)
    and the exact files (what the LoRAs take now)."""
    total = sum(spectrum.total for spectrum in spectra)

    def share(kept_energy: float) -> float:
        return 1.0 if total <= 0 else min(1.0, kept_energy / total)

    levels = []
    for key, name, quality in LEVELS:
        chosen = [allocate(spectrum, quality / 1000) for spectrum in spectra]
        levels.append({
            "key": key, "label": name, "quality": quality,
            "kept": share(sum(spectrum.kept_energy(ranks) for spectrum, ranks in zip(spectra, chosen))),
            "bytes": sum(spectrum.size(ranks) for spectrum, ranks in zip(spectra, chosen)),
            "rank": max((max(ranks, default=0) for ranks in chosen), default=0),
        })
    matched = [spectrum.uniform(spectrum.default_rank) for spectrum in spectra]
    return {
        "levels": levels,
        "matched": {
            "rank": max((spectrum.default_rank for spectrum in spectra), default=0),
            "kept": share(sum(spectrum.kept_energy(ranks) for spectrum, ranks in zip(spectra, matched))),
            "bytes": sum(spectrum.size(ranks) for spectrum, ranks in zip(spectra, matched)),
        },
        "exact": {"bytes": sum(spectrum.size(spectrum.exact()) for spectrum in spectra)},
    }


def measure_key(members: list[Member], sided: bool) -> str:
    """One name per set of members as they are on disk and what each
    brings: a measurement is reused only for exactly that."""
    lines = [f"sided={int(bool(sided))}"]
    for member in members:
        try:
            stat = os.stat(member.path)
            stamp = f"{stat.st_size}:{stat.st_mtime_ns}"
        except OSError:
            stamp = "missing"
        lines.append(f"{os.path.realpath(member.path)}\t{stamp}\t{member.mode}\t{float(member.weight)!r}")
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()[:16]


#: Measurements made this session, newest last.  A large stack's is a few MB.
_MEASURED: OrderedDict[str, Spectrum] = OrderedDict()
_MEASURED_LOCK = threading.Lock()
_MEASURED_KEEP = 24


def remember(key: str, spectrum: Spectrum) -> None:
    with _MEASURED_LOCK:
        _MEASURED[key] = spectrum
        _MEASURED.move_to_end(key)
        while len(_MEASURED) > _MEASURED_KEEP:
            _MEASURED.popitem(last=False)


def recall(key: str) -> Spectrum | None:
    with _MEASURED_LOCK:
        found = _MEASURED.get(key)
        if found is not None:
            _MEASURED.move_to_end(key)
        return found


# ------------------------------------------------------------------ maths


def _fit(module: _Module, k: int, handles, plan_: Plan):
    """``(down, up, alpha, energy, kept energy)`` for one module at rank ``k``."""
    parts = module.factored
    downs = [_load(part.down, handles[part.member]) for part in parts]
    ups = [_load(part.up, handles[part.member]) for part in parts]
    scales = [_scale(part) * _weight(plan_, part.member) for part in parts]
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


def _weight(plan_: Plan, member: int) -> float:
    return float(plan_.members[member].weight)


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
