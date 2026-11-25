# -*- coding: utf-8 -*-
"""
Deterministic model assembler for external validation.
=============================================================
Purpose
    Convert a"motif specification"(A preprocessing, B backbone, C forecast head, E composition) into
    a PyTorch candidate_model.py without an LLM call.
    Named motifs in law-guided recommendations use fixed, reproducible implementations.

Module interfaces used by the training pipeline
    CandidateModel(time_lag, pre_len, station_num)
    forward(x): x [B, time_lag, station_num] -> [B, pre_len, station_num]
    - A: [B,L,C] -> [B,L,C]. Cache reversible state in self._* for stage D.
    - B: [B,L,C] -> [B,L,C]
    - C: [B,L,C] -> [B,pre_len,C]
    - E: compose multiple B blocks
    - D: invert normalization, differencing and decomposition in reverse A order

Usage
    python model_assembler.py --selftest   # Self-test: generate example specifications and validate them with the runtime pipeline.
"""

import os
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Module source fragments, indexed by canonical motif name.
# candidate_model.py includes the required fragments as _A_xxx, _B_xxx, _C_xxx and _D_xxx.
# ---------------------------------------------------------------------------

# Scalable widths: default hidden and n_basis dimensions for B/C motifs.
# Entries map a motif to (parameter name, default). Other motifs such as Linear Block and Linear Head have fixed capacity.
_B_SCALE_ARGS = {
    "MLP Block": ("hidden", 64),
    "CNN Block": ("hidden", 32),
    "TCN Block": ("hidden", 32),
    "GRU Block": ("hidden", 64),
    "LSTM Block": ("hidden", 64),
    "State Space Model Block": ("hidden", 32),
    "Transformer Encoder Block": ("hidden", 64),
    "Basis Block": ("n_basis", 32),
}
_C_SCALE_ARGS = {
    "MLP Head": ("hidden", 128),
    "Basis Projection": ("n_basis", 32),
}

# Branch order in the generated _build_b and _build_c functions (motif name to class suffix).
_B_BUILD_ORDER = [
    ("Linear Block", "Linear"), ("MLP Block", "MLP"), ("CNN Block", "CNN"),
    ("TCN Block", "TCN"), ("GRU Block", "GRU"), ("LSTM Block", "LSTM"),
    ("State Space Model Block", "SSM"), ("Transformer Encoder Block", "Transformer"),
    ("Basis Block", "Basis"),
]
_C_BUILD_ORDER = [
    ("Linear Head", "LinearHead"), ("MLP Head", "MLPHead"),
    ("Basis Projection", "BasisProjection"),
]

_HEADER = '''\
import torch
import torch.nn as nn
import torch.nn.functional as F

'''

# ---- A: preprocessing ----

_A_IDENTITY = '''\
class _A_Identity(nn.Module):
    def forward(self, x):
        return x
'''

_A_ZSCORE = '''\
class _A_ZScore(nn.Module):
    """Normalize each channel within the window; stage D restores the original scale."""
    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True).detach()
        std = x.std(dim=1, keepdim=True, correction=0).clamp_min(1e-5).detach()
        self._mean = mean
        self._std = std
        return (x - mean) / std
    def inverse(self, y):
        return y * self._std + self._mean
'''

_A_REVIN = '''\
class _A_RevIN(nn.Module):
    """RevIN: per-instance, per-variable normalization with learnable affine transforms."""
    def __init__(self, station_num, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.affine_weight = nn.Parameter(torch.ones(1, 1, station_num))
        self.affine_bias = nn.Parameter(torch.zeros(1, 1, station_num))
    def forward(self, x):
        self._mean = x.mean(dim=1, keepdim=True).detach()
        self._std = x.std(dim=1, keepdim=True, correction=0).clamp_min(self.eps).detach()
        z = (x - self._mean) / self._std
        return z * self.affine_weight + self.affine_bias
    def inverse(self, y):
        z = (y - self.affine_bias) / (self.affine_weight + self.eps)
        return z * self._std + self._mean
'''

_A_DIFFERENCING = '''\
class _A_Differencing(nn.Module):
    """First-order differencing; stage D reconstructs values by cumulative summation."""
    def forward(self, x):
        self._last = x[:, -1:, :]
        diff = x[:, 1:, :] - x[:, :-1, :]
        return torch.cat([torch.zeros_like(self._last), diff], dim=1)
    def inverse(self, y):
        return torch.cumsum(y, dim=1) + self._last
'''

_A_SERIESDECOMP = '''\
class _A_SeriesDecomposition(nn.Module):
    """DLinear moving-average decomposition: the backbone models residuals; a shared linear head predicts the trend."""
    def __init__(self, time_lag, pre_len, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        self.trend_proj = nn.Linear(time_lag, pre_len)
        nn.init.constant_(self.trend_proj.weight, 1.0 / time_lag)
        nn.init.zeros_(self.trend_proj.bias)
    def forward(self, x):
        x_t = x.transpose(1, 2)
        padded = F.pad(x_t, (self.padding, self.padding), mode="replicate")
        trend_t = F.avg_pool1d(padded, kernel_size=self.kernel_size, stride=1)
        self._trend = trend_t
        return x - trend_t.transpose(1, 2)
    def inverse(self, y):
        trend_future = self.trend_proj(self._trend).transpose(1, 2)
        return y + trend_future
'''

_A_PATCH = '''\
class _A_Patch(nn.Module):
    """Apply a shared local linear transform to overlapping patches and restore sequence length with overlap-add."""
    def __init__(self, time_lag, station_num, patch_len=4, step=2):
        super().__init__()
        self.time_lag = time_lag
        self.patch_len = min(patch_len, time_lag)
        self.step = step
        self.n_patch = max(1, (time_lag - self.patch_len + step - 1) // step + 1)
        self.covered_len = (self.n_patch - 1) * step + self.patch_len
        self.proj = nn.Linear(self.patch_len, self.patch_len)
        nn.init.eye_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
    def forward(self, x):
        B, L, C = x.shape
        x_t = x.transpose(1, 2)
        if self.covered_len > L:
            x_t = F.pad(x_t, (0, self.covered_len - L), mode="replicate")
        patches = x_t.unfold(-1, self.patch_len, self.step)
        patches = self.proj(patches)
        out = x_t.new_zeros(B, C, self.covered_len)
        count = x_t.new_zeros(1, 1, self.covered_len)
        for i in range(self.n_patch):
            start = i * self.step
            out[:, :, start:start + self.patch_len] += patches[:, :, i, :]
            count[:, :, start:start + self.patch_len] += 1
        return (out / count.clamp_min(1))[:, :, :L].transpose(1, 2)
'''

_A_MASK = '''\
class _A_Mask(nn.Module):
    """Randomly mask inputs during training; preserve inputs during inference."""
    def __init__(self, station_num, mask_ratio=0.1):
        super().__init__()
        self.mask_ratio = mask_ratio
    def forward(self, x):
        if not self.training or self.mask_ratio <= 0:
            return x
        keep = torch.rand_like(x) >= self.mask_ratio
        return x * keep.to(x.dtype) / (1.0 - self.mask_ratio)
'''

# ---- B: backbone ----

_B_LINEAR = '''\
class _B_Linear(nn.Module):
    """Channel-wise temporal linear block (DLinear style)."""
    def __init__(self, time_lag, station_num):
        super().__init__()
        self.linear = nn.Linear(time_lag, time_lag)
    def forward(self, x):
        return self.linear(x.transpose(1, 2)).transpose(1, 2)
'''

_B_MLP = '''\
class _B_MLP(nn.Module):
    """Shared temporal MLP applied to each variable with a residual connection."""
    def __init__(self, time_lag, station_num, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(time_lag, hidden),
                                 nn.GELU(),
                                 nn.Linear(hidden, time_lag))
    def forward(self, x):
        return x + self.mlp(x.transpose(1, 2)).transpose(1, 2)
'''

_B_CNN = '''\
class _B_CNN(nn.Module):
    """Temporal convolution with channel mixing and residual connections."""
    def __init__(self, time_lag, station_num, hidden=32, kernel=3):
        super().__init__()
        self.conv1 = nn.Conv1d(station_num, hidden, kernel, padding=kernel // 2)
        self.conv2 = nn.Conv1d(hidden, station_num, kernel, padding=kernel // 2)
    def forward(self, x):
        h = F.gelu(self.conv1(x.transpose(1, 2))).transpose(1, 2)
        return x + self.conv2(h.transpose(1, 2)).transpose(1, 2)
'''

_B_TCN = '''\
class _B_TCN(nn.Module):
    """Two-layer residual causal TCN with dilation factors of 1 and 2."""
    def __init__(self, time_lag, station_num, hidden=32, kernel=3):
        super().__init__()
        self.kernel = kernel
        self.conv1 = nn.Conv1d(station_num, hidden, kernel, dilation=1)
        self.conv2 = nn.Conv1d(hidden, station_num, kernel, dilation=2)
    def forward(self, x):
        h = F.pad(x.transpose(1, 2), ((self.kernel - 1), 0))
        h = F.gelu(self.conv1(h))
        h = F.pad(h, (2 * (self.kernel - 1), 0))
        h = self.conv2(h)
        return x + h.transpose(1, 2)
'''

_B_GRU = '''\
class _B_GRU(nn.Module):
    def __init__(self, time_lag, station_num, hidden=64, num_layers=1):
        super().__init__()
        self.rnn = nn.GRU(station_num, hidden, num_layers, batch_first=True)
        self.proj = nn.Linear(hidden, station_num)
    def forward(self, x):
        h, _ = self.rnn(x)
        return x + self.proj(h)
'''

_B_LSTM = '''\
class _B_LSTM(nn.Module):
    def __init__(self, time_lag, station_num, hidden=64, num_layers=1):
        super().__init__()
        self.rnn = nn.LSTM(station_num, hidden, num_layers, batch_first=True)
        self.proj = nn.Linear(hidden, station_num)
    def forward(self, x):
        h, _ = self.rnn(x)
        return x + self.proj(h)
'''

_B_SSM = '''\
class _B_SSM(nn.Module):
    """Stable diagonal state-space block with decay factors in (0,1), input projections and residual projections."""
    def __init__(self, time_lag, station_num, hidden=32):
        super().__init__()
        self.hidden = hidden
        self.decay_logit = nn.Parameter(torch.full((hidden,), 2.0))
        self.in_proj = nn.Linear(station_num, hidden, bias=False)
        self.out_proj = nn.Linear(hidden, station_num)
    def forward(self, x):
        B, L, C = x.shape
        h = x.new_zeros(B, self.hidden)
        decay = torch.sigmoid(self.decay_logit).view(1, -1)
        outs = []
        for t in range(L):
            u = self.in_proj(x[:, t, :])
            h = decay * h + (1.0 - decay) * u
            outs.append(self.out_proj(h))
        return x + torch.stack(outs, dim=1)
'''

_B_TRANSFORMER = '''\
class _B_Transformer(nn.Module):
    def __init__(self, time_lag, station_num, hidden=64, num_heads=4, n_layers=1):
        super().__init__()
        self.in_proj = nn.Linear(station_num, hidden)
        self.position = nn.Parameter(torch.zeros(1, time_lag, hidden))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=num_heads,
                                           dim_feedforward=hidden * 2, dropout=0.1,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.out_proj = nn.Linear(hidden, station_num)
    def forward(self, x):
        h = self.in_proj(x) + self.position[:, :x.size(1), :]
        return x + self.out_proj(self.enc(h))
'''

_B_BASIS = '''\
class _B_Basis(nn.Module):
    """Per-variable basis block: generate coefficients independently and reconstruct historical representations."""
    def __init__(self, time_lag, station_num, n_basis=32):
        super().__init__()
        self.n_basis = n_basis
        self.mlp = nn.Sequential(nn.Linear(time_lag, 64),
                                 nn.GELU(),
                                 nn.Linear(64, n_basis))
        self.basis = nn.Parameter(torch.randn(n_basis, time_lag) * 0.1)
    def forward(self, x):
        coeffs = self.mlp(x.transpose(1, 2))            # [B, C, n_basis]
        out = coeffs @ self.basis                       # [B, C, L]
        return out.transpose(1, 2)
'''

# ---- C: forecast heads ----

_C_LINEAR = '''\
class _C_LinearHead(nn.Module):
    def __init__(self, time_lag, pre_len, station_num):
        super().__init__()
        self.proj = nn.Linear(time_lag, pre_len)
    def forward(self, z):
        return self.proj(z.transpose(1, 2)).transpose(1, 2)
'''

_C_MLP = '''\
class _C_MLPHead(nn.Module):
    def __init__(self, time_lag, pre_len, station_num, hidden=128):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(time_lag, hidden),
                                  nn.GELU(),
                                  nn.Linear(hidden, pre_len))
    def forward(self, z):
        return self.proj(z.transpose(1, 2)).transpose(1, 2)
'''

_C_BASIS = '''\
class _C_BasisProjection(nn.Module):
    """Per-variable basis projection: history -> basis coefficients -> future basis."""
    def __init__(self, time_lag, pre_len, station_num, n_basis=32):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(time_lag, 64),
                                 nn.GELU(),
                                 nn.Linear(64, n_basis))
        self.basis = nn.Parameter(torch.randn(n_basis, pre_len) * 0.1)
    def forward(self, z):
        coeffs = self.mlp(z.transpose(1, 2))            # [B, C, n_basis]
        out = coeffs @ self.basis                       # [B, C, pre_len]
        return out.transpose(1, 2)
'''

# ---- E: backbone composition ----

_E = '''\
class _E_Composition(nn.Module):
    def __init__(self, b_modules, e_name):
        super().__init__()
        valid = {"Single Block", "Sequential Composition", "Parallel Additive Branches",
                 "Residual Modeling", "Gated Fusion"}
        if e_name not in valid:
            raise ValueError("unknown E: " + e_name)
        if e_name == "Single Block" and len(b_modules) != 1:
            raise ValueError("Single Block requires exactly one B module")
        if e_name != "Single Block" and len(b_modules) != 2:
            raise ValueError(e_name + " requires exactly two B modules")
        self.e_name = e_name
        self.blocks = nn.ModuleList(b_modules)
        if e_name == "Gated Fusion":
            self.gate = nn.Parameter(torch.zeros(len(b_modules)))
    def forward(self, x):
        if self.e_name == "Single Block":
            return self.blocks[0](x)
        if self.e_name == "Sequential Composition":
            out = x
            for b in self.blocks:
                out = b(out)
            return out
        if self.e_name == "Parallel Additive Branches":
            return sum(b(x) for b in self.blocks)
        if self.e_name == "Residual Modeling":
            main = self.blocks[0](x)
            correction = x
            for b in self.blocks[1:]:
                correction = b(correction)
            return main + correction
        if self.e_name == "Gated Fusion":
            g = torch.softmax(self.gate, dim=0)
            outs = torch.stack([b(x) for b in self.blocks], dim=0)
            return (g.view(-1, 1, 1, 1) * outs).sum(dim=0)
'''

# ---- D: output restoration ----

_D = '''\
class _D_OutputAdapt(nn.Module):
    """Apply inverse transforms and trend recomposition in reverse stage-A order."""
    def __init__(self, invertible):
        super().__init__()
        self.invertible = nn.ModuleList(invertible)
    def forward(self, out):
        for m in reversed(self.invertible):
            out = m.inverse(out)
        return out
'''

# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

_MODULE_SOURCES = {
    # A
    "Identity": _A_IDENTITY,
    "Z-score": _A_ZSCORE,
    "RevIN": _A_REVIN,
    "Differencing": _A_DIFFERENCING,
    "Series Decomposition": _A_SERIESDECOMP,
    "Patch": _A_PATCH,
    "Mask": _A_MASK,
    # B
    "Linear Block": _B_LINEAR,
    "MLP Block": _B_MLP,
    "CNN Block": _B_CNN,
    "TCN Block": _B_TCN,
    "GRU Block": _B_GRU,
    "LSTM Block": _B_LSTM,
    "State Space Model Block": _B_SSM,
    "Transformer Encoder Block": _B_TRANSFORMER,
    "Basis Block": _B_BASIS,
    # C
    "Linear Head": _C_LINEAR,
    "MLP Head": _C_MLP,
    "Basis Projection": _C_BASIS,
}

# Inverse mapping from the grammar's auto_output_adaptation.
_INVERSE_OF = {"Z-score", "RevIN", "Differencing", "Series Decomposition"}


def _scaled_widths(width_scale):
    """Use width_scale to return {motif name: scaled integer hidden or n_basis dimension}for scalable motifs only."""
    def _sc(base):
        return max(1, int(round(base * width_scale)))

    widths = {}
    for name, (kw, base) in _B_SCALE_ARGS.items():
        v = _sc(base)
        if name == "Transformer Encoder Block":
            v = max(8, v + (4 - v % 4) % 4)   # Make hidden divisible by num_heads=4.
        widths[name] = v
    for name, (kw, base) in _C_SCALE_ARGS.items():
        widths[name] = _sc(base)
    return widths


def _build_b_source(time_lag, station_num, widths):
    """Generate _build_b with scaled width literals."""
    lines = ["def _build_b(name):"]
    for m, cls in _B_BUILD_ORDER:
        lines.append(f'    if name == "{m}":')
        arg = ""
        if m in widths:
            kw = _B_SCALE_ARGS[m][0]
            arg = f", {kw}={widths[m]}"
        lines.append(f'        return _B_{cls}({time_lag}, {station_num}{arg})')
    lines.append('    raise ValueError("unknown B: " + name)')
    return "\n".join(lines)


def _build_c_source(time_lag, pre_len, station_num, widths):
    """Generate _build_c with scaled width literals."""
    lines = ["def _build_c(name):"]
    for m, cls in _C_BUILD_ORDER:
        lines.append(f'    if name == "{m}":')
        arg = ""
        if m in widths:
            kw = _C_SCALE_ARGS[m][0]
            arg = f", {kw}={widths[m]}"
        lines.append(f'        return _C_{cls}({time_lag}, {pre_len}, {station_num}{arg})')
    lines.append('    raise ValueError("unknown C: " + name)')
    return "\n".join(lines)


def build_model_code(spec, time_lag, pre_len, station_num, width_scale=None):
    """
    spec: {"A": [motif names...], "B": [motif names...], "C": motif name, "E": composition name}
    width_scale scales hidden/n_basis in B/C while preserving A preprocessing.
                 If None, read spec["width_scale"]; use 1.0 (default width) if absent.
    Return the complete candidate_model.py source.
    """
    if width_scale is None:
        width_scale = float(spec.get("width_scale", 1.0)) if isinstance(spec, dict) else 1.0
    width_scale = max(0.05, float(width_scale))

    a_list = list(spec.get("A", []) or [])
    b_list = list(spec.get("B", []) or [])
    c_name = spec.get("C") or "Linear Head"
    e_name = spec.get("E") or "Sequential Composition"

    if not b_list:
        raise ValueError("B (backbone) must contain at least one block")

    used = set(a_list) | set(b_list) | {c_name}
    parts = [_HEADER]
    for name in ["Identity", "Z-score", "RevIN", "Differencing",
                 "Series Decomposition", "Patch", "Mask",
                 "Linear Block", "MLP Block", "CNN Block", "TCN Block",
                 "GRU Block", "LSTM Block", "State Space Model Block",
                 "Transformer Encoder Block", "Basis Block",
                 "Linear Head", "MLP Head", "Basis Projection"]:
        if name in used:
            parts.append(_MODULE_SOURCES[name])
            parts.append("\n")
    parts.append(_E)
    parts.append(_D)

    # Build module initialization statements for CandidateModel.__init__.
    a_insts = [f'        self.a_{i} = _build_a("{m}")' for i, m in enumerate(a_list)]
    invertible_refs = [f"self.a_{i}" for i, m in enumerate(a_list) if m in _INVERSE_OF]

    b_insts = [f'        self.b_{i} = _build_b("{m}")' for i, m in enumerate(b_list)]
    b_refs = [f"self.b_{i}" for i in range(len(b_list))]

    init_lines = "\n".join(a_insts + b_insts) or "        pass"
    inv_list = ", ".join(invertible_refs) if invertible_refs else ""

    code = "\n".join(parts)

    forward_chain = "\n".join(
        f"        x = self.a_{i}(x)" for i in range(len(a_list))) or "        pass"
    b_backbone = ", ".join(b_refs)

    widths = _scaled_widths(width_scale)
    b_code = _build_b_source(time_lag, station_num, widths)
    c_code = _build_c_source(time_lag, pre_len, station_num, widths)

    model_code = code + f'''

def _build_a(name):
    if name == "Identity":
        return _A_Identity()
    if name == "Z-score":
        return _A_ZScore()
    if name == "RevIN":
        return _A_RevIN({station_num})
    if name == "Differencing":
        return _A_Differencing()
    if name == "Series Decomposition":
        return _A_SeriesDecomposition({time_lag}, {pre_len})
    if name == "Patch":
        return _A_Patch({time_lag}, {station_num})
    if name == "Mask":
        return _A_Mask({station_num})
    raise ValueError("unknown A: " + name)


{b_code}


{c_code}


class CandidateModel(nn.Module):
    def __init__(self, time_lag, pre_len, station_num):
        super().__init__()
        self.time_lag = time_lag
        self.pre_len = pre_len
        self.station_num = station_num
{init_lines}
        self.backbone = _E_Composition([{b_backbone}], "{e_name}")
        self.head = _build_c("{c_name}")
        self.output_adapt = _D_OutputAdapt([{inv_list}])

    def forward(self, x):
{forward_chain}
        z = self.backbone(x)
        out = self.head(z)
        out = self.output_adapt(out)
        return out
'''
    return model_code


def write_candidate_model(spec, time_lag, pre_len, station_num, path):
    """Assemble and save candidate_model.py using width_scale from the specification."""
    ws = float(spec.get("width_scale", 1.0)) if isinstance(spec, dict) else 1.0
    code = build_model_code(spec, time_lag, pre_len, station_num, width_scale=ws)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    return path


# ---------------------------------------------------------------------------
# Parameter counts and width scaling
# ---------------------------------------------------------------------------

def count_params_for_spec(spec, time_lag, pre_len, station_num, width_scale=None):
    """Build the model from spec and optional width_scale and count its parameters.

    Execute and instantiate in memory without temporary files or repeated imports.
    This avoids stale candidate_utils entries for"modules at the same path"when the generated code changes.
    Return the integer parameter count.
    """
    code = build_model_code(spec, time_lag, pre_len, station_num, width_scale=width_scale)
    ns = {}
    exec(code, ns)
    model = ns["CandidateModel"](time_lag, pre_len, station_num)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def scale_to_target_params(spec, time_lag, pre_len, station_num, target,
                           lo=0.5, hi=2.0, scale_min=0.05, scale_max=40.0,
                           max_iter=40):
    """
    Binary-search width_scale for a count in [target*lo, target*hi] as close as possible
    to target. Align the F architecture capacity with the rule-guided L model.

    Return (scaled_spec, n_params):
      - scaled_spec is a copy with width_scale for the assembler.
      - When a specification contains only fixed-width modules such as Linear Block and Linear Head,
        return a copy and its actual parameter count.
    """
    def _count(s):
        return count_params_for_spec(spec, time_lag, pre_len, station_num,
                                     width_scale=s)

    n_default = _count(1.0)
    try:
        n_at_max = _count(scale_max)
    except Exception:
        n_at_max = n_default

    # Parameter count is independent of width for this specification.
    if abs(n_at_max - n_default) < max(10, target * 0.01):
        return dict(spec), n_default

    # Keep the default width if already within range.
    if target * lo <= n_default <= target * hi:
        return dict(spec), n_default

    # Parameter count is nondecreasing in scale.
    s_lo, s_hi = scale_min, scale_max
    for _ in range(max_iter):
        mid = 0.5 * (s_lo + s_hi)
        if _count(mid) < target:
            s_lo = mid
        else:
            s_hi = mid
    best_s, best_n = min([(s_lo, _count(s_lo)), (s_hi, _count(s_hi))],
                         key=lambda t: abs(t[1] - target))

    scaled = dict(spec)
    rounded_s = round(best_s, 4)
    scaled["width_scale"] = rounded_s
    # Recount with the four-decimal scale actually written to spec;
    # rounding discrete hidden/n_basis dimensions may change the parameter count.
    actual_n = _count(rounded_s)
    print(f"    [width-scale] {spec.get('B')} params {n_default} → {actual_n} "
          f"(target {target}, scale={rounded_s:.4f})")
    return scaled, actual_n


# ---------------------------------------------------------------------------
# Self-test: generate specifications and run pipeline validation.
# ---------------------------------------------------------------------------

def self_test():
    import tempfile
    import config
    sys.path.insert(0, str(config.RUN_PIPELINE_DIR))
    from candidate_utils import validate_candidate_model_runtime

    specs = [
        {"A": ["RevIN"], "B": ["LSTM Block"], "C": "Linear Head", "E": "Single Block"},
        {"A": ["Series Decomposition", "Z-score"], "B": ["Linear Block"], "C": "Linear Head", "E": "Single Block"},
        {"A": ["Differencing"], "B": ["Transformer Encoder Block", "Linear Block"], "C": "MLP Head", "E": "Gated Fusion"},
        {"A": [], "B": ["State Space Model Block"], "C": "Basis Projection", "E": "Single Block"},
        {"A": ["Mask"], "B": ["CNN Block", "GRU Block"], "C": "Linear Head", "E": "Parallel Additive Branches"},
        {"A": ["Patch"], "B": ["Basis Block"], "C": "Linear Head", "E": "Single Block"},
    ]
    for station_num in [1, 7]:
        for spec in specs:
            for pre_len in [3, 12]:
                path = write_candidate_model(spec, 12, pre_len, station_num,
                                             os.path.join(tempfile.gettempdir(), "candidate_selftest.py"))
                try:
                    validate_candidate_model_runtime(path, 12, pre_len, station_num,
                                                     batch_sizes=(1, 2, 4), device="cpu",
                                                     auto_repair=False)
                    print(f"[OK] station={station_num} pre={pre_len} {spec['A']}|{spec['B']}|{spec['C']}|{spec['E']}")
                except Exception as e:
                    print(f"[FAIL] station={station_num} pre={pre_len} {spec['A']}|{spec['B']}|{spec['C']}|{spec['E']}: {type(e).__name__}: {str(e)[:120]}")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        self_test()
    else:
        print("Usage: python model_assembler.py --selftest")
