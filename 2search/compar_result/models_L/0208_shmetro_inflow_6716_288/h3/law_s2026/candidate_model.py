import torch
import torch.nn as nn
import torch.nn.functional as F


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



class _C_MLPHead(nn.Module):
    def __init__(self, time_lag, pre_len, station_num, hidden=128):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(time_lag, hidden),
                                  nn.GELU(),
                                  nn.Linear(hidden, pre_len))
    def forward(self, z):
        return self.proj(z.transpose(1, 2)).transpose(1, 2)



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

class _D_OutputAdapt(nn.Module):
    """Apply inverse transforms and trend recomposition in reverse stage-A order."""
    def __init__(self, invertible):
        super().__init__()
        self.invertible = nn.ModuleList(invertible)
    def forward(self, out):
        for m in reversed(self.invertible):
            out = m.inverse(out)
        return out


def _build_a(name):
    if name == "Identity":
        return _A_Identity()
    if name == "Z-score":
        return _A_ZScore()
    if name == "RevIN":
        return _A_RevIN(288)
    if name == "Differencing":
        return _A_Differencing()
    if name == "Series Decomposition":
        return _A_SeriesDecomposition(12, 3)
    if name == "Patch":
        return _A_Patch(12, 288)
    if name == "Mask":
        return _A_Mask(288)
    raise ValueError("unknown A: " + name)


def _build_b(name):
    if name == "Linear Block":
        return _B_Linear(12, 288)
    if name == "MLP Block":
        return _B_MLP(12, 288, hidden=64)
    if name == "CNN Block":
        return _B_CNN(12, 288, hidden=32)
    if name == "TCN Block":
        return _B_TCN(12, 288, hidden=32)
    if name == "GRU Block":
        return _B_GRU(12, 288, hidden=64)
    if name == "LSTM Block":
        return _B_LSTM(12, 288, hidden=64)
    if name == "State Space Model Block":
        return _B_SSM(12, 288, hidden=32)
    if name == "Transformer Encoder Block":
        return _B_Transformer(12, 288, hidden=64)
    if name == "Basis Block":
        return _B_Basis(12, 288, n_basis=32)
    raise ValueError("unknown B: " + name)


def _build_c(name):
    if name == "Linear Head":
        return _C_LinearHead(12, 3, 288)
    if name == "MLP Head":
        return _C_MLPHead(12, 3, 288, hidden=128)
    if name == "Basis Projection":
        return _C_BasisProjection(12, 3, 288, n_basis=32)
    raise ValueError("unknown C: " + name)


class CandidateModel(nn.Module):
    def __init__(self, time_lag, pre_len, station_num):
        super().__init__()
        self.time_lag = time_lag
        self.pre_len = pre_len
        self.station_num = station_num
        self.a_0 = _build_a("Patch")
        self.b_0 = _build_b("TCN Block")
        self.backbone = _E_Composition([self.b_0], "Single Block")
        self.head = _build_c("MLP Head")
        self.output_adapt = _D_OutputAdapt([])

    def forward(self, x):
        x = self.a_0(x)
        z = self.backbone(x)
        out = self.head(z)
        out = self.output_adapt(out)
        return out
