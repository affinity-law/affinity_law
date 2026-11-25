import torch
import torch.nn as nn
import torch.nn.functional as F


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
        return _A_RevIN(1)
    if name == "Differencing":
        return _A_Differencing()
    if name == "Series Decomposition":
        return _A_SeriesDecomposition(12, 3)
    if name == "Patch":
        return _A_Patch(12, 1)
    if name == "Mask":
        return _A_Mask(1)
    raise ValueError("unknown A: " + name)


def _build_b(name):
    if name == "Linear Block":
        return _B_Linear(12, 1)
    if name == "MLP Block":
        return _B_MLP(12, 1, hidden=64)
    if name == "CNN Block":
        return _B_CNN(12, 1, hidden=32)
    if name == "TCN Block":
        return _B_TCN(12, 1, hidden=32)
    if name == "GRU Block":
        return _B_GRU(12, 1, hidden=64)
    if name == "LSTM Block":
        return _B_LSTM(12, 1, hidden=64)
    if name == "State Space Model Block":
        return _B_SSM(12, 1, hidden=32)
    if name == "Transformer Encoder Block":
        return _B_Transformer(12, 1, hidden=64)
    if name == "Basis Block":
        return _B_Basis(12, 1, n_basis=32)
    raise ValueError("unknown B: " + name)


def _build_c(name):
    if name == "Linear Head":
        return _C_LinearHead(12, 3, 1)
    if name == "MLP Head":
        return _C_MLPHead(12, 3, 1, hidden=128)
    if name == "Basis Projection":
        return _C_BasisProjection(12, 3, 1, n_basis=32)
    raise ValueError("unknown C: " + name)


class CandidateModel(nn.Module):
    def __init__(self, time_lag, pre_len, station_num):
        super().__init__()
        self.time_lag = time_lag
        self.pre_len = pre_len
        self.station_num = station_num
        self.a_0 = _build_a("RevIN")
        self.b_0 = _build_b("State Space Model Block")
        self.backbone = _E_Composition([self.b_0], "Single Block")
        self.head = _build_c("Basis Projection")
        self.output_adapt = _D_OutputAdapt([self.a_0])

    def forward(self, x):
        x = self.a_0(x)
        z = self.backbone(x)
        out = self.head(z)
        out = self.output_adapt(out)
        return out
