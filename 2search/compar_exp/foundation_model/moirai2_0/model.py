"""Moirai 2.0 with a 64-step context: four complete 16-step patches."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "_runtime"))
import torch
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module


class CandidateModel(torch.nn.Module):
    time_lag = 64
    input_patch_length = 16

    def __init__(self, horizon=3, station_num=1, *, checkpoint=None,
                 device="cuda", batch_size=256):
        super().__init__()
        if horizon not in (3, 12):
            raise ValueError("Horizon must be 3 or 12")
        self.horizon, self.station_num, self.batch_size = horizon, station_num, batch_size
        module = Moirai2Module.from_pretrained(
            str(checkpoint or HERE / "checkpoint"), local_files_only=True)
        if module.patch_size != self.input_patch_length:
            raise ValueError("Checkpoint input patch length is not 16")
        if self.time_lag // self.input_patch_length != 4:
            raise ValueError("This adapter requires exactly four input patches")
        self.forecaster = Moirai2Forecast(
            module=module, prediction_length=horizon, context_length=self.time_lag,
            target_dim=1, feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0).to(device).eval()
        self.median_index = list(module.quantile_levels).index(0.5)

    @torch.inference_mode()
    def forward(self, x):
        expected = (self.time_lag, self.station_num)
        if x.ndim != 3 or tuple(x.shape[1:]) != expected:
            raise ValueError(f"Expected [B, {expected[0]}, {expected[1]}], got {tuple(x.shape)}")
        if not torch.isfinite(x).all():
            raise ValueError("Inputs contain NaN/Inf")
        batch, _, channels = x.shape
        device = next(self.forecaster.parameters()).device
        flat = x.permute(0, 2, 1).reshape(-1, self.time_lag, 1).float().to(device)
        forecasts = []
        with self.forecaster.hparams_context(prediction_length=self.horizon):
            for values in flat.split(self.batch_size):
                quantiles = self.forecaster(
                    past_target=values,
                    past_observed_target=torch.ones_like(values, dtype=torch.bool),
                    past_is_pad=torch.zeros(values.shape[:2], dtype=torch.bool, device=device))
                forecasts.append(quantiles[:, self.median_index].reshape(len(values), self.horizon))
        return torch.cat(forecasts).reshape(batch, channels, self.horizon).permute(0, 2, 1).to(
            device=x.device, dtype=x.dtype)
