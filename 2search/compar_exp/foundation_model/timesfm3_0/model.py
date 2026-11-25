"""TimesFM 3.0 with a 64-step context: two complete 32-step patches."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "_runtime"))
import numpy as np
import torch
from timesfm3 import ModelConfig, TimesFM3Evaluator


class CandidateModel(torch.nn.Module):
    time_lag = 64
    input_patch_length = 32

    def __init__(self, horizon=3, station_num=1, *, checkpoint=None,
                 device="cuda", batch_size=32):
        super().__init__()
        if horizon not in (3, 12):
            raise ValueError("Horizon must be 3 or 12")
        self.horizon, self.station_num = horizon, station_num
        self.forecaster = TimesFM3Evaluator(ModelConfig(
            checkpoint_path=str(checkpoint or HERE / "checkpoint"),
            per_core_batch_size=batch_size, device=device, local_files_only=True))
        self.backbone = self.forecaster.model
        if self.forecaster.config.input_patch_length != self.input_patch_length:
            raise ValueError("Checkpoint input patch length is not 32")
        if self.time_lag // self.input_patch_length != 2:
            raise ValueError("This adapter requires exactly two input patches")

    @torch.inference_mode()
    def forward(self, x):
        expected = (self.time_lag, self.station_num)
        if x.ndim != 3 or tuple(x.shape[1:]) != expected:
            raise ValueError(f"Expected [B, {expected[0]}, {expected[1]}], got {tuple(x.shape)}")
        if not torch.isfinite(x).all():
            raise ValueError("Inputs contain NaN/Inf")
        self.forecaster.device = next(self.backbone.parameters()).device
        outputs = list(self.forecaster.predict_batch(
            [a.T for a in x.detach().float().cpu().numpy()], horizon=self.horizon,
            return_quantiles=False, use_symmetric_averaging=True,
            make_positive=False, sort_quantiles=True, use_znorm=False,
            univariate=False))
        prediction = np.stack([out.forecast.T for out in outputs])
        return torch.as_tensor(prediction, dtype=x.dtype, device=x.device)
