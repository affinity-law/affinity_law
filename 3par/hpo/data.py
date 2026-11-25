"""Chronological splits and sliding windows with test data isolated from tuning."""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


TARGET_START_OFFSET = 12
TARGET_ALIGNMENT = "split_start_plus_12_with_past_context_v1"


def input_time_lag(config, settings):
    lag = (config or {}).get("time_lag", settings["time_lag"])
    if type(lag) is not int or lag < 1:
        raise ValueError("Input time_lag must be a positive integer")
    return lag


def input_window_details(config, settings):
    """Only new configurations add alignment fields to frozen request identities."""
    if not config or "time_lag" not in config:
        return {}
    lag = input_time_lag(config, settings)
    if lag < TARGET_START_OFFSET:
        raise ValueError("Aligned input windows require time_lag of at least 12")
    return {"time_lag": lag, "target_start_offset": TARGET_START_OFFSET,
            "target_alignment": TARGET_ALIGNMENT}


def evaluation_segment(data, start, end, config, settings):
    """Use past context so every input length predicts the original split dates."""
    window = input_window_details(config, settings)
    context = max(0, window["time_lag"] - TARGET_START_OFFSET) if window else 0
    if start < context:
        raise ValueError("Insufficient past context to preserve the fixed evaluation target dates")
    return np.asarray(data[start - context:end], dtype=np.float32)


def series_view(path):
    data = np.load(path, mmap_mode="r", allow_pickle=False)
    if data.ndim == 1:
        return data[:, None]
    if data.ndim == 2:
        return data
    if data.ndim == 3 and data.shape[2] == 1:
        return data[:, :, 0]
    raise ValueError(f"Unsupported data shape: {data.shape}")


def split_bounds(length, settings):
    train_end = int(length * settings["train_ratio"])
    val_end = train_end + int(length * settings["val_ratio"])
    return train_end, val_end


def effective_series(path, settings):
    data = series_view(path)
    if settings.get("smoke"):
        # Engineering smoke test only: use a short prefix of the original training segment.
        end, _ = split_bounds(len(data), settings)
        data = data[:min(end, settings["smoke_prefix_points"])]
    return data


class WindowDataset(Dataset):
    def __init__(self, data, lag, horizon):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.lag, self.horizon = lag, horizon
        self.length = len(self.data) - lag - horizon + 1
        if self.length <= 0:
            raise ValueError("A split is too short to produce one window")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        middle = index + self.lag
        return self.data[index:middle], self.data[middle:middle + self.horizon]


def loader(data, task, settings, batch_size, shuffle, seed, config=None):
    return DataLoader(WindowDataset(data, input_time_lag(config, settings), task["horizon"]),
                      batch_size=batch_size, shuffle=shuffle, drop_last=False,
                      num_workers=0, generator=torch.Generator().manual_seed(seed))


def train_val_loaders(path, task, settings, batch_size, seed, config=None):
    data = effective_series(path, settings)
    train_end, val_end = split_bounds(len(data), settings)
    train = np.asarray(data[:train_end], dtype=np.float32)
    val = evaluation_segment(data, train_end, val_end, config, settings)
    if not np.isfinite(train).all() or not np.isfinite(val).all():
        raise ValueError("The training or validation segment contains NaN/Inf; cross-split interpolation is disabled")
    mean, std = train.mean(axis=0, keepdims=True), train.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    train_loader = loader((train - mean) / std, task, settings, batch_size, True, seed, config)
    val_loader = loader((val - mean) / std, task, settings, batch_size, False, seed, config)
    return train_loader, val_loader, mean, std


def validation_diagnostics(path, settings):
    """Check whether the validation segment has enough variation for tuning."""
    data = effective_series(path, settings)
    train_end, val_end = split_bounds(len(data), settings)
    train = np.asarray(data[:train_end], dtype=np.float64)
    val = np.asarray(data[train_end:val_end], dtype=np.float64)
    differences = np.diff(val, axis=0)
    difference_rmse = float(np.sqrt(np.mean(differences ** 2))) if len(differences) else 0.0
    train_scale = float(np.sqrt(np.mean((train - train.mean(axis=0, keepdims=True)) ** 2)))
    threshold = max(1e-12, train_scale * 1e-8)
    return {"validation_points": len(val), "temporal_difference_rmse": difference_rmse,
            "train_scale": train_scale, "minimum_signal": threshold,
            "tuning_allowed": difference_rmse > threshold,
            "reason": "usable" if difference_rmse > threshold else "validation_segment_has_no_temporal_signal"}


def test_loader(path, task, settings, batch_size, seed, mean, std, config=None):
    data = effective_series(path, settings)
    _, val_end = split_bounds(len(data), settings)
    test = evaluation_segment(data, val_end, len(data), config, settings)
    if not np.isfinite(test).all():
        raise ValueError("The test segment contains NaN/Inf")
    return loader((test - mean) / std, task, settings, batch_size, False, seed, config)
