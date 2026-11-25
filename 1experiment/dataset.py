import math

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class TimeSeriesDataset(Dataset):
    """Slice a continuous time series into sliding window samples."""

    def __init__(self, data, time_lag, pre_len):
        data = np.asarray(data)
        if data.ndim == 1:
            data = data[:, None]
        if data.ndim != 2:
            raise ValueError(f"data shape must be [T, C], got {data.shape}")

        self.data = torch.tensor(data, dtype=torch.float32)
        self.time_lag = time_lag
        self.pre_len = pre_len
        self.length = len(self.data) - time_lag - pre_len + 1
        if self.length <= 0:
            raise ValueError(f"Data too short: len={len(self.data)}, time_lag={time_lag}, pre_len={pre_len}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x_start = idx
        x_end = idx + self.time_lag
        y_end = x_end + self.pre_len
        return self.data[x_start:x_end], self.data[x_end:y_end]


def load_npy_data(npy_path, univariate=False):
    """Read a time series npy file with shape [T], [T, C], or [T, C, 1]."""
    data = np.load(npy_path, allow_pickle=True)
    if not isinstance(data, np.ndarray):
        raise TypeError("Loaded npy is not a numpy.ndarray")

    print("Original data shape:", data.shape)

    if data.ndim == 1:
        data = data[:, None]

    elif data.ndim == 2:
        pass

    elif data.ndim == 3:
        # Support the common traffic data format [T, N, F]
        # When F=1, squeeze to [T, N]
        if data.shape[2] == 1:
            data = data[:, :, 0]
            print("Detected [T, C, 1] data, squeezed to [T, C].")
        else:
            raise ValueError(
                f"Detected 3D data shape={data.shape}, but the last dimension feature count is not 1."
                f"The current project only supports [T], [T, C], or [T, C, 1]."
            )

    else:
        raise ValueError(f"Shape={data.shape} is not currently supported, please reshape to [T], [T, C], or [T, C, 1]")

    if data.shape[0] <= data.shape[1]:
        print(f"Warning: shape={data.shape}, please ensure dimension 0 is the time dimension.")

    if univariate:
        data = data[:, :1]
        print("Univariate mode enabled, using only the 0th variable.")

    print("Data shape before standardization:", data.shape)
    return data.astype(np.float32)

def split_series(data, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    """Split data into training, validation, and test sets in chronological order."""
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, rel_tol=1e-6):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1")

    train_end = int(len(data) * train_ratio)
    val_end = train_end + int(len(data) * val_ratio)
    return data[:train_end], data[train_end:val_end], data[val_end:]


def standardize_by_train(train_data, val_data, test_data):
    """Standardize using only the training set mean and variance to prevent data leakage."""
    if train_data.ndim == 1:
        train_data, val_data, test_data = train_data[:, None], val_data[:, None], test_data[:, None]

    mean = train_data.mean(axis=0, keepdims=True)
    std = train_data.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return (train_data - mean) / std, (val_data - mean) / std, (test_data - mean) / std, mean, std


def _prepare_data(npy_path, train_ratio, val_ratio, test_ratio, univariate):
    """Shared data preparation pipeline for training, validation, and testing: read -> split -> standardize using training stats."""
    data = load_npy_data(npy_path, univariate=univariate)
    train_data, val_data, test_data = split_series(data, train_ratio, val_ratio, test_ratio)
    return standardize_by_train(train_data, val_data, test_data)


def _make_loader(data, time_lag, pre_len, batch_size, shuffle, num_workers, seed=None):
    """Wrap a segment of the time series into a PyTorch DataLoader. When seed is not None, a fixed generator is used to ensure reproducible shuffle."""
    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )
    if seed is not None:
        kwargs["generator"] = torch.Generator().manual_seed(seed)
    return DataLoader(
        TimeSeriesDataset(data, time_lag, pre_len),
        **kwargs,
    )


def create_train_val_loaders(
    npy_path,
    time_lag=96,
    pre_len=24,
    batch_size=32,
    train_ratio=0.7,
    val_ratio=0.1,
    test_ratio=0.2,
    num_workers=0,
    univariate=False,
    seed=None,
):
    train_data, val_data, _, mean, std = _prepare_data(
        npy_path, train_ratio, val_ratio, test_ratio, univariate
    )
    return (
        _make_loader(train_data, time_lag, pre_len, batch_size, True, num_workers, seed=seed),
        _make_loader(val_data, time_lag, pre_len, batch_size, False, num_workers, seed=seed),
        train_data.shape[1],
        mean,
        std,
    )


def create_test_loader(
    npy_path,
    time_lag=96,
    pre_len=24,
    batch_size=32,
    train_ratio=0.7,
    val_ratio=0.1,
    test_ratio=0.2,
    num_workers=0,
    univariate=False,
    seed=None,
):
    _, _, test_data, mean, std = _prepare_data(
        npy_path, train_ratio, val_ratio, test_ratio, univariate
    )
    return _make_loader(test_data, time_lag, pre_len, batch_size, False, num_workers, seed=seed), test_data.shape[1], mean, std
