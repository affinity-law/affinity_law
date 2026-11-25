import csv
import os
import random

import numpy as np
import torch
import torch.nn as nn

import project_config as cfg
from candidate_utils import load_candidate_model_class, validate_candidate_model_runtime
from dataset import create_train_val_loaders


class WeightedMSEMAELoss(nn.Module):
    """Weighted combination loss of MSE and MAE."""

    def __init__(self, mse_weight=0.5, mae_weight=0.5):
        super().__init__()
        self.mse_weight = float(mse_weight)
        self.mae_weight = float(mae_weight)
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

    def forward(self, pred, target):
        return self.mse_weight * self.mse(pred, target) + self.mae_weight * self.mae(pred, target)


def build_criterion():
    """Create the training objective based on TRAIN_LOSS_TYPE in project_config.py."""
    loss_type = str(getattr(cfg, "TRAIN_LOSS_TYPE", "mse_mae")).lower().strip()

    if loss_type == "mse_mae":
        mse_weight = float(getattr(cfg, "MSE_LOSS_WEIGHT", 0.5))
        mae_weight = float(getattr(cfg, "MAE_LOSS_WEIGHT", 0.5))
        print(f"Training loss function: MSE + MAE, mse_weight={mse_weight}, mae_weight={mae_weight}")
        return WeightedMSEMAELoss(mse_weight=mse_weight, mae_weight=mae_weight)

    if loss_type == "huber":
        beta = float(getattr(cfg, "HUBER_BETA", 0.5))
        print(f"Training loss function: Huber/SmoothL1, beta={beta}")
        return nn.SmoothL1Loss(beta=beta)

    raise ValueError("TRAIN_LOSS_TYPE only supports 'mse_mae' or 'huber'.")


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = total_samples = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = total_mae = total_se = total_samples = total_elems = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        batch_size = x.size(0)

        total_loss += criterion(pred, y).item() * batch_size
        total_mae += torch.mean(torch.abs(pred - y)).item() * batch_size
        total_se += torch.sum((pred - y) ** 2).item()
        total_elems += pred.numel()
        total_samples += batch_size

    n = max(total_samples, 1)
    val_loss = total_loss / n
    val_mae = total_mae / n
    val_rmse = float(np.sqrt(total_se / max(total_elems, 1)))
    return val_loss, val_mae, val_rmse


def save_history(history, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_mae", "val_rmse", "best_val_loss"])
        writer.writeheader()
        writer.writerows(history)


def save_best_model(model, model_path, stats_path, mean, std, station_num):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    np.savez(
        stats_path,
        mean=mean,
        std=std,
        time_lag=cfg.TIME_LAG,
        pre_len=cfg.PRE_LEN,
        station_num=station_num,
        univariate=cfg.UNIVARIATE,
    )
    print("Best model saved to:", model_path)
    print("Stats saved to:", stats_path)


def main():
    output_dir = getattr(cfg, "TRAIN_OUTPUT_DIR", "") or os.path.join(cfg.RESULT_DIR, "manual_run", "train")
    candidate_model_path = getattr(cfg, "CANDIDATE_MODEL_PATH", "") or os.path.join(cfg.RESULT_DIR, "manual_run", "train", cfg.GENERATED_MODEL_FILE)
    seed = getattr(cfg, "SEARCH_SEED", None)
    os.makedirs(output_dir, exist_ok=True)

    best_model_path = os.path.join(output_dir, cfg.BEST_MODEL_FILE)
    train_stats_path = os.path.join(output_dir, cfg.TRAIN_STATS_FILE)
    train_history_path = os.path.join(output_dir, cfg.TRAIN_HISTORY_FILE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("Candidate model file:", candidate_model_path)
    print("Training output directory:", output_dir)

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        print("Random seed:", seed)

    # Create training and validation DataLoaders. mean/std is computed from the training set only to prevent data leakage.
    train_loader, val_loader, station_num, mean, std = create_train_val_loaders(
        npy_path=cfg.NPY_PATH,
        time_lag=cfg.TIME_LAG,
        pre_len=cfg.PRE_LEN,
        batch_size=cfg.BATCH_SIZE,
        train_ratio=cfg.TRAIN_RATIO,
        val_ratio=cfg.VAL_RATIO,
        test_ratio=cfg.TEST_RATIO,
        num_workers=0,
        univariate=cfg.UNIVARIATE,
        seed=seed,
    )
    print("station_num:", station_num)
    print("train batches:", len(train_loader))
    print("val batches:", len(val_loader))

    validate_candidate_model_runtime(
        candidate_model_path,
        time_lag=cfg.TIME_LAG,
        pre_len=cfg.PRE_LEN,
        station_num=station_num,
        batch_sizes=(1, 2, cfg.BATCH_SIZE),
        device="cpu",
        auto_repair=bool(getattr(cfg, "AUTO_REPAIR_CANDIDATE_SYNTAX", True)),
    )
    # Dynamically load the LLM-generated CandidateModel class.
    CandidateModel = load_candidate_model_class(
        candidate_model_path,
        auto_repair=bool(getattr(cfg, "AUTO_REPAIR_CANDIDATE_SYNTAX", True)),
    )
    model = CandidateModel(cfg.TIME_LAG, cfg.PRE_LEN, station_num).to(device)
    criterion = build_criterion()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)

    best_val_loss = float("inf")
    no_improve_epochs = 0
    history = []

    # Main training loop: train for one epoch, then evaluate on the validation set to decide whether to save the best model.
    for epoch in range(1, cfg.EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_mae, val_rmse = evaluate(model, val_loader, criterion, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epochs = 0
            save_best_model(model, best_model_path, train_stats_path, mean, std, station_num)
        else:
            no_improve_epochs += 1

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mae": val_mae,
            "val_rmse": val_rmse,
            "best_val_loss": best_val_loss,
        })

        print(
            f"Epoch [{epoch:03d}/{cfg.EPOCHS:03d}] "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"val_mae={val_mae:.6f} val_rmse={val_rmse:.6f} best_val_loss={best_val_loss:.6f}"
        )

        if cfg.EARLY_STOP_PATIENCE > 0 and no_improve_epochs >= cfg.EARLY_STOP_PATIENCE:
            print(f"Validation loss has not improved for {cfg.EARLY_STOP_PATIENCE} consecutive epochs, stopping early.")
            break

    # Write the full training history once at the end (avoids per-epoch disk I/O).
    save_history(history, train_history_path)


if __name__ == "__main__":
    main()
