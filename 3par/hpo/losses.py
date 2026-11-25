"""Named normalized training losses; the original L loss remains the default."""

DEFAULT_LOSS_PROFILE = "mae_heavy"
LOSS_PROFILES = {
    "mae_heavy": (0.2, 0.8),
    "balanced": (0.5, 0.5),
    "mse": (1.0, 0.0),
}


def loss_details(config=None, settings=None):
    """Resolve old configurations without a loss field to the original loss."""
    profile = (config or {}).get("loss_profile", DEFAULT_LOSS_PROFILE)
    if not isinstance(profile, str) or profile not in LOSS_PROFILES:
        raise ValueError(f"Unknown loss_profile: {profile!r}")
    mse, mae = LOSS_PROFILES[profile]
    return {"loss_profile": profile, "mse_weight": mse, "mae_weight": mae,
            "checkpoint_metric": f"normalized_validation_{mse:g}_mse_{mae:g}_mae_loss"}
