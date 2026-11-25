# -*- coding: utf-8 -*-
"""
ts_meta_feature_extractor.py
Reads npy time-series files and outputs:
1) raw_indicators.csv: the 39 data features in Supplementary Table 3
2) meta_features.csv: 7 meta-features

Data shape: expected [samples, features]; 1-D arrays are automatically reshaped to [samples, 1].

MAX_ACF_LAG=200, TREND_WINDOW=201, ROLL_VAR_WINDOW=201, CHANGEPOINT_WINDOW=201 should be adjusted
according to the sampling frequency.
"""

import os
import math
import tempfile
import warnings
from collections import Counter

import numpy as np
import pandas as pd


# =======================
# Local constants (not included in project_config)
# =======================
HORIZON = 1                        # Forecast horizon; keep 1 when forecast step is not considered
MAX_LAG_HURST = 200                 # Hurst maximum lag; log-spaced lags are used in practice
PERMUTATION_ORDER = 5               # Permutation entropy order
NEAR_ZERO_ABS_TOL = 1e-8
PLATEAU_ABS_TOL = 1e-8
EPS = 1e-8
RANDOM_SEED = 2026                  # Deterministic seed for multivariate pair sampling

# ── Read configurable hyperparameters from project_config ──
try:
    import project_config as _cfg
    SEASONAL_PERIOD = int(getattr(_cfg, "SEASONAL_PERIOD", 0))
    AUTO_ESTIMATE_SEASONAL_PERIOD = bool(getattr(_cfg, "AUTO_ESTIMATE_SEASONAL_PERIOD", True))
    MIN_AUTO_SEASONAL_PERIOD = int(getattr(_cfg, "MIN_AUTO_SEASONAL_PERIOD", 2))
    MAX_AUTO_SEASONAL_PERIOD_RATIO = float(getattr(_cfg, "MAX_AUTO_SEASONAL_PERIOD_RATIO", 1.0 / 3.0))
    REQUIRE_STAT_TESTS = bool(getattr(_cfg, "REQUIRE_STAT_TESTS", True))
    MAX_ACF_LAG = int(getattr(_cfg, "MAX_ACF_LAG", 48))
    MAX_CROSS_LAG = int(getattr(_cfg, "MAX_CROSS_LAG", 24))
    CORR_THRESHOLD = float(getattr(_cfg, "CORR_THRESHOLD", 0.30))
    MAX_CROSS_PAIRS = int(getattr(_cfg, "MAX_CROSS_PAIRS", 100))
    ENTROPY_WINDOW = int(getattr(_cfg, "ENTROPY_WINDOW", 200))
    MAX_ENTROPY_WINDOWS = int(getattr(_cfg, "MAX_ENTROPY_WINDOWS", 8))
    MAX_ENTROPY_POINTS = int(getattr(_cfg, "MAX_ENTROPY_POINTS", 144))
    HIGH_FREQ_MAX_PERIOD = int(getattr(_cfg, "HIGH_FREQ_MAX_PERIOD", 8))
    TREND_WINDOW = int(getattr(_cfg, "TREND_WINDOW", 101))
    ROLL_VAR_WINDOW = int(getattr(_cfg, "ROLL_VAR_WINDOW", 51))
    CHANGEPOINT_WINDOW = int(getattr(_cfg, "CHANGEPOINT_WINDOW", 51))
except ImportError:
    pass


# =======================
# Utility Functions
# =======================
def finite_float(x):
    try:
        x = float(x)
        return x if math.isfinite(x) else np.nan
    except Exception:
        return np.nan


def clip01(x, default=0.5):
    if not np.isfinite(x):
        return default
    return float(np.clip(x, 0.0, 1.0))


def score_pos(x, scale=1.0, default=0.5):
    """Squash a non-negative indicator to [0,1]; larger values approach 1."""
    if not np.isfinite(x):
        return default
    return clip01(1.0 - np.exp(-abs(float(x)) / (scale + EPS)))


def score01(x, default=0.5):
    """Use when the indicator is already approximately in [0,1]."""
    return clip01(x, default=default)


def mean_finite(values, default=0.5):
    """Average only over finite values, avoiding NaN silently being treated as 0.5."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return default
    return float(np.mean(arr))


def nanmedian(xs, default=np.nan):
    xs = np.asarray(xs, dtype=float)
    xs = xs[np.isfinite(xs)]
    return float(np.median(xs)) if len(xs) else default


def iqr(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    q75, q25 = np.percentile(x, [75, 25])
    return float(q75 - q25)


def mad(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    m = np.median(x)
    return float(np.median(np.abs(x - m)))


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = min(len(a), len(b))
    if n < 3:
        return 0.0
    a, b = a[:n], b[:n]
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3:
        return 0.0
    if np.std(a) < EPS or np.std(b) < EPS:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def acf(x, lag):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if lag <= 0:
        return 1.0
    if lag >= n - 1 or np.var(x) < EPS:
        return 0.0
    z = x - np.mean(x)
    return float(np.sum(z[:-lag] * z[lag:]) / (np.sum(z ** 2) + EPS))


def roll_mean(x, win):
    win = int(max(1, min(win, len(x))))
    return pd.Series(x).rolling(win, center=True, min_periods=1).mean().to_numpy()


def roll_var(x, win):
    win = int(max(2, min(win, len(x))))
    return pd.Series(x).rolling(win, center=True, min_periods=2).var().fillna(0).to_numpy()


def fixed_window(n, win, min_win=3):
    """Fixed window length; automatically truncated to n when the series is short."""
    if n <= 0:
        return 1
    return int(max(min_win, min(int(win), int(n))))


def make_odd(x):
    """Odd window sizes work better with center=True in rolling operations."""
    x = int(max(1, x))
    return x if x % 2 == 1 else x + 1


def get_effective_lags(n, seasonal_period=0):
    """
    Automatically set several effective lags/windows for long series.
    This preserves the original hyperparameters while avoiding inappropriate
    fixed values for long series.
    """
    if seasonal_period and seasonal_period > 1:
        # Cover at least 3 cycles, but not exceeding n/4.
        acf_lag = min(n // 4, max(MAX_ACF_LAG, 3 * int(seasonal_period)))
        trend_win = make_odd(min(n, max(TREND_WINDOW, int(seasonal_period))))
        roll_win = make_odd(min(n, max(ROLL_VAR_WINDOW, int(seasonal_period))))
        cp_win = min(n // 4, max(CHANGEPOINT_WINDOW, int(seasonal_period)))
    else:
        # No explicit seasonal period: scale windows with series length.
        # Config values act as FLOORS; n-based fractions act as CEILINGS.
        # This ensures reasonable behavior from n=800 to n=50000+.
        acf_lag = min(n // 4, max(MAX_ACF_LAG, min(2000, n // 10)))
        trend_win = make_odd(min(n // 4, max(TREND_WINDOW, n // 10)))
        roll_win = make_odd(min(n // 4, max(ROLL_VAR_WINDOW, n // 20)))
        cp_win = min(n // 4, max(CHANGEPOINT_WINDOW, n // 20))

    acf_lag = int(max(1, acf_lag))
    trend_win = fixed_window(n, trend_win, min_win=3)
    roll_win = fixed_window(n, roll_win, min_win=2)
    cp_win = int(max(5, min(cp_win, n // 4))) if n >= 20 else 5
    cross_lag = int(max(1, min(48, max(MAX_CROSS_LAG, n // 40))))

    return {
        "acf_lag": acf_lag,
        "trend_win": trend_win,
        "roll_var_win": roll_win,
        "changepoint_win": cp_win,
        "cross_lag": cross_lag,
    }


def load_npy(path):
    """
    Read an npy file.
    This version assumes your data has already been preprocessed and does not
    output additional missing-value quality indicators.
    If the data still contain NaN/inf, a simple interpolation is performed
    to prevent computational failures.
    """
    y = np.load(path).astype(float)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    elif y.ndim > 2:
        y = y.reshape(y.shape[0], -1)

    y[~np.isfinite(y)] = np.nan
    df = pd.DataFrame(y).interpolate(axis=0, limit_direction="both")
    df = df.fillna(df.median(numeric_only=True)).fillna(0.0)
    return df.to_numpy(float)


# =======================
# Univariate Features
# =======================
def nonzero_frequency_spectrum(x):
    """Return positive FFT frequencies and their power, excluding the mean."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 4 or np.std(x) < EPS:
        return np.empty(0), np.empty(0)

    power = np.abs(np.fft.rfft(x - np.mean(x))) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0)
    return freqs[1:], power[1:]


def fft_features(x):
    """Compute spectral peak ratio, spectral entropy and high-frequency energy."""
    freqs, power = nonzero_frequency_spectrum(x)
    total = np.sum(power)
    if total < EPS or len(power) == 0:
        return 0.0, 1.0, 0.0

    peak_ratio = float(np.max(power) / total)

    q = power / total
    spec_entropy = float(-np.sum(q * np.log(q + EPS)) / np.log(len(q) + EPS))

    cutoff_freq = 1.0 / max(float(HIGH_FREQ_MAX_PERIOD), 2.0)
    high_ratio = float(np.sum(power[freqs >= cutoff_freq]) / total)
    return peak_ratio, spec_entropy, high_ratio



def estimate_seasonal_period_from_fft(x):
    """
    When SEASONAL_PERIOD=0, automatically estimate a candidate period from the spectrum.

    Note: Automatic estimation is only a fallback.
    If you know the sampling frequency of your traffic data, it is best to set it explicitly:
    5 min = 288, 15 min = 96, hourly = 24.
    """
    x = np.asarray(x, float)
    n = len(x)
    if n < 12 or np.std(x) < EPS:
        return 0.0

    power = np.abs(np.fft.rfft(x - np.mean(x))) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0)

    # Remove the zero-frequency (mean) component.
    power = power[1:]
    freqs = freqs[1:]
    if len(power) == 0 or np.sum(power) < EPS:
        return 0.0

    periods = 1.0 / (freqs + EPS)
    max_period = max(MIN_AUTO_SEASONAL_PERIOD, int(n * MAX_AUTO_SEASONAL_PERIOD_RATIO))

    # The period must not be too short or too long; if too long, there will be only
    # 1-2 cycles and seasonal stability becomes meaningless.
    ok = (periods >= MIN_AUTO_SEASONAL_PERIOD) & (periods <= max_period)
    if not np.any(ok):
        return 0.0

    cand_power = power[ok]
    cand_periods = periods[ok]
    k = int(np.argmax(cand_power))
    return finite_float(cand_periods[k])


def seasonal_features(x, period, trend):
    """Seasonal strength, periodic ACF peak, seasonal stability, seasonal component."""
    n, period = len(x), int(round(period))
    if period < 2 or n < 2 * period:
        return 0.0, 0.0, 0.0, np.zeros(n)

    detrended = x - trend
    profile = np.array([np.mean(detrended[i::period]) for i in range(period)])
    profile -= np.mean(profile)
    seasonal = profile[np.arange(n) % period]
    residual = detrended - seasonal

    strength = max(0.0, 1.0 - np.var(residual) / (np.var(detrended) + EPS))
    acf_peak = abs(acf(x, period))

    cycles_n = n // period
    if cycles_n < 3:
        stability = 0.0
    else:
        cycles = detrended[:cycles_n * period].reshape(cycles_n, period)
        cs = [corr(cycles[i], cycles[j]) for i in range(cycles_n) for j in range(i + 1, cycles_n)]
        stability = (nanmedian(cs, 0.0) + 1.0) / 2.0

    return finite_float(strength), finite_float(acf_peak), finite_float(stability), seasonal


def hurst(x):
    """Lightweight Hurst exponent. Long series use log-spaced lags to avoid overweighting short lags."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 32 or np.std(x) < EPS:
        return 0.5

    max_lag = min(MAX_LAG_HURST, n // 4)
    if max_lag < 4:
        return 0.5

    lags = np.unique(np.logspace(np.log10(2), np.log10(max_lag), 50).astype(int))
    tau = np.array([np.std(x[k:] - x[:-k]) for k in lags])
    ok = tau > EPS
    if np.sum(ok) < 3:
        return 0.5
    h = np.polyfit(np.log(lags[ok]), np.log(tau[ok]), 1)[0]
    return finite_float(np.clip(h, 0.0, 1.0))


def _sample_entropy_core(x, m=2):
    """Sample entropy for a single window, O(n^2); computed only within the window."""
    x = np.asarray(x, float)
    n = len(x)
    if n < m + 5 or np.std(x) < EPS:
        return 0.0

    # If the window is still too long, subsample evenly to keep computation manageable.
    if n > MAX_ENTROPY_POINTS:
        idx = np.linspace(0, n - 1, MAX_ENTROPY_POINTS).astype(int)
        x = x[idx]
        n = len(x)

    r = 0.2 * np.std(x)

    def count(order):
        temp = np.array([x[i:i + order] for i in range(n - order + 1)])
        c = 0
        for i in range(len(temp) - 1):
            d = np.max(np.abs(temp[i + 1:] - temp[i]), axis=1)
            c += int(np.sum(d <= r))
        return c

    b, a = count(m), count(m + 1)
    return finite_float(-np.log((a + 1.0) / (b + 1.0)))


def sample_entropy(x, m=2):
    """Windowed sample entropy. Long series take the median over several windows, which is more suitable for scientific reports."""
    x = np.asarray(x, float)
    n = len(x)
    if n <= ENTROPY_WINDOW:
        return _sample_entropy_core(x, m=m)

    starts = np.linspace(0, n - ENTROPY_WINDOW, MAX_ENTROPY_WINDOWS).astype(int)
    vals = []
    for s in starts:
        vals.append(_sample_entropy_core(x[s:s + ENTROPY_WINDOW], m=m))
    return nanmedian(vals, 0.0)


def _permutation_entropy_core(x, order=PERMUTATION_ORDER):
    """Permutation entropy for a single window, output approximately in [0,1]."""
    x = np.asarray(x, float)
    if len(x) < order + 2 or np.std(x) < EPS:
        return 0.0

    # Stable handling of ties: add a tiny deterministic jitter to each position
    # to avoid unstable sorting due to many equal values.
    jitter = np.linspace(0, EPS, len(x))
    x = x + jitter

    patterns = [tuple(np.argsort(x[i:i + order])) for i in range(len(x) - order + 1)]
    counts = np.array(list(Counter(patterns).values()), float)
    p = counts / np.sum(counts)
    return finite_float(-np.sum(p * np.log(p + EPS)) / np.log(math.factorial(order)))


def permutation_entropy(x, order=PERMUTATION_ORDER):
    """Windowed permutation entropy."""
    x = np.asarray(x, float)
    n = len(x)
    if n <= ENTROPY_WINDOW:
        return _permutation_entropy_core(x, order=order)

    starts = np.linspace(0, n - ENTROPY_WINDOW, MAX_ENTROPY_WINDOWS).astype(int)
    vals = []
    for s in starts:
        vals.append(_permutation_entropy_core(x[s:s + ENTROPY_WINDOW], order=order))
    return nanmedian(vals, 0.0)


def adf_kpss(x):
    """
    ADF p-value and KPSS statistic.
    Returns: adf_pvalue, kpss_statistic.
    """
    if len(x) < 20 or np.std(x) < EPS:
        return np.nan, np.nan
    try:
        from statsmodels.tsa.stattools import adfuller, kpss
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf_p = finite_float(adfuller(x, autolag="AIC")[1])
            kpss_stat = finite_float(kpss(x, regression="ct", nlags="auto")[0])
        return adf_p, kpss_stat
    except ImportError as e:
        if REQUIRE_STAT_TESTS:
            raise ImportError("Scientific experiments are recommended to install statsmodels, otherwise ADF/KPSS will not be reproducible.") from e
        return np.nan, np.nan
    except Exception:
        return np.nan, np.nan


def changepoint_density(x, win):
    """Lightweight changepoint density: a candidate changepoint is flagged when the difference between left and right window means exceeds a threshold."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 20 or np.std(x) < EPS:
        return 0.0

    win = int(max(5, min(win, n // 4)))
    if 2 * win >= n:
        return 0.0

    threshold = 1.5 * np.std(x)
    flags = []
    for t in range(win, n - win):
        left_mean = np.mean(x[t - win:t])
        right_mean = np.mean(x[t:t + win])
        flags.append(abs(right_mean - left_mean) > threshold)

    cnt, inside = 0, False
    for f in flags:
        if f and not inside:
            cnt, inside = cnt + 1, True
        elif not f:
            inside = False
    return finite_float(cnt / n)


def raw_ratio_features(x_raw):
    """Indicators that should only be computed on the original physical quantities."""
    x_raw = np.asarray(x_raw, float)
    finite_x = x_raw[np.isfinite(x_raw)]
    if len(finite_x) == 0:
        return {
            "near_zero_ratio": np.nan,
            "plateau_ratio": np.nan,
        }

    near_zero = float(np.mean(np.abs(finite_x) <= NEAR_ZERO_ABS_TOL))

    ok = np.isfinite(x_raw[:-1]) & np.isfinite(x_raw[1:]) if len(x_raw) >= 2 else np.array([])
    if len(ok) == 0 or np.sum(ok) == 0:
        plateau = np.nan
    else:
        plateau = float(np.mean(np.abs(np.diff(x_raw)[ok]) <= PLATEAU_ABS_TOL))

    return {
        "near_zero_ratio": near_zero,
        "plateau_ratio": plateau,
    }


def univar_features(x_clean, x_raw=None):
    """Compute univariate indicators. x_clean is used for most calculations; x_raw is used for physical-quantity indicators like missing/zero/plateau ratios."""
    x = np.asarray(x_clean, float)
    n = len(x)
    lags = get_effective_lags(n, SEASONAL_PERIOD)

    med, x_iqr, x_mad, x_std = np.median(x), iqr(x), mad(x), np.std(x)
    var_x = np.var(x)

    if x_std < EPS:
        skew, kurt = 0.0, 0.0
    else:
        z = (x - np.mean(x)) / (x_std + EPS)
        skew, kurt = float(np.mean(z ** 3)), float(np.mean(z ** 4) - 3.0)

    robust_cv = x_mad / (abs(med) + EPS)
    range_iqr = (np.max(x) - np.min(x)) / (x_iqr + EPS)
    outlier = float(np.mean(np.abs(x - med) > 3.0 * (x_mad + EPS)))

    # near-zero and plateau use the original physical quantities; fall back to clean data if x_raw is not provided.
    raw_part = raw_ratio_features(x_raw if x_raw is not None else x)
    near_zero = raw_part["near_zero_ratio"]
    plateau = raw_part["plateau_ratio"]

    # Trend: window adapts to period/length for long series.
    trend = roll_mean(x, lags["trend_win"])
    trend_strength = max(0.0, 1.0 - np.var(x - trend) / (var_x + EPS)) if var_x > EPS else 0.0

    if n >= 3 and x_iqr > EPS:
        t = np.linspace(0, 1, n)
        slope = np.polyfit(t, x, 1)[0] / (x_iqr + EPS)
        curvature = abs(np.polyfit(t, x, 2)[0]) / (x_iqr + EPS)
    else:
        slope, curvature = 0.0, 0.0

    monotonic = abs(corr(pd.Series(np.arange(n)).rank(), pd.Series(x).rank()))
    mean_drift = (np.max(trend) - np.min(trend)) / (x_iqr + EPS)

    # Spectral features and seasonal-period estimation use the detrended series.
    peak_ratio, spec_entropy, high_ratio = fft_features(x - trend)

    if SEASONAL_PERIOD and SEASONAL_PERIOD > 1:
        # Recommended: explicitly provided period.
        period = SEASONAL_PERIOD
    elif AUTO_ESTIMATE_SEASONAL_PERIOD:
        # Fallback: automatically estimate the period from the detrended series.
        # This way, when SEASONAL_PERIOD=0, seasonal_strength is not forcibly zeroed because period=0.
        period = estimate_seasonal_period_from_fft(x - trend)
    else:
        period = 0

    seas_strength, periodic_peak, seas_stability, seasonal = seasonal_features(x, period, trend)

    # Memory: acf_energy is changed to mean squared ACF for comparability across different MAX_ACF_LAG values.
    max_lag = lags["acf_lag"]
    acfs = [acf(x, k) for k in range(1, max_lag + 1)]
    acf1 = acfs[0] if len(acfs) >= 1 else 0.0
    acf2 = acfs[1] if len(acfs) >= 2 else 0.0
    acf_energy = float(np.mean(np.asarray(acfs) ** 2)) if len(acfs) else 0.0

    # Second-order partial autocorrelation; first-order PACF equals ACF1 and would be redundant with acf1 as an indicator.
    denom = 1.0 - acf1 ** 2
    if abs(denom) < EPS:
        pacf2 = 0.0
    else:
        pacf2 = (acf2 - acf1 ** 2) / (denom + EPS)
    pacf2 = float(np.clip(pacf2, -1.0, 1.0))

    if abs(acf1) < EPS:
        half_life = 0.0
    else:
        half_life = float(max_lag)
        for k, v in enumerate(acfs, start=1):
            if abs(v) < 0.5 * abs(acf1):
                half_life = float(k)
                break

    # Noise and volatility
    residual = x - trend - seasonal
    res_var_ratio = np.var(residual) / (var_x + EPS)
    diff1 = np.diff(x) if n >= 2 else np.array([0.0])
    diff2 = np.diff(x, n=2) if n >= 3 else np.array([0.0])
    roughness = np.mean(np.abs(diff1)) / (x_iqr + EPS)
    second_energy = np.mean(diff2 ** 2) / (var_x + EPS)
    rv = roll_var(x, lags["roll_var_win"])
    roll_var_cv = np.std(rv) / (np.mean(rv) + EPS)
    sqdiff_acf1 = abs(acf(diff1 ** 2, 1)) if len(diff1) >= 3 else 0.0

    # Stationarity, changepoints, shocks, complexity
    adf_p, kpss_stat = adf_kpss(x)

    # Key change: changepoints are computed on the residual to avoid spurious changepoints caused by strong seasonality.
    cp_density = changepoint_density(residual, lags["changepoint_win"])

    r_mad, r_med = mad(residual), np.median(residual)
    shock = float(np.mean(np.abs(residual - r_med) > 3.0 * (r_mad + EPS)))

    return {
        "robust_cv": finite_float(robust_cv),
        "range_iqr_ratio": finite_float(range_iqr),
        "skewness": finite_float(skew),
        "excess_kurtosis": finite_float(kurt),
        "outlier_ratio": finite_float(outlier),
        "near_zero_ratio": finite_float(near_zero),
        "plateau_ratio": finite_float(plateau),

        "trend_strength": finite_float(trend_strength),
        "normalized_slope": finite_float(slope),
        "monotonicity": finite_float(monotonic),
        "curvature": finite_float(curvature),
        "rolling_mean_drift": finite_float(mean_drift),

        "seasonal_strength": finite_float(seas_strength),
        "periodic_acf_peak": finite_float(periodic_peak),
        "spectral_peak_ratio": finite_float(peak_ratio),
        "spectral_entropy": finite_float(spec_entropy),
        "seasonal_stability": finite_float(seas_stability),

        "acf1": finite_float(acf1),
        "acf_energy": finite_float(acf_energy),
        "pacf2": finite_float(pacf2),
        "acf_half_life": finite_float(half_life),
        "hurst_exponent": finite_float(hurst(x)),

        "residual_variance_ratio": finite_float(res_var_ratio),
        "high_freq_energy_ratio": finite_float(high_ratio),
        "roughness_index": finite_float(roughness),
        "second_difference_energy": finite_float(second_energy),
        "rolling_variance_cv": finite_float(roll_var_cv),
        "squared_diff_acf1": finite_float(sqdiff_acf1),

        "adf_pvalue": finite_float(adf_p),
        "kpss_statistic": finite_float(kpss_stat),
        "changepoint_density": finite_float(cp_density),
        "shock_burst_ratio": finite_float(shock),
        "sample_entropy": finite_float(sample_entropy(residual)),
        "permutation_entropy": finite_float(permutation_entropy(residual)),
    }


# =======================
# Multivariate Indicators
# =======================
def robust_scale_each_feature(Y):
    """Apply robust standardization to each variable. Used only for PCA/multivariate structure, not for overriding the original data."""
    Y = np.asarray(Y, float)
    Z = np.zeros_like(Y, dtype=float)
    for j in range(Y.shape[1]):
        col = Y[:, j]
        med = np.median(col)
        scale = iqr(col)
        if not np.isfinite(scale) or scale < EPS:
            scale = np.std(col)
        Z[:, j] = (col - med) / (scale + EPS)
    return Z


def select_pairs(p, max_pairs):
    """
    Deterministic uniform sampling of variable pairs.
    The original code simply took the first max_pairs, which was biased toward the first few variables.
    """
    all_pairs = [(i, j) for i in range(p) for j in range(i + 1, p)]
    if len(all_pairs) <= max_pairs:
        return all_pairs

    # Uniform sampling ensures consistent results across runs and is easy for beginners to understand.
    idx = np.linspace(0, len(all_pairs) - 1, max_pairs).astype(int)
    return [all_pairs[i] for i in idx]


def multivar_features(Y_clean, uni_list):
    """Compute multivariate structural indicators."""
    n, p = Y_clean.shape
    if p < 2:
        return {
            "mean_absolute_correlation": 0.0,
            "correlation_network_density": 0.0,
            "median_lagged_cross_correlation": 0.0,
            "first_pc_variance_ratio": 0.0,
            "cross_feature_heterogeneity": 0.0,
        }

    # Multivariate correlation/PCA uses robustly standardized data to avoid scale effects.
    Y_scaled = robust_scale_each_feature(Y_clean)

    C = np.nan_to_num(np.corrcoef(Y_scaled, rowvar=False), nan=0.0)
    off = np.abs(C[np.triu_indices(p, k=1)])
    mean_corr = float(np.mean(off)) if len(off) else 0.0
    density = float(np.mean(off > CORR_THRESHOLD)) if len(off) else 0.0

    # Lagged cross-correlation: no longer limited to the first 300 pairs; output both max and median for robustness in research.
    pairs = select_pairs(p, MAX_CROSS_PAIRS)
    L = get_effective_lags(n, SEASONAL_PERIOD)["cross_lag"]
    best_corrs = []

    for i, j in pairs:
        a, b = Y_scaled[:, i], Y_scaled[:, j]
        pair_best = 0.0
        for lag in range(-L, L + 1):
            if lag < 0:
                c = abs(corr(a[:lag], b[-lag:]))
            elif lag > 0:
                c = abs(corr(a[lag:], b[:-lag]))
            else:
                c = abs(corr(a, b))
            pair_best = max(pair_best, c)
        best_corrs.append(pair_best)

    max_lag_corr = float(np.max(best_corrs)) if len(best_corrs) else 0.0
    median_lag_corr = float(np.median(best_corrs)) if len(best_corrs) else 0.0

    # Fraction of variance explained by the first principal component
    try:
        s = np.linalg.svd(Y_scaled, compute_uv=False)
        pc1 = float((s[0] ** 2) / (np.sum(s ** 2) + EPS))
    except Exception:
        pc1 = 0.0

    # Cross-feature heterogeneity: larger differences in univariate features across variables yield larger values.
    def robust_standardize_feature_matrix(F):
        """Apply robust cross-variable standardization to each column of the feature matrix F."""
        F = np.asarray(F, float).copy()

        for cidx in range(F.shape[1]):
            col = F[:, cidx]
            med = nanmedian(col, 0.0)
            col = np.where(np.isfinite(col), col, med)

            scale = iqr(col)
            if not np.isfinite(scale) or scale < EPS:
                scale = np.std(col)

            F[:, cidx] = (col - med) / (scale + EPS)

        return F

    def heterogeneity_from_keys(uni_list, keys):
        valid_keys = [
            k for k in keys
            if any(np.isfinite(u.get(k, np.nan)) for u in uni_list)
        ]

        if len(valid_keys) == 0:
            return np.nan

        F = np.asarray([[u.get(k, np.nan) for k in valid_keys] for u in uni_list], float)

        if F.shape[0] < 2 or F.shape[1] == 0:
            return np.nan

        Z = robust_standardize_feature_matrix(F)
        center = np.median(Z, axis=0)
        dist = np.sqrt(np.mean((Z - center) ** 2, axis=1))
        return finite_float(np.median(dist))

    def grouped_cross_feature_heterogeneity(uni_list):
        group_scores = []
        for _, keys in HETERO_FEATURE_GROUPS.items():
            h = heterogeneity_from_keys(uni_list, keys)
            if np.isfinite(h):
                group_scores.append(h)

        if len(group_scores) == 0:
            return 0.0
        return finite_float(np.mean(group_scores))

    HETERO_FEATURE_GROUPS = {
        "distribution": [
            "robust_cv", "normalized_iqr", "range_iqr_ratio", "skewness",
            "excess_kurtosis", "outlier_ratio", "near_zero_ratio", "plateau_ratio",
        ],
        "trend": [
            "trend_strength", "normalized_slope", "monotonicity", "curvature", "rolling_mean_drift",
        ],
        "periodicity": [
            "seasonal_strength", "periodic_acf_peak", "spectral_peak_ratio",
            "spectral_entropy", "seasonal_stability",
        ],
        "memory": [
            "acf1", "acf_energy", "pacf2", "acf_half_life", "hurst_exponent",
        ],
        "noise_volatility": [
            "residual_variance_ratio", "high_freq_energy_ratio", "roughness_index",
            "second_difference_energy", "rolling_variance_cv", "squared_diff_acf1",
        ],
        "stationarity": [
            "adf_pvalue", "kpss_statistic", "changepoint_density", "shock_burst_ratio",
        ],
        "entropy_complexity": [
            "sample_entropy", "permutation_entropy",
        ],
    }

    # Table 3, row 39 includes normalized IQR in the distribution profile only.
    profiles = [
        {**features, "normalized_iqr": finite_float(iqr(col) / (abs(np.median(col)) + EPS))}
        for features, col in zip(uni_list, Y_clean.T)
    ]
    hetero = grouped_cross_feature_heterogeneity(profiles)

    return {
        "mean_absolute_correlation": finite_float(mean_corr),
        "correlation_network_density": finite_float(density),
        "median_lagged_cross_correlation": finite_float(median_lag_corr),
        "first_pc_variance_ratio": finite_float(pc1),
        "cross_feature_heterogeneity": finite_float(hetero),
    }


# =======================
# Supplementary Table 3, rows 1-39
# =======================
RAW_ORDER = [
    "robust_cv", "range_iqr_ratio", "skewness", "excess_kurtosis",
    "outlier_ratio", "near_zero_ratio", "plateau_ratio",
    "trend_strength", "normalized_slope", "monotonicity", "curvature", "rolling_mean_drift",
    "seasonal_strength", "periodic_acf_peak", "spectral_peak_ratio",
    "spectral_entropy", "seasonal_stability",
    "acf1", "acf_energy", "pacf2", "acf_half_life", "hurst_exponent",
    "residual_variance_ratio", "high_freq_energy_ratio", "roughness_index",
    "second_difference_energy", "rolling_variance_cv", "squared_diff_acf1",
    "adf_pvalue", "kpss_statistic", "changepoint_density", "shock_burst_ratio",
    "sample_entropy", "permutation_entropy",
    "mean_absolute_correlation", "correlation_network_density",
    "median_lagged_cross_correlation", "first_pc_variance_ratio", "cross_feature_heterogeneity",
]


def compute_raw(Y_clean):
    """For multivariate data, univariate indicators are summarized by the median across variables."""
    n, p = Y_clean.shape
    uni_list = [univar_features(Y_clean[:, j], Y_clean[:, j]) for j in range(p)]

    raw = {}

    # The original code only outputs the median. This is preserved for minimal changes.
    # For a paper, it is recommended to additionally output the IQR of each indicator across variables.
    for k in uni_list[0].keys():
        raw[k] = nanmedian([u[k] for u in uni_list], np.nan)

    raw.update(multivar_features(Y_clean, uni_list))
    return {k: finite_float(raw[k]) for k in RAW_ORDER}


# =======================
# 7 Meta-Features (zero shared raw indicators)
# =======================
def compute_meta(r):
    """
    7 meta-features with zero shared raw indicators between any two features.
    Each raw indicator contributes to exactly one meta-feature, ensuring clean
    differentiation for experimental pattern-finding.

    1. distribution_shape_score    — value distribution characteristics
    2. trend_strength_score        — overall directional tendency
    3. stationarity_score          — degree of deviation from a stationary process
    4. periodicity_score           — strength of cyclic/seasonal patterns
    5. temporal_dependence_score   — how strongly the past predicts the future
    6. noise_volatility_score      — unpredictable / high-frequency component
    7. multivariate_structure_score — cross-variable correlation structure
    """

    # 1. distribution_shape_score
    zero_plateau_score = mean_finite([
        score01(r["near_zero_ratio"]),
        score01(r["plateau_ratio"]),
    ])
    distribution_shape = mean_finite([
        score_pos(r["robust_cv"], 1),
        score_pos(r["range_iqr_ratio"], 10),
        score_pos(abs(r["skewness"]), 2),
        score_pos(r["excess_kurtosis"], 5),
        score_pos(r["outlier_ratio"], 0.05),
        zero_plateau_score,
    ])

    # 2. trend_strength_score
    trend = mean_finite([
        score01(r["trend_strength"]),
        score_pos(abs(r["normalized_slope"]), 1),
        score01(abs(r["monotonicity"])),
    ])

    # 3. stationarity_score (merged: trend_instability + nonstationarity_regime)
    stationarity = mean_finite([
        score_pos(r["curvature"], 1),
        score_pos(r["rolling_mean_drift"], 1),
        score_pos(r["changepoint_density"], 0.05),
        score01(r["adf_pvalue"], default=np.nan),
        score_pos(r["kpss_statistic"], 1, default=np.nan),
        score_pos(r["shock_burst_ratio"], 0.05),
    ])

    # 4. periodicity_score
    periodicity = mean_finite([
        score01(r["seasonal_strength"]),
        score01(abs(r["periodic_acf_peak"])),
        score_pos(r["spectral_peak_ratio"], 0.10),
        1 - score01(r["spectral_entropy"]),
        score01(r["seasonal_stability"]),
    ])

    # 5. temporal_dependence_score
    temporal_dependence = mean_finite([
        score01(abs(r["acf1"])),
        score_pos(r["acf_energy"], 0.25),
        score01(abs(r["pacf2"])),
        score_pos(r["acf_half_life"], max(3, MAX_ACF_LAG / 2)),
        score_pos(abs(r["hurst_exponent"] - 0.5), 0.25),
    ])

    # 6. noise_volatility_score (absorbed entropy indicators from nonlinear_complexity)
    noise = mean_finite([
        score01(r["residual_variance_ratio"]),
        score01(r["high_freq_energy_ratio"]),
        score_pos(r["roughness_index"], 1),
        score_pos(r["second_difference_energy"], 2),
        score_pos(r["rolling_variance_cv"], 1),
        score01(abs(r["squared_diff_acf1"])),
        score_pos(r["sample_entropy"], 2),
        score01(r["permutation_entropy"]),
    ])

    # 7. multivariate_structure_score
    shared = mean_finite([
        score01(r["mean_absolute_correlation"]),
        score01(r["correlation_network_density"]),
        score01(r["median_lagged_cross_correlation"]),
        score01(r["first_pc_variance_ratio"]),
    ])
    multi = clip01(shared * (1 - score_pos(r["cross_feature_heterogeneity"], 2, 0)))

    return {
        "distribution_shape_score": finite_float(distribution_shape),
        "trend_strength_score": finite_float(trend),
        "stationarity_score": finite_float(stationarity),
        "periodicity_score": finite_float(periodicity),
        "temporal_dependence_score": finite_float(temporal_dependence),
        "noise_volatility_score": finite_float(noise),
        "multivariate_structure_score": finite_float(multi),
    }


def save_csv(d, path):
    pd.DataFrame({"indicator_name": list(d.keys()), "value": list(d.values())}).to_csv(
        path, index=False, encoding="utf-8-sig"
    )


def reuse_cached_metrics(raw_path, meta_path):
    """Reuse a complete feature table, projecting legacy rows without rounding values."""
    if not (os.path.isfile(raw_path) and os.path.isfile(meta_path)):
        return False
    try:
        frame = pd.read_csv(raw_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return False
    if list(frame.columns) != ["indicator_name", "value"]:
        return False
    names = frame.indicator_name.tolist()
    if names == RAW_ORDER:
        return True
    legacy = set(RAW_ORDER) | {"iqr_scale_ratio", "dominant_period"}
    if len(names) != len(legacy) or set(names) != legacy:
        return False
    retained = frame.set_index("indicator_name").loc[RAW_ORDER]
    with tempfile.NamedTemporaryFile(dir=os.path.dirname(raw_path), delete=False) as handle:
        temporary = handle.name
    try:
        retained.to_csv(temporary, encoding="utf-8-sig")
        os.replace(temporary, raw_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return True


def compute_metrics_for_npy(npy_path, output_dir, seasonal_period=288):
    """
    Compute raw_indicators.csv and meta_features.csv for a single npy file.

    Parameters
    ----------
    npy_path : str
        Full path to the .npy file.
    output_dir : str
        Output directory; will be created automatically.
    seasonal_period : int
        Seasonal period. Traffic data at 5-min resolution = 288; set to 0 to enable automatic estimation when unknown.
    """
    global SEASONAL_PERIOD
    SEASONAL_PERIOD = int(seasonal_period)

    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"npy file not found: {npy_path}")

    os.makedirs(output_dir, exist_ok=True)

    raw_path = os.path.join(output_dir, "raw_indicators.csv")
    meta_path = os.path.join(output_dir, "meta_features.csv")
    if reuse_cached_metrics(raw_path, meta_path):
        print(f"[compute_metrics] Skipping: raw_indicators.csv and meta_features.csv already exist in {output_dir}")
        return None, None

    Y_clean = load_npy(npy_path)
    print(f"[compute_metrics] Data shape: {Y_clean.shape}")

    raw = compute_raw(Y_clean)
    meta = compute_meta(raw)

    save_csv(raw, raw_path)
    save_csv(meta, meta_path)

    print(f"[compute_metrics] Done: {npy_path} -> {output_dir}")
    print(f"  Raw indicators: {raw_path}")
    print(f"  Meta-features: {meta_path}")

    if SEASONAL_PERIOD <= 1:
        print("  Note: SEASONAL_PERIOD <= 1, the code will auto-estimate the period. For traffic data, explicit setting is recommended (5 min = 288).")

    if not np.isfinite(raw.get("adf_pvalue", np.nan)):
        print("  Note: ADF/KPSS not computed. Scientific experiments are recommended to install statsmodels.")

    return raw, meta


def main(argv=None):
    """Rebuild feature tables for selected files in 0dataset/train or test."""
    import argparse
    from pathlib import Path
    from prepare_numeric_input import compatible_input

    ROOT = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Rebuild feature tables with the original metric implementation.")
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--datasets", default="all", help="all, firstN, or comma-separated dataset names")
    parser.add_argument("--output-dir", type=Path, help="Use a new directory to recompute frozen features")
    parser.add_argument("--seasonal-period", type=int, default=0)
    args = parser.parse_args(argv)
    data_root = ROOT / "0dataset" / args.split
    files = sorted(data_root.rglob("*.npy"))
    if args.datasets.startswith("first"):
        count = int(args.datasets[5:])
        if count < 1:
            parser.error("firstN requires N >= 1")
        files = files[:count]
    elif args.datasets != "all":
        requested = set(args.datasets.split(","))
        missing = requested - {p.stem for p in files}
        if missing:
            parser.error("Unknown datasets: " + ", ".join(sorted(missing)))
        files = [p for p in files if p.stem in requested]
    archive = ROOT / "0dataset" / (args.split + "metrics")
    output = args.output_dir or archive
    use_categories = output.resolve() == archive.resolve()
    for file in files:
        category = file.parent.relative_to(data_root)
        target = (output / category if use_categories else output) / file.stem
        if reuse_cached_metrics(target / "raw_indicators.csv", target / "meta_features.csv"):
            print(f"Existing features kept: {file.stem}")
            continue
        with compatible_input(file) as prepared:
            compute_metrics_for_npy(str(prepared), str(target), args.seasonal_period)
    print(f"Checked {len(files)} datasets; output: {output.resolve()}")


if __name__ == "__main__":
    main()
