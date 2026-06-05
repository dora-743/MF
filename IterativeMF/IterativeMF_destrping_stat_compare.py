"""Iterative Matched Filter with two-direction diagonal destriping and statistic comparison.

It keeps the original one-direction and two-direction destriping scripts separate,
and adds a statistic-comparison workflow for two-direction diagonal destriping.
The workflow compares stripe-offset statistics such as median, mean,
trimmed mean, histogram mode, and sigma-clipped mean.

The destriping directions are:

1. ``y_minus_x``: lines parallel to y=x, represented by row - col = constant.
2. ``y_plus_x``: lines parallel to y=-x, represented by row + col = constant.

The default correction order is ``y_minus_x`` followed by ``y_plus_x``.

Notes
-----
Two-direction destriping can be strong. Always compare the raw alpha map,
the total stripe map, each direction-wise stripe map, the corrected alpha map,
and the plume mask before interpreting the corrected result. A real elongated
plume can be partly removed if it aligns with the stripe model.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Default configuration

ROI_CSV = Path("all_roi_spectra200x200.csv")
MODTRAN_CSV = Path("CH4c.csv")
OUTPUT_DIR = Path("outputs_two_direction_diagonal_destripe_stat_compare")

WL_MIN = 2100.0
WL_MAX = 2450.0
FWHM_NM = 12.5

# Low-alpha MODTRAN range used to estimate the unit absorption spectrum.
UAS_ALPHA_MIN = 0.0
UAS_ALPHA_MAX = 0.5

N_ITER = 5
NSIGMA = 3.0
REG = 1e-6
RCOND = 1e-8

NODATA_VALUES = [0, -9999]
REQUIRE_POSITIVE = True
MIN_VALID_FRACTION = 1.0

# Every statistic listed here is run as both final_only and each_iter.
STRIPE_STAT_METHODS = [
    ("median", {}),
    ("mean", {}),
    ("trimmed_mean", {"trim_fraction": 0.1}),
    ("mode", {"mode_bins": 64}),
    ("sigma_clipped_mean", {"sigma_clip_nsigma": 3.0, "sigma_clip_max_iter": 3}),
]


def make_default_destripe_params(nsigma: float = NSIGMA) -> dict:
    """Return recommended two-direction destriping parameters."""
    return {
        # Correction directions.
        # y=x stripes:  "y_minus_x" -> row - col = const.
        # y=-x stripes: "y_plus_x"  -> row + col = const.
        # This default corrects y=x first, then y=-x.
        "directions": ["y_minus_x", "y_plus_x"],

        # Statistic used for estimating a stripe offset on each line.
        # Available values: median, mean, trimmed_mean, mode, sigma_clipped_mean.
        "method": "median",

        # Minimum pixels per diagonal line for estimating an offset.
        "min_pixels_per_line": 5,

        # If True, subtract line_stat - global_stat to preserve the global alpha baseline.
        "preserve_global_stat": True,

        # Half-window for median smoothing of neighboring diagonal offsets. Try 0, 1, or 2.
        "smooth_half_window": 0,

        # Exclusion mode for pixels used in stripe estimation.
        # Available values: none, robust_high, previous_plume, previous_plume_or_high.
        "exclude_mode": "robust_high",
        "exclude_nsigma": nsigma,

        # Recompute the exclusion mask before estimating the second direction.
        "recompute_exclude_each_direction": True,

        # If too few pixels remain after exclusion, re-estimate from all valid pixels.
        "fallback_to_valid": True,

        # Parameters for alternative line statistics.
        "trim_fraction": 0.1,
        "mode_bins": 64,
        "sigma_clip_nsigma": 3.0,
        "sigma_clip_max_iter": 3,

        # For each_iter, choose whether plume thresholding uses corrected or raw alpha.
        "threshold_source": "corrected",
    }


def make_experiments(nsigma: float = NSIGMA) -> dict[str, dict]:
    """Return the default experiment set used by the statistic-comparison workflow."""
    default = make_default_destripe_params(nsigma=nsigma)

    experiments: dict[str, dict] = {
        "baseline_no_destripe": {
            "destripe_when": "none",
            "destripe_params": None,
        },

        # y=x only: useful for comparison against the two-direction correction.
        "median_yx_final_only": {
            "destripe_when": "final_only",
            "destripe_params": {**default, "directions": ["y_minus_x"], "method": "median"},
        },
        "median_yx_each_iter": {
            "destripe_when": "each_iter",
            "destripe_params": {**default, "directions": ["y_minus_x"], "method": "median"},
        },
    }

    # Two-direction correction, y=x -> y=-x, for every statistic and timing mode.
    for method, overrides in STRIPE_STAT_METHODS:
        for when in ("final_only", "each_iter"):
            experiments[f"{method}_yx_then_ynegx_{when}"] = {
                "destripe_when": when,
                "destripe_params": {**default, "method": method, **overrides},
            }

    # Optional comparison: destripe only after iteration 3.
    experiments["median_yx_then_ynegx_iter_3_to_5"] = {
        "destripe_when": [3, 4, 5],
        "destripe_params": {**default, "method": "median"},
    }

    return experiments


DEFAULT_DESTRIPE_PARAMS = make_default_destripe_params(NSIGMA)
EXPERIMENTS = make_experiments(NSIGMA)


# 1. Data loading helpers

def get_wave_columns(df: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    """Return wavelength columns sorted by wavelength."""
    wave_cols: list[str] = []
    wavelengths: list[float] = []

    pattern = re.compile(r"^wave_([0-9]+(?:\.[0-9]+)?)nm$")
    for col in df.columns:
        match = pattern.match(col)
        if match is not None:
            wave_cols.append(col)
            wavelengths.append(float(match.group(1)))

    if len(wave_cols) == 0:
        raise ValueError("No wavelength columns like wave_2300nm were found.")

    wavelengths_arr = np.asarray(wavelengths, dtype=float)
    order = np.argsort(wavelengths_arr)
    wavelengths_arr = wavelengths_arr[order]
    wave_cols_sorted = [wave_cols[i] for i in order]
    return wave_cols_sorted, wavelengths_arr


def load_roi_spectra_csv(path: str | Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load ROI spectra CSV with y, x, and wave_XXXXnm columns."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ROI CSV not found: {path}")

    df = pd.read_csv(path)
    if "y" not in df.columns or "x" not in df.columns:
        raise ValueError("ROI CSV must contain columns 'y' and 'x'.")

    wave_cols, wavelengths = get_wave_columns(df)
    spectra = df[wave_cols].to_numpy(dtype=float)
    return df, wavelengths, spectra


def spectra_to_cube(
    df: pd.DataFrame,
    spectra: np.ndarray,
    fill_value: float = np.nan,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert table spectra to image cube [H, W, B]."""
    ys = np.sort(df["y"].unique())
    xs = np.sort(df["x"].unique())

    y_to_row = {y: i for i, y in enumerate(ys)}
    x_to_col = {x: j for j, x in enumerate(xs)}

    H, W = len(ys), len(xs)
    B = spectra.shape[1]
    cube = np.full((H, W, B), fill_value, dtype=float)

    for row_idx, row in df.iterrows():
        r = y_to_row[row["y"]]
        c = x_to_col[row["x"]]
        cube[r, c, :] = spectra[row_idx, :]

    return cube, ys, xs


def band_mask(
    wavelengths: np.ndarray,
    wl_min: Optional[float] = None,
    wl_max: Optional[float] = None,
    exclude_ranges: Optional[Sequence[tuple[float, float]]] = None,
) -> np.ndarray:
    mask = np.ones_like(wavelengths, dtype=bool)

    if wl_min is not None:
        mask &= wavelengths >= wl_min
    if wl_max is not None:
        mask &= wavelengths <= wl_max

    if exclude_ranges is not None:
        for a, b in exclude_ranges:
            mask &= ~((wavelengths >= a) & (wavelengths <= b))

    return mask


def select_bands(
    cube: np.ndarray,
    wavelengths: np.ndarray,
    wl_min: float,
    wl_max: float,
    exclude_ranges: Optional[Sequence[tuple[float, float]]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = band_mask(wavelengths, wl_min=wl_min, wl_max=wl_max, exclude_ranges=exclude_ranges)
    return cube[:, :, mask], wavelengths[mask], mask


# 2. Plotting helpers


def finite_values(img: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    arr = np.asarray(img, dtype=float)
    if mask is None:
        vals = arr[np.isfinite(arr)]
    else:
        vals = arr[np.isfinite(arr) & np.asarray(mask, dtype=bool)]
    return vals


def robust_limits(
    images: Union[np.ndarray, Sequence[np.ndarray]],
    mask: Optional[np.ndarray] = None,
    q_low: float = 2,
    q_high: float = 98,
) -> tuple[float, float]:
    if isinstance(images, np.ndarray):
        images = [images]

    vals_list = []
    for img in images:
        vals = finite_values(img, mask=mask)
        if vals.size > 0:
            vals_list.append(vals)

    if len(vals_list) == 0:
        return -1.0, 1.0

    vals_all = np.concatenate(vals_list)
    lo, hi = np.nanpercentile(vals_all, [q_low, q_high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(vals_all)), float(np.nanmax(vals_all))
        if hi <= lo:
            hi = lo + 1.0
    return float(lo), float(hi)


def robust_scale_image(img: np.ndarray, pmin: float = 2, pmax: float = 98) -> np.ndarray:
    vals = finite_values(img)
    if vals.size == 0:
        return np.zeros_like(img, dtype=float)
    lo, hi = np.nanpercentile(vals, [pmin, pmax])
    if hi <= lo:
        return np.zeros_like(img, dtype=float)
    return np.clip((img - lo) / (hi - lo), 0, 1)


def make_rgb_from_cube(
    cube: np.ndarray,
    wavelengths: np.ndarray,
    r_wl: float = 650,
    g_wl: float = 550,
    b_wl: float = 460,
) -> np.ndarray:
    idx_r = int(np.argmin(np.abs(wavelengths - r_wl)))
    idx_g = int(np.argmin(np.abs(wavelengths - g_wl)))
    idx_b = int(np.argmin(np.abs(wavelengths - b_wl)))

    r = robust_scale_image(cube[:, :, idx_r])
    g = robust_scale_image(cube[:, :, idx_g])
    b = robust_scale_image(cube[:, :, idx_b])
    return np.dstack([r, g, b])


def plot_map(
    img: np.ndarray,
    title: str = "Map",
    mask: Optional[np.ndarray] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "viridis",
    colorbar_label: Optional[str] = None,
    figsize: tuple[float, float] = (5, 5),
):
    if vmin is None or vmax is None:
        auto_vmin, auto_vmax = robust_limits(img, mask=mask)
        if vmin is None:
            vmin = auto_vmin
        if vmax is None:
            vmax = auto_vmax

    plt.figure(figsize=figsize)
    im = plt.imshow(img, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar(im, label=colorbar_label)
    plt.show()


def plot_mean_spectrum(cube: np.ndarray, wavelengths: np.ndarray, mask: Optional[np.ndarray] = None, xlim=None):
    if mask is None:
        X = cube.reshape(-1, cube.shape[2])
        spec = np.nanmean(X, axis=0)
    else:
        spec = np.nanmean(cube[mask], axis=0)

    plt.figure(figsize=(8, 4))
    plt.plot(wavelengths, spec, marker="o", ms=3)
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Radiance / Reflectance")
    plt.title("Mean spectrum")
    if xlim is not None:
        plt.xlim(*xlim)
    plt.grid(True)
    plt.show()


def plot_uas(wavelengths: np.ndarray, uas: np.ndarray, title: str = "UAS", xlim=None):
    plt.figure(figsize=(8, 4))
    plt.plot(wavelengths, uas, marker="o", ms=3)
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("UAS")
    plt.title(title)
    if xlim is not None:
        plt.xlim(*xlim)
    plt.grid(True)
    plt.show()


# 3. MODTRAN / UAS helpers


def load_ch4_modtran_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load MODTRAN spectra CSV.

    Expected format:
        wavelength, 0.0, 0.5, 1.0, ...

    Returns:
        mod_wave: (n_mod_wave,)
        alpha_grid: (n_alpha,)
        spectra_grid: (n_alpha, n_mod_wave)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MODTRAN CSV not found: {path}")

    df_mod = pd.read_csv(path)
    if "wavelength" not in df_mod.columns:
        raise ValueError("MODTRAN CSV must contain a 'wavelength' column.")

    mod_wave = df_mod["wavelength"].to_numpy(dtype=float)
    alpha_cols = [c for c in df_mod.columns if c != "wavelength"]

    try:
        alpha_grid = np.array([float(c) for c in alpha_cols], dtype=float)
    except ValueError as exc:
        raise ValueError("MODTRAN columns other than 'wavelength' must be numeric alpha values.") from exc

    order = np.argsort(alpha_grid)
    alpha_grid = alpha_grid[order]
    alpha_cols = [alpha_cols[i] for i in order]

    spectra_grid = df_mod[alpha_cols].to_numpy(dtype=float).T
    return mod_wave, alpha_grid, spectra_grid


def gaussian_srf_resample(
    mod_wave: np.ndarray,
    mod_spectra: np.ndarray,
    sensor_wave: np.ndarray,
    fwhm_nm: Union[float, np.ndarray],
) -> np.ndarray:
    """Resample high-resolution MODTRAN spectra to sensor wavelengths by Gaussian SRF."""
    mod_wave = np.asarray(mod_wave, dtype=float)
    mod_spectra = np.asarray(mod_spectra, dtype=float)
    sensor_wave = np.asarray(sensor_wave, dtype=float)

    if np.isscalar(fwhm_nm):
        fwhm_arr = np.full(sensor_wave.shape, float(fwhm_nm), dtype=float)
    else:
        fwhm_arr = np.asarray(fwhm_nm, dtype=float)
        if fwhm_arr.shape != sensor_wave.shape:
            raise ValueError("fwhm_nm must be scalar or same shape as sensor_wave.")

    out = np.full((mod_spectra.shape[0], sensor_wave.size), np.nan, dtype=float)

    for j, center in enumerate(sensor_wave):
        sigma = fwhm_arr[j] / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        use = np.abs(mod_wave - center) <= 4.0 * sigma

        if np.sum(use) < 2:
            # fallback: simple interpolation for each alpha spectrum
            for i in range(mod_spectra.shape[0]):
                out[i, j] = np.interp(center, mod_wave, mod_spectra[i])
            continue

        weights = np.exp(-0.5 * ((mod_wave[use] - center) / sigma) ** 2)
        weights = weights / np.sum(weights)
        out[:, j] = mod_spectra[:, use] @ weights

    return out


def compute_uas_log_slope(
    alpha_grid: np.ndarray,
    spectra_grid: np.ndarray,
    alpha_min: Optional[float] = None,
    alpha_max: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute UAS by fitting log(spectrum) = intercept - UAS * alpha."""
    alpha_grid = np.asarray(alpha_grid, dtype=float)
    spectra_grid = np.asarray(spectra_grid, dtype=float)

    use = np.ones_like(alpha_grid, dtype=bool)
    if alpha_min is not None:
        use &= alpha_grid >= alpha_min
    if alpha_max is not None:
        use &= alpha_grid <= alpha_max

    a = alpha_grid[use]
    if a.size < 2:
        raise ValueError("Need at least two alpha values to compute UAS.")

    Y = np.log(np.maximum(spectra_grid[use], 1e-30))
    A = np.vstack([np.ones_like(a), a]).T
    coeff, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)

    intercept = coeff[0]
    slope = coeff[1]
    uas = -slope
    return uas, intercept


# 4. Matched Filter helpers


def make_valid_pixel_mask(
    cube: np.ndarray,
    nodata_values: Optional[Sequence[float]] = None,
    require_positive: bool = True,
    min_valid_fraction: float = 1.0,
) -> np.ndarray:
    valid_band = np.isfinite(cube)

    if nodata_values is not None:
        for value in nodata_values:
            valid_band &= cube != value

    if require_positive:
        valid_band &= cube > 0

    valid_fraction = np.mean(valid_band, axis=2)
    return valid_fraction >= min_valid_fraction


def estimate_background_mean_cov_from_cube(
    cube: np.ndarray,
    background_mask: np.ndarray,
    reg: float = 1e-6,
    min_pixels: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate background mean and covariance from selected pixels."""
    H, W, B = cube.shape

    if min_pixels is None:
        min_pixels = max(B + 5, 30)

    X = cube[np.asarray(background_mask, dtype=bool)]
    X = X[np.all(np.isfinite(X), axis=1)]

    if X.shape[0] < min_pixels:
        raise ValueError(f"Too few background pixels: {X.shape[0]} < {min_pixels}")

    mu = np.mean(X, axis=0)
    Xc = X - mu
    cov = (Xc.T @ Xc) / max(X.shape[0] - 1, 1)

    # A small diagonal regularization. Scale-aware enough for most reflectance/radiance cases.
    scale = np.nanmean(np.diag(cov))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    cov = cov + reg * scale * np.eye(B)

    return mu, cov, X


def make_methane_target(mu: np.ndarray, uas: np.ndarray, positive_alpha: bool = True) -> np.ndarray:
    """MF target spectrum. positive_alpha=True follows the convention used in the original notebook."""
    mu = np.asarray(mu, dtype=float).reshape(-1)
    uas = np.asarray(uas, dtype=float).reshape(-1)
    return -mu * uas if positive_alpha else mu * uas


def matched_filter_alpha_map(
    cube: np.ndarray,
    uas: np.ndarray,
    valid_mask: np.ndarray,
    background_mask: np.ndarray,
    reg: float = 1e-6,
    rcond: float = 1e-8,
    min_background_pixels: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute MF alpha map using current background pixels."""
    H, W, B = cube.shape
    uas = np.asarray(uas, dtype=float).reshape(-1)
    if uas.size != B:
        raise ValueError(f"uas length {uas.size} does not match cube bands {B}.")

    valid_mask = np.asarray(valid_mask, dtype=bool)
    background_mask = valid_mask & np.asarray(background_mask, dtype=bool)

    mu, cov, _ = estimate_background_mean_cov_from_cube(
        cube=cube,
        background_mask=background_mask,
        reg=reg,
        min_pixels=min_background_pixels,
    )

    target = make_methane_target(mu, uas, positive_alpha=True)
    inv_cov = np.linalg.pinv(cov, rcond=rcond)

    denom = float(target.T @ inv_cov @ target)
    if abs(denom) < 1e-15:
        raise ValueError("MF denominator is too small. Check UAS, covariance, and wavelength selection.")

    alpha_map = np.full((H, W), np.nan, dtype=float)
    X = cube[valid_mask]
    diff = X - mu
    alpha_values = (diff @ inv_cov @ target) / denom
    alpha_map[valid_mask] = alpha_values

    return alpha_map, mu, cov, target


def robust_threshold_from_alpha(alpha_values: np.ndarray, nsigma: float = 4.0) -> tuple[float, float, float]:
    """Median + nsigma * 1.4826*MAD threshold."""
    values = np.asarray(alpha_values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("No finite alpha values for thresholding.")

    med = float(np.nanmedian(values))
    mad = float(np.nanmedian(np.abs(values - med)))
    robust_std = 1.4826 * mad + 1e-12
    threshold = med + nsigma * robust_std
    return float(threshold), med, float(robust_std)


def plume_mask_from_alpha(
    alpha_map: np.ndarray,
    valid_mask: np.ndarray,
    nsigma: float = 4.0,
) -> tuple[np.ndarray, dict]:
    threshold, med, robust_std = robust_threshold_from_alpha(alpha_map[valid_mask], nsigma=nsigma)
    plume_mask = np.zeros_like(valid_mask, dtype=bool)
    plume_mask[valid_mask] = alpha_map[valid_mask] > threshold

    meta = {
        "threshold": threshold,
        "median": med,
        "robust_std": robust_std,
        "n_plume": int(np.sum(plume_mask)),
    }
    return plume_mask, meta


# 5. Improved diagonal destriping helpers


def line_id_map(shape: tuple[int, int], direction: str = "y_minus_x") -> np.ndarray:
    """Return line IDs for directional line grouping.

    Image coordinate convention:
        row = y, positive downward
        col = x, positive rightward

    direction="y_minus_x": lines parallel to y=x,  row - col = const
    direction="y_plus_x" : lines parallel to y=-x, row + col = const
    """
    rows, cols = np.indices(shape)

    if direction == "y_minus_x":
        return rows - cols
    if direction == "y_plus_x":
        return rows + cols

    raise ValueError("direction must be 'y_minus_x' or 'y_plus_x'.")


def normalize_directions(destripe_params: Optional[dict]) -> list[str]:
    """Accept either 'directions' or old-style 'direction'."""
    if destripe_params is None:
        return []

    if "directions" in destripe_params:
        dirs = destripe_params["directions"]
    else:
        dirs = destripe_params.get("direction", "y_minus_x")

    if isinstance(dirs, str):
        dirs = [dirs]

    dirs = list(dirs)
    for d in dirs:
        if d not in {"y_minus_x", "y_plus_x"}:
            raise ValueError("directions must contain only 'y_minus_x' and/or 'y_plus_x'.")
    return dirs


def moving_nanmedian_1d(values: np.ndarray, half_window: int = 0) -> np.ndarray:
    """Median-smooth a 1D array while ignoring NaN."""
    values = np.asarray(values, dtype=float)
    if half_window <= 0:
        return values.copy()

    out = np.full_like(values, np.nan, dtype=float)
    for i in range(values.size):
        lo = max(0, i - half_window)
        hi = min(values.size, i + half_window + 1)
        win = values[lo:hi]
        finite = np.isfinite(win)
        if np.any(finite):
            out[i] = np.nanmedian(win[finite])
    return out


def statistic_1d(
    values: np.ndarray,
    method: str = "median",
    trim_fraction: float = 0.1,
    mode_bins: int = 64,
    sigma_clip_nsigma: float = 3.0,
    sigma_clip_max_iter: int = 3,
) -> float:
    """Compute 1D statistic for stripe offset estimation."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return np.nan

    if method == "median":
        return float(np.nanmedian(values))

    if method == "mean":
        return float(np.nanmean(values))

    if method == "trimmed_mean":
        v = np.sort(values)
        k = int(np.floor(trim_fraction * v.size))
        if 2 * k >= v.size:
            return float(np.nanmean(v))
        return float(np.nanmean(v[k:v.size-k]))

    if method == "mode":
        # Continuous values do not have an exact mode, so use the center of the most populated histogram bin.
        if values.size == 1 or np.allclose(values, values[0]):
            return float(values[0])
        counts, edges = np.histogram(values, bins=mode_bins)
        idx = int(np.argmax(counts))
        return float(0.5 * (edges[idx] + edges[idx + 1]))

    if method == "sigma_clipped_mean":
        clipped = values.copy()
        for _ in range(int(sigma_clip_max_iter)):
            if clipped.size < 3:
                break
            med = float(np.nanmedian(clipped))
            mad = float(np.nanmedian(np.abs(clipped - med)))
            robust_std = 1.4826 * mad
            if not np.isfinite(robust_std) or robust_std <= 0:
                break
            keep = np.abs(clipped - med) <= sigma_clip_nsigma * robust_std
            if np.all(keep) or not np.any(keep):
                break
            clipped = clipped[keep]
        return float(np.nanmean(clipped))

    raise ValueError("method must be 'median', 'mean', 'trimmed_mean', 'mode', or 'sigma_clipped_mean'.")



def make_exclude_mask_for_destriping(
    alpha_map: np.ndarray,
    valid_mask: np.ndarray,
    plume_mask: Optional[np.ndarray] = None,
    exclude_mode: str = "robust_high",
    exclude_nsigma: float = 4.0,
) -> tuple[np.ndarray, dict]:
    """Build mask of pixels excluded from stripe estimation."""
    valid_mask = np.asarray(valid_mask, dtype=bool) & np.isfinite(alpha_map)
    exclude = np.zeros_like(valid_mask, dtype=bool)
    meta: dict = {"exclude_mode": exclude_mode}

    if exclude_mode == "none" or exclude_mode is None:
        meta["n_excluded"] = 0
        return exclude, meta

    if "high" in exclude_mode:
        threshold, med, robust_std = robust_threshold_from_alpha(alpha_map[valid_mask], nsigma=exclude_nsigma)
        high_mask = np.zeros_like(valid_mask, dtype=bool)
        high_mask[valid_mask] = alpha_map[valid_mask] > threshold
        exclude |= high_mask
        meta.update({
            "high_threshold": float(threshold),
            "high_median": float(med),
            "high_robust_std": float(robust_std),
            "n_high_excluded": int(np.sum(high_mask)),
        })

    if "plume" in exclude_mode:
        if plume_mask is not None:
            plume_mask = np.asarray(plume_mask, dtype=bool)
            exclude |= plume_mask
            meta["n_plume_excluded"] = int(np.sum(plume_mask))
        else:
            meta["n_plume_excluded"] = 0

    if exclude_mode not in {"robust_high", "previous_plume", "previous_plume_or_high", "none", None}:
        raise ValueError(
            "exclude_mode must be 'none', 'robust_high', 'previous_plume', or 'previous_plume_or_high'."
        )

    exclude &= valid_mask
    meta["n_excluded"] = int(np.sum(exclude))
    return exclude, meta


def destripe_by_directional_lines(
    alpha_map: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    exclude_mask: Optional[np.ndarray] = None,
    direction: str = "y_minus_x",
    method: str = "median",
    min_pixels_per_line: int = 5,
    preserve_global_stat: bool = True,
    smooth_half_window: int = 0,
    fallback_to_valid: bool = True,
    trim_fraction: float = 0.1,
    mode_bins: int = 64,
    sigma_clip_nsigma: float = 3.0,
    sigma_clip_max_iter: int = 3,
) -> dict:
    """Estimate directional stripe offsets and subtract them from alpha_map.

    If preserve_global_stat=True:
        corrected = alpha - (line_stat - global_stat)

    If preserve_global_stat=False:
        corrected = alpha - line_stat
    """
    alpha = np.asarray(alpha_map, dtype=float)
    if alpha.ndim != 2:
        raise ValueError("alpha_map must be 2D.")

    finite = np.isfinite(alpha)
    if valid_mask is None:
        valid = finite.copy()
    else:
        valid = np.asarray(valid_mask, dtype=bool) & finite

    if not np.any(valid):
        raise ValueError("No valid finite pixels for destriping.")

    if exclude_mask is None:
        estimate_mask = valid.copy()
    else:
        estimate_mask = valid & (~np.asarray(exclude_mask, dtype=bool))

    if not np.any(estimate_mask):
        if fallback_to_valid:
            estimate_mask = valid.copy()
        else:
            raise ValueError("No pixels remain after exclusion for stripe estimation.")

    ids = line_id_map(alpha.shape, direction=direction)
    id_values = np.arange(int(np.nanmin(ids)), int(np.nanmax(ids)) + 1)

    global_stat = statistic_1d(
        alpha[estimate_mask],
        method=method,
        trim_fraction=trim_fraction,
        mode_bins=mode_bins,
        sigma_clip_nsigma=sigma_clip_nsigma,
        sigma_clip_max_iter=sigma_clip_max_iter,
    )

    raw_stats = np.full(id_values.shape, np.nan, dtype=float)
    counts_used = np.zeros(id_values.shape, dtype=int)
    used_fallback = np.zeros(id_values.shape, dtype=bool)

    for k, line_id in enumerate(id_values):
        m = (ids == line_id) & estimate_mask
        count = int(np.sum(m))

        if count < min_pixels_per_line and fallback_to_valid:
            mf = (ids == line_id) & valid
            count_f = int(np.sum(mf))
            if count_f >= min_pixels_per_line:
                m = mf
                count = count_f
                used_fallback[k] = True

        counts_used[k] = count
        if count >= min_pixels_per_line:
            raw_stats[k] = statistic_1d(
                alpha[m],
                method=method,
                trim_fraction=trim_fraction,
                mode_bins=mode_bins,
                sigma_clip_nsigma=sigma_clip_nsigma,
                sigma_clip_max_iter=sigma_clip_max_iter,
            )

    smoothed_stats = moving_nanmedian_1d(raw_stats, half_window=smooth_half_window)

    if preserve_global_stat and np.isfinite(global_stat):
        offsets = smoothed_stats - global_stat
    else:
        offsets = smoothed_stats.copy()

    # Too-short or all-NaN lines are left uncorrected.
    offsets_filled = np.where(np.isfinite(offsets), offsets, 0.0)

    stripe_map = np.zeros_like(alpha, dtype=float)
    for line_id, offset in zip(id_values, offsets_filled):
        stripe_map[ids == line_id] = float(offset)

    corrected = alpha - stripe_map
    corrected[~finite] = np.nan

    line_table = pd.DataFrame({
        "line_id": id_values,
        "n_pixels_used": counts_used,
        "used_fallback_to_valid": used_fallback,
        "line_stat_raw": raw_stats,
        "line_stat_after_smoothing": smoothed_stats,
        "stripe_offset_subtracted": offsets_filled,
        "global_stat": global_stat,
        "direction": direction,
        "method": method,
    })

    return {
        "corrected": corrected,
        "stripe_map": stripe_map,
        "line_table": line_table,
        "global_stat": global_stat,
        "estimate_mask": estimate_mask,
    }


def destripe_by_sequential_directions(
    alpha_map: np.ndarray,
    valid_mask: np.ndarray,
    plume_mask: Optional[np.ndarray] = None,
    destripe_params: Optional[dict] = None,
    nsigma: float = 4.0,
) -> dict:
    """Apply directional destriping sequentially.

    Example:
        directions=["y_minus_x", "y_plus_x"]

    This means:
        1. remove stripes parallel to y=x
        2. from the corrected result, remove stripes parallel to y=-x
    """
    if destripe_params is None:
        destripe_params = {}

    directions = normalize_directions(destripe_params)
    if len(directions) == 0:
        zero = np.zeros_like(alpha_map, dtype=float)
        return {
            "corrected": np.asarray(alpha_map, dtype=float).copy(),
            "stripe_map": zero,
            "directional_stripe_maps": {},
            "line_table": pd.DataFrame(),
            "exclude_meta": [],
        }

    current = np.asarray(alpha_map, dtype=float).copy()
    total_stripe = np.zeros_like(current, dtype=float)
    directional_stripe_maps: dict[str, np.ndarray] = {}
    line_tables = []
    exclude_metas = []

    recompute_exclude = destripe_params.get("recompute_exclude_each_direction", True)
    fixed_exclude_mask = None
    fixed_exclude_meta = None

    if not recompute_exclude:
        fixed_exclude_mask, fixed_exclude_meta = make_exclude_mask_for_destriping(
            alpha_map=current,
            valid_mask=valid_mask,
            plume_mask=plume_mask,
            exclude_mode=destripe_params.get("exclude_mode", "robust_high"),
            exclude_nsigma=destripe_params.get("exclude_nsigma", nsigma),
        )

    for pass_index, direction in enumerate(directions, start=1):
        if recompute_exclude:
            exclude_mask, exclude_meta = make_exclude_mask_for_destriping(
                alpha_map=current,
                valid_mask=valid_mask,
                plume_mask=plume_mask,
                exclude_mode=destripe_params.get("exclude_mode", "robust_high"),
                exclude_nsigma=destripe_params.get("exclude_nsigma", nsigma),
            )
        else:
            exclude_mask = fixed_exclude_mask
            exclude_meta = dict(fixed_exclude_meta)

        out = destripe_by_directional_lines(
            alpha_map=current,
            valid_mask=valid_mask,
            exclude_mask=exclude_mask,
            direction=direction,
            method=destripe_params.get("method", "median"),
            min_pixels_per_line=destripe_params.get("min_pixels_per_line", 5),
            preserve_global_stat=destripe_params.get("preserve_global_stat", True),
            smooth_half_window=destripe_params.get("smooth_half_window", 0),
            fallback_to_valid=destripe_params.get("fallback_to_valid", True),
            trim_fraction=destripe_params.get("trim_fraction", 0.1),
            mode_bins=destripe_params.get("mode_bins", 64),
            sigma_clip_nsigma=destripe_params.get("sigma_clip_nsigma", 3.0),
            sigma_clip_max_iter=destripe_params.get("sigma_clip_max_iter", 3),
        )

        current = out["corrected"]
        total_stripe = total_stripe + out["stripe_map"]
        directional_stripe_maps[direction] = out["stripe_map"].copy()

        table = out["line_table"].copy()
        table.insert(0, "pass_index", pass_index)
        line_tables.append(table)

        exclude_meta = dict(exclude_meta)
        exclude_meta.update({
            "pass_index": pass_index,
            "direction": direction,
        })
        exclude_metas.append(exclude_meta)

    line_table_all = pd.concat(line_tables, ignore_index=True) if len(line_tables) > 0 else pd.DataFrame()

    return {
        "corrected": current,
        "stripe_map": total_stripe,
        "directional_stripe_maps": directional_stripe_maps,
        "line_table": line_table_all,
        "exclude_meta": exclude_metas,
    }


# 6. Iterative MF with optional destriping


def should_apply_destriping(iteration_number: int, destripe_when) -> bool:
    if destripe_when is None or destripe_when == "none":
        return False
    if destripe_when == "each_iter":
        return True
    if destripe_when == "final_only":
        return False
    if isinstance(destripe_when, (list, tuple, set, np.ndarray)):
        return int(iteration_number) in {int(v) for v in destripe_when}
    raise ValueError("destripe_when must be 'none', 'each_iter', 'final_only', or a list of iteration numbers.")


def run_iterative_mf_with_optional_destriping(
    cube: np.ndarray,
    uas: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    initial_background_mask: Optional[np.ndarray] = None,
    n_iter: int = 5,
    nsigma: float = 4.0,
    reg: float = 1e-6,
    rcond: float = 1e-8,
    min_background_pixels: Optional[int] = None,
    destripe_when = "none",
    destripe_params: Optional[dict] = None,
    verbose: bool = True,
) -> dict:
    """Run Iterative MF, optionally inserting sequential directional destriping into the loop."""
    H, W, B = cube.shape

    if valid_mask is None:
        valid_mask = np.all(np.isfinite(cube), axis=2)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    if initial_background_mask is None:
        background_mask = valid_mask.copy()
    else:
        background_mask = valid_mask & np.asarray(initial_background_mask, dtype=bool)

    if min_background_pixels is None:
        min_background_pixels = max(B + 5, 30)

    if destripe_params is None:
        destripe_params = {}

    alpha_raw_history = []
    alpha_corrected_history = []
    alpha_used_history = []
    stripe_history = []
    directional_stripe_history = []
    plume_mask_history = []
    background_mask_history = []
    threshold_meta_history = []
    line_table_history = []
    exclude_meta_history = []
    mu_history = []
    cov_history = []

    prev_plume_mask = None
    converged_iter = None

    directions = normalize_directions(destripe_params) if destripe_params else []

    for it in range(1, n_iter + 1):
        n_bg = int(np.sum(background_mask))
        if n_bg < min_background_pixels:
            raise ValueError(f"Background pixels too few at iter {it}: {n_bg} < {min_background_pixels}")

        alpha_raw, mu, cov, target = matched_filter_alpha_map(
            cube=cube,
            uas=uas,
            valid_mask=valid_mask,
            background_mask=background_mask,
            reg=reg,
            rcond=rcond,
            min_background_pixels=min_background_pixels,
        )

        alpha_corrected = alpha_raw.copy()
        alpha_used = alpha_raw.copy()
        stripe_map = np.zeros((H, W), dtype=float)
        directional_stripe_maps = {}
        line_table = pd.DataFrame()
        exclude_meta = []

        apply_now = should_apply_destriping(it, destripe_when)
        if apply_now:
            out = destripe_by_sequential_directions(
                alpha_map=alpha_raw,
                valid_mask=valid_mask,
                plume_mask=prev_plume_mask,
                destripe_params=destripe_params,
                nsigma=nsigma,
            )

            alpha_corrected = out["corrected"]
            stripe_map = out["stripe_map"]
            directional_stripe_maps = out["directional_stripe_maps"]
            line_table = out["line_table"]
            exclude_meta = out["exclude_meta"]

            threshold_source = destripe_params.get("threshold_source", "corrected")
            if threshold_source == "corrected":
                alpha_used = alpha_corrected.copy()
            elif threshold_source == "raw":
                alpha_used = alpha_raw.copy()
            else:
                raise ValueError("threshold_source must be 'corrected' or 'raw'.")

        plume_mask, threshold_meta = plume_mask_from_alpha(alpha_used, valid_mask, nsigma=nsigma)
        new_background_mask = valid_mask & (~plume_mask)

        alpha_raw_history.append(alpha_raw.copy())
        alpha_corrected_history.append(alpha_corrected.copy())
        alpha_used_history.append(alpha_used.copy())
        stripe_history.append(stripe_map.copy())
        directional_stripe_history.append({k: v.copy() for k, v in directional_stripe_maps.items()})
        plume_mask_history.append(plume_mask.copy())
        background_mask_history.append(background_mask.copy())
        threshold_meta_history.append(threshold_meta.copy())
        line_table_history.append(line_table.copy())
        exclude_meta_history.append(exclude_meta.copy() if hasattr(exclude_meta, 'copy') else exclude_meta)
        mu_history.append(mu.copy())
        cov_history.append(cov.copy())

        if verbose:
            print(
                f"iter {it:02d} | bg={n_bg:6d} | "
                f"thr={threshold_meta['threshold']:+.6e} | "
                f"med={threshold_meta['median']:+.6e} | "
                f"rstd={threshold_meta['robust_std']:.6e} | "
                f"plume={threshold_meta['n_plume']:6d} | "
                f"destripe={apply_now} | directions={directions if apply_now else []}"
            )

        if prev_plume_mask is not None and np.array_equal(plume_mask, prev_plume_mask):
            converged_iter = it
            background_mask = new_background_mask
            if verbose:
                print(f"Converged at iteration {it}.")
            break

        prev_plume_mask = plume_mask.copy()
        background_mask = new_background_mask

    # Final-only destriping: apply after the iterative background/mask loop.
    final_only_post = None
    if destripe_when == "final_only":
        final_raw = alpha_raw_history[-1]
        final_loop_plume = plume_mask_history[-1]

        out = destripe_by_sequential_directions(
            alpha_map=final_raw,
            valid_mask=valid_mask,
            plume_mask=final_loop_plume,
            destripe_params=destripe_params,
            nsigma=nsigma,
        )

        final_corr = out["corrected"]
        final_plume, final_meta = plume_mask_from_alpha(final_corr, valid_mask, nsigma=nsigma)

        alpha_corrected_history[-1] = final_corr.copy()
        alpha_used_history[-1] = final_corr.copy()
        stripe_history[-1] = out["stripe_map"].copy()
        directional_stripe_history[-1] = {k: v.copy() for k, v in out["directional_stripe_maps"].items()}
        plume_mask_history[-1] = final_plume.copy()
        threshold_meta_history[-1] = final_meta.copy()
        line_table_history[-1] = out["line_table"].copy()
        exclude_meta_history[-1] = out["exclude_meta"]

        final_only_post = {
            "threshold_meta": final_meta,
            "exclude_meta": out["exclude_meta"],
        }

    result = {
        "alpha_raw_history": alpha_raw_history,
        "alpha_corrected_history": alpha_corrected_history,
        "alpha_used_history": alpha_used_history,
        "stripe_history": stripe_history,
        "directional_stripe_history": directional_stripe_history,
        "plume_mask_history": plume_mask_history,
        "background_mask_history": background_mask_history,
        "threshold_meta_history": threshold_meta_history,
        "line_table_history": line_table_history,
        "exclude_meta_history": exclude_meta_history,
        "mu_history": mu_history,
        "cov_history": cov_history,
        "valid_mask": valid_mask,
        "background_mask_final": background_mask,
        "converged_iter": converged_iter,
        "destripe_when": destripe_when,
        "destripe_params": destripe_params,
        "final_only_post": final_only_post,
    }

    # Convenience aliases
    result["alpha_final_raw"] = alpha_raw_history[-1]
    result["alpha_final_corrected"] = alpha_corrected_history[-1]
    result["alpha_final_used"] = alpha_used_history[-1]
    result["stripe_map_final"] = stripe_history[-1]
    result["directional_stripe_maps_final"] = directional_stripe_history[-1]
    result["plume_mask_final"] = plume_mask_history[-1]
    result["threshold_meta_final"] = threshold_meta_history[-1]
    result["line_table_final"] = line_table_history[-1]

    return result


# 7. Result comparison / saving helpers


def get_result_directions(res: dict):
    params = res.get("destripe_params")
    if params is None:
        return None
    return normalize_directions(params)


def summarize_results_table(results: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for name, res in results.items():
        meta = res["threshold_meta_final"]
        params = res.get("destripe_params")
        rows.append({
            "name": name,
            "destripe_when": res["destripe_when"],
            "method": None if params is None else params.get("method"),
            "directions": None if params is None else " -> ".join(normalize_directions(params)),
            "smooth_half_window": None if params is None else params.get("smooth_half_window"),
            "exclude_mode": None if params is None else params.get("exclude_mode"),
            "converged_iter": res["converged_iter"],
            "threshold": meta["threshold"],
            "median": meta["median"],
            "robust_std": meta["robust_std"],
            "plume_pixels": int(np.sum(res["plume_mask_final"])),
            "valid_pixels": int(np.sum(res["valid_mask"])),
        })
    return pd.DataFrame(rows)


def plot_single_result(res: dict, title_prefix: str = "result"):
    valid = res["valid_mask"]
    raw = res["alpha_final_raw"]
    corr = res["alpha_final_corrected"]
    stripe = res["stripe_map_final"]
    plume = res["plume_mask_final"]
    directional_maps = res.get("directional_stripe_maps_final", {})

    vmin, vmax = robust_limits([raw, corr], mask=valid, q_low=2, q_high=98)
    svmin, svmax = robust_limits(stripe, mask=valid, q_low=2, q_high=98)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    im0 = axes[0].imshow(raw, origin="upper", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"{title_prefix}\nraw alpha")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, label="alpha")

    im1 = axes[1].imshow(stripe, origin="upper", vmin=svmin, vmax=svmax)
    axes[1].set_title("total estimated stripe")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, label="offset")

    im2 = axes[2].imshow(corr, origin="upper", vmin=vmin, vmax=vmax)
    axes[2].set_title("corrected alpha")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, label="alpha")

    im3 = axes[3].imshow(plume, origin="upper")
    axes[3].set_title("plume mask")
    axes[3].set_xlabel("x")
    axes[3].set_ylabel("y")
    plt.colorbar(im3, ax=axes[3], fraction=0.046, label="candidate")

    plt.tight_layout()
    plt.show()

    # Direction-wise stripe maps
    if len(directional_maps) > 0:
        n = len(directional_maps)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
        if n == 1:
            axes = [axes]
        for ax, (direction, smap) in zip(axes, directional_maps.items()):
            lo, hi = robust_limits(smap, mask=valid, q_low=2, q_high=98)
            im = ax.imshow(smap, origin="upper", vmin=lo, vmax=hi)
            ax.set_title(f"stripe: {direction}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            plt.colorbar(im, ax=ax, fraction=0.046, label="offset")
        plt.tight_layout()
        plt.show()

    line_table = res["line_table_final"]
    if line_table is not None and len(line_table) > 0:
        plt.figure(figsize=(8, 4))
        for direction, df_dir in line_table.groupby("direction"):
            plt.plot(
                df_dir["line_id"],
                df_dir["stripe_offset_subtracted"],
                marker=".",
                linewidth=1,
                label=direction,
            )
        plt.axhline(0, color="black", linewidth=1)
        plt.xlabel("line_id")
        plt.ylabel("subtracted offset")
        plt.title(f"{title_prefix}: directional stripe offset")
        plt.grid(True)
        plt.legend()
        plt.show()


def plot_experiment_grid(results: dict[str, dict], names: Optional[Sequence[str]] = None):
    if names is None:
        names = list(results.keys())

    valid = next(iter(results.values()))["valid_mask"]
    maps = []
    for name in names:
        maps.append(results[name]["alpha_final_raw"])
        maps.append(results[name]["alpha_final_corrected"])
    vmin, vmax = robust_limits(maps, mask=valid, q_low=2, q_high=98)

    n = len(names)
    fig, axes = plt.subplots(3, n, figsize=(4 * n, 12))
    if n == 1:
        axes = axes.reshape(3, 1)

    for j, name in enumerate(names):
        res = results[name]
        raw = res["alpha_final_raw"]
        corr = res["alpha_final_corrected"]
        plume = res["plume_mask_final"]

        im0 = axes[0, j].imshow(raw, origin="upper", vmin=vmin, vmax=vmax)
        axes[0, j].set_title(f"{name}\nraw")
        axes[0, j].set_xlabel("x")
        axes[0, j].set_ylabel("y")

        im1 = axes[1, j].imshow(corr, origin="upper", vmin=vmin, vmax=vmax)
        axes[1, j].set_title("corrected")
        axes[1, j].set_xlabel("x")
        axes[1, j].set_ylabel("y")

        im2 = axes[2, j].imshow(plume, origin="upper")
        axes[2, j].set_title("plume mask")
        axes[2, j].set_xlabel("x")
        axes[2, j].set_ylabel("y")

    fig.colorbar(im1, ax=axes[0:2, :].ravel().tolist(), fraction=0.02, label="alpha")
    plt.tight_layout()
    plt.show()



def stat_case_name(method: str, when: str) -> str:
    return f"{method}_yx_then_ynegx_{when}"



def available_stat_methods(results: dict[str, dict], methods: Optional[Sequence[str]] = None) -> list[str]:
    if methods is None:
        methods = [m for m, _ in STRIPE_STAT_METHODS]
    out = []
    for method in methods:
        if stat_case_name(method, "final_only") in results or stat_case_name(method, "each_iter") in results:
            out.append(method)
    return out



def plot_when_comparison_by_stat(
    results: dict[str, dict],
    when: str,
    methods: Optional[Sequence[str]] = None,
    include_baseline: bool = True,
):
    """Show one overview image for final_only or each_iter across all statistics."""
    if when not in {"final_only", "each_iter"}:
        raise ValueError("when must be 'final_only' or 'each_iter'.")

    methods = available_stat_methods(results, methods)
    names = [stat_case_name(method, when) for method in methods if stat_case_name(method, when) in results]
    if include_baseline and "baseline_no_destripe" in results:
        names = ["baseline_no_destripe"] + names

    if len(names) == 0:
        print(f"No cases found for {when}.")
        return

    print(f"Overview: {when} / " + ", ".join(names))
    plot_experiment_grid(results, names=names)



def plot_stat_method_results_separately(
    results: dict[str, dict],
    methods: Optional[Sequence[str]] = None,
    include_pair_grid: bool = True,
    show_single_result: bool = True,
):
    """For each statistic, show final_only and each_iter as separate detailed images."""
    methods = available_stat_methods(results, methods)

    for method in methods:
        names = [
            stat_case_name(method, "final_only"),
            stat_case_name(method, "each_iter"),
        ]
        names = [name for name in names if name in results]
        if len(names) == 0:
            continue

        print("\n" + "-" * 80)
        print(f"Statistic: {method}")
        print("-" * 80)

        if include_pair_grid and len(names) >= 2:
            plot_experiment_grid(results, names=names)

        if show_single_result:
            for name in names:
                plot_single_result(results[name], title_prefix=name)



def plot_stat_threshold_histories(results: dict[str, dict], methods: Optional[Sequence[str]] = None):
    """Show threshold histories for final_only and each_iter, grouped by statistic."""
    methods = available_stat_methods(results, methods)
    plt.figure(figsize=(10, 6))
    for method in methods:
        for when, linestyle in [("final_only", "--"), ("each_iter", "-")]:
            name = stat_case_name(method, when)
            if name not in results:
                continue
            th = [m["threshold"] for m in results[name]["threshold_meta_history"]]
            plt.plot(np.arange(1, len(th) + 1), th, marker="o", linestyle=linestyle, label=name)
    plt.xlabel("Iteration")
    plt.ylabel("Threshold")
    plt.title("Threshold history by statistic")
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.show()



def plot_threshold_histories(results: dict[str, dict]):
    plt.figure(figsize=(8, 5))
    for name, res in results.items():
        th = [m["threshold"] for m in res["threshold_meta_history"]]
        plt.plot(np.arange(1, len(th) + 1), th, marker="o", label=name)
    plt.xlabel("Iteration")
    plt.ylabel("Threshold")
    plt.title("Threshold history")
    plt.grid(True)
    plt.legend()
    plt.show()


def save_case_outputs(
    result: dict,
    case_name: str,
    output_dir: str | Path,
    ys: Optional[np.ndarray] = None,
    xs: Optional[np.ndarray] = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = result["alpha_final_raw"]
    corr = result["alpha_final_corrected"]
    stripe = result["stripe_map_final"]
    plume = result["plume_mask_final"]
    valid = result["valid_mask"]
    directional_maps = result.get("directional_stripe_maps_final", {})

    np.save(output_dir / f"{case_name}_alpha_raw.npy", raw)
    np.save(output_dir / f"{case_name}_alpha_corrected.npy", corr)
    np.save(output_dir / f"{case_name}_stripe_map_total.npy", stripe)
    np.save(output_dir / f"{case_name}_plume_mask.npy", plume)

    for direction, smap in directional_maps.items():
        np.save(output_dir / f"{case_name}_stripe_map_{direction}.npy", smap)

    H, W = raw.shape
    rows, cols = np.indices((H, W))

    if ys is not None and len(ys) == H:
        y_values = np.asarray(ys)[rows.ravel()]
    else:
        y_values = rows.ravel()

    if xs is not None and len(xs) == W:
        x_values = np.asarray(xs)[cols.ravel()]
    else:
        x_values = cols.ravel()

    y_minus_x_ids = line_id_map(raw.shape, direction="y_minus_x")
    y_plus_x_ids = line_id_map(raw.shape, direction="y_plus_x")

    pixel_df = pd.DataFrame({
        "row": rows.ravel(),
        "col": cols.ravel(),
        "y": y_values,
        "x": x_values,
        "line_id_y_minus_x": y_minus_x_ids.ravel(),
        "line_id_y_plus_x": y_plus_x_ids.ravel(),
        "is_valid": valid.ravel(),
        "alpha_raw": raw.ravel(),
        "stripe_offset_total_subtracted": stripe.ravel(),
        "alpha_corrected": corr.ravel(),
        "is_plume": plume.ravel(),
    })

    for direction, smap in directional_maps.items():
        pixel_df[f"stripe_offset_{direction}"] = smap.ravel()

    pixel_df.to_csv(output_dir / f"{case_name}_pixel_results.csv", index=False)

    line_table = result["line_table_final"]
    if line_table is not None and len(line_table) > 0:
        line_table.to_csv(output_dir / f"{case_name}_line_table.csv", index=False)

    return pixel_df

USAGE_NOTES = """
Usage and statistic notes
=========================

Images to inspect first
-----------------------
This version can display these figures when --show-plots is used:

- one overview for final_only across statistics,
- one overview for each_iter across statistics,
- separate detailed figures for final_only and each_iter for each statistic:
  median, mean, trimmed_mean, mode, and sigma_clipped_mean.

Each detailed figure shows raw alpha, total estimated stripe, corrected alpha,
and plume mask. Direction-wise stripe maps are also shown, so inspect both
stripe: y_minus_x and stripe: y_plus_x before trusting the corrected alpha map.

final_only vs each_iter
-----------------------
final_only leaves the Iterative MF background-mask update untouched and applies
destriping only to the last alpha map. This is the safest first check when you
only want to see whether the visible stripe can be removed as a post-process.

each_iter applies destriping before plume thresholding at every iteration. This
can help when stripe artifacts are being classified as plume candidates and
removed from the background. It is stronger, but it can also weaken a real plume
if the plume is elongated in the same direction as the estimated stripe.

Statistic choice
----------------
median is robust to outliers and is usually the best first candidate when
high-alpha plume pixels are sparse along a line.

mean follows broad line-wide offsets well, but it is sensitive to plume pixels
and other high-alpha outliers. Check whether exclude_mode="robust_high" is
excluding enough pixels.

mode can work well when the true background forms the densest peak in the
line-wise alpha distribution. Because it is histogram-based for continuous alpha
values, test mode_bins values such as 32, 64, and 128 and look for stability.

trimmed_mean discards a fraction of the lowest and highest values before
averaging. It is often a good middle ground: smoother than median, less
outlier-sensitive than mean.

sigma_clipped_mean clips outliers using median + MAD and then averages the
remaining pixels. It is useful when stripes affect most pixels in a line while
plume pixels appear as a minority of high values. Try sigma_clip_nsigma around
2.5 to 4.0.

Caution
-------
Two-direction destriping is powerful. Judge the result using the corrected alpha
map, the total stripe map, the direction-wise stripe maps, and the plume mask
together. A real elongated plume can be partially removed if it aligns with the
stripe model.
"""



# Workflow runner



def prepare_cube_and_uas(
    roi_csv: str | Path,
    modtran_csv: str | Path,
    wl_min: float = WL_MIN,
    wl_max: float = WL_MAX,
    fwhm_nm: float = FWHM_NM,
    uas_alpha_min: float = UAS_ALPHA_MIN,
    uas_alpha_max: float = UAS_ALPHA_MAX,
    nodata_values: Optional[Sequence[float]] = None,
    require_positive: bool = REQUIRE_POSITIVE,
    min_valid_fraction: float = MIN_VALID_FRACTION,
    show_plots: bool = False,
) -> dict:
    """Load ROI and MODTRAN data, then prepare the selected cube and UAS template."""
    if nodata_values is None:
        nodata_values = NODATA_VALUES

    roi_df, wavelengths, spectra = load_roi_spectra_csv(roi_csv)
    cube, ys, xs = spectra_to_cube(roi_df, spectra)

    print(f"ROI table shape: {roi_df.shape}")
    print(f"Cube shape: {cube.shape}")
    print(f"Wavelength range: {wavelengths[0]:.2f} - {wavelengths[-1]:.2f} nm")

    if show_plots:
        plot_mean_spectrum(cube, wavelengths, xlim=(wl_min, wl_max))
        try:
            rgb = make_rgb_from_cube(cube, wavelengths)
            plt.figure(figsize=(5, 5))
            plt.imshow(rgb, origin="upper")
            plt.title("RGB preview")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.show()
        except Exception as exc:
            print(f"RGB preview skipped: {exc}")

    mod_wave, alpha_grid, mod_spectra = load_ch4_modtran_csv(modtran_csv)
    print(f"MODTRAN wavelength range: {mod_wave[0]:.2f} - {mod_wave[-1]:.2f} nm")
    print("alpha_grid:", alpha_grid)

    mod_sensor = gaussian_srf_resample(
        mod_wave=mod_wave,
        mod_spectra=mod_spectra,
        sensor_wave=wavelengths,
        fwhm_nm=fwhm_nm,
    )

    uas_all, intercept = compute_uas_log_slope(
        alpha_grid=alpha_grid,
        spectra_grid=mod_sensor,
        alpha_min=uas_alpha_min,
        alpha_max=uas_alpha_max,
    )

    if show_plots:
        plot_uas(wavelengths, uas_all, title="CH4 UAS from MODTRAN", xlim=(wl_min, wl_max))

    cube_sel, wave_sel, band_sel = select_bands(cube, wavelengths, wl_min=wl_min, wl_max=wl_max)
    uas_sel = uas_all[band_sel]

    valid_mask = make_valid_pixel_mask(
        cube_sel,
        nodata_values=nodata_values,
        require_positive=require_positive,
        min_valid_fraction=min_valid_fraction,
    )

    print(f"Selected cube shape: {cube_sel.shape}")
    print(f"Valid pixels: {int(np.sum(valid_mask))}")

    if show_plots:
        plot_map(valid_mask.astype(float), title="Valid pixel mask", cmap="gray", colorbar_label="valid")

    return {
        "roi_df": roi_df,
        "cube": cube,
        "wavelengths": wavelengths,
        "ys": ys,
        "xs": xs,
        "mod_wave": mod_wave,
        "alpha_grid": alpha_grid,
        "mod_spectra": mod_spectra,
        "mod_sensor": mod_sensor,
        "uas_all": uas_all,
        "uas_intercept": intercept,
        "cube_sel": cube_sel,
        "wave_sel": wave_sel,
        "band_sel": band_sel,
        "uas_sel": uas_sel,
        "valid_mask": valid_mask,
    }


def run_experiments(
    cube: np.ndarray,
    uas: np.ndarray,
    valid_mask: np.ndarray,
    experiments: Optional[dict[str, dict]] = None,
    n_iter: int = N_ITER,
    nsigma: float = NSIGMA,
    reg: float = REG,
    rcond: float = RCOND,
    verbose: bool = True,
) -> dict[str, dict]:
    """Run all configured destriping experiments."""
    if experiments is None:
        experiments = make_experiments(nsigma=nsigma)

    results: dict[str, dict] = {}

    for name, cfg in experiments.items():
        print("\n" + "=" * 80)
        print(f"Running experiment: {name}")
        print("=" * 80)

        res = run_iterative_mf_with_optional_destriping(
            cube=cube,
            uas=uas,
            valid_mask=valid_mask,
            initial_background_mask=None,
            n_iter=n_iter,
            nsigma=nsigma,
            reg=reg,
            rcond=rcond,
            min_background_pixels=None,
            destripe_when=cfg.get("destripe_when", "none"),
            destripe_params=cfg.get("destripe_params", None),
            verbose=verbose,
        )
        results[name] = res

    return results


def save_selected_case_outputs(
    results: dict[str, dict],
    output_dir: str | Path,
    ys: Optional[np.ndarray] = None,
    xs: Optional[np.ndarray] = None,
) -> list[str]:
    """Save baseline, one-direction median cases, and all two-direction statistic cases."""
    cases_to_save = [
        "baseline_no_destripe",
        "median_yx_final_only",
        "median_yx_each_iter",
    ]

    for method, _ in STRIPE_STAT_METHODS:
        cases_to_save.extend([
            f"{method}_yx_then_ynegx_final_only",
            f"{method}_yx_then_ynegx_each_iter",
        ])

    saved_cases: list[str] = []
    for case_name in cases_to_save:
        if case_name in results:
            save_case_outputs(
                result=results[case_name],
                case_name=case_name,
                output_dir=output_dir,
                ys=ys,
                xs=xs,
            )
            print(f"Saved: {case_name}")
            saved_cases.append(case_name)
        else:
            print(f"Skipped missing case: {case_name}")
    return saved_cases


def plot_comparison_figures(results: dict[str, dict]) -> None:
    """Show the comparison plots used in the original notebook."""
    summary_df = summarize_results_table(results)
    print(summary_df.to_string(index=False))

    plot_threshold_histories(results)
    plot_stat_threshold_histories(results)
    plot_when_comparison_by_stat(results, when="final_only")
    plot_when_comparison_by_stat(results, when="each_iter")
    plot_stat_method_results_separately(
        results,
        methods=[m for m, _ in STRIPE_STAT_METHODS],
        include_pair_grid=True,
        show_single_result=True,
    )

    primary_names = [
        "baseline_no_destripe",
        "median_yx_final_only",
        "median_yx_then_ynegx_final_only",
        "median_yx_each_iter",
        "median_yx_then_ynegx_each_iter",
    ]
    primary_names = [name for name in primary_names if name in results]
    plot_experiment_grid(results, names=primary_names)


def run_workflow(
    roi_csv: str | Path = ROI_CSV,
    modtran_csv: str | Path = MODTRAN_CSV,
    output_dir: str | Path = OUTPUT_DIR,
    wl_min: float = WL_MIN,
    wl_max: float = WL_MAX,
    fwhm_nm: float = FWHM_NM,
    uas_alpha_min: float = UAS_ALPHA_MIN,
    uas_alpha_max: float = UAS_ALPHA_MAX,
    n_iter: int = N_ITER,
    nsigma: float = NSIGMA,
    reg: float = REG,
    rcond: float = RCOND,
    show_plots: bool = False,
    save_outputs: bool = True,
    verbose: bool = True,
) -> dict:
    """Run the complete statistic-comparison workflow."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = prepare_cube_and_uas(
        roi_csv=roi_csv,
        modtran_csv=modtran_csv,
        wl_min=wl_min,
        wl_max=wl_max,
        fwhm_nm=fwhm_nm,
        uas_alpha_min=uas_alpha_min,
        uas_alpha_max=uas_alpha_max,
        nodata_values=NODATA_VALUES,
        require_positive=REQUIRE_POSITIVE,
        min_valid_fraction=MIN_VALID_FRACTION,
        show_plots=show_plots,
    )

    experiments = make_experiments(nsigma=nsigma)
    results = run_experiments(
        cube=data["cube_sel"],
        uas=data["uas_sel"],
        valid_mask=data["valid_mask"],
        experiments=experiments,
        n_iter=n_iter,
        nsigma=nsigma,
        reg=reg,
        rcond=rcond,
        verbose=verbose,
    )

    summary_df = summarize_results_table(results)
    summary_path = output_dir / "stat_compare_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")
    print(summary_df.to_string(index=False))

    if show_plots:
        plot_comparison_figures(results)

    saved_cases: list[str] = []
    if save_outputs:
        saved_cases = save_selected_case_outputs(
            results=results,
            output_dir=output_dir,
            ys=data["ys"],
            xs=data["xs"],
        )

    return {
        "data": data,
        "experiments": experiments,
        "results": results,
        "summary_df": summary_df,
        "saved_cases": saved_cases,
        "output_dir": output_dir,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Iterative MF with two-direction diagonal destriping and statistic comparison."
    )
    parser.add_argument("--roi-csv", type=Path, default=ROI_CSV, help="ROI spectra CSV path.")
    parser.add_argument("--modtran-csv", type=Path, default=MODTRAN_CSV, help="MODTRAN methane spectra CSV path.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--wl-min", type=float, default=WL_MIN, help="Minimum wavelength [nm].")
    parser.add_argument("--wl-max", type=float, default=WL_MAX, help="Maximum wavelength [nm].")
    parser.add_argument("--fwhm-nm", type=float, default=FWHM_NM, help="Sensor FWHM used for MODTRAN resampling [nm].")
    parser.add_argument("--uas-alpha-min", type=float, default=UAS_ALPHA_MIN, help="Minimum alpha used for UAS fitting.")
    parser.add_argument("--uas-alpha-max", type=float, default=UAS_ALPHA_MAX, help="Maximum alpha used for UAS fitting.")
    parser.add_argument("--n-iter", type=int, default=N_ITER, help="Number of Iterative MF iterations.")
    parser.add_argument("--nsigma", type=float, default=NSIGMA, help="Robust threshold multiplier.")
    parser.add_argument("--reg", type=float, default=REG, help="Covariance regularization.")
    parser.add_argument("--rcond", type=float, default=RCOND, help="Pseudo-inverse rcond value.")
    parser.add_argument("--show-plots", action="store_true", help="Show notebook-style diagnostic plots.")
    parser.add_argument("--no-save", action="store_true", help="Do not save per-case output files.")
    parser.add_argument("--quiet", action="store_true", help="Reduce iteration logs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_workflow(
        roi_csv=args.roi_csv,
        modtran_csv=args.modtran_csv,
        output_dir=args.output_dir,
        wl_min=args.wl_min,
        wl_max=args.wl_max,
        fwhm_nm=args.fwhm_nm,
        uas_alpha_min=args.uas_alpha_min,
        uas_alpha_max=args.uas_alpha_max,
        n_iter=args.n_iter,
        nsigma=args.nsigma,
        reg=args.reg,
        rcond=args.rcond,
        show_plots=args.show_plots,
        save_outputs=not args.no_save,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
