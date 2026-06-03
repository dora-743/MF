from __future__ import annotations

import re
from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# CSV and cube utilities

# returns wavelength column names and wavelengths sorted by wavelength
def get_wave_columns(df: pd.DataFrame) -> Tuple[list[str], np.ndarray]:
    wave_cols: list[str] = []
    wavelengths: list[float] = []
    pattern = re.compile(r"wave_([0-9.]+)nm")

    for col in df.columns:
        match = pattern.match(col)
        if match is not None:
            wave_cols.append(col)
            wavelengths.append(float(match.group(1)))

    wavelengths_arr = np.asarray(wavelengths, dtype=float)
    order = np.argsort(wavelengths_arr)

    return [wave_cols[i] for i in order], wavelengths_arr[order]


# load CSV with `y`, `x`, and wavelength columns, returning the DataFrame, wavelengths, and spectra array
def load_roi_spectra_csv(path: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:

    df = pd.read_csv(path)
    if "y" not in df.columns or "x" not in df.columns:
        raise ValueError("The CSV file must contain `y` and `x` columns.")

    wave_cols, wavelengths = get_wave_columns(df)
    if not wave_cols:
        raise ValueError("No wavelength columns were found in the CSV file.")

    spectra = df[wave_cols].to_numpy(dtype=float)
    if not np.all(np.isfinite(spectra)):
        raise ValueError("Spectra contain non-finite values.")

    return df, wavelengths, spectra


# convert between `(N, B)` spectra and `(H, W, B)` cube using `y` and `x` columns in the DataFrame, filling missing pixels with `fill_value`
def spectra_to_cube(
    df: pd.DataFrame,
    spectra: np.ndarray,
    fill_value: float = np.nan,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    ys = np.sort(df["y"].unique())
    xs = np.sort(df["x"].unique())

    y_to_i = {y: i for i, y in enumerate(ys)}
    x_to_j = {x: j for j, x in enumerate(xs)}

    height, width = len(ys), len(xs)
    n_bands = spectra.shape[1]
    cube = np.full((height, width, n_bands), fill_value, dtype=float)

    for row_idx, row in df.iterrows():
        i = y_to_i[row["y"]]
        j = x_to_j[row["x"]]
        cube[i, j, :] = spectra[row_idx, :]

    return cube, ys, xs


def cube_to_spectra(cube: np.ndarray) -> np.ndarray:
    """Flatten a cube from `(H, W, B)` to `(H*W, B)`."""
    height, width, n_bands = cube.shape
    return cube.reshape(height * width, n_bands)


# Wavelength selection

# create a boolean mask for selecting wavelength bands based on min/max and exclusion ranges
def band_mask(
    wavelengths: np.ndarray,
    wl_min: Optional[float] = None,
    wl_max: Optional[float] = None,
    exclude_ranges: Optional[Sequence[Tuple[float, float]]] = None,
) -> np.ndarray:
    mask = np.ones_like(wavelengths, dtype=bool)

    if wl_min is not None:
        mask &= wavelengths >= wl_min
    if wl_max is not None:
        mask &= wavelengths <= wl_max
    if exclude_ranges is not None:
        for start, end in exclude_ranges:
            mask &= ~((wavelengths >= start) & (wavelengths <= end))

    return mask


# select bands from a cube and wavelengths using the band_mask function
def select_bands(
    cube: np.ndarray,
    wavelengths: np.ndarray,
    wl_min: float,
    wl_max: float,
    exclude_ranges: Optional[Sequence[tuple[float, float]]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = band_mask(wavelengths, wl_min, wl_max, exclude_ranges)
    return cube[:, :, mask], wavelengths[mask], mask


# Plot utilities


# plot a spectrum with optional x-axis limits
def plot_spectrum(
    wavelengths: np.ndarray,
    spectrum: np.ndarray,
    title: str = "Spectrum",
    xlim: Optional[Tuple[float, float]] = None,
) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(wavelengths, spectrum)
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Radiance / Reflectance")
    plt.title(title)
    if xlim is not None:
        plt.xlim(*xlim)
    plt.grid(True)
    plt.show()


# plot the spectrum of a single pixel from the cube at coordinates (y, x)
def plot_pixel_spectrum(
    cube: np.ndarray,
    wavelengths: np.ndarray,
    y: int,
    x: int,
    xlim: Optional[Tuple[float, float]] = None,
) -> None:
    
    plot_spectrum(
        wavelengths,
        cube[y, x, :],
        title=f"Spectrum at y={y}, x={x}",
        xlim=xlim,
    )


# plot the mean spectrum of all pixels in the cube, optionally using a mask to select specific pixels
def plot_mean_spectrum(
    cube: np.ndarray,
    wavelengths: np.ndarray,
    mask: Optional[np.ndarray] = None,
    xlim: Optional[Tuple[float, float]] = None,
) -> None:
    spectra = cube.reshape(-1, cube.shape[-1]) if mask is None else cube[mask]
    mean_spec = np.nanmean(spectra, axis=0)
    plot_spectrum(wavelengths, mean_spec, title="Mean Spectrum", xlim=xlim)


# plot an image of a single band from the cube at the wavelength closest to target_wl
def plot_band_image(
    cube: np.ndarray,
    wavelengths: np.ndarray,
    target_wl: float,
    title: Optional[str] = None,
) -> None:
    idx = int(np.argmin(np.abs(wavelengths - target_wl)))
    image = cube[:, :, idx]

    plt.figure(figsize=(5, 5))
    plt.imshow(image)
    plt.colorbar(label="Value")
    plt.title(title or f"Band image: {wavelengths[idx]:.2f} nm")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


# plot the Matched Filter alpha map as an image, with optional vmin/vmax for color scaling
def plot_alpha_map(
    alpha_map: np.ndarray,
    title: str = "Matched Filter Output",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    plt.figure(figsize=(5, 5))
    plt.imshow(alpha_map, vmin=vmin, vmax=vmax)
    plt.colorbar(label=r"$\hat{\alpha}$")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


# scale an image robustly by clipping to the percentiles and rescaling to [0, 1]
def robust_scale_image(img: np.ndarray, pmin: float = 2, pmax: float = 98) -> np.ndarray:
    lo, hi = np.nanpercentile(img, [pmin, pmax])
    if hi <= lo:
        return np.zeros_like(img)
    return np.clip((img - lo) / (hi - lo), 0, 1)


# create a pseudo-RGB image from a cube by selecting bands near specified R, G, B wavelengths and scaling them robustly
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


# plot an RGB image created from the cube using make_rgb_from_cube
def plot_rgb(rgb: np.ndarray, title: str = "RGB") -> None:
    plt.figure(figsize=(5, 5))
    plt.imshow(rgb)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


# plot the UAS spectrum as a line plot, with optional x-axis limits
def plot_uas(
    wavelengths: np.ndarray,
    uas: np.ndarray,
    title: str = "CH4 UAS",
    xlim: Optional[Tuple[float, float]] = None,
) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(wavelengths, uas)
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("UAS")
    plt.title(title)
    if xlim is not None:
        plt.xlim(*xlim)
    plt.grid(True)
    plt.show()


# plot the Matched Filter target spectrum as a line plot, with optional x-axis limits
def plot_target(
    wavelengths: np.ndarray,
    target: np.ndarray,
    title: str = "MF target spectrum",
    xlim: Optional[Tuple[float, float]] = None,
) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(wavelengths, target)
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Target")
    plt.title(title)
    if xlim is not None:
        plt.xlim(*xlim)
    plt.grid(True)
    plt.show()


# plot multiple MODTRAN spectra for selected alpha values, with optional x-axis limits
def compare_modtran_spectra(
    wavelengths: np.ndarray,
    alpha_grid: np.ndarray,
    spectra_grid: np.ndarray,
    alpha_list: Optional[Sequence[float]] = None,
    xlim: Optional[Tuple[float, float]] = None,
) -> None:
    if alpha_list is None:
        alpha_list = [alpha_grid[0], alpha_grid[len(alpha_grid) // 2], alpha_grid[-1]]

    plt.figure(figsize=(8, 4))
    for alpha in alpha_list:
        idx = int(np.argmin(np.abs(alpha_grid - alpha)))
        plt.plot(wavelengths, spectra_grid[idx], label=f"alpha={alpha_grid[idx]:g}")

    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Radiance")
    plt.title("MODTRAN CH4 spectra")
    if xlim is not None:
        plt.xlim(*xlim)
    plt.legend()
    plt.grid(True)
    plt.show()


# Background statistics and Matched Filter functions


# estimate the background mean and covariance from a set of spectra, optionally using a mask to select specific spectra and adding regularization to the covariance
def estimate_background_mean_cov(
    spectra: np.ndarray,
    mask: Optional[np.ndarray] = None,
    reg: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    x = spectra[mask] if mask is not None else spectra
    good = np.all(np.isfinite(x), axis=1)
    x = x[good]

    if x.size == 0:
        raise ValueError("No valid spectra are available for background estimation.")

    mu = np.nanmean(x, axis=0)
    x_centered = x - mu
    cov = np.cov(x_centered, rowvar=False)
    cov = cov + reg * np.eye(cov.shape[0])
    return mu, cov


# estimate the background mean and covariance from a cube using a valid pixel mask, with regularization and a minimum pixel count requirement
def estimate_background_mean_cov_from_cube(
    cube: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    reg: float = 1e-6,
    min_pixels: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, _, n_bands = cube.shape

    if valid_mask is None:
        valid_mask = np.all(np.isfinite(cube), axis=2)

    x = cube[valid_mask]
    good = np.all(np.isfinite(x), axis=1)
    x = x[good]

    if min_pixels is None:
        min_pixels = max(n_bands + 5, 30)

    if x.shape[0] < min_pixels:
        raise ValueError(
            f"Too few valid background pixels: {x.shape[0]} pixels, "
            f"required >= {min_pixels}."
        )

    mu = np.mean(x, axis=0)
    x_centered = x - mu
    cov = np.cov(x_centered, rowvar=False)
    cov = cov + reg * np.eye(n_bands)

    return mu, cov, x


# compute the pseudo-inverse of the covariance matrix with a specified rcond for numerical stability
def invert_cov(cov: np.ndarray, rcond: float = 1e-8) -> np.ndarray:
    return np.linalg.pinv(cov, rcond=rcond)


# buikd the Matched Filter target spectrum from the background mean and UAS, with an optional sign to control the direction of the target
def make_target_from_uas(mu: np.ndarray, uas: np.ndarray, sign: float = -1.0) -> np.ndarray:
    return sign * mu * uas


# build the Matched Filter target spectrum specifically for methane detection, using the background mean and UAS, with an option to control the sign of the target
def make_methane_target(mu: np.ndarray, uas: np.ndarray, positive_alpha: bool = True) -> np.ndarray:
    return -mu * uas if positive_alpha else mu * uas


# approximate the UAS by finite-differencing the log of two spectra at different alpha values, with a small epsilon to avoid division by zero
def finite_difference_uas(L0: np.ndarray, L1: np.ndarray, delta_alpha: float) -> np.ndarray:
    eps = 1e-12
    ratio = np.clip(L1 / np.maximum(L0, eps), eps, None)
    return -np.log(ratio) / delta_alpha


# appply the standard Matched Filter to a set of spectra with a given target, using the background mean and covariance, and returning the alpha values for each spectrum
def matched_filter(
    spectra: np.ndarray,
    target: np.ndarray,
    mu: Optional[np.ndarray] = None,
    cov: Optional[np.ndarray] = None,
    background_mask: Optional[np.ndarray] = None,
    reg: float = 1e-6,
    rcond: float = 1e-8,
) -> np.ndarray:
    if mu is None or cov is None:
        mu, cov = estimate_background_mean_cov(
            spectra,
            mask=background_mask,
            reg=reg,
        )

    inv_cov = invert_cov(cov, rcond=rcond)
    target = target.reshape(-1)
    diff = spectra - mu

    numerator = diff @ (inv_cov @ target)
    denominator = target.T @ inv_cov @ target

    if np.abs(denominator) < 1e-12:
        raise ValueError("Denominator is too small. Check the target spectrum and covariance matrix.")

    return numerator / denominator


# apply the Matched Filter to a hyperspectral cube, using a valid pixel mask and optionally a background mask for estimating the mean and covariance, returning the alpha map and the estimated background mean and covariance
def matched_filter_map_from_cube(
    cube: np.ndarray,
    uas: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    reg: float = 1e-6,
    background_mask: Optional[np.ndarray] = None,
    rcond: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width, n_bands = cube.shape
    uas = np.asarray(uas, dtype=float).reshape(-1)
    if uas.shape[0] != n_bands:
        raise ValueError(f"UAS length {uas.shape[0]} does not match cube bands {n_bands}.")

    if valid_mask is None:
        valid_mask = np.all(np.isfinite(cube), axis=2)

    if background_mask is None:
        background_mask = valid_mask
    else:
        background_mask = background_mask & valid_mask

    mu, cov, _ = estimate_background_mean_cov_from_cube(
        cube,
        valid_mask=background_mask,
        reg=reg,
    )
    target = make_methane_target(mu, uas, positive_alpha=True)
    inv_cov = np.linalg.pinv(cov, rcond=rcond)

    alpha_map = np.full((height, width), np.nan, dtype=float)
    x = cube[valid_mask]
    diff = x - mu

    numerator = diff @ inv_cov @ target
    denominator = target.T @ inv_cov @ target

    if np.abs(denominator) < 1e-12:
        raise ValueError("Denominator is too small. Check the target spectrum and covariance matrix.")

    alpha_map[valid_mask] = numerator / denominator
    return alpha_map, mu, cov


# reshape a flat alpha array back to the original cube dimensions using the height and width
def alpha_to_map(alpha: np.ndarray, height: int, width: int) -> np.ndarray:
    return alpha.reshape(height, width)


# MODTRAN and UAS utilities


# load MODTRAN spectra from a CSV file with a `wavelength` column and additional columns for different alpha values, returning the wavelengths, sorted alpha values, and spectra grid
def load_ch4_modtran_csv(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    df_mod = pd.read_csv(path)
    if "wavelength" not in df_mod.columns:
        raise ValueError("The MODTRAN CSV file must contain a `wavelength` column.")

    mod_wave = df_mod["wavelength"].to_numpy(dtype=float)
    alpha_cols = [col for col in df_mod.columns if col != "wavelength"]

    try:
        alpha_grid = np.asarray([float(col) for col in alpha_cols], dtype=float)
    except ValueError as exc:
        raise ValueError(
            "Column names other than `wavelength` must be numeric alpha values."
        ) from exc

    order = np.argsort(alpha_grid)
    alpha_grid = alpha_grid[order]
    alpha_cols = [alpha_cols[i] for i in order]

    spectra_grid = df_mod[alpha_cols].to_numpy(dtype=float).T
    if not np.all(np.isfinite(spectra_grid)):
        raise ValueError("MODTRAN spectra contain NaN or inf values.")

    return mod_wave, alpha_grid, spectra_grid

 # resample the MODTRAN spectra to the sensor wavelengths using a Gaussian SRF with specified FWHM, returning the resampled spectra grid
def gaussian_srf_resample(
    mod_wave: np.ndarray,
    mod_spectra: np.ndarray,
    sensor_wave: np.ndarray,
    fwhm_nm: float | np.ndarray,
) -> np.ndarray:
    mod_wave = np.asarray(mod_wave, dtype=float)
    sensor_wave = np.asarray(sensor_wave, dtype=float)
    mod_spectra = np.asarray(mod_spectra, dtype=float)

    if np.isscalar(fwhm_nm):
        fwhm_arr = np.full_like(sensor_wave, float(fwhm_nm), dtype=float)
    else:
        fwhm_arr = np.asarray(fwhm_nm, dtype=float)
        if len(fwhm_arr) != len(sensor_wave):
            raise ValueError("fwhm_nm must be a scalar or an array with the same length as sensor_wave.")

    output = np.zeros((mod_spectra.shape[0], len(sensor_wave)), dtype=float)

    for j, center in enumerate(sensor_wave):
        sigma = fwhm_arr[j] / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        use = np.abs(mod_wave - center) <= 4.0 * sigma

        if np.sum(use) < 2:
            output[:, j] = np.interp(center, mod_wave, mod_spectra)
            continue

        weights = np.exp(-0.5 * ((mod_wave[use] - center) / sigma) ** 2)
        weights = weights / np.sum(weights)
        output[:, j] = mod_spectra[:, use] @ weights

    return output


# compute the UAS by fitting a line to the log of the spectra across the alpha grid, optionally using only a subset of alpha values defined by alpha_min and alpha_max, and returning the UAS (negative slope) and intercept
def compute_uas_log_slope(
    alpha_grid: np.ndarray,
    spectra_grid: np.ndarray,
    alpha_min: Optional[float] = None,
    alpha_max: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    alpha_grid = np.asarray(alpha_grid, dtype=float)
    spectra_grid = np.asarray(spectra_grid, dtype=float)

    use = np.ones_like(alpha_grid, dtype=bool)
    if alpha_min is not None:
        use &= alpha_grid >= alpha_min
    if alpha_max is not None:
        use &= alpha_grid <= alpha_max

    alpha = alpha_grid[use]
    log_spectra = np.log(np.maximum(spectra_grid[use], 1e-30))

    if len(alpha) < 2:
        raise ValueError("Not enough alpha points in the specified range for the log-slope method.")

    design = np.vstack([np.ones_like(alpha), alpha]).T
    coeff, _, _, _ = np.linalg.lstsq(design, log_spectra, rcond=None)

    intercept = coeff[0]
    slope = coeff[1]
    uas = -slope

    return uas, intercept


# compute the UAS by finite-differencing the log of two spectra at specified reference and perturbation alpha values, with a small epsilon to avoid division by zero, and returning the UAS
def compute_uas_two_spectra(
    alpha_grid: np.ndarray,
    spectra_grid: np.ndarray,
    alpha_ref: float = 0.0,
    alpha_pert: float = 1.0,
) -> np.ndarray:
    idx_ref = int(np.argmin(np.abs(alpha_grid - alpha_ref)))
    idx_pert = int(np.argmin(np.abs(alpha_grid - alpha_pert)))

    a0 = alpha_grid[idx_ref]
    a1 = alpha_grid[idx_pert]
    if np.isclose(a1, a0):
        raise ValueError("alpha_ref and alpha_pert are too close to compute finite-difference UAS.")

    l0 = spectra_grid[idx_ref]
    l1 = spectra_grid[idx_pert]
    ratio = np.maximum(l1, 1e-30) / np.maximum(l0, 1e-30)

    return -np.log(ratio) / (a1 - a0)


# Valid masks, robust thresholding, and iterative MF


# create a boolean mask of valid pixels in the cube based on finite values, optional nodata values, positivity requirement, and minimum valid fraction across bands
def make_valid_pixel_mask(
    cube: np.ndarray,
    nodata_values: Optional[Sequence[float]] = None,
    require_positive: bool = True,
    min_valid_fraction: float = 1.0,
) -> np.ndarray:
    finite = np.isfinite(cube)
    valid_band = finite.copy()

    if nodata_values is not None:
        for value in nodata_values:
            valid_band &= cube != value

    if require_positive:
        valid_band &= cube > 0

    valid_fraction = np.mean(valid_band, axis=2)
    return valid_fraction >= min_valid_fraction


# compute a robust threshold for alpha values using the median and MAD, with an option to specify the number of robust standard deviations (nsigma) for the threshold, and returning the threshold, median, and robust standard deviation
def robust_threshold_from_alpha(
    alpha_values: np.ndarray,
    nsigma: float = 3.0,
) -> Tuple[float, float, float]:
    alpha_values = np.asarray(alpha_values, dtype=float)
    alpha_values = alpha_values[np.isfinite(alpha_values)]

    if alpha_values.size == 0:
        raise ValueError("No finite alpha values are available for thresholding.")

    median = float(np.nanmedian(alpha_values))
    mad = float(np.nanmedian(np.abs(alpha_values - median)))
    robust_std = 1.4826 * mad

    if robust_std == 0.0:
        robust_std = float(np.nanstd(alpha_values))

    threshold = median + nsigma * robust_std
    return threshold, median, robust_std


# run the iterative Matched Filter algorithm on a hyperspectral cube, starting with an initial background mask (or using all valid pixels if none is provided), and iteratively estimating the background statistics, updating the target, applying the Matched Filter, and updating the plume and background masks for a specified number of iterations or until convergence, returning a dictionary containing the final alpha map, plume mask, background mask, and histories of these variables across iterations
def iterative_matched_filter_map_from_cube(
    cube: np.ndarray,
    uas: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    initial_background_mask: Optional[np.ndarray] = None,
    n_iter: int = 5,
    nsigma: float = 3.0,
    reg: float = 1e-6,
    rcond: float = 1e-8,
    min_background_pixels: Optional[int] = None,
    verbose: bool = True,
) -> dict[str, object]:
    height, width, n_bands = cube.shape
    uas = np.asarray(uas, dtype=float).reshape(-1)

    if uas.shape[0] != n_bands:
        raise ValueError(f"UAS length {uas.shape[0]} does not match cube bands {n_bands}.")

    if valid_mask is None:
        valid_mask = np.all(np.isfinite(cube), axis=2)
    valid_mask = valid_mask.astype(bool)

    if initial_background_mask is None:
        background_mask = valid_mask.copy()
    else:
        background_mask = initial_background_mask.astype(bool) & valid_mask

    if min_background_pixels is None:
        min_background_pixels = max(n_bands + 5, 30)

    alpha_history: list[np.ndarray] = []
    plume_mask_history: list[np.ndarray] = []
    background_mask_history: list[np.ndarray] = []
    threshold_history: list[float] = []
    mu_history: list[np.ndarray] = []
    cov_history: list[np.ndarray] = []

    previous_plume_mask: Optional[np.ndarray] = None

    for iteration in range(n_iter):
        n_background = int(np.sum(background_mask))
        if n_background < min_background_pixels:
            raise ValueError(
                f"Background pixels are too few at iteration {iteration + 1}: "
                f"{n_background} pixels, required >= {min_background_pixels}."
            )

        # Estimate background statistics from the current background mask.
        mu, cov, _ = estimate_background_mean_cov_from_cube(
            cube,
            valid_mask=background_mask,
            reg=reg,
            min_pixels=min_background_pixels,
        )

        # Update the methane target because the background mean has changed.
        target = make_methane_target(mu=mu, uas=uas, positive_alpha=True)
        inv_cov = np.linalg.pinv(cov, rcond=rcond)

        alpha_map = np.full((height, width), np.nan, dtype=float)
        x = cube[valid_mask]
        diff = x - mu

        numerator = diff @ inv_cov @ target
        denominator = target.T @ inv_cov @ target
        if np.abs(denominator) < 1e-12:
            raise ValueError("Denominator is too small. Check the target spectrum and covariance.")

        alpha_map[valid_mask] = numerator / denominator

        # Detect plume candidates using a robust median/MAD threshold.
        threshold, median, robust_std = robust_threshold_from_alpha(
            alpha_map[valid_mask],
            nsigma=nsigma,
        )
        plume_mask = np.zeros((height, width), dtype=bool)
        plume_mask[valid_mask] = alpha_map[valid_mask] > threshold

        # Exclude plume candidates from the next background estimation.
        new_background_mask = valid_mask & (~plume_mask)

        alpha_history.append(alpha_map.copy())
        plume_mask_history.append(plume_mask.copy())
        background_mask_history.append(background_mask.copy())
        threshold_history.append(float(threshold))
        mu_history.append(mu.copy())
        cov_history.append(cov.copy())

        if verbose:
            print(
                f"iter {iteration + 1:02d}: "
                f"background={n_background}, "
                f"threshold={threshold:.6g}, "
                f"median={median:.6g}, "
                f"robust_std={robust_std:.6g}, "
                f"plume_pixels={int(np.sum(plume_mask))}"
            )

        if previous_plume_mask is not None and np.array_equal(plume_mask, previous_plume_mask):
            if verbose:
                print(f"Converged at iter {iteration + 1}.")
            background_mask = new_background_mask
            break

        previous_plume_mask = plume_mask.copy()
        background_mask = new_background_mask

    return {
        "alpha_map": alpha_history[-1],
        "plume_mask": plume_mask_history[-1],
        "background_mask": background_mask,
        "alpha_history": alpha_history,
        "plume_mask_history": plume_mask_history,
        "background_mask_history": background_mask_history,
        "threshold_history": threshold_history,
        "mu_history": mu_history,
        "cov_history": cov_history,
    }


# plot the results of the iterative Matched Filter algorithm, including the final alpha map, plume candidate mask, background mask, and threshold history across iterations, with appropriate titles and colorbars
def plot_iterative_mf_result(result: dict[str, object]) -> None:
    alpha_map = result["alpha_map"]
    plume_mask = result["plume_mask"]
    background_mask = result["background_mask"]
    threshold_history = result["threshold_history"]

    if not isinstance(alpha_map, np.ndarray):
        raise TypeError("result['alpha_map'] must be a NumPy array.")
    if not isinstance(plume_mask, np.ndarray):
        raise TypeError("result['plume_mask'] must be a NumPy array.")
    if not isinstance(background_mask, np.ndarray):
        raise TypeError("result['background_mask'] must be a NumPy array.")

    plot_alpha_map(
        alpha_map,
        title="Iterative MF output",
        vmin=np.nanpercentile(alpha_map, 2),
        vmax=np.nanpercentile(alpha_map, 98),
    )

    plt.figure(figsize=(5, 5))
    plt.imshow(plume_mask)
    plt.title("Plume candidate mask from Iterative MF")
    plt.colorbar(label="Plume candidate")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(threshold_history, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Threshold")
    plt.title("Threshold history")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(5, 5))
    plt.imshow(background_mask)
    plt.title("Final background mask")
    plt.colorbar(label="Background")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
