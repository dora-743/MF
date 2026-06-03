from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


DEFAULT_CSV_PATH = Path(r"D:\research\code\all_roi_spectra200x200.csv")
DEFAULT_OUT_DIR = Path(r"D:\research\code\diagonal_spike_tolerant_output")

DEFAULT_WL_MIN = 2100.0
DEFAULT_WL_MAX = 2400.0
DEFAULT_N_SHOW_PC = 8
DEFAULT_N_PC_TOTAL = 12
DEFAULT_TARGET_WLS = (2125.03, 2174.99, 2387.32)


# regex pattern for wavelength columns: wave_2125.03nm, wave_2174.99nm, etc.
def get_wave_columns(df: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    wave_cols: list[str] = []
    wavelengths: list[float] = []
    pattern = re.compile(r"^wave_([0-9]+(?:\.[0-9]+)?)nm$")

    for col in df.columns:
        match = pattern.match(str(col))
        if match is not None:
            wave_cols.append(str(col))
            wavelengths.append(float(match.group(1)))

    if not wave_cols:
        raise ValueError("No wavelength columns found. Expected columns like wave_2125.03nm.")

    wavelengths_arr = np.asarray(wavelengths, dtype=float)
    order = np.argsort(wavelengths_arr)
    return [wave_cols[i] for i in order], wavelengths_arr[order]


# convert from long-form DataFrame with (y, x, wave_XXXnm) columns to a 3D cube (y, x, wavelength) and coordinate arrays
def spectra_to_cube(
    df: pd.DataFrame,
    wave_cols: list[str],
    fill_value: float = np.nan,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if "y" not in df.columns or "x" not in df.columns:
        raise ValueError("The input DataFrame must contain `y` and `x` columns.")

    ys = np.sort(df["y"].unique())
    xs = np.sort(df["x"].unique())
    y_to_i = {y: i for i, y in enumerate(ys)}
    x_to_j = {x: j for j, x in enumerate(xs)}

    cube = np.full((len(ys), len(xs), len(wave_cols)), fill_value, dtype=float)
    spectra = df[wave_cols].to_numpy(dtype=float)

    for row_idx, row in df.iterrows():
        i = y_to_i[row["y"]]
        j = x_to_j[row["x"]]
        cube[i, j, :] = spectra[row_idx, :]

    return cube, ys, xs


# load the cube and wavelength info from a CSV, using the above helper functions
def load_cube_from_csv(csv_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    wave_cols, wavelengths = get_wave_columns(df)
    cube, ys, xs = spectra_to_cube(df, wave_cols)
    return cube, wavelengths, ys, xs, df


# compute a robust z-score for a 1D array, using the median and MAD for scaling
def robust_zscore_1d(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    median = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - median))
    scale = mad / 0.6745

    if not np.isfinite(scale) or scale < 1e-12:
        scale = np.nanstd(values)
    if not np.isfinite(scale) or scale < 1e-12:
        return np.zeros_like(values, dtype=float)

    return (values - median) / scale


# return the index of the band whose wavelength is closest to the target wavelength
def nearest_band_index(target_wl: float, wavelengths: np.ndarray) -> int:
    wavelengths = np.asarray(wavelengths, dtype=float)
    return int(np.argmin(np.abs(wavelengths - target_wl)))


# select a subset of the cube and wavelengths based on a specified wavelength range
def select_wavelength_range(
    cube: np.ndarray,
    wavelengths: np.ndarray,
    wl_min: float,
    wl_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = (wavelengths >= wl_min) & (wavelengths <= wl_max)
    if not np.any(mask):
        raise ValueError(f"No bands selected in {wl_min} to {wl_max} nm.")
    return cube[:, :, mask], wavelengths[mask], mask


# flatten valid pixels and standardize each band before PCA, returning the standardized data, valid pixel mask, and mean/std for each band
def standardize_valid_pixels(cube: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width, n_bands = cube.shape
    raw = cube.reshape(height * width, n_bands)
    valid_pixel_mask = np.all(np.isfinite(raw), axis=1)
    valid = raw[valid_pixel_mask]

    if valid.shape[0] == 0:
        raise ValueError("No valid pixels found for PCA.")

    mean = np.nanmean(valid, axis=0)
    std = np.nanstd(valid, axis=0, ddof=1)
    std[std < 1e-12] = 1.0
    standardized = (valid - mean) / std

    return standardized, valid_pixel_mask, mean, std


# plot the PCA score maps for the first n_show PCs, using a robust z-score for color scaling and showing the explained variance ratio in the title
def plot_pca_score_maps(
    scores: np.ndarray,
    explained_variance_ratio: np.ndarray,
    valid_pixel_mask: np.ndarray,
    image_shape: tuple[int, int],
    out_png: str | Path,
    n_show: int = DEFAULT_N_SHOW_PC,
) -> None:
    height, width = image_shape
    n_show = min(n_show, scores.shape[1])
    ncols = 4
    nrows = math.ceil(n_show / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.asarray(axes).reshape(-1)

    for i in range(n_show):
        score_full = np.full(height * width, np.nan, dtype=float)
        score_full[valid_pixel_mask] = scores[:, i]
        score_img = score_full.reshape(height, width)
        score_z = robust_zscore_1d(score_img.ravel()).reshape(height, width)

        finite = score_z[np.isfinite(score_z)]
        vmax = np.nanpercentile(np.abs(finite), 99) if finite.size else 1.0
        vmin = -vmax

        ax = axes[i]
        image = ax.imshow(score_z, origin="upper", cmap="RdBu_r", vmin=vmin, vmax=vmax)
        ax.set_title(f"PC{i + 1} score map\nEVR={explained_variance_ratio[i]:.4f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    for j in range(n_show, len(axes)):
        axes[j].axis("off")

    fig.suptitle("PCA score maps", fontsize=16)
    fig.tight_layout()
    out_png = Path(out_png)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out_png}")


# plot images at specific target wavelengths, using a robust percentile-based scaling for visualization and showing the actual wavelength in the title
def plot_target_wavelength_images(
    cube: np.ndarray,
    wavelengths: np.ndarray,
    target_wls: Iterable[float],
    out_png: str | Path,
) -> None:
    target_wls = list(target_wls)
    if not target_wls:
        return

    ncols = min(3, len(target_wls))
    nrows = math.ceil(len(target_wls) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = np.asarray(axes).reshape(-1)

    for ax, wl in zip(axes, target_wls):
        idx = nearest_band_index(float(wl), wavelengths)
        image = cube[:, :, idx]
        vmin, vmax = np.nanpercentile(image, [2, 98])
        im = ax.imshow(image, origin="upper", cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(f"{wavelengths[idx]:.2f} nm")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes[len(target_wls):]:
        ax.axis("off")

    fig.tight_layout()
    out_png = Path(out_png)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out_png}")


# run the full PCA analysis pipeline, including loading the cube, selecting wavelength range, standardizing, running PCA, and plotting results
def run_pca_analysis(
    csv_path: str | Path = DEFAULT_CSV_PATH,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    wl_min: float = DEFAULT_WL_MIN,
    wl_max: float = DEFAULT_WL_MAX,
    n_components: int = DEFAULT_N_PC_TOTAL,
    n_show_pc: int = DEFAULT_N_SHOW_PC,
    target_wls: Iterable[float] = DEFAULT_TARGET_WLS,
) -> dict[str, object]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cube_all, wavelengths_all, ys, xs, df = load_cube_from_csv(csv_path)
    cube, wavelengths, _ = select_wavelength_range(cube_all, wavelengths_all, wl_min, wl_max)
    standardized, valid_pixel_mask, mean, std = standardize_valid_pixels(cube)

    n_components = min(n_components, standardized.shape[0], standardized.shape[1])
    if n_components < 1:
        raise ValueError("n_components must be at least 1 after checking data dimensions.")

    print("PCA input shape:", standardized.shape)
    print("Wavelength range:", wavelengths[0], "to", wavelengths[-1], "nm")
    print("Valid pixels:", int(valid_pixel_mask.sum()))

    pca = PCA(n_components=n_components, random_state=0)
    scores = pca.fit_transform(standardized)

    explained = pd.DataFrame({
        "PC": np.arange(1, n_components + 1),
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative": np.cumsum(pca.explained_variance_ratio_),
    })
    print(explained)

    explained_path = out_dir / "pca_explained_variance.csv"
    explained.to_csv(explained_path, index=False)
    print(f"Saved: {explained_path}")

    plot_pca_score_maps(
        scores=scores,
        explained_variance_ratio=pca.explained_variance_ratio_,
        valid_pixel_mask=valid_pixel_mask,
        image_shape=cube.shape[:2],
        out_png=out_dir / "pca_score_maps_PC1_PC8.png",
        n_show=n_show_pc,
    )

    plot_target_wavelength_images(
        cube=cube_all,
        wavelengths=wavelengths_all,
        target_wls=target_wls,
        out_png=out_dir / "target_wavelength_images.png",
    )

    return {
        "df": df,
        "cube": cube,
        "wavelengths": wavelengths,
        "ys": ys,
        "xs": xs,
        "valid_pixel_mask": valid_pixel_mask,
        "mean": mean,
        "std": std,
        "pca": pca,
        "scores": scores,
        "explained": explained,
        "out_dir": out_dir,
    }


# create an argument parser for running the PCA analysis from the command line, with options for input CSV, output directory, wavelength range, number of PCA components, and number of PCs to show in the score maps
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PCA diagnostics for an ROI spectra CSV.")
    parser.add_argument("--csv", default=str(DEFAULT_CSV_PATH), help="Input ROI spectra CSV path.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory.")
    parser.add_argument("--wl-min", type=float, default=DEFAULT_WL_MIN, help="Minimum wavelength [nm].")
    parser.add_argument("--wl-max", type=float, default=DEFAULT_WL_MAX, help="Maximum wavelength [nm].")
    parser.add_argument("--n-components", type=int, default=DEFAULT_N_PC_TOTAL, help="Number of PCA components.")
    parser.add_argument("--n-show-pc", type=int, default=DEFAULT_N_SHOW_PC, help="Number of PC maps to plot.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_pca_analysis(
        csv_path=args.csv,
        out_dir=args.out_dir,
        wl_min=args.wl_min,
        wl_max=args.wl_max,
        n_components=args.n_components,
        n_show_pc=args.n_show_pc,
    )


if __name__ == "__main__":
    main()
