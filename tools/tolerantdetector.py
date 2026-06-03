from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_CSV_PATH = Path(r"D:\research\code\all_roi_spectra200x200.csv")
DEFAULT_OUT_DIR = Path(r"D:\research\code\diagonal_spike_tolerant_output")

DEFAULT_WL_MIN = 2100.0
DEFAULT_WL_MAX = 2400.0
DEFAULT_PIXEL_Z_THRESHOLD = 4.0
DEFAULT_DIAG_HALF_WIDTH = 2
DEFAULT_MIN_CENTER_LINE_LENGTH = 80
DEFAULT_MIN_HIT_FRACTION = 0.20
DEFAULT_MIN_MEAN_Z = 1.0
DEFAULT_MANUAL_WLS = (2125.03,)


# find columns matching the pattern "wave_XXXnm", extract the wavelength values, and return sorted lists of column names and wavelengths
def parse_wavelengths(columns: Iterable[object]) -> tuple[list[str], np.ndarray]:
    wave_cols: list[str] = []
    wavelengths: list[float] = []
    pattern = re.compile(r"^wave_([0-9]+(?:\.[0-9]+)?)nm$")

    for col in columns:
        match = pattern.fullmatch(str(col))
        if match is not None:
            wave_cols.append(str(col))
            wavelengths.append(float(match.group(1)))

    if not wave_cols:
        raise ValueError("No wavelength columns found. Expected columns like wave_2125.03nm.")

    wavelengths_arr = np.asarray(wavelengths, dtype=float)
    order = np.argsort(wavelengths_arr)
    return [wave_cols[i] for i in order], wavelengths_arr[order]


def load_cube(csv_path: str | Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "y" not in df.columns or "x" not in df.columns:
        raise ValueError("The CSV must contain `y` and `x` columns.")

    wave_cols, wavelengths = parse_wavelengths(df.columns)
    ys = np.sort(df["y"].unique())
    xs = np.sort(df["x"].unique())
    y_to_i = {y: i for i, y in enumerate(ys)}
    x_to_j = {x: j for j, x in enumerate(xs)}

    cube = np.full((len(ys), len(xs), len(wave_cols)), np.nan, dtype=np.float32)
    spectra = df[wave_cols].to_numpy(dtype=np.float32)

    for row_idx, row in df.iterrows():
        cube[y_to_i[row["y"]], x_to_j[row["x"]], :] = spectra[row_idx, :]

    return cube, wavelengths, df, ys, xs


def compute_spectral_spike(cube: np.ndarray, wavelengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if cube.shape[2] < 3:
        raise ValueError("At least three wavelength bands are required to compute spectral spikes.")
    spike = cube[:, :, 1:-1] - 0.5 * (cube[:, :, :-2] + cube[:, :, 2:])
    spike_wavelengths = wavelengths[1:-1]
    return spike, spike_wavelengths


# compute a robust z-score for each band of the spike cube, using the median and MAD across all pixels, and return a z-score cube of the same shape
def robust_zscore_per_band(spike: np.ndarray) -> np.ndarray:
    median = np.nanmedian(spike, axis=(0, 1))
    mad = np.nanmedian(np.abs(spike - median[None, None, :]), axis=(0, 1))
    scale = 1.4826 * mad
    scale[~np.isfinite(scale) | (scale < 1e-12)] = np.nan

    zscore = (spike - median[None, None, :]) / scale[None, None, :]
    zscore = np.where(np.isfinite(zscore), zscore, 0.0)
    return zscore


# find the index of the wavelength in the array that is closest to the target wavelength
def nearest_band_index(target_wl: float, wavelengths: np.ndarray) -> int:
    wavelengths = np.asarray(wavelengths, dtype=float)
    return int(np.argmin(np.abs(wavelengths - target_wl)))


# detect diagonal lines in the spike z-score map that have a certain fraction of pixels above the threshold, and return a DataFrame of detected lines and the diag array for plotting
def detect_tolerant_diagonal_lines(
    spike_z: np.ndarray,
    spike_wavelengths: np.ndarray,
    pixel_z_threshold: float = DEFAULT_PIXEL_Z_THRESHOLD,
    diag_half_width: int = DEFAULT_DIAG_HALF_WIDTH,
    min_center_line_length: int = DEFAULT_MIN_CENTER_LINE_LENGTH,
    min_hit_fraction: float = DEFAULT_MIN_HIT_FRACTION,
    min_mean_z: float = DEFAULT_MIN_MEAN_Z,
) -> tuple[pd.DataFrame, np.ndarray]:
    height, width, _ = spike_z.shape
    yy, xx = np.indices((height, width))
    diag = yy - xx

    rows: list[dict[str, object]] = []
    for d in range(-(width - 1), height):
        center_len = int((diag == d).sum())
        if center_len < min_center_line_length:
            continue

        band_mask = np.abs(diag - d) <= diag_half_width
        values = spike_z[band_mask, :]
        hit_fraction = (values > pixel_z_threshold).mean(axis=0)
        mean_z = values.mean(axis=0)
        median_z = np.median(values, axis=0)
        p90_z = np.percentile(values, 90, axis=0)
        max_z = values.max(axis=0)

        for i, wl in enumerate(spike_wavelengths):
            if hit_fraction[i] >= min_hit_fraction and mean_z[i] >= min_mean_z:
                rows.append({
                    "diag_center_d_y_minus_x": int(d),
                    "wavelength_nm": float(wl),
                    "center_line_length_px": center_len,
                    "diagonal_band_half_width_px": int(diag_half_width),
                    "diagonal_band_n_px": int(values.shape[0]),
                    "hit_fraction": float(hit_fraction[i]),
                    "mean_z": float(mean_z[i]),
                    "median_z": float(median_z[i]),
                    "p90_z": float(p90_z[i]),
                    "max_z": float(max_z[i]),
                })

    detected = pd.DataFrame(rows)
    if len(detected) > 0:
        detected = detected.sort_values(["hit_fraction", "mean_z"], ascending=False).reset_index(drop=True)

    return detected, diag


# build a table of individual pixels that are above the z-score threshold and belong to the detected diagonal lines, including their coordinates, diagonal values, wavelength, and spike z-score, and return as a DataFrame
def build_detected_pixel_table(
    detected: pd.DataFrame,
    spike_z: np.ndarray,
    spike_wavelengths: np.ndarray,
    diag: np.ndarray,
    pixel_z_threshold: float = DEFAULT_PIXEL_Z_THRESHOLD,
) -> pd.DataFrame:
    pixel_rows: list[dict[str, object]] = []

    for _, line in detected.iterrows():
        d = int(line["diag_center_d_y_minus_x"])
        half = int(line["diagonal_band_half_width_px"])
        wl = float(line["wavelength_nm"])
        wi = nearest_band_index(wl, spike_wavelengths)
        mask = (np.abs(diag - d) <= half) & (spike_z[:, :, wi] > pixel_z_threshold)
        ys, xs = np.where(mask)

        for y, x in zip(ys, xs):
            pixel_rows.append({
                "y": int(y),
                "x": int(x),
                "diag_d_y_minus_x": int(y - x),
                "diag_center_d_y_minus_x": d,
                "wavelength_nm": wl,
                "spike_z": float(spike_z[y, x, wi]),
            })

    pixels = pd.DataFrame(pixel_rows)
    if len(pixels) > 0:
        pixels = pixels.sort_values(
            ["wavelength_nm", "diag_center_d_y_minus_x", "y", "x"]
        ).reset_index(drop=True)

    return pixels


# plot the spike z-score map at a specific wavelength with an overlay of the detected diagonal lines, and save the figure as a PNG
def plot_spike_z_overlay(
    spike_z: np.ndarray,
    spike_wavelengths: np.ndarray,
    diag_values: Sequence[int],
    out_dir: str | Path,
    target_wl: float | None = None,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if target_wl is None:
        target_wl = float(spike_wavelengths[spike_z.shape[2] // 2])

    wi = nearest_band_index(target_wl, spike_wavelengths)
    zmap = spike_z[:, :, wi]
    height, width = zmap.shape

    fig, ax = plt.subplots(figsize=(7, 6))
    vmax = np.nanpercentile(zmap, 99.5)
    image = ax.imshow(zmap, origin="upper", vmin=0, vmax=vmax)
    fig.colorbar(image, ax=ax, label="spectral spike robust z-score")
    ax.set_title(f"Spike z-map at {spike_wavelengths[wi]:.2f} nm")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    x_values = np.arange(width)
    for d in diag_values:
        y_values = x_values + int(d)
        ok = (y_values >= 0) & (y_values < height)
        if np.any(ok):
            ax.plot(x_values[ok], y_values[ok], linewidth=1, label=f"y-x={int(d)}")

    if len(diag_values) > 0:
        ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    out_png = out_dir / f"spike_z_map_{spike_wavelengths[wi]:.2f}nm_diag_overlay.png"
    fig.savefig(out_png, dpi=200)
    plt.show()
    print(f"Saved: {out_png}")
    return out_png


# plot images at specific target wavelengths, using a robust percentile-based scaling for visualization and showing the actual wavelength in the title
def plot_wavelength_images(
    cube: np.ndarray,
    wavelengths: np.ndarray,
    display_wls: Iterable[float],
    out_png: str | Path,
) -> None:
    display_wls = list(display_wls)
    if not display_wls:
        print("No wavelengths to plot.")
        return

    height, width, _ = cube.shape
    ncols = 2
    nrows = math.ceil(len(display_wls) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 5 * nrows))
    axes = np.asarray(axes).reshape(-1)

    for ax, wl in zip(axes, display_wls):
        band_idx = nearest_band_index(float(wl), wavelengths)
        image = cube[:, :, band_idx]
        vmin, vmax = np.nanpercentile(image, [2, 98])
        im = ax.imshow(image, cmap="gray", origin="upper", vmin=vmin, vmax=vmax)
        ax.set_title(f"{wavelengths[band_idx]:.2f} nm")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(-0.5, width - 0.5)
        ax.set_ylim(height - 0.5, -0.5)
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes[len(display_wls):]:
        ax.axis("off")

    fig.tight_layout()
    out_png = Path(out_png)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out_png}")


# run the full tolerant diagonal spike detection pipeline, including loading the cube, computing the spike and z-score, detecting diagonal lines, building the pixel table, and plotting results, and return a dictionary of outputs
def run_tolerant_detector(
    csv_path: str | Path = DEFAULT_CSV_PATH,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    wl_min: float = DEFAULT_WL_MIN,
    wl_max: float = DEFAULT_WL_MAX,
    pixel_z_threshold: float = DEFAULT_PIXEL_Z_THRESHOLD,
    diag_half_width: int = DEFAULT_DIAG_HALF_WIDTH,
    min_center_line_length: int = DEFAULT_MIN_CENTER_LINE_LENGTH,
    min_hit_fraction: float = DEFAULT_MIN_HIT_FRACTION,
    min_mean_z: float = DEFAULT_MIN_MEAN_Z,
    manual_wls: Iterable[float] = DEFAULT_MANUAL_WLS,
) -> dict[str, object]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cube, wavelengths, df, ys, xs = load_cube(csv_path)
    spike, spike_wavelengths = compute_spectral_spike(cube, wavelengths)

    wl_mask = (spike_wavelengths >= wl_min) & (spike_wavelengths <= wl_max)
    if not np.any(wl_mask):
        raise ValueError(f"No spike bands selected in {wl_min} to {wl_max} nm.")

    spike_band = spike[:, :, wl_mask]
    spike_wavelengths_band = spike_wavelengths[wl_mask]
    spike_z = robust_zscore_per_band(spike_band)

    print("Cube shape:", cube.shape)
    print("Spike wavelength range:", spike_wavelengths_band[0], spike_wavelengths_band[-1])
    print("Spike bands:", len(spike_wavelengths_band))

    detected, diag = detect_tolerant_diagonal_lines(
        spike_z=spike_z,
        spike_wavelengths=spike_wavelengths_band,
        pixel_z_threshold=pixel_z_threshold,
        diag_half_width=diag_half_width,
        min_center_line_length=min_center_line_length,
        min_hit_fraction=min_hit_fraction,
        min_mean_z=min_mean_z,
    )

    detected_path = out_dir / "tolerant_detected_diagonal_lines.csv"
    detected.to_csv(detected_path, index=False)
    print(f"Saved: {detected_path}")

    pixels = build_detected_pixel_table(
        detected=detected,
        spike_z=spike_z,
        spike_wavelengths=spike_wavelengths_band,
        diag=diag,
        pixel_z_threshold=pixel_z_threshold,
    )
    pixels_path = out_dir / "tolerant_detected_anomaly_pixels.csv"
    pixels.to_csv(pixels_path, index=False)
    print(f"Saved: {pixels_path}")

    manual_wls = list(manual_wls)
    detected_wls = [] if len(detected) == 0 else list(np.sort(detected["wavelength_nm"].unique()))
    display_wls = np.array(sorted(set(np.round(detected_wls + manual_wls, 2))))
    print("Display wavelengths:", display_wls)

    if len(detected) > 0:
        top_wl = float(detected.iloc[0]["wavelength_nm"])
        top_diag = int(detected.iloc[0]["diag_center_d_y_minus_x"])
        diag_values = list(range(top_diag - diag_half_width, top_diag + diag_half_width + 1))
    else:
        top_wl = float(manual_wls[0]) if manual_wls else float(spike_wavelengths_band[len(spike_wavelengths_band) // 2])
        diag_values = []

    plot_spike_z_overlay(
        spike_z=spike_z,
        spike_wavelengths=spike_wavelengths_band,
        diag_values=diag_values,
        out_dir=out_dir,
        target_wl=top_wl,
    )

    plot_wavelength_images(
        cube=cube,
        wavelengths=wavelengths,
        display_wls=display_wls,
        out_png=out_dir / "anomalous_wavelength_images_with_manual_2125_03nm.png",
    )

    return {
        "df": df,
        "cube": cube,
        "wavelengths": wavelengths,
        "ys": ys,
        "xs": xs,
        "spike_z": spike_z,
        "spike_wavelengths": spike_wavelengths_band,
        "detected": detected,
        "pixels": pixels,
        "diag": diag,
        "out_dir": out_dir,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run tolerant diagonal spike detection.")
    parser.add_argument("--csv", default=str(DEFAULT_CSV_PATH), help="Input ROI spectra CSV path.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory.")
    parser.add_argument("--wl-min", type=float, default=DEFAULT_WL_MIN, help="Minimum wavelength [nm].")
    parser.add_argument("--wl-max", type=float, default=DEFAULT_WL_MAX, help="Maximum wavelength [nm].")
    parser.add_argument("--pixel-z-threshold", type=float, default=DEFAULT_PIXEL_Z_THRESHOLD)
    parser.add_argument("--diag-half-width", type=int, default=DEFAULT_DIAG_HALF_WIDTH)
    parser.add_argument("--min-center-line-length", type=int, default=DEFAULT_MIN_CENTER_LINE_LENGTH)
    parser.add_argument("--min-hit-fraction", type=float, default=DEFAULT_MIN_HIT_FRACTION)
    parser.add_argument("--min-mean-z", type=float, default=DEFAULT_MIN_MEAN_Z)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_tolerant_detector(
        csv_path=args.csv,
        out_dir=args.out_dir,
        wl_min=args.wl_min,
        wl_max=args.wl_max,
        pixel_z_threshold=args.pixel_z_threshold,
        diag_half_width=args.diag_half_width,
        min_center_line_length=args.min_center_line_length,
        min_hit_fraction=args.min_hit_fraction,
        min_mean_z=args.min_mean_z,
    )


if __name__ == "__main__":
    main()
