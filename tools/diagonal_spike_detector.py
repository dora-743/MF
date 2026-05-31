from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    from IPython.display import display
except Exception:
    display = print


# ============================================================
# CONFIG
# ============================================================

CSV_PATH = r"D:\research\code\all_roi_spectra200x200.csv"
OUT_DIR = r"D:\research\code\diagonal_spike_check"

# User-described target range
WL_MIN = 2100.0
WL_MAX = 2400.0
EXCLUDE_RANGES: Optional[Sequence[Tuple[float, float]]] = None

# Adjacent-band spike definition:
# center band is compared with center-neighbor_offset and center+neighbor_offset.
NEIGHBOR_OFFSET = 1

# Pixel-level spike threshold.
# Raise this if many false positive pixels appear.
PIXEL_Z_THRESHOLD = 4.0

# Line-level criteria.
# Because the artifact is known to form y=x-parallel lines, we aggregate spike pixels
# by diagonal id = row - col.
MIN_LINE_PIXELS = 20       # ignore very short diagonals near image corners
MIN_HIT_PIXELS = 6         # at least this many spike pixels on one diagonal/wavelength
MIN_HIT_FRACTION = 0.10    # fraction of valid pixels on the diagonal that are spike pixels
LINE_PERCENTILE = 80.0     # percentile of spike_z along a diagonal
LINE_SCORE_THRESHOLD = 2.5 # required percentile score

# If True, keep only the strongest wavelength candidate within +/- BAND_SEPARATION bands
# for each diagonal. Useful when the same one-band artifact leaks into neighboring bands.
COLLAPSE_ADJACENT_BANDS = True
BAND_SEPARATION = 2

# Diagonal definition:
# "row_col": diagonal id = row - col. Recommended for regular image grids.
# "coord":   diagonal id = y - x. Use this if raw CSV coordinates are guaranteed regular.
DIAG_MODE = "row_col"

REQUIRE_POSITIVE = True


# Loading and cube construction

def get_wave_columns(df: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    """Find columns like wave_2300.00nm and return them sorted by wavelength."""
    wave_cols: list[str] = []
    wavelengths: list[float] = []
    pattern = re.compile(r"^wave_([0-9]+(?:\.[0-9]+)?)nm$")

    for col in df.columns:
        m = pattern.match(str(col))
        if m is not None:
            wave_cols.append(col)
            wavelengths.append(float(m.group(1)))

    if len(wave_cols) == 0:
        raise ValueError("No wavelength columns found. Expected columns like wave_2300.00nm.")

    wavelengths_arr = np.asarray(wavelengths, dtype=float)
    order = np.argsort(wavelengths_arr)
    wavelengths_arr = wavelengths_arr[order]
    wave_cols = [wave_cols[i] for i in order]
    return wave_cols, wavelengths_arr


def load_roi_spectra_csv(path: str | Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)

    if "y" not in df.columns or "x" not in df.columns:
        raise ValueError("The CSV must contain 'y' and 'x' columns.")

    wave_cols, wavelengths = get_wave_columns(df)
    spectra = df[wave_cols].to_numpy(dtype=float)
    return df, wavelengths, spectra, wave_cols


def spectra_to_cube(
    df: pd.DataFrame,
    spectra: np.ndarray,
    fill_value: float = np.nan,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ys = np.sort(df["y"].unique())
    xs = np.sort(df["x"].unique())

    y_to_row = {y: i for i, y in enumerate(ys)}
    x_to_col = {x: j for j, x in enumerate(xs)}

    H, W = len(ys), len(xs)
    B = spectra.shape[1]
    cube = np.full((H, W, B), fill_value, dtype=float)

    # Keep this simple and safe. 200x200 is fine with iterrows.
    for row_idx, row in df.iterrows():
        r = y_to_row[row["y"]]
        c = x_to_col[row["x"]]
        cube[r, c, :] = spectra[row_idx, :]

    return cube, ys, xs


def band_mask(
    wavelengths: np.ndarray,
    wl_min: Optional[float] = None,
    wl_max: Optional[float] = None,
    exclude_ranges: Optional[Sequence[Tuple[float, float]]] = None,
) -> np.ndarray:
    """Create wavelength selection mask."""
    mask = np.ones_like(wavelengths, dtype=bool)
    if wl_min is not None:
        mask &= wavelengths >= wl_min
    if wl_max is not None:
        mask &= wavelengths <= wl_max
    if exclude_ranges is not None:
        for a, b in exclude_ranges:
            mask &= ~((wavelengths >= a) & (wavelengths <= b))
    return mask


def make_valid_mask(cube: np.ndarray, require_positive: bool = True) -> np.ndarray:
    """Valid if all selected-band values are finite and optionally positive."""
    valid = np.all(np.isfinite(cube), axis=2)
    if require_positive:
        valid &= np.all(cube > 0, axis=2)
    return valid


# Spectral one-band spike score

def robust_z_by_band(
    values: np.ndarray,
    valid3: np.ndarray,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Robust z-score for values[H, W, B] independently for each band.
    z = (value - median_band) / (1.4826 * MAD_band)
    """
    masked = np.where(valid3, values, np.nan)
    med = np.nanmedian(masked, axis=(0, 1))
    mad = np.nanmedian(np.abs(masked - med[None, None, :]), axis=(0, 1))
    scale = 1.4826 * mad + eps
    z = (values - med[None, None, :]) / scale[None, None, :]
    z[~valid3] = np.nan
    return z, med, scale


def compute_spectral_spike_cube(
    cube: np.ndarray,
    wavelengths: np.ndarray,
    valid_mask: np.ndarray,
    neighbor_offset: int = 1,
) -> dict:
    if neighbor_offset < 1:
        raise ValueError("neighbor_offset must be >= 1.")

    H, W, B = cube.shape
    k = neighbor_offset
    if B <= 2 * k:
        raise ValueError("Not enough bands for the requested neighbor_offset.")

    left = cube[:, :, 0:B - 2 * k]
    center = cube[:, :, k:B - k]
    right = cube[:, :, 2 * k:B]

    center_wavelengths = wavelengths[k:B - k]
    center_band_indices = np.arange(k, B - k)

    local_baseline = 0.5 * (left + right)
    spike_height = center - local_baseline
    spike_ratio = center / np.maximum(local_baseline, 1e-12)
    is_local_peak = (center > left) & (center > right)

    valid3 = (
        valid_mask[:, :, None]
        & np.isfinite(left)
        & np.isfinite(center)
        & np.isfinite(right)
        & np.isfinite(spike_height)
    )

    spike_z, spike_med, spike_scale = robust_z_by_band(spike_height, valid3)

    return {
        "left": left,
        "center": center,
        "right": right,
        "local_baseline": local_baseline,
        "spike_height": spike_height,
        "spike_ratio": spike_ratio,
        "spike_z": spike_z,
        "is_local_peak": is_local_peak,
        "valid3": valid3,
        "center_wavelengths": center_wavelengths,
        "center_band_indices": center_band_indices,
        "spike_median_by_band": spike_med,
        "spike_scale_by_band": spike_scale,
    }


# Diagonal line aggregation

def make_diagonal_id_map(
    H: int,
    W: int,
    ys: np.ndarray,
    xs: np.ndarray,
    mode: str = "row_col",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create diagonal ids for y=x-parallel lines.

    row_col mode:
        diag_id = row - col
        robust if the image grid is regular.

    coord mode:
        diag_id = y - x
        matches raw coordinate equation directly if y and x have the same spacing.
    """
    rr, cc = np.indices((H, W))

    if mode == "row_col":
        diag_id = rr - cc
    elif mode == "coord":
        diag_id = ys[rr] - xs[cc]
    else:
        raise ValueError("mode must be 'row_col' or 'coord'.")

    y_minus_x = ys[rr] - xs[cc]
    return diag_id, y_minus_x


def collapse_adjacent_band_candidates(
    df_lines: pd.DataFrame,
    band_separation: int = 2,
) -> pd.DataFrame:
    """
    For each diagonal, keep only the strongest candidates that are separated
    by more than band_separation in center-band index.
    """
    if len(df_lines) == 0:
        return df_lines

    kept = []
    sort_cols = ["diag_id", "line_score", "hit_fraction", "n_hit"]
    df_sorted = df_lines.sort_values(sort_cols, ascending=[True, False, False, False])

    for diag_id, g in df_sorted.groupby("diag_id", sort=False):
        selected_band_indices = []
        for _, row in g.iterrows():
            b = int(row["center_band_index"])
            if all(abs(b - sb) > band_separation for sb in selected_band_indices):
                kept.append(row)
                selected_band_indices.append(b)

    out = pd.DataFrame(kept)
    out = out.sort_values(["line_score", "n_hit"], ascending=False).reset_index(drop=True)
    return out


def detect_diagonal_spike_lines(
    spike: dict,
    valid_mask: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    diag_mode: str = "row_col",
    pixel_z_threshold: float = 5.0,
    min_line_pixels: int = 20,
    min_hit_pixels: int = 6,
    min_hit_fraction: float = 0.20,
    line_percentile: float = 80.0,
    line_score_threshold: float = 3.0,
    collapse_adjacent_bands: bool = True,
    band_separation: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    spike_z = spike["spike_z"]
    spike_height = spike["spike_height"]
    is_local_peak = spike["is_local_peak"]
    center_wavelengths = spike["center_wavelengths"]
    center_band_indices = spike["center_band_indices"]
    center = spike["center"]
    left = spike["left"]
    right = spike["right"]
    baseline = spike["local_baseline"]

    H, W, Bc = spike_z.shape
    diag_id_map, y_minus_x_map = make_diagonal_id_map(H, W, ys, xs, mode=diag_mode)

    line_rows = []
    unique_diags = np.unique(diag_id_map[valid_mask])

    for diag_id in unique_diags:
        diag_mask = (diag_id_map == diag_id) & valid_mask
        n_valid = int(np.sum(diag_mask))
        if n_valid < min_line_pixels:
            continue

        for kb in range(Bc):
            z_vals = spike_z[:, :, kb][diag_mask]
            peak_vals = is_local_peak[:, :, kb][diag_mask]
            finite = np.isfinite(z_vals)
            if np.sum(finite) < min_line_pixels:
                continue

            hit_vals = finite & peak_vals & (z_vals >= pixel_z_threshold)
            n_hit = int(np.sum(hit_vals))
            hit_fraction = n_hit / max(1, int(np.sum(finite)))

            if n_hit == 0:
                max_z = float(np.nanmax(z_vals[finite])) if np.any(finite) else np.nan
            else:
                max_z = float(np.nanmax(z_vals[hit_vals]))

            p_score = float(np.nanpercentile(z_vals[finite], line_percentile))
            median_z = float(np.nanmedian(z_vals[finite]))

            if (
                n_hit >= min_hit_pixels
                and hit_fraction >= min_hit_fraction
                and p_score >= line_score_threshold
            ):
                line_rows.append({
                    "diag_id": diag_id,
                    "diag_mode": diag_mode,
                    "center_band_index": int(center_band_indices[kb]),
                    "center_wavelength_nm": float(center_wavelengths[kb]),
                    "n_valid_on_line": int(np.sum(finite)),
                    "n_hit": n_hit,
                    "hit_fraction": float(hit_fraction),
                    "line_score": p_score,
                    "line_percentile": line_percentile,
                    "median_z_on_line": median_z,
                    "max_z_on_line": max_z,
                })

    df_lines = pd.DataFrame(line_rows)

    if len(df_lines) > 0:
        df_lines = df_lines.sort_values(
            ["line_score", "hit_fraction", "n_hit"],
            ascending=False
        ).reset_index(drop=True)

        if collapse_adjacent_bands:
            df_lines = collapse_adjacent_band_candidates(df_lines, band_separation=band_separation)

    # Pixel table and maps for accepted line/wavelength pairs
    rr_grid, cc_grid = np.indices((H, W))
    detected_mask = np.zeros((H, W), dtype=bool)
    best_z_map = np.full((H, W), np.nan, dtype=float)
    wavelength_map = np.full((H, W), np.nan, dtype=float)
    diag_map_out = np.full((H, W), np.nan, dtype=float)

    pixel_rows = []

    for line_idx, line in df_lines.iterrows():
        diag_id = line["diag_id"]
        center_band_index = int(line["center_band_index"])
        # Convert original selected-band index to spike-cube index.
        kb_matches = np.where(center_band_indices == center_band_index)[0]
        if len(kb_matches) == 0:
            continue
        kb = int(kb_matches[0])

        hit_mask = (
            (diag_id_map == diag_id)
            & valid_mask
            & is_local_peak[:, :, kb]
            & np.isfinite(spike_z[:, :, kb])
            & (spike_z[:, :, kb] >= pixel_z_threshold)
        )

        rows, cols = np.where(hit_mask)

        for r, c in zip(rows, cols):
            z_val = float(spike_z[r, c, kb])
            detected_mask[r, c] = True

            if not np.isfinite(best_z_map[r, c]) or z_val > best_z_map[r, c]:
                best_z_map[r, c] = z_val
                wavelength_map[r, c] = float(center_wavelengths[kb])
                diag_map_out[r, c] = float(diag_id)

            pixel_rows.append({
                "line_rank": int(line_idx),
                "row": int(r),
                "col": int(c),
                "y": ys[r],
                "x": xs[c],
                "diag_id": diag_id,
                "y_minus_x": y_minus_x_map[r, c],
                "center_band_index": int(center_band_indices[kb]),
                "center_wavelength_nm": float(center_wavelengths[kb]),
                "spike_z": z_val,
                "spike_height": float(spike_height[r, c, kb]),
                "left_value": float(left[r, c, kb]),
                "center_value": float(center[r, c, kb]),
                "right_value": float(right[r, c, kb]),
                "local_baseline": float(baseline[r, c, kb]),
            })

    df_pixels = pd.DataFrame(pixel_rows)
    if len(df_pixels) > 0:
        df_pixels = df_pixels.sort_values(
            ["line_rank", "diag_id", "center_wavelength_nm", "row", "col"]
        ).reset_index(drop=True)

    max_spike_z_map = np.nanmax(spike_z, axis=2)

    maps = {
        "detected_mask": detected_mask,
        "best_z_map": best_z_map,
        "wavelength_map": wavelength_map,
        "diag_id_map": diag_id_map,
        "detected_diag_map": diag_map_out,
        "max_spike_z_map": max_spike_z_map,
        "y_minus_x_map": y_minus_x_map,
    }

    return df_lines, df_pixels, maps


# Plotting / exporting

def save_map_csv(arr: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    pd.DataFrame(arr).to_csv(path, index=False, header=False)
    print(f"Saved: {path}")


def plot_map(
    arr: np.ndarray,
    title: str,
    out_png: str | Path,
    colorbar_label: str = "value",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    plt.figure(figsize=(7, 6))
    im = plt.imshow(arr, origin="upper", vmin=vmin, vmax=vmax)
    plt.colorbar(im, label=colorbar_label)
    plt.xlabel("x / col")
    plt.ylabel("y / row")
    plt.title(title)
    plt.tight_layout()
    out_png = Path(out_png)
    plt.savefig(out_png, dpi=200)
    plt.show()
    print(f"Saved: {out_png}")


def plot_detected_overlay(
    base_map: np.ndarray,
    detected_mask: np.ndarray,
    out_png: str | Path,
    title: str = "Detected diagonal spike pixels",
) -> None:
    finite = base_map[np.isfinite(base_map)]
    vmin = np.nanpercentile(finite, 2) if finite.size else None
    vmax = np.nanpercentile(finite, 98) if finite.size else None

    plt.figure(figsize=(7, 6))
    plt.imshow(base_map, origin="upper", vmin=vmin, vmax=vmax)
    yy, xx = np.where(detected_mask)
    plt.scatter(xx, yy, s=8, facecolors="none", edgecolors="red", linewidths=0.7)
    plt.colorbar(label="max spectral spike z-score")
    plt.xlabel("x / col")
    plt.ylabel("y / row")
    plt.title(title)
    plt.tight_layout()
    out_png = Path(out_png)
    plt.savefig(out_png, dpi=200)
    plt.show()
    print(f"Saved: {out_png}")


def plot_line_spectrum_examples(
    cube: np.ndarray,
    wavelengths: np.ndarray,
    df_pixels: pd.DataFrame,
    out_dir: str | Path,
    max_examples: int = 10,
) -> None:
    """Plot spectra for the strongest detected pixels."""
    if len(df_pixels) == 0:
        print("No detected pixels to plot.")
        return

    out_dir = Path(out_dir)
    examples = df_pixels.sort_values("spike_z", ascending=False).head(max_examples)

    for _, row in examples.iterrows():
        r = int(row["row"])
        c = int(row["col"])
        wl = float(row["center_wavelength_nm"])
        spec = cube[r, c, :]

        plt.figure(figsize=(9, 4))
        plt.plot(wavelengths, spec, marker="o", linewidth=1)
        plt.axvline(wl, linestyle="--", linewidth=1)
        plt.xlabel("Wavelength [nm]")
        plt.ylabel("Radiance / Reflectance")
        plt.title(f"Detected spike: y={row['y']}, x={row['x']}, wl={wl:.2f} nm, z={row['spike_z']:.2f}")
        plt.grid(True)
        plt.tight_layout()

        out_png = out_dir / f"spectrum_y{row['y']}_x{row['x']}_wl{wl:.2f}.png".replace(".", "p")
        plt.savefig(out_png, dpi=200)
        plt.show()
        print(f"Saved: {out_png}")


# Main runner


def run_detector(
    csv_path: str | Path = CSV_PATH,
    out_dir: str | Path = OUT_DIR,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading CSV...")
    df, wavelengths_all, spectra_all, wave_cols_all = load_roi_spectra_csv(csv_path)
    cube_all, ys, xs = spectra_to_cube(df, spectra_all)

    print(f"CSV rows: {len(df)}")
    print(f"Cube shape: {cube_all.shape}")
    print(f"Original y range: {ys[0]} to {ys[-1]}")
    print(f"Original x range: {xs[0]} to {xs[-1]}")
    print(f"Wavelength range: {wavelengths_all[0]:.2f} to {wavelengths_all[-1]:.2f} nm")

    bmask = band_mask(wavelengths_all, WL_MIN, WL_MAX, EXCLUDE_RANGES)
    cube = cube_all[:, :, bmask]
    wavelengths = wavelengths_all[bmask]

    if cube.shape[2] == 0:
        raise ValueError("No bands selected. Check WL_MIN/WL_MAX/EXCLUDE_RANGES.")

    print(f"Selected wavelength range: {wavelengths[0]:.2f} to {wavelengths[-1]:.2f} nm")
    print(f"Selected bands: {cube.shape[2]}")

    valid_mask = make_valid_mask(cube, require_positive=REQUIRE_POSITIVE)
    print(f"Valid pixels: {np.sum(valid_mask)} / {valid_mask.size}")

    print("Computing spectral one-band spike score...")
    spike = compute_spectral_spike_cube(
        cube=cube,
        wavelengths=wavelengths,
        valid_mask=valid_mask,
        neighbor_offset=NEIGHBOR_OFFSET,
    )

    print("Aggregating by y=x-parallel diagonals...")
    df_lines, df_pixels, maps = detect_diagonal_spike_lines(
        spike=spike,
        valid_mask=valid_mask,
        ys=ys,
        xs=xs,
        diag_mode=DIAG_MODE,
        pixel_z_threshold=PIXEL_Z_THRESHOLD,
        min_line_pixels=MIN_LINE_PIXELS,
        min_hit_pixels=MIN_HIT_PIXELS,
        min_hit_fraction=MIN_HIT_FRACTION,
        line_percentile=LINE_PERCENTILE,
        line_score_threshold=LINE_SCORE_THRESHOLD,
        collapse_adjacent_bands=COLLAPSE_ADJACENT_BANDS,
        band_separation=BAND_SEPARATION,
    )

    print(f"Detected line/wavelength pairs: {len(df_lines)}")
    print(f"Detected pixels: {len(df_pixels)}")

    # Save outputs
    df_lines.to_csv(out_dir / "detected_diagonal_lines.csv", index=False)
    df_pixels.to_csv(out_dir / "detected_anomaly_pixels.csv", index=False)
    print(f"Saved: {out_dir / 'detected_diagonal_lines.csv'}")
    print(f"Saved: {out_dir / 'detected_anomaly_pixels.csv'}")

    save_map_csv(maps["detected_mask"].astype(int), out_dir / "detected_mask.csv")
    save_map_csv(maps["best_z_map"], out_dir / "detected_best_spike_z_map.csv")
    save_map_csv(maps["wavelength_map"], out_dir / "detected_wavelength_map.csv")
    save_map_csv(maps["max_spike_z_map"], out_dir / "max_spike_z_map.csv")

    finite = maps["max_spike_z_map"][np.isfinite(maps["max_spike_z_map"])]
    vmax = np.nanpercentile(finite, 99) if finite.size else None

    plot_map(
        maps["max_spike_z_map"],
        title="Maximum spectral one-band spike z-score",
        out_png=out_dir / "max_spike_z_map.png",
        colorbar_label="max spike z",
        vmin=0,
        vmax=vmax,
    )

    plot_map(
        maps["wavelength_map"],
        title="Detected artifact wavelength map",
        out_png=out_dir / "detected_wavelength_map.png",
        colorbar_label="wavelength [nm]",
    )

    plot_detected_overlay(
        base_map=maps["max_spike_z_map"],
        detected_mask=maps["detected_mask"],
        out_png=out_dir / "detected_overlay.png",
    )

    plot_line_spectrum_examples(
        cube=cube,
        wavelengths=wavelengths,
        df_pixels=df_pixels,
        out_dir=out_dir,
        max_examples=10,
    )

    return {
        "df": df,
        "cube": cube,
        "wavelengths": wavelengths,
        "ys": ys,
        "xs": xs,
        "valid_mask": valid_mask,
        "spike": spike,
        "df_lines": df_lines,
        "df_pixels": df_pixels,
        "maps": maps,
        "out_dir": out_dir,
    }


result = run_detector(
    csv_path=CSV_PATH,
    out_dir=OUT_DIR,
)

display(result["df_lines"].head(30))
display(result["df_pixels"].head(30))

