"""Inspect one pixel spectrum against background and methane target spectra."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# plotand optionally save diagnostic spectra for one selected pixel, comparing it to the background mean, its difference and ratio to the background, and the normalized CH4 target shape when a UAS is provided, with appropriate titles, labels, and error handling for input types and shapes
def plot_spectrum_at_yx(
    cube: np.ndarray,
    wavelengths: np.ndarray,
    y: int,
    x: int,
    background_mask: Optional[np.ndarray] = None,
    alpha_map: Optional[np.ndarray] = None,
    uas: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    xlim: Optional[Tuple[float, float]] = None,
    save_csv: Optional[str | Path] = None,
) -> np.ndarray:
    height, width, _ = cube.shape

    if not (0 <= y < height and 0 <= x < width):
        raise ValueError(f"(y, x)=({y}, {x}) is outside cube shape H={height}, W={width}.")

    spec = cube[y, x, :]
    alpha_value = alpha_map[y, x] if alpha_map is not None else None

    # Plot the selected pixel spectrum and the background mean spectrum.
    plt.figure(figsize=(8, 4))
    plt.plot(wavelengths, spec, marker="o", label=f"pixel ({y}, {x})")

    bg_mean = None
    if background_mask is not None:
        bg_mean = np.nanmean(cube[background_mask], axis=0)
        plt.plot(wavelengths, bg_mean, marker="o", label="background mean")

    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Radiance / Reflectance")

    if title is None:
        if alpha_value is None:
            title = f"Spectrum at (y={y}, x={x})"
        else:
            title = f"Spectrum at (y={y}, x={x}), alpha={alpha_value:.4g}"

    plt.title(title)
    if xlim is not None:
        plt.xlim(*xlim)
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot the difference and ratio relative to the background mean.
    if bg_mean is not None:
        diff = spec - bg_mean
        ratio = spec / np.maximum(bg_mean, 1e-12)

        plt.figure(figsize=(8, 4))
        plt.plot(wavelengths, diff, marker="o", label="pixel - background mean")
        plt.axhline(0, linewidth=1)
        plt.xlabel("Wavelength [nm]")
        plt.ylabel("Difference")
        plt.title(f"Difference from background at (y={y}, x={x})")
        if xlim is not None:
            plt.xlim(*xlim)
        plt.grid(True)
        plt.legend()
        plt.show()

        plt.figure(figsize=(8, 4))
        plt.plot(wavelengths, ratio, marker="o", label="pixel / background mean")
        plt.axhline(1, linewidth=1)
        plt.xlabel("Wavelength [nm]")
        plt.ylabel("Ratio")
        plt.title(f"Ratio to background at (y={y}, x={x})")
        if xlim is not None:
            plt.xlim(*xlim)
        plt.grid(True)
        plt.legend()
        plt.show()

    # Plot the normalized pixel difference against the normalized methane target.
    if bg_mean is not None and uas is not None:
        diff = spec - bg_mean
        diff_norm = diff / (np.nanmax(np.abs(diff)) + 1e-12)

        target = -bg_mean * uas
        target_norm = target / (np.nanmax(np.abs(target)) + 1e-12)

        plt.figure(figsize=(8, 4))
        plt.plot(wavelengths, diff_norm, marker="o", label="pixel - background, normalized")
        plt.plot(wavelengths, target_norm, marker="o", label="CH4 target, normalized")
        plt.axhline(0, linewidth=1)
        plt.xlabel("Wavelength [nm]")
        plt.ylabel("Normalized value")
        plt.title(f"Difference shape vs CH4 target at (y={y}, x={x})")
        if xlim is not None:
            plt.xlim(*xlim)
        plt.grid(True)
        plt.legend()
        plt.show()

    # Save diagnostic data for external inspection.
    if save_csv is not None:
        df = pd.DataFrame(
            {
                "wavelength_nm": wavelengths,
                "spectrum": spec,
                "y": y,
                "x": x,
            }
        )

        if bg_mean is not None:
            df["background_mean"] = bg_mean
            df["difference_from_background"] = spec - bg_mean
            df["ratio_to_background"] = spec / np.maximum(bg_mean, 1e-12)

        if uas is not None:
            df["uas"] = uas

        if alpha_value is not None:
            df["alpha"] = alpha_value

        save_csv = Path(save_csv)
        save_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_csv, index=False)
        print(f"Saved: {save_csv}")

    return spec
