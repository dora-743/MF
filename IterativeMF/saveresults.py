"""Save pixel-wise Iterative Matched Filter results to CSV files.

This module is intentionally import-safe: it defines functions only and does
not execute notebook-specific variables at import time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# save pixel-wise results from an Iterative Matched Filter analysis to a CSV file, including row and column indices, spatial coordinates (if provided), alpha values, plume candidate flags, and optional background and valid pixel flags, with the output sorted by alpha in descending order
def save_imf_pixel_results_csv(
    alpha_map: np.ndarray,
    plume_mask: np.ndarray,
    background_mask: Optional[np.ndarray] = None,
    valid_mask: Optional[np.ndarray] = None,
    ys: Optional[np.ndarray] = None,
    xs: Optional[np.ndarray] = None,
    out_csv: str | Path = "imf_pixel_results.csv",
) -> pd.DataFrame:
    alpha_map = np.asarray(alpha_map)
    plume_mask = np.asarray(plume_mask).astype(bool)

    if alpha_map.shape != plume_mask.shape:
        raise ValueError("alpha_map and plume_mask must have the same shape.")

    height, width = alpha_map.shape
    row_idx, col_idx = np.indices((height, width))

    if ys is not None:
        y_coord = np.asarray(ys)[row_idx.ravel()]
    else:
        y_coord = row_idx.ravel()

    if xs is not None:
        x_coord = np.asarray(xs)[col_idx.ravel()]
    else:
        x_coord = col_idx.ravel()

    df = pd.DataFrame(
        {
            "row": row_idx.ravel(),
            "col": col_idx.ravel(),
            "y": y_coord,
            "x": x_coord,
            "alpha": alpha_map.ravel(),
            "is_plume": plume_mask.ravel(),
        }
    )

    if background_mask is not None:
        background_mask = np.asarray(background_mask).astype(bool)
        if background_mask.shape != alpha_map.shape:
            raise ValueError("background_mask must have the same shape as alpha_map.")
        df["is_background"] = background_mask.ravel()

    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask).astype(bool)
        if valid_mask.shape != alpha_map.shape:
            raise ValueError("valid_mask must have the same shape as alpha_map.")
        df["is_valid"] = valid_mask.ravel()

    df = df.sort_values("alpha", ascending=False)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print(f"Saved: {out_csv}")
    print(f"Total pixels: {len(df)}")
    print(f"Plume candidate pixels: {int(df['is_plume'].sum())}")

    return df


# save only the plume candidate pixels from the full pixel results DataFrame to a separate CSV file, ensuring that the input DataFrame contains an 'is_plume' column and printing the number of plume candidate pixels saved
def save_plume_pixels_csv(
    pixel_results: pd.DataFrame,
    out_csv: str | Path = "imf_plume_pixels_only.csv",
) -> pd.DataFrame:
    if "is_plume" not in pixel_results.columns:
        raise ValueError("pixel_results must contain an `is_plume` column.")

    plume_df = pixel_results[pixel_results["is_plume"]].copy()
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    plume_df.to_csv(out_csv, index=False)

    print(f"Saved: {out_csv}")
    print(f"Plume candidate pixels: {len(plume_df)}")

    return plume_df

 # save the results of an Iterative Matched Filter analysis to CSV files, including a full pixel results CSV with alpha values and plume flags, and a separate CSV containing only plume candidate pixels, while also saving relevant NumPy arrays for later analysis and optionally plotting the results, with appropriate error handling for input types and shapes
def save_imf_result_csvs(
    result: dict[str, object],
    valid_mask: Optional[np.ndarray] = None,
    ys: Optional[np.ndarray] = None,
    xs: Optional[np.ndarray] = None,
    pixel_csv: str | Path = "imf_pixel_results.csv",
    plume_csv: str | Path = "imf_plume_pixels_only.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    alpha_map = result["alpha_map"]
    plume_mask = result["plume_mask"]
    background_mask = result.get("background_mask")

    if not isinstance(alpha_map, np.ndarray):
        raise TypeError("result['alpha_map'] must be a NumPy array.")
    if not isinstance(plume_mask, np.ndarray):
        raise TypeError("result['plume_mask'] must be a NumPy array.")
    if background_mask is not None and not isinstance(background_mask, np.ndarray):
        raise TypeError("result['background_mask'] must be a NumPy array when provided.")

    pixel_df = save_imf_pixel_results_csv(
        alpha_map=alpha_map,
        plume_mask=plume_mask,
        background_mask=background_mask,
        valid_mask=valid_mask,
        ys=ys,
        xs=xs,
        out_csv=pixel_csv,
    )
    plume_df = save_plume_pixels_csv(pixel_df, out_csv=plume_csv)

    return pixel_df, plume_df
