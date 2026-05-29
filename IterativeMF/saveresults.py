# save pixel-wise results to CSV
import numpy as np
import pandas as pd
from pathlib import Path
def save_imf_pixel_results_csv(
    alpha_map,
    plume_mask,
    background_mask=None,
    valid_mask=None,
    ys=None,
    xs=None,
    out_csv="imf_pixel_results.csv"
):

    H, W = alpha_map.shape
    row_idx, col_idx = np.indices((H, W))

    if ys is not None:
        y_coord = np.asarray(ys)[row_idx.ravel()]
    else:
        y_coord = row_idx.ravel()

    if xs is not None:
        x_coord = np.asarray(xs)[col_idx.ravel()]
    else:
        x_coord = col_idx.ravel()

    df = pd.DataFrame({
        "row": row_idx.ravel(),
        "col": col_idx.ravel(),
        "y": y_coord,
        "x": x_coord,
        "alpha": alpha_map.ravel(),
        "is_plume": plume_mask.ravel().astype(bool),
    })

    if background_mask is not None:
        df["is_background"] = background_mask.ravel().astype(bool)

    if valid_mask is not None:
        df["is_valid"] = valid_mask.ravel().astype(bool)

    df = df.sort_values("alpha", ascending=False)

    out_csv = Path(out_csv)
    df.to_csv(out_csv, index=False)

    print(f"Saved: {out_csv}")
    print(f"Total pixels: {len(df)}")
    print(f"Plume candidate pixels: {df['is_plume'].sum()}")

    return df

df_imf = save_imf_pixel_results_csv(
    alpha_map=alpha_imf_map,
    plume_mask=plume_imf_mask,
    background_mask=background_imf_mask,
    valid_mask=valid_mask_2300,
    ys=None,   
    xs=None,   
    out_csv="imf_pixel_results.csv"
)

# save only plume candidate pixels to a separate CSV
df_plume = df_imf[df_imf["is_plume"]].copy()
df_plume.to_csv("imf_plume_pixels_only.csv", index=False)

print("Saved: imf_plume_pixels_only.csv")
print(df_plume.head(20))

