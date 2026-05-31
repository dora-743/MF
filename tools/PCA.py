import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import math
from pathlib import Path
from sklearn.decomposition import PCA
from IPython.display import display
CSV_PATH = Path(r"D:\research\code\all_roi_spectra200x200.csv")
OUT_DIR = Path(r"D:\research\code\diagonal_spike_tolerant_output")
OUT_DIR = Path(OUT_DIR)
OUT_DIR.mkdir(parents=True, exist_ok=True)
# settings
WL_MIN = 2100
WL_MAX = 2400

# number of PCs to show
N_SHOW_PC = 8

# number of PCs to use for PCA reconstruction
N_PC_TOTAL = 12

# number of PCs to use for PCA residual calculation
# too small -> local anomalies remain in residual, too large -> residual mostly noise
N_PC_RECON = 3

# wavelengths to visualize
TARGET_WLS = [2125.03, 2174.99, 2387.32]


def robust_zscore_1d(v):
    v = np.asarray(v, dtype=float)
    med = np.nanmedian(v)
    mad = np.nanmedian(np.abs(v - med))
    scale = mad / 0.6745

    if not np.isfinite(scale) or scale < 1e-12:
        scale = np.nanstd(v)

    if not np.isfinite(scale) or scale < 1e-12:
        return np.zeros_like(v)

    return (v - med) / scale


def nearest_band_index(target_wl, wavelengths):
    wavelengths = np.asarray(wavelengths, dtype=float)
    return int(np.argmin(np.abs(wavelengths - target_wl)))


# if "cube" not in globals():

if "cube" not in globals():
    wave_cols = [c for c in df.columns if c.startswith("wave_")]

    wavelengths = np.array([
        float(re.search(r"wave_([0-9.]+)nm", c).group(1))
        for c in wave_cols
    ])

    height = int(df["y"].max()) + 1
    width = int(df["x"].max()) + 1

    cube = np.full((height, width, len(wave_cols)), np.nan, dtype=float)
    cube[
        df["y"].to_numpy(dtype=int),
        df["x"].to_numpy(dtype=int),
        :
    ] = df[wave_cols].to_numpy(dtype=float)

else:
    height, width, n_bands = cube.shape

# make sure wavelengths is numpy array

wl_mask = (wavelengths >= WL_MIN) & (wavelengths <= WL_MAX)
wl_pca = wavelengths[wl_mask]

X_raw = cube[:, :, wl_mask].reshape(-1, wl_mask.sum())

valid_pixel_mask = np.all(np.isfinite(X_raw), axis=1)
X_valid = X_raw[valid_pixel_mask]

# normolize per band
mu = np.nanmean(X_valid, axis=0)
sd = np.nanstd(X_valid, axis=0, ddof=1)
sd[sd < 1e-12] = 1.0

X_std = (X_valid - mu) / sd

print("PCA input shape:", X_std.shape)
print("Wavelength range:", wl_pca[0], "to", wl_pca[-1], "nm")
print("Valid pixels:", valid_pixel_mask.sum())

# calculate PCA scores

pca = PCA(n_components=N_PC_TOTAL, random_state=0)
scores = pca.fit_transform(X_std)

explained = pd.DataFrame({
    "PC": np.arange(1, N_PC_TOTAL + 1),
    "explained_variance_ratio": pca.explained_variance_ratio_,
    "cumulative": np.cumsum(pca.explained_variance_ratio_)
})

display(explained)

explained.to_csv(OUT_DIR / "pca_explained_variance.csv", index=False)

# show PCA score maps

n_show = min(N_SHOW_PC, scores.shape[1])

ncols = 4
nrows = math.ceil(n_show / ncols)

fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
axes = np.array(axes).reshape(-1)

for i in range(n_show):
    score_full = np.full(height * width, np.nan, dtype=float)
    score_full[valid_pixel_mask] = scores[:, i]

    score_img = score_full.reshape(height, width)
    score_z = robust_zscore_1d(score_img.ravel()).reshape(height, width)

    vals = score_z[np.isfinite(score_z)]
    vmax = np.nanpercentile(np.abs(vals), 99)
    vmin = -vmax

    ax = axes[i]
    im = ax.imshow(
        score_z,
        origin="upper",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax
    )

    ax.set_title(
        f"PC{i+1} score map\n"
        f"EVR={pca.explained_variance_ratio_[i]:.4f}"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

for j in range(n_show, len(axes)):
    axes[j].axis("off")

fig.suptitle("PCA score maps", fontsize=16)
fig.tight_layout()

save_path = OUT_DIR / "pca_score_maps_PC1_PC8.png"
fig.savefig(save_path, dpi=200, bbox_inches="tight")
plt.show()

print(f"Saved: {save_path}")

