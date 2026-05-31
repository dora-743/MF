from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = Path(r"D:\research\code\all_roi_spectra200x200.csv")
OUT_DIR = Path(r"D:\research\code\diagonal_spike_tolerant_output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

WL_MIN = 2100.0
WL_MAX = 2400.0
PIXEL_Z_THRESHOLD = 4.0
DIAG_HALF_WIDTH = 2
MIN_CENTER_LINE_LENGTH = 80
MIN_HIT_FRACTION = 0.20
MIN_MEAN_Z = 1.0

def parse_wavelengths(columns):
    wave_cols = []
    wavelengths = []
    for c in columns:
        m = re.fullmatch(r'wave_([0-9]+(?:\.[0-9]+)?)nm', str(c))
        if m is not None:
            wave_cols.append(c)
            wavelengths.append(float(m.group(1)))
    return wave_cols, np.asarray(wavelengths, dtype=float)

def robust_zscore_per_band(spike):
    med = np.nanmedian(spike, axis=(0, 1))
    mad = np.nanmedian(np.abs(spike - med), axis=(0, 1))
    return (spike - med) / (1.4826 * (mad + 1e-12))

def load_cube(csv_path):
    df = pd.read_csv(csv_path)
    wave_cols, wavelengths = parse_wavelengths(df.columns)
    df = df.sort_values(['y', 'x']).reset_index(drop=True)
    height = int(df['y'].max()) + 1
    width = int(df['x'].max()) + 1
    cube = df[wave_cols].to_numpy(dtype=np.float32).reshape(height, width, len(wave_cols))
    return cube, wavelengths, df

def compute_spectral_spike(cube, wavelengths):
    spike = cube[:, :, 1:-1] - 0.5 * (cube[:, :, :-2] + cube[:, :, 2:])
    spike_wavelengths = wavelengths[1:-1]
    return spike, spike_wavelengths

cube, wavelengths, df = load_cube(CSV_PATH)
spike, spike_wavelengths = compute_spectral_spike(cube, wavelengths)

wl_mask = (spike_wavelengths >= WL_MIN) & (spike_wavelengths <= WL_MAX)
spike_band = spike[:, :, wl_mask]
spike_wavelengths_band = spike_wavelengths[wl_mask]
spike_z = robust_zscore_per_band(spike_band)

print(cube.shape)
print(spike_wavelengths_band[0], spike_wavelengths_band[-1], len(spike_wavelengths_band))

height, width, _ = spike_z.shape
yy, xx = np.indices((height, width))
diag = yy - xx

rows = []
for d in range(-(width - 1), height):
    center_len = int((diag == d).sum())
    if center_len < MIN_CENTER_LINE_LENGTH:
        continue

    band_mask = np.abs(diag - d) <= DIAG_HALF_WIDTH
    vals = spike_z[band_mask, :]
    hit_fraction = (vals > PIXEL_Z_THRESHOLD).mean(axis=0)
    mean_z = vals.mean(axis=0)
    median_z = np.median(vals, axis=0)
    p90_z = np.percentile(vals, 90, axis=0)
    max_z = vals.max(axis=0)

    for i, wl in enumerate(spike_wavelengths_band):
        if hit_fraction[i] >= MIN_HIT_FRACTION and mean_z[i] >= MIN_MEAN_Z:
            rows.append({
                'diag_center_d_y_minus_x': d,
                'wavelength_nm': wl,
                'center_line_length_px': center_len,
                'diagonal_band_half_width_px': DIAG_HALF_WIDTH,
                'diagonal_band_n_px': int(vals.shape[0]),
                'hit_fraction': float(hit_fraction[i]),
                'mean_z': float(mean_z[i]),
                'median_z': float(median_z[i]),
                'p90_z': float(p90_z[i]),
                'max_z': float(max_z[i]),
            })

detected = pd.DataFrame(rows).sort_values(['hit_fraction', 'mean_z'], ascending=False).reset_index(drop=True)
detected.to_csv(OUT_DIR / 'tolerant_detected_diagonal_lines.csv', index=False)
detected

zmap = spike_z[:, :, wi]
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(zmap, origin='upper', vmin=0, vmax=np.nanpercentile(zmap, 99.5))
fig.colorbar(im, ax=ax, label='spectral spike robust z-score')
ax.set_title(f'Spike z-map at {spike_wavelengths_band[wi]:.2f} nm')
ax.set_xlabel('x')
ax.set_ylabel('y')
xs = np.arange(width)
for d in [50, 51, 52, 53, 54]:
    ys = xs + d
    ok = (ys >= 0) & (ys < height)
    ax.plot(xs[ok], ys[ok], linewidth=1, label=f'y-x={d}')
ax.legend(loc='lower right', fontsize=8)
fig.tight_layout()
plot_path = OUT_DIR / f'spike_z_map_{spike_wavelengths_band[wi]:.2f}nm_diag_overlay.png'
fig.savefig(plot_path, dpi=200)
plot_path

# Pixel-level list for the detected diagonal bands
pixel_rows = []
for _, line in detected.iterrows():
    d = int(line['diag_center_d_y_minus_x'])
    half = int(line['diagonal_band_half_width_px'])
    wl = float(line['wavelength_nm'])
    wi = int(np.argmin(np.abs(spike_wavelengths_band - wl)))
    mask = (np.abs(diag - d) <= half) & (spike_z[:, :, wi] > PIXEL_Z_THRESHOLD)
    ys, xs = np.where(mask)
    for y, x in zip(ys, xs):
        pixel_rows.append({
            'y': int(y),
            'x': int(x),
            'diag_d_y_minus_x': int(y - x),
            'diag_center_d_y_minus_x': d,
            'wavelength_nm': wl,
            'spike_z': float(spike_z[y, x, wi]),
        })

pixels = pd.DataFrame(pixel_rows).sort_values(['wavelength_nm', 'diag_center_d_y_minus_x', 'y', 'x']).reset_index(drop=True)
pixels.to_csv(OUT_DIR / 'tolerant_detected_anomaly_pixels.csv', index=False)
pixels.head()

OUT_DIR = Path(OUT_DIR)
OUT_DIR.mkdir(parents=True, exist_ok=True)

if "detected" not in globals():
    detected = pd.read_csv(OUT_DIR / "tolerant_detected_diagonal_lines.csv")

def nearest_band_index(target_wl, wavelengths):
    wavelengths = np.asarray(wavelengths, dtype=float)
    return int(np.argmin(np.abs(wavelengths - target_wl)))

detected_wls = list(np.sort(detected["wavelength_nm"].unique()))

manual_wls = [2125.03]

display_wls = np.array(sorted(set(np.round(detected_wls + manual_wls, 2))))

print("show wavelengths:", display_wls)

height, width, n_bands = cube.shape

n = len(display_wls)
ncols = 2
nrows = math.ceil(n / ncols)

fig, axes = plt.subplots(nrows, ncols, figsize=(10, 5 * nrows))
axes = np.array(axes).reshape(-1)

for ax, wl in zip(axes, display_wls):
    band_idx = nearest_band_index(wl, wavelengths)
    img = cube[:, :, band_idx]

    vmin, vmax = np.nanpercentile(img, [2, 98])

    im = ax.imshow(
        img,
        cmap="gray",
        origin="upper",
        vmin=vmin,
        vmax=vmax
    )

    ax.set_title(f"{wavelengths[band_idx]:.2f} nm")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)
    ax.set_aspect("equal")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

for k in range(len(display_wls), len(axes)):
    axes[k].axis("off")

fig.tight_layout()

save_path = OUT_DIR / "anomalous_wavelength_images_with_manual_2125_03nm.png"
fig.savefig(save_path, dpi=200, bbox_inches="tight")
plt.show()

print(f"Saved: {save_path}")
