# plot spectrum at selected pixel with background comparison and UAS comparison, and save the spectrum data to CSV
def plot_spectrum_at_yx(
    cube,
    wavelengths,
    y,
    x,
    background_mask=None,
    alpha_map=None,
    uas=None,
    title=None,
    xlim=None,
    save_csv=None
):

    H, W, B = cube.shape

    if not (0 <= y < H and 0 <= x < W):
        raise ValueError(f"(y, x)=({y}, {x}) is outside cube shape H={H}, W={W}.")

    spec = cube[y, x, :]

    if alpha_map is not None:
        alpha_value = alpha_map[y, x]
    else:
        alpha_value = None

# 1. Plot spectrum with background mean comparison
    plt.figure(figsize=(8, 4))
    plt.plot(wavelengths, spec, marker="o", label=f"pixel ({y}, {x})")

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

# 2. Plot difference and ratio to background mean
    if background_mask is not None:
        bg_mean = np.nanmean(cube[background_mask], axis=0)

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

# 3. Plot comparison to UAS-based target spectrum
    if background_mask is not None and uas is not None:
        bg_mean = np.nanmean(cube[background_mask], axis=0)
        diff = spec - bg_mean

        diff_norm = diff / (np.nanmax(np.abs(diff)) + 1e-12)

        # For absorption, methane target in radiance space is roughly -mu * uas
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

# 4. Save spectrum data to CSV
    if save_csv is not None:
        df = pd.DataFrame({
            "wavelength_nm": wavelengths,
            "spectrum": spec
        })

        if background_mask is not None:
            bg_mean = np.nanmean(cube[background_mask], axis=0)
            df["background_mean"] = bg_mean
            df["difference_from_background"] = spec - bg_mean
            df["ratio_to_background"] = spec / np.maximum(bg_mean, 1e-12)

        if uas is not None:
            df["uas"] = uas

        df["y"] = y
        df["x"] = x

        if alpha_value is not None:
            df["alpha"] = alpha_value

        df.to_csv(save_csv, index=False)
        print(f"Saved: {save_csv}")

    return spec
spec = plot_spectrum_at_yx(
    cube=cube_2300,
    wavelengths=wave_2300,
    y=159,
    x=35,
    background_mask=background_imf_mask,
    alpha_map=alpha_imf_map,
    uas=uas_2300,
    xlim=(2100, 2450),
    save_csv="suspicious_pixel_y159_x35.csv"
)
