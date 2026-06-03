"""Command-line example for running Iterative MF on ROI and MODTRAN CSV files.

Example:
    python run_iterative_mf.py \
        --roi-csv all_roi_spectra200x200.csv \
        --modtran-csv CH4c.csv \
        --output-dir outputs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from Iterative_MF import (
    compute_uas_log_slope,
    gaussian_srf_resample,
    iterative_matched_filter_map_from_cube,
    load_ch4_modtran_csv,
    load_roi_spectra_csv,
    make_valid_pixel_mask,
    plot_alpha_map,
    plot_iterative_mf_result,
    select_bands,
    spectra_to_cube,
)
from saveresults import save_imf_result_csvs


# parse command-line arguments for running the Iterative MF workflow, including paths to ROI and MODTRAN CSV files, output directory, wavelength selection parameters, alpha range for UAS fitting, number of iterations, robust threshold multiplier, and an option to disable plots
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Iterative MF for methane plume detection.")
    parser.add_argument("--roi-csv", required=True, help="ROI spectra CSV path.")
    parser.add_argument("--modtran-csv", required=True, help="CH4 MODTRAN CSV path.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for CSV outputs.")
    parser.add_argument("--wl-min", type=float, default=2100.0, help="Minimum wavelength [nm].")
    parser.add_argument("--wl-max", type=float, default=2450.0, help="Maximum wavelength [nm].")
    parser.add_argument("--fwhm", type=float, default=12.5, help="Sensor FWHM [nm].")
    parser.add_argument("--alpha-min", type=float, default=0.0, help="Minimum alpha for UAS fitting.")
    parser.add_argument("--alpha-max", type=float, default=0.5, help="Maximum alpha for UAS fitting.")
    parser.add_argument("--n-iter", type=int, default=5, help="Maximum number of IMF iterations.")
    parser.add_argument("--nsigma", type=float, default=3.0, help="Robust threshold multiplier.")
    parser.add_argument("--no-plots", action="store_true", help="Disable interactive plots.")
    return parser.parse_args()


# run the main workflow for loading ROI and MODTRAN spectra, preparing the data cube, selecting bands, estimating UAS, running the iterative Matched Filter algorithm, saving results, and plotting outputs if enabled
def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load ROI spectra and convert them to an image cube.
    df, wavelengths, spectra = load_roi_spectra_csv(args.roi_csv)
    cube, ys, xs = spectra_to_cube(df, spectra)

    print(f"ROI table shape: {df.shape}")
    print(f"Cube shape: {cube.shape}")
    print(f"Wavelength range: {wavelengths[0]:.2f} - {wavelengths[-1]:.2f} nm")

    # Load and resample MODTRAN spectra to the sensor wavelengths.
    mod_wave, alpha_grid, mod_spectra = load_ch4_modtran_csv(args.modtran_csv)
    mod_sensor = gaussian_srf_resample(
        mod_wave=mod_wave,
        mod_spectra=mod_spectra,
        sensor_wave=wavelengths,
        fwhm_nm=args.fwhm,
    )

    # Estimate UAS from the selected alpha range.
    uas_all, _ = compute_uas_log_slope(
        alpha_grid=alpha_grid,
        spectra_grid=mod_sensor,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
    )

    # Select the methane-sensitive wavelength window.
    cube_sel, wave_sel, mask_sel = select_bands(
        cube,
        wavelengths,
        wl_min=args.wl_min,
        wl_max=args.wl_max,
    )
    uas_sel = uas_all[mask_sel]

    # Build a valid pixel mask and run Iterative MF.
    valid_mask = make_valid_pixel_mask(
        cube_sel,
        nodata_values=[0, -9999],
        require_positive=True,
        min_valid_fraction=1.0,
    )

    result = iterative_matched_filter_map_from_cube(
        cube=cube_sel,
        uas=uas_sel,
        valid_mask=valid_mask,
        initial_background_mask=None,
        n_iter=args.n_iter,
        nsigma=args.nsigma,
        reg=1e-6,
        verbose=True,
    )

    # Save pixel-wise results.
    save_imf_result_csvs(
        result,
        valid_mask=valid_mask,
        ys=ys,
        xs=xs,
        pixel_csv=output_dir / "imf_pixel_results.csv",
        plume_csv=output_dir / "imf_plume_pixels_only.csv",
    )

    # Save arrays for later analysis.
    np.save(output_dir / "alpha_imf_map.npy", result["alpha_map"])
    np.save(output_dir / "plume_imf_mask.npy", result["plume_mask"])
    np.save(output_dir / "background_imf_mask.npy", result["background_mask"])
    np.save(output_dir / "wave_selected.npy", wave_sel)
    np.save(output_dir / "uas_selected.npy", uas_sel)
    print(f"Saved NumPy arrays to: {output_dir}")

    if not args.no_plots:
        plot_iterative_mf_result(result)
        alpha_map = result["alpha_map"]
        if isinstance(alpha_map, np.ndarray):
            for idx, alpha_iter in enumerate(result["alpha_history"]):
                plot_alpha_map(
                    alpha_iter,
                    title=f"Iterative MF iter {idx + 1}",
                    vmin=np.nanpercentile(alpha_iter, 2),
                    vmax=np.nanpercentile(alpha_iter, 98),
                )
        plt.show()


if __name__ == "__main__":
    main()
