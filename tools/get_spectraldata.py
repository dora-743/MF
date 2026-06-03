from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile

try:
    from osgeo import osr  # type: ignore
except Exception:  # pragma: no cover - GDAL is optional.
    osr = None


DEFAULT_FILE = Path(
    r"E:\メタン\2025_HISUI_72_The Permian Basin-論文照合用"
    r"\HSHL1G_N320W1032_20221030160051_20231127193053"
    r"\HSHL1G_N320W1032_20221030160051_20231127193053.tif"
)
DEFAULT_CENTER_Y = 1066
DEFAULT_CENTER_X = 1463
DEFAULT_HALF_SIZE = 100
DEFAULT_OUTPUT_CSV = Path("all_roi_spectra200x200.csv")


# read the band parameters from the _B.csv file and return as a numpy array. The CSV has columns: band_index, wavelength_nm, fwhm_nm, gain, offset. We will read up to max_bands rows and return an array of shape (max_bands, 5) with the values. If there are fewer than max_bands rows in the CSV, we will fill the remaining rows with zeros.
def read_bfile(file_path: str | Path, max_bands: int = 185) -> np.ndarray:
    file_path = Path(file_path)
    band_csv = file_path.with_name(file_path.stem + "_B.csv")
    if not band_csv.exists():
        raise FileNotFoundError(f"Band parameter CSV not found: {band_csv}")

    df = pd.read_csv(band_csv)
    n_rows = min(max_bands, len(df))
    param = np.zeros((max_bands, 5), dtype=float)

    for i in range(n_rows):
        for j in range(5):
            param[i, j] = float(df.iloc[i, j + 1])

    return param


# read the radiometric coefficients from the _T.txt file. The text file has lines in the format "key=value". We will read the values for the keys: RadianceMultiVNIR, RadianceMultiSWIR, RadianceAddVNIR, RadianceAddSWIR. We will return these four values as a tuple of floats.
def read_tfile(file_path: str | Path) -> tuple[float, float, float, float]:
    file_path = Path(file_path)
    text_file = file_path.with_suffix(".txt")
    if not text_file.exists():
        raise FileNotFoundError(f"Metadata text file not found: {text_file}")

    values: dict[str, float] = {}
    with text_file.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if "=" not in line:
                continue
            key, value = line.rstrip().split("=", 1)
            values[key.strip()] = float(value)

    required = {
        "RadianceMultiVNIR": "rad_multi_vnir",
        "RadianceAddVNIR": "rad_add_vnir",
        "RadianceMultiSWIR": "rad_multi_swir",
        "RadianceAddSWIR": "rad_add_swir",
    }

    missing = [key for key in required if key not in values]
    if missing:
        raise ValueError(f"Missing radiometric coefficients in {text_file}: {missing}")

    return (
        values["RadianceMultiVNIR"],
        values["RadianceMultiSWIR"],
        values["RadianceAddVNIR"],
        values["RadianceAddSWIR"],
    )


# apply the radiometric correction to the image using the coefficients. The image is a 3D numpy array with shape (height, width, bands). We will apply the correction as follows:
# For the VNIR bands (0 to vnir_bands-1), we will apply: image = image * rad_multi_vnir + rad_add_vnir
# For the SWIR bands (vnir_bands to end), we will apply: image = image * rad_multi_swir + rad_add_swir
def apply_radiometric(
    image: np.ndarray,
    rad_multi_vnir: float,
    rad_multi_swir: float,
    rad_add_vnir: float,
    rad_add_swir: float,
    vnir_bands: int = 58,
) -> np.ndarray:
    image = image.astype(float, copy=True)
    valid_area = np.ones(image.shape[:2], dtype=bool)

    if image.shape[2] > 10:
        valid_area &= image[:, :, 10] != 0

    n_bands = image.shape[2]
    vnir_end = min(vnir_bands, n_bands)

    image[:, :, :vnir_end] = image[:, :, :vnir_end] * rad_multi_vnir + rad_add_vnir
    if n_bands > vnir_end:
        image[:, :, vnir_end:] = image[:, :, vnir_end:] * rad_multi_swir + rad_add_swir

    image[~valid_area] = 0
    return image


# convert image pixel coordinates to geospatial coordinates using the GDAL geotransform. The function takes the GDAL dataset and pixel coordinates (x, y) and returns the corresponding geospatial coordinates (X, Y).
def show_xy(src, x: float, y: float) -> tuple[float, float]:
    transform = src.GetGeoTransform()
    X = transform[0] + x * transform[1] + y * transform[2]
    Y = transform[3] + x * transform[4] + y * transform[5]
    return X, Y


# convert image pixel coordinates to latitude and longitude using GDAL/OSR. The function takes the GDAL dataset and pixel coordinates (x, y) and returns the corresponding latitude and longitude (lat, lon). It uses the geotransform to get the geospatial coordinates and then transforms them to WGS84 lat/lon.
def show_latlon(src, x: float, y: float) -> tuple[float, float, float]:
    if osr is None:
        raise ImportError("GDAL/OSGeo is required for show_latlon().")

    old_cs = osr.SpatialReference()
    old_cs.ImportFromWkt(src.GetProjectionRef())

    wgs84_wkt = """
        GEOGCS["WGS 84",
            DATUM["WGS_1984",
                SPHEROID["WGS 84",6378137,298.257223563,
                    AUTHORITY["EPSG","7030"]],
                AUTHORITY["EPSG","6326"]],
            PRIMEM["Greenwich",0,
                AUTHORITY["EPSG","8901"]],
            UNIT["degree",0.01745329251994328,
                AUTHORITY["EPSG","9122"]],
            AUTHORITY["EPSG","4326"]]"""
    new_cs = osr.SpatialReference()
    new_cs.ImportFromWkt(wgs84_wkt)
    transform = osr.CoordinateTransformation(old_cs, new_cs)

    X, Y = show_xy(src, x, y)
    return transform.TransformPoint(X, Y)


# create an RGB image from the specified bands of the input image. The input image is a 3D numpy array with shape (height, width, bands). The function takes the band indices for blue, green, and red channels and returns a 3-channel RGB image normalized to [0, 1]. The function also checks that the specified band indices are within the valid range of the image bands.
def get_rgb(image: np.ndarray, b: int = 8, g: int = 18, r: int = 28) -> np.ndarray:
    for idx in (b, g, r):
        if idx < 0 or idx >= image.shape[2]:
            raise IndexError(f"Band index {idx} is outside image bands 0 to {image.shape[2] - 1}.")

    rgb = np.zeros((image.shape[0], image.shape[1], 3), dtype=float)
    rgb[:, :, 0] = image[:, :, r]
    rgb[:, :, 1] = image[:, :, g]
    rgb[:, :, 2] = image[:, :, b]

    scale = np.nanmax(rgb) / 3.0
    if not np.isfinite(scale) or scale < 1e-12:
        scale = 1.0
    rgb = np.clip(rgb / scale, 0.0, 1.0)
    return rgb


# show an image using matplotlib. The input image is a 3D numpy array with shape (height, width, channels). The function displays the image and optionally sets the title.
def show_img(image: np.ndarray, title: Optional[str] = None) -> None:
    _, ax = plt.subplots()
    ax.imshow(image)
    if title is not None:
        ax.set_title(title)
    plt.show()


# return the wavelengths and radiance values for a specific pixel (y, x) in the image. The function takes the image, the band parameters, and the pixel coordinates. It returns a 2D array with two columns: wavelength and radiance for each band starting from swir_start index.
def get_radiance(image: np.ndarray, param: np.ndarray, y: int, x: int, swir_start: int = 58) -> np.ndarray:
    wavelengths = param[swir_start:image.shape[2], 0]
    radiance = image[y, x, swir_start:image.shape[2]]
    return np.column_stack([wavelengths, radiance])


# extract a square region of interest (ROI) from the image centered at (center_y, center_x) with the specified half_size. The function checks that the requested ROI is within the bounds of the image and returns the extracted ROI as a 3D numpy array.
def extract_roi(image: np.ndarray, center_y: int, center_x: int, half_size: int) -> np.ndarray:
    if half_size <= 0:
        raise ValueError("half_size must be positive.")

    y0 = max(0, center_y - half_size)
    y1 = min(image.shape[0], center_y + half_size)
    x0 = max(0, center_x - half_size)
    x1 = min(image.shape[1], center_x + half_size)

    if y0 >= y1 or x0 >= x1:
        raise ValueError("The requested ROI is outside the image.")

    return image[y0:y1, x0:x1, :]


# save the spectra for each pixel in the ROI to a CSV file. The CSV will have columns: y, x, wave_XXXnm for each wavelength. The function takes the image slice (ROI), the band parameters, and the output CSV path. It returns a DataFrame with the saved spectra.
def save_roi_spectra_csv(
    image_slice: np.ndarray,
    param: np.ndarray,
    output_csv: str | Path = DEFAULT_OUTPUT_CSV,
) -> pd.DataFrame:
    height, width, n_bands = image_slice.shape
    wavelengths = param[:n_bands, 0]
    columns = ["y", "x"] + [f"wave_{wl:.2f}nm" for wl in wavelengths]

    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    spectra = image_slice.reshape(-1, n_bands)
    data = np.column_stack([yy.reshape(-1), xx.reshape(-1), spectra])

    df = pd.DataFrame(data, columns=columns)
    df["y"] = df["y"].astype(int)
    df["x"] = df["x"].astype(int)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")
    return df


# read the HISUI image, apply radiometric correction, extract the RGB image and the ROI, save the spectra for the ROI to a CSV file, and optionally show the RGB images. The function returns a dictionary with the processed data and paths.
def process_hisui_file(
    file_path: str | Path = DEFAULT_FILE,
    center_y: int = DEFAULT_CENTER_Y,
    center_x: int = DEFAULT_CENTER_X,
    half_size: int = DEFAULT_HALF_SIZE,
    output_csv: str | Path = DEFAULT_OUTPUT_CSV,
    show_images: bool = False,
) -> dict[str, object]:
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Image file not found: {file_path}")

    image = tifffile.imread(file_path)
    param = read_bfile(file_path)
    rad_multi_vnir, rad_multi_swir, rad_add_vnir, rad_add_swir = read_tfile(file_path)
    image = apply_radiometric(image, rad_multi_vnir, rad_multi_swir, rad_add_vnir, rad_add_swir)

    rgb = get_rgb(image, b=8, g=18, r=28)
    image_slice = extract_roi(image, center_y=center_y, center_x=center_x, half_size=half_size)
    rgb_slice = extract_roi(rgb, center_y=center_y, center_x=center_x, half_size=half_size)

    if show_images:
        show_img(rgb, title="RGB image")
        show_img(rgb_slice, title="ROI RGB image")

    roi_df = save_roi_spectra_csv(image_slice, param, output_csv=output_csv)

    return {
        "image": image,
        "param": param,
        "rgb": rgb,
        "image_slice": image_slice,
        "rgb_slice": rgb_slice,
        "roi_df": roi_df,
        "output_csv": Path(output_csv),
    }


# create command-line interface for running the ROI extraction on a specified HISUI file and parameters
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export ROI spectra from a HISUI image.")
    parser.add_argument("--file", default=str(DEFAULT_FILE), help="Input HISUI TIFF file.")
    parser.add_argument("--center-y", type=int, default=DEFAULT_CENTER_Y, help="ROI center row.")
    parser.add_argument("--center-x", type=int, default=DEFAULT_CENTER_X, help="ROI center column.")
    parser.add_argument("--half-size", type=int, default=DEFAULT_HALF_SIZE, help="Half-size of the square ROI.")
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV), help="Output ROI spectra CSV.")
    parser.add_argument("--show-images", action="store_true", help="Show RGB preview images.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    process_hisui_file(
        file_path=args.file,
        center_y=args.center_y,
        center_x=args.center_x,
        half_size=args.half_size,
        output_csv=args.output_csv,
        show_images=args.show_images,
    )


if __name__ == "__main__":
    main()
