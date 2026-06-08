# Line Detection

This directory contains an integrated workflow for detecting and analyzing line-like residual artifacts or remaining plume-like line structures after Matched Filter-based methane detection.

The scripts are designed to work together:

1. run an Iterative Matched Filter workflow with sensor-geometry-aware destriping,
2. detect remaining line-like plume candidates,
3. extract spectra from selected line pixels and nearby control pixels,
4. summarize line angles and spectral behavior.


---

## Files

| File | Role |
|---|---|
| `paper_sensor_geometry_iterative_mf_destriping_full.py` | Main Iterative MF workflow with sensor-geometry-aware destriping and angle-sweep cleanup. |
| `analyze_remaining_plume_lines_and_spectra.py` | Detects remaining line-like plume candidates and extracts spectra from line and control pixels. |
| `mf_line_spectrum_pipeline.py` | Integrated runner that executes the MF stage and the remaining-line spectrum-analysis stage in sequence. |

---

## Required Inputs

The workflow expects the following inputs.

### 1. ROI or map spectra CSV

A CSV file containing one row per pixel and wavelength columns.

Expected format:

```text
y,x,wave_405.00nm,wave_415.00nm,...,wave_2450.00nm
```

The `y` and `x` columns are pixel coordinates.

This file is used by the MF workflow and also by the spectrum extraction stage when:

```python
SPECTRA_SOURCE = "map_csv"
```

### 2. MODTRAN methane simulation CSV

A CSV file containing simulated methane spectra for several methane enhancement values.

Expected format:

```text
wavelength,0.0,0.5,1.0,2.0,...
```

where each non-wavelength column name represents a methane enhancement value.

This file is used to estimate the unit absorption spectrum, or UAS.

### 3. HISUI metadata text file

The sensor-geometry-aware destriping workflow uses HISUI metadata to approximate sensor-line and sensor-column directions.

The metadata path is configured in:

```python
METADATA_TXT
```

inside `paper_sensor_geometry_iterative_mf_destriping_full.py`.

If you use `mf_line_spectrum_pipeline.py`, you can override this path with:

```python
METADATA_TXT_OVERRIDE
```

---

## Main Workflow

The recommended entry point is:

```bash
python mf_line_spectrum_pipeline.py
```

This script runs two stages.

### Stage 1: Iterative MF with destriping

This stage is implemented in:

```text
paper_sensor_geometry_iterative_mf_destriping_full.py
```

It performs:

1. spectra CSV loading,
2. wavelength band selection,
3. MODTRAN-based UAS estimation,
4. Iterative Matched Filter processing,
5. sensor-geometry-aware destriping,
6. angle-sweep cleanup for remaining broad line artifacts,
7. saving alpha maps, plume masks, and summary tables.

### Stage 2: Remaining-line and spectrum analysis

This stage is implemented in:

```text
analyze_remaining_plume_lines_and_spectra.py
```

It performs:

1. detection of remaining line-like plume pixels,
2. line equation estimation,
3. line-angle diagnostics,
4. line-pixel and control-pixel sampling,
5. spectrum extraction,
6. spectrum plots,
7. overlay plots.

---

## How to Configure

Open `mf_line_spectrum_pipeline.py` and edit the user settings near the top.

Important settings include:

```python
ROI_CSV = Path(r"D:/research/code/all_roi_spectra200x200.csv")
MODTRAN_CSV = Path(r"E:/refit/CH4c.csv")
RESULT_DIR = Path(r"D:/research/code/outputs_paper_sensor_geometry_destripe")

CASE_NAME = "paper_sensor_line_then_column_each_iter_angle125_focus"
REFERENCE_CASE_NAME = "paper_sensor_line_then_column_each_iter_angle2475_focus"
```

If the MF stage has already been run and you only want to rerun the remaining-line analysis, set:

```python
RUN_MF_STAGE = False
RUN_LINE_SPECTRUM_STAGE = True
```

If you want to run both stages, use:

```python
RUN_MF_STAGE = True
RUN_LINE_SPECTRUM_STAGE = True
```

---

## Output Files

The MF stage writes files such as:

```text
summary_df.csv
<case_name>_alpha_raw.npy
<case_name>_alpha_corrected.npy
<case_name>_plume_mask.npy
<case_name>_background_mask.npy
<case_name>_stripe_map.npy
<case_name>_threshold_history.csv
```

The remaining-line analysis writes files such as:

```text
detected_remaining_line_equations_with_angles.csv
detected_remaining_line_candidates_with_angles.csv
detected_remaining_line_angle_summary.csv
remaining_line_candidates_ranked.csv
remaining_line_sample_pixels.csv
remaining_line_sample_spectra_from_map_csv.csv
remaining_line_spectra_all_wavelengths.png
remaining_line_spectra_swir_mf_range.png
remaining_line_overlay.png
```

These generated outputs can become large. It is recommended to keep them out of Git using `.gitignore`.


---

## Notes on the Three Scripts

### `paper_sensor_geometry_iterative_mf_destriping_full.py`

This is the main MF and destriping script.

It includes:

- data loading helpers,
- MODTRAN and UAS helpers,
- Matched Filter helpers,
- robust thresholding,
- image-space diagonal destriping,
- sensor-geometry destriping,
- angle-sweep cleanup,
- experiment execution and output saving.

This script can be run directly:

```bash
python paper_sensor_geometry_iterative_mf_destriping_full.py
```

However, for the full line-analysis workflow, using `mf_line_spectrum_pipeline.py` is recommended.

### `analyze_remaining_plume_lines_and_spectra.py`

This script analyzes remaining line-like detections after the MF/destriping workflow.

It can:

- detect line candidates from plume masks,
- estimate line equations,
- choose representative line pixels,
- choose nearby control pixels,
- extract spectra from a map CSV or HISUI TIF,
- save line overlays and spectrum plots.

It can be run directly after the MF outputs already exist:

```bash
python analyze_remaining_plume_lines_and_spectra.py
```

### `mf_line_spectrum_pipeline.py`

This is the integrated runner.

It imports the two other scripts from the same directory and configures them from one place.

Use this script when you want to reproduce the full workflow from MF processing to remaining-line spectral analysis.

```bash
python mf_line_spectrum_pipeline.py
```


---

## Research Interpretation

The outputs from this workflow should be interpreted carefully.

Line-like detections may correspond to:

- real plume structures,
- residual striping,
- sensor-geometry artifacts,
- geolocation or resampling artifacts,
- one-band spectral anomalies,
- surface reflectance structures.

Therefore, a line-like detection should not be interpreted as methane only from the MF score. It should be checked using:

- spectrum shape,
- spatial continuity,
- comparison with control pixels,
- consistency with wind direction,
- source location,
- comparison across destriping cases.

---

## Minimal Run Example

```bash
cd Line_detection
python mf_line_spectrum_pipeline.py
```

Before running, edit the path settings in `mf_line_spectrum_pipeline.py`.

If you only want to analyze already generated MF results:

```python
RUN_MF_STAGE = False
RUN_LINE_SPECTRUM_STAGE = True
```

Then run:

```bash
python mf_line_spectrum_pipeline.py
```
