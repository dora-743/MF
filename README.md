# MF

This repository contains Python implementations and experimental tools for methane plume detection from hyperspectral imagery using Matched Filter-based methods.

The main target of this repository is the detection of methane absorption signals in shortwave infrared hyperspectral data. The implementation is based on the Matched Filter framework and related extensions discussed in papers A–E listed below.

---

## Overview

Methane plumes can be detected from hyperspectral imagery because methane has characteristic absorption features in the shortwave infrared region, especially around the 2.3 μm wavelength range.

In hyperspectral methane detection, each pixel has a spectrum. The goal is to determine whether the observed spectrum contains a methane-like absorption signature compared with the surrounding background spectra.

This repository mainly focuses on:

* standard Matched Filter detection,
* methane target spectrum construction using unit absorption spectra,
* MODTRAN-based methane absorption simulation,
* Iterative Matched Filter processing,
* artifact checking such as diagonal spike detection,
* PCA-based inspection of spectral anomalies.

---

## Matched Filter

The Matched Filter, or MF, is a detection method used to find a known target spectral signature in noisy background data.

For each pixel spectrum (x), the MF output (\hat{\alpha}) is computed as:

```math
\hat{\alpha}
=
\frac{(x-\mu)^T \Sigma^{-1} t}
{t^T \Sigma^{-1} t}
```

where:

| Symbol         | Meaning                      |
| -------------- | ---------------------------- |
| $(x)$            | observed pixel spectrum      |
| $(\mu)$          | background mean spectrum     |
| $(\Sigma)$       | background covariance matrix |
| $(t)$            | methane target spectrum      |
| $(\hat{\alpha})$ | methane enhancement score    |

A larger value of (\hat{\alpha}) means that the pixel spectrum is more similar to the methane absorption target.

---

## Methane Target Spectrum

Methane absorption is approximated using the Beer-Lambert law:

```math
L \approx L_0 \exp(-\alpha s)
```

where:

| Symbol   | Meaning                                    |
| -------- | ------------------------------------------ |
| $(L)$      | observed radiance after methane absorption |
| $(L_0)$    | background radiance                        |
| $(\alpha)$ | methane enhancement amount                 |
| $(s)$      | unit absorption spectrum                   |

For small methane enhancement, this can be linearized as:

```math
L \approx L_0 - \alpha L_0 s
```

Therefore, the methane-induced spectral change is approximately:

```math
\Delta L \approx -\alpha L_0 s
```

In this repository, the background radiance (L_0) is approximated by the background mean spectrum (\mu). Therefore, the methane target spectrum is defined as:

```math
t = -\mu s
```

The unit absorption spectrum (s) can be estimated from simulated methane spectra, for example by using MODTRAN spectra with different methane enhancement values.

---

## MODTRAN-Based Unit Absorption Spectrum

This repository includes functions to load MODTRAN-simulated methane spectra and convert them into the wavelength grid of the hyperspectral sensor.

The typical workflow is:

1. Load MODTRAN spectra with different methane enhancement values.
2. Resample the MODTRAN spectra to the sensor wavelength bands.
3. Estimate the unit absorption spectrum using a log-slope method or finite difference method.
4. Construct the methane target spectrum.
5. Apply the Matched Filter to the hyperspectral image cube.

The log-slope method fits the following relationship for each wavelength band:

```math
\ln L = c - \alpha s
```

The unit absorption spectrum is then obtained from the negative slope.

---

## Iterative Matched Filter

In the standard MF, the background mean and covariance are estimated only once. However, if methane plume pixels are included in the background estimation, the methane absorption signal can contaminate the background statistics.

Iterative Matched Filter reduces this problem by repeating the following steps:

1. Estimate background statistics from valid background pixels.
2. Construct the methane target spectrum.
3. Apply the Matched Filter.
4. Detect high-score plume candidate pixels.
5. Exclude plume candidates from the background mask.
6. Re-estimate the background statistics.
7. Repeat until the plume mask converges or the maximum number of iterations is reached.

This process can improve plume contrast because the final background statistics are less affected by methane plume pixels.

---

## Artifact and Anomaly Checking

Hyperspectral data can contain non-methane artifacts that may produce false positives in MF outputs.

Examples include:

* bad pixels,
* bad columns,
* striping noise,
* one-band spectral spikes,
* diagonal line artifacts,
* surface reflectance anomalies,
* correction artifacts.

This repository includes additional tools for checking such artifacts:

| File                               | Purpose                                                      |
| ---------------------------------- | ------------------------------------------------------------ |
| `tools/PCA.py`                     | PCA-based spectral anomaly inspection                        |
| `tools/tolerantdetector.py`        | tolerant diagonal spike detection                            |
| `tools/diagonal_spike_detector.py` | diagonal one-band spike detection                            |
| `tools/get_spectraldata.py`        | extraction of ROI spectra from HISUI-like hyperspectral data |
| `tools/MF_function.py`             | basic MF-related functions                                   |
| `IterativeMF/Iterative_MF.py`      | standard and iterative MF processing                         |
| `IterativeMF/saveresults.py`       | saving pixel-wise MF results                                 |
| `IterativeMF/show_difference.py`   | spectrum comparison for selected pixels                      |

These tools are intended to support interpretation of methane detection results and to identify possible false positives.

---

## Reference Papers

This repository is developed with reference to the following papers.

### A. He et al. (2026) — SC-LMMF

**Improved Quantification of Methane Point-Source Emissions from Hyperspectral Imagery Using a Spectrally Corrected Levenberg–Marquardt Matched Filter**

This paper is used as a reference for spectrally corrected matched filtering and methane plume quantification.

### B. Pei et al. (2023) — ILMF

**Improving Quantification of Methane Point Source Emissions from Imaging Spectroscopy**

This paper is used as a reference for iterative matched filtering and improved methane enhancement estimation.

### C. Li et al. (2025) — SSRMF

**SSRMF: A Sparse Spectral Reconstruction Enhanced Matched Filter for Improving Point-Source Methane Emission Detection in Complex Terrain**

This paper is used as a reference for improving MF robustness in complex terrain and difficult background conditions.

### D. Roger et al. (2024) — Combo-MF

**Exploiting the Entire Near-Infrared Spectral Range to Improve the Detection of Methane Plumes with High-Resolution Imaging Spectrometers**

This paper is used as a reference for using a wider spectral range and combining methane absorption information across wavelengths.

### E. Liang et al. (2025) — MLMF

**An Effective Quantification of Methane Point-Source Emissions with the Multi-Level Matched Filter from Hyperspectral Imagery**

This paper is used as a reference for multi-level matched filtering and methane plume quantification.

---

## Notes

The code in this repository is an experimental research implementation. It is not intended to be a complete reproduction of any single paper. Instead, it collects and tests core ideas from the referenced Matched Filter-based methane detection methods.

The results should be interpreted carefully. High MF values do not always indicate methane. Detected plume candidates should be checked using additional information such as:

* spectral shape,
* spatial continuity,
* wind direction,
* source location,
* comparison with neighboring wavelength bands,
* artifact detection results.

---

## Typical Workflow

A typical workflow is:

1. Extract hyperspectral ROI spectra from the original image.
2. Convert the spectra into an image cube.
3. Load or simulate methane absorption spectra.
4. Estimate the unit absorption spectrum.
5. Select the target wavelength range.
6. Apply standard MF or Iterative MF.
7. Save the methane enhancement map and plume candidate pixels.
8. Check suspicious detections using PCA and diagonal spike detection tools.

---
