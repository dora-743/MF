## Iterative Matched Filter

### Overview

Iterative Matched Filter (Iterative MF) is an extension of the standard Matched Filter used for methane plume detection from hyperspectral imagery.

In a standard Matched Filter, the background mean spectrum and covariance matrix are estimated only once, and then the filter is applied to all pixels. However, if methane plume pixels are included in the background pixels, the background statistics can be contaminated by methane absorption features. This may reduce the sensitivity of the filter and lead to underestimation of methane enhancement.

To reduce this effect, Iterative MF repeatedly detects plume candidate pixels and excludes them from the background estimation in the next iteration.

---

### Standard Matched Filter

For each pixel spectrum (x), the Matched Filter output (\hat{\alpha}) is computed as:

[
\hat{\alpha}
============

\frac{(x-\mu)^T \Sigma^{-1} t}
{t^T \Sigma^{-1} t}
]

where:

| Symbol         | Description                         |
| -------------- | ----------------------------------- |
| (x)            | observed pixel spectrum             |
| (\mu)          | background mean spectrum            |
| (\Sigma)       | background covariance matrix        |
| (t)            | methane target spectrum             |
| (\hat{\alpha}) | estimated methane enhancement score |

A larger (\hat{\alpha}) indicates that the pixel spectrum is more similar to the methane absorption target.

---

### Methane Target Spectrum

The methane absorption effect can be approximated as:

[
L \approx L_0 \exp(-\alpha s)
]

where (L_0) is the background radiance spectrum, (s) is the unit absorption spectrum, and (\alpha) is the methane enhancement amount.

For small (\alpha), this can be linearized as:

[
L \approx L_0 - \alpha L_0 s
]

Therefore, the methane target spectrum is defined as:

[
t = -\mu s
]

where (\mu) is the current background mean spectrum and (s) is the unit absorption spectrum.

In Iterative MF, (\mu) is updated at each iteration, so the target spectrum (t) is also updated.

---

### Algorithm

The Iterative MF procedure is as follows:

```text
Input:
    Hyperspectral image cube X
    Unit absorption spectrum s
    Valid pixel mask M
    Number of iterations K

Initialize:
    Background mask B0 = M

For k = 1, ..., K:

    1. Estimate background statistics
       using pixels in Bk-1:
           μk, Σk

    2. Construct methane target spectrum:
           tk = - μk × s

    3. Apply Matched Filter to all valid pixels:
           αk = MF(X; μk, Σk, tk)

    4. Detect plume candidate pixels
       using a robust threshold:
           threshold = median(αk) + nσ × 1.4826 × MAD(αk)

    5. Define plume candidate mask:
           Pk = αk > threshold

    6. Update background mask:
           Bk = M \ Pk

    7. Stop if the plume mask no longer changes.

Output:
    Final methane enhancement map α
    Final plume candidate mask P
    Final background mask B
```

---

### Robust Thresholding

In this implementation, plume candidate pixels are extracted using a robust threshold based on the median and Median Absolute Deviation (MAD):

[
\mathrm{threshold}
==================

\mathrm{median}(\alpha)
+
n_\sigma
\times
1.4826
\times
\mathrm{MAD}(\alpha)
]

where:

[
\mathrm{MAD}(\alpha)
====================

\mathrm{median}
\left(
|\alpha - \mathrm{median}(\alpha)|
\right)
]

The factor (1.4826) converts MAD to a standard-deviation-like scale under the assumption of a normal distribution.

This robust threshold is used instead of the mean and standard deviation because the MF output may contain strong plume pixels or abnormal pixels. These outliers can strongly affect the mean and standard deviation, while the median and MAD are more robust.

---

### Purpose of the Iteration

The main purpose of the iteration is to reduce contamination of background statistics by plume pixels.

In the standard MF, methane plume pixels may be included when estimating (\mu) and (\Sigma). As a result, methane absorption features can be partially treated as background variation.

Iterative MF reduces this problem by:

1. detecting high-score methane candidate pixels,
2. excluding them from the background mask,
3. re-estimating the background mean and covariance,
4. re-applying the Matched Filter.

This process makes the background statistics less affected by methane plume pixels and can improve plume contrast in the final MF output.

---

### Notes

Although Iterative MF can reduce plume contamination in the background statistics, it can also enhance non-methane spectral anomalies if their spectral shape is correlated with the methane target spectrum.

Examples of possible false positives include:

* bad pixels,
* bad columns,
* striping noise,
* isolated spectral spikes,
* surface reflectance anomalies,
* correction artifacts.

Therefore, detected plume candidates should be checked using additional information such as spectral shape, spatial continuity, wind direction, and source location.

In this study, suspicious line-like detections are further investigated by checking whether specific pixels show abnormal radiance at only one wavelength band compared with neighboring wavelength bands.

