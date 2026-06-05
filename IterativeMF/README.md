## Iterative Matched Filter

### Overview

Iterative Matched Filter, or Iterative MF, is an extension of the standard Matched Filter used for methane plume detection from hyperspectral imagery.

In a standard Matched Filter, the background mean spectrum and covariance matrix are estimated only once. The filter is then applied to all pixels using the fixed background statistics.

However, if methane plume pixels are included in the background pixels, the estimated background statistics can be contaminated by methane absorption features. As a result, methane absorption may be partially treated as background variation, which can reduce the detection sensitivity.

Iterative MF addresses this problem by repeatedly detecting plume candidate pixels and excluding them from the background estimation in the next iteration.

---

### Standard Matched Filter

For each pixel spectrum $x$, the Matched Filter output $\hat{\alpha}$ is computed as:

```math
\hat{\alpha}
=
\frac{(x-\mu)^T \Sigma^{-1} t}
{t^T \Sigma^{-1} t}
```
where:

| Symbol | Description |
|---|---|
| $x$ | observed pixel spectrum |
| $\mu$ | background mean spectrum |
| $\Sigma$ | background covariance matrix |
| $t$ | methane target spectrum |
| $\hat{\alpha}$ | methane enhancement score |

A larger value of $\hat{\alpha}$ indicates that the pixel spectrum is more similar to the methane absorption target.

---

### Methane Target Spectrum

The methane absorption effect is approximated as:

$$
L \approx L_0 \exp(-\alpha s)
$$

where:

| Symbol | Description |
|---|---|
| $L$ | observed radiance spectrum after methane absorption |
| $L_0$ | background radiance spectrum |
| $\alpha$ | methane enhancement amount |
| $s$ | unit absorption spectrum |

For small $\alpha$, this model can be linearized as:

$$
L \approx L_0 - \alpha L_0 s
$$

Therefore, the methane-induced spectral change is approximately:

$$
\Delta L \approx -\alpha L_0 s
$$

In this implementation, the background spectrum $L_0$ is approximated by the background mean spectrum $\mu$. Therefore, the methane target spectrum is defined as:

$$
t = -\mu s
$$

In Iterative MF, $\mu$ is updated at each iteration. Therefore, the target spectrum $t$ is also updated at each iteration.

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
       using pixels in B(k-1):
           mu_k, Sigma_k

    2. Construct methane target spectrum:
           t_k = - mu_k * s

    3. Apply Matched Filter to all valid pixels:
           alpha_k = MF(X; mu_k, Sigma_k, t_k)

    4. Detect plume candidate pixels
       using a robust threshold:
           threshold = median(alpha_k)
                     + n_sigma * 1.4826 * MAD(alpha_k)

    5. Define plume candidate mask:
           P_k = alpha_k > threshold

    6. Update background mask:
           B_k = M excluding P_k

    7. Stop if the plume mask no longer changes.

Output:
    Final methane enhancement map alpha
    Final plume candidate mask P

    Final background mask B
```



### Robust Thresholding

Plume candidate pixels are extracted using a robust threshold based on the median and Median Absolute Deviation, MAD.

```math
\mathrm{threshold}
=
\mathrm{median}(\alpha)
+
n_\sigma
\times
1.4826
\times
\mathrm{MAD}(\alpha)
```

where:

```math
\mathrm{MAD}(\alpha)
=
\mathrm{median}
\left(
|\alpha - \mathrm{median}(\alpha)|
\right)
```

The factor `1.4826` converts MAD to a standard-deviation-like scale under the assumption of a normal distribution.

This robust threshold is used instead of the mean and standard deviation because the MF output may contain strong plume pixels or abnormal pixels. These outliers can strongly affect the mean and standard deviation, while the median and MAD are more robust.

In this implementation, pixels satisfying the following condition are treated as plume candidates:

```math
\alpha > \mathrm{threshold}
```

These plume candidate pixels are excluded from the background estimation in the next iteration.

---

### Purpose of the Iteration

The main purpose of the iteration is to reduce contamination of the background statistics by plume pixels.

In the standard MF, methane plume pixels may be included when estimating the background mean spectrum and covariance matrix. As a result, methane absorption features can be partially treated as background variation.

Iterative MF reduces this problem by repeating the following steps:

1. detecting high-score methane candidate pixels,
2. excluding them from the background mask,
3. re-estimating the background mean spectrum and covariance matrix,
4. reconstructing the methane target spectrum,
5. re-applying the Matched Filter.

This process makes the background statistics less affected by methane plume pixels and can improve plume contrast in the final MF output.

---

### Notes

Although Iterative MF can reduce plume contamination in the background statistics, it can also enhance non-methane spectral anomalies if their spectral shape is correlated with the methane target spectrum.

Examples of possible false positives include:

- bad pixels,
- bad columns,
- striping noise,
- isolated spectral spikes,
- surface reflectance anomalies,
- correction artifacts.

Therefore, detected plume candidates should be checked using additional information such as:

- spectral shape,
- spatial continuity,
- wind direction,
- source location.

In this study, suspicious line-like detections are further investigated by checking whether specific pixels show abnormal radiance at only one wavelength band compared with neighboring wavelength bands.

## Improved Iterative MF with Diagonal Destriping

`improved_iterative_mf_diagonal_destriping.py` provides an extended Iterative Matched Filter workflow that can apply directional destriping to MF alpha maps.

The main use case is reducing line-like artifacts that appear along image diagonals such as `row - col = const`. The script supports several destriping timings:

- `none`: run Iterative MF without destriping,
- `final_only`: apply destriping only after the Iterative MF loop,
- `each_iter`: apply destriping during every iteration before plume thresholding,
- a list such as `[3, 4, 5]`: apply destriping only at selected iterations.

The stripe offset for each diagonal can be estimated using:

- `median`,
- `mean`,
- `trimmed_mean`,
- `mode`.

A typical command is:

```bash
python IterativeMF/improved_iterative_mf_diagonal_destriping.py \
  --roi-csv all_roi_spectra200x200.csv \
  --modtran-csv CH4c.csv \
  --output-dir outputs_improved_diagonal_destripe
```

The script saves experiment summaries, selected wavelength arrays, selected UAS arrays, alpha maps, corrected alpha maps, stripe maps, plume masks, and per-pixel CSV outputs.

## Improved Iterative MF with two-direction diagonal destriping

`improved_iterative_mf_two_direction_destriping.py` is an additional version of the improved Iterative Matched Filter workflow.
The original one-direction destriping file is kept unchanged. This two-direction version estimates and subtracts stripe-like offsets in two line directions:

- `y_minus_x`: lines parallel to `y=x`, represented by `row - col = const`
- `y_plus_x`: lines parallel to `y=-x`, represented by `row + col = const`

The default direction order is:

```python
["y_minus_x", "y_plus_x"]
```

This means the script first corrects `y=x`-parallel stripes and then corrects `y=-x`-parallel stripes on the already corrected alpha map.

Typical execution:

```bash
python IterativeMF/improved_iterative_mf_two_direction_destriping.py   --roi-csv all_roi_spectra200x200.csv   --modtran-csv CH4c.csv   --output-dir outputs_two_direction_diagonal_destripe
```

Useful options:

```bash
# Run only the baseline and the main two-direction final-only case
python IterativeMF/improved_iterative_mf_two_direction_destriping.py   --experiments baseline_no_destripe,median_yx_then_ynegx_final_only

# Show diagnostic plots
python IterativeMF/improved_iterative_mf_two_direction_destriping.py   --show-plots
```

Because two-direction destriping is stronger than one-direction destriping, always compare:

- the raw alpha map,
- the total stripe map,
- each direction-wise stripe map,
- the corrected alpha map,
- the plume candidate mask.

If the correction looks too strong, prefer `final_only` first, or reduce the aggressiveness by changing `exclude_mode`, `smooth_half_window`, or the selected experiment set.
