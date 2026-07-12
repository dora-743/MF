## 二吸収帯整合型 SC-LMMF

本研究では、ハイパースペクトル画像からのメタン濃度増分推定を高精度化するため、1.6 µm帯と2.3 µm帯の二つのメタン吸収帯を同時に利用する「二吸収帯整合型 SC-LMMF（Dual-Window Consistency-Constrained SC-LMMF）」を検討している。

従来のMatched Filter（MF）では、単一の吸収帯を用いる場合が多く、地表面反射率の変動、センサノイズ、大気条件の不一致、吸収の非線形性などによって、偽陽性や濃度推定誤差が生じることがある。

メタンによる濃度増分は、1.6 µm帯と2.3 µm帯の両方に共通して現れる物理量である。そこで本手法では、両吸収帯に共通のメタン濃度増分を仮定し、二つの波長領域を同時にフィッティングする。

### 基本モデル

背景大気中のメタン濃度を $c_{\mathrm{bg}}$、背景濃度からのメタン濃度増分を $\Delta c_i$ とする。

画素 $i$ の波長帯 $b$ における観測スペクトルは、次のようにモデル化する。

```math
L_i^{(b)}(\lambda)
=
R_i^{(b)}(\lambda)
T^{(b)}(\lambda,\Delta c_i)
+
\varepsilon_i^{(b)}(\lambda)
```

ここで、

* $L_i^{(b)}$：観測放射輝度
* $R_i^{(b)}$：メタン濃度増分が存在しない場合の背景スペクトル
* $\Delta c_i$：背景からのメタン濃度増分
* $T^{(b)}(\lambda,\Delta c_i)$：MODTRAN LUTから求めた放射輝度比
* $\varepsilon_i^{(b)}$：背景モデル誤差および観測ノイズ

である。

MODTRAN LUTの濃度軸には、大気中メタンの絶対濃度を使用する。放射輝度比は、背景メタン濃度 $c_{\mathrm{bg}}$ を基準として次式で計算する。

```math
T(\lambda,\Delta c)
=
\frac{
L_{\mathrm{MODTRAN}}
\left(\lambda,c_{\mathrm{bg}}+\Delta c\right)
}{
L_{\mathrm{MODTRAN}}
\left(\lambda,c_{\mathrm{bg}}\right)
}
```

この比を用いることで、MODTRANと観測データの放射輝度単位に一定倍率の差がある場合でも、その倍率は分子と分母で相殺される。

### 二吸収帯の同時推定

1.6 µm帯と2.3 µm帯に共通の濃度増分 $\Delta c_i$ を仮定し、次の目的関数を最小化する。

```math
\begin{aligned}
J_i(\Delta c_i)
={}&
\frac{1}{2}
\left\|
L_i^{(1.6)}
-
R_i^{(1.6)}
\odot
T^{(1.6)}(\Delta c_i)
\right\|_{\Sigma_{1.6}^{-1}}^2
\\
&+
\frac{1}{2}
\left\|
L_i^{(2.3)}
-
R_i^{(2.3)}
\odot
T^{(2.3)}(\Delta c_i)
\right\|_{\Sigma_{2.3}^{-1}}^2 .
\end{aligned}
```

ここで、$`\odot`$ は要素ごとの積を表し、$`\Sigma_{1.6}`$ および $`\Sigma_{2.3}`$ は各吸収帯における背景スペクトルの共分散行列を表す。

推定値は次式で表される。

```math
\widehat{\Delta c}_i
=
\underset{\Delta c_i}{\mathrm{arg\,min}}
\;
J_i(\Delta c_i)
```


同一の $\Delta c_i$ を二つの吸収帯に適用することで、片方の吸収帯だけに現れる地表面由来のスペクトル変動やノイズの影響を抑制することを目指す。

### 吸収帯間の整合性評価

各吸収帯から独立に推定した濃度増分を、それぞれ次のように表す。

```math
\widehat{\Delta c}_i^{(1.6)},
\qquad
\widehat{\Delta c}_i^{(2.3)}
```

両者の差を用いて、吸収帯間の不整合スコアを計算する。

```math
C_i
=
\frac{
\left|
\widehat{\Delta c}_i^{(1.6)}
-
\widehat{\Delta c}_i^{(2.3)}
\right|
}{
\sigma_{1.6}
+
\sigma_{2.3}
+
\epsilon
}
```

ここで、$`\sigma_{1.6}`$ および $`\sigma_{2.3}`$ は各吸収帯における濃度推定値の不確かさを表し、$`\epsilon`$ はゼロ除算を防ぐための微小値である。

不整合スコアが小さい画素は、二つのメタン吸収帯が同じ濃度増分を示しているため、メタン検出としての信頼性が高いと考えられる。

一方、不整合スコアが大きい画素では、地表面アーティファクト、低SNR、大気条件の不一致、背景スペクトル推定誤差などが疑われる。

### 合成メタンプルームによる評価

本リポジトリでは、実際のHISUI観測スペクトルに既知のメタン濃度増分を注入する評価機能を実装している。

合成プルームは、注入前のHISUIスペクトル $L_{\mathrm{bg}}$ にMODTRAN LUTから求めた放射輝度比を乗じて生成する。

```math
L_{\mathrm{syn}}(x,y,\lambda)
=
L_{\mathrm{bg}}(x,y,\lambda)
T\left(
\lambda,
\Delta c_{\mathrm{true}}(x,y)
\right)
```

この方法により、実観測画像が持つ地表面スペクトル、センサノイズ、空間的不均一性を保持したまま、真値が既知のメタンプルームを生成できる。

推定結果は、注入した真値 $\Delta c_{\mathrm{true}}$ と比較し、次の指標で評価する。

* Bias
* MAE
* RMSE
* 回帰直線の傾き
* 決定係数 $R^2$
* Precision
* Recall
* F1 score

### 現在の実装状況

現在の実装には、次の処理が含まれている。

* MODTRAN絶対メタン濃度LUTの読み込み
* HISUIの波長・装置関数への変換
* 背景濃度を基準とした濃度増分LUTの生成
* 実HISUI画像への合成メタンプルーム注入
* 1.6 µm帯単独の線形MF
* 2.3 µm帯単独の線形MF
* 二吸収帯共通濃度によるLUTグリッド探索
* 非線形最小二乗法による連続濃度値への精密化
* 吸収帯間不整合スコアの計算
* 真値に対する定量評価

現在は、注入前のHISUIスペクトルを真の背景スペクトルとして利用するOracle背景評価を行っている。

この段階では、背景推定誤差を含めずに、MODTRAN LUT、吸収モデル、二吸収帯融合および非線形推定の妥当性を評価できる。

### 今後の拡張

完全な二吸収帯整合型SC-LMMFに向けて、次の機能を追加する予定である。

* Iterative MFによる背景画素の選別
* SVDまたはSSRMF型の画素別背景再構成
* フル共分散行列または縮小共分散行列の導入
* 波長帯ごとの放射輝度ゲイン補正
* スペクトル傾きおよびオフセット補正
* 波長シフトおよび装置関数幅の補正
* 水蒸気などの干渉成分との同時推定
* 二吸収帯間の整合性ペナルティ
* 画素ごとの推定不確かさに基づく帯域重み付け
* 風向情報を利用した空間正則化

現在のコードは、二吸収帯整合型SC-LMMFの中核となる共通濃度推定と、合成メタンプルームによる検証基盤を提供するものである。


## Dual-Window Consistency-Constrained SC-LMMF

This project investigates a Dual-Window Consistency-Constrained Spectrally Corrected Levenberg–Marquardt Matched Filter (SC-LMMF) for improving methane-enhancement retrieval from hyperspectral imagery.

Conventional matched-filter approaches often use a single methane absorption window. Their retrieval accuracy can be degraded by surface reflectance variability, sensor noise, atmospheric mismatch, background contamination, and the nonlinear relationship between methane concentration and radiance.

A methane enhancement is a common physical quantity that affects both the 1.6 µm and 2.3 µm methane absorption windows. The proposed method therefore estimates a shared methane enhancement by fitting both spectral windows simultaneously.

### Basic model

Let $`c_{\mathrm{bg}}`$ denote the background atmospheric methane concentration, and let $`\Delta c_i`$ denote the methane enhancement above the background concentration at pixel $`i`$.

The observed spectrum in spectral window $`b`$ is modeled as

```math
L_i^{(b)}(\lambda)
=
R_i^{(b)}(\lambda)
T^{(b)}(\lambda,\Delta c_i)
+
\varepsilon_i^{(b)}(\lambda)
```

where

* $`L_i^{(b)}`$ is the observed radiance,
* $`R_i^{(b)}`$ is the plume-free background spectrum,
* $`\Delta c_i`$ is the methane enhancement above the background concentration,
* $`T^{(b)}(\lambda,\Delta c_i)`$ is a radiance ratio derived from a MODTRAN lookup table,
* $`\varepsilon_i^{(b)}`$ represents background-model error and observation noise.

The concentration axis of the MODTRAN lookup table represents the absolute atmospheric methane concentration. The radiance ratio is calculated relative to the background concentration:

```math
T(\lambda,\Delta c)
=
\frac{
L_{\mathrm{MODTRAN}}
\left(\lambda,c_{\mathrm{bg}}+\Delta c\right)
}{
L_{\mathrm{MODTRAN}}
\left(\lambda,c_{\mathrm{bg}}\right)
}
```

Because the same radiance scaling factor appears in both the numerator and denominator, a constant unit-conversion factor between the MODTRAN and HISUI radiances is canceled in the ratio.

### Joint retrieval using two absorption windows

A common methane enhancement $`\Delta c_i`$ is used for both the 1.6 µm and 2.3 µm windows. The retrieval minimizes the following objective function:

```math
\begin{aligned}
J_i(\Delta c_i)
={}&
\frac{1}{2}
\left\lVert
L_i^{(1.6)}
-
R_i^{(1.6)}
\odot
T^{(1.6)}(\Delta c_i)
\right\rVert_{\Sigma_{1.6}^{-1}}^2
\\
&+
\frac{1}{2}
\left\lVert
L_i^{(2.3)}
-
R_i^{(2.3)}
\odot
T^{(2.3)}(\Delta c_i)
\right\rVert_{\Sigma_{2.3}^{-1}}^2 .
\end{aligned}
```

Here, $`\odot`$ denotes element-wise multiplication, and $`\Sigma_{1.6}`$ and $`\Sigma_{2.3}`$ denote the covariance matrices for the two absorption windows.

The methane-enhancement estimate is

```math
\widehat{\Delta c}_i
=
\underset{\Delta c_i}{\mathrm{arg\,min}}
\;
J_i(\Delta c_i)
```

Using the same methane enhancement in both windows introduces a physical consistency constraint. The goal is to suppress spectral anomalies that appear in only one absorption window, such as surface-related artifacts or wavelength-dependent noise.

### Inter-window consistency

Independent methane-enhancement estimates can also be obtained from the two absorption windows:

```math
\widehat{\Delta c}_i^{(1.6)},
\qquad
\widehat{\Delta c}_i^{(2.3)}
```

An inter-window inconsistency score is then defined as

```math
C_i
=
\frac{
\left|
\widehat{\Delta c}_i^{(1.6)}
-
\widehat{\Delta c}_i^{(2.3)}
\right|
}{
\sigma_{1.6}
+
\sigma_{2.3}
+
\epsilon
}
```

Here, $`\sigma_{1.6}`$ and $`\sigma_{2.3}`$ represent the uncertainties of the methane-enhancement estimates obtained from the two absorption windows. The parameter $`\epsilon`$ is a small positive value used to prevent division by zero.

A small inconsistency score indicates that both methane absorption windows support a similar methane enhancement and therefore increases confidence in the detection.

A large inconsistency score may indicate surface artifacts, a low signal-to-noise ratio, atmospheric mismatch, or errors in background-spectrum estimation.

### Synthetic methane plume injection

This repository includes a synthetic methane-plume injection framework for evaluating methane-retrieval algorithms using real HISUI background spectra.

A synthetic plume is generated by multiplying the original HISUI background spectrum by a MODTRAN-derived radiance ratio:

```math
L_{\mathrm{syn}}(x,y,\lambda)
=
L_{\mathrm{bg}}(x,y,\lambda)
T\left(
\lambda,
\Delta c_{\mathrm{true}}(x,y)
\right)
```

This approach preserves the real surface spectra, sensor noise, and spatial variability of the HISUI image while adding a methane plume with a known enhancement field.

The retrieved enhancement is compared with the injected ground truth $`\Delta c_{\mathrm{true}}`$ using the following metrics:

* Bias
* Mean absolute error
* Root mean squared error
* Regression slope
* Coefficient of determination $`R^2`$
* Precision
* Recall
* F1 score

### Current implementation

The current implementation includes:

* loading a MODTRAN lookup table based on absolute methane concentration,
* convolution and resampling to the HISUI spectral response,
* construction of a methane-enhancement radiance-ratio lookup table,
* injection of synthetic methane plumes into real HISUI observations,
* linear matched-filter retrieval using the 1.6 µm window,
* linear matched-filter retrieval using the 2.3 µm window,
* grid-search retrieval using a shared methane enhancement,
* nonlinear least-squares refinement of the methane enhancement,
* calculation of an inter-window inconsistency score,
* quantitative comparison with the injected ground truth.

The current evaluation uses the original plume-free HISUI spectrum as the true pixel-wise background. This is referred to as an oracle-background evaluation.

The oracle experiment isolates errors associated with the MODTRAN lookup table, the methane absorption model, dual-window fusion, and nonlinear optimization from errors caused by background estimation.

### Planned extensions

The following extensions are planned toward a complete Dual-Window Consistency-Constrained SC-LMMF:

* background-pixel selection using Iterative MF,
* pixel-wise background reconstruction using SVD or an SSRMF-type model,
* full or shrinkage covariance matrices,
* radiance-gain correction for each spectral window,
* spectral-slope and offset correction,
* wavelength-shift and spectral-response-width correction,
* joint retrieval of interfering atmospheric components such as water vapor,
* an explicit inter-window consistency penalty,
* uncertainty-based adaptive weighting of the two absorption windows,
* wind-direction-aware spatial regularization.

The current code provides the core shared-enhancement retrieval framework and a synthetic-plume evaluation environment for developing and validating the complete Dual-Window SC-LMMF.
