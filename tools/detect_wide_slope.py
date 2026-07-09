from pathlib import Path

# 解析したい2D alpha画像
ALPHA_PATH = Path(r"D:/research/code/outputs_paper_sensor_geometry_destripe/baseline_no_destripe_alpha_corrected.npy")

# 出力先
OUTPUT_DIR = Path(r"D:/research/code/outputs_general_scene_slope_detection")

# 必要なら有効画素マスクを指定。不要なら None。
VALID_MASK_PATH = None

# CSV入力の場合だけ、alpha値の列名を明示。npyなら None のまま
CSV_VALUE_COLUMN = None

# 正の傾きを角度で全探索。1-89 degで正の有限傾きを全域探索。
ANGLE_MIN_DEG = 10.0
ANGLE_MAX_DEG = 80.0
ANGLE_STEP_DEG = 0.1

# 1.22と1.23のように近い傾きを同じピークとして扱うための分離幅。
MIN_ANGLE_SEPARATION_DEG = 1.0
MIN_SLOPE_SEPARATION = 0.05

# 細い高alpha線の検出設定。1-2 px程度の線を拾うため、thin側は細いbin幅・細かいsampleに
THIN_LINE_BIN_WIDTH = 2.0
THIN_SAMPLE_STEP = 1
THIN_HIGH_NSIGMA = 4.0
THIN_LINES_PER_SLOPE = 6
THIN_MIN_HIGH_PIXELS_PER_LINE = 4

BROAD_LINE_BIN_WIDTH = 18.0
BROAD_MIN_PIXELS_PER_LINE = 80

# 計算量調整
BROAD_SAMPLE_STEP = 4

# 線ごとの代表値。 median を標準
STATISTIC_METHOD = "median"

# 高alpha線そのものを検出したい場合は False。実プルームが強すぎる場合は True を
EXCLUDE_HIGH_ALPHA_FROM_OFFSET_ESTIMATE = False

# 右肩上がりの背景トレンドではなく、局所的に突出したピークだけを選ぶ設定。
SELECT_LOCAL_PEAKS_ONLY = True
PEAK_TREND_WINDOW_DEG = 8.0
PEAK_LOCAL_WINDOW_DEG = 1.0
PEAK_EDGE_EXCLUSION_DEG = 1.0
PEAK_MIN_PROMINENCE = None

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ALPHA_PATH, OUTPUT_DIR

import importlib.util
import sys

script_candidates = [
    Path.cwd() / "detect_general_scene_stripe_slopes.py",
    Path.cwd() / "outputs" / "detect_general_scene_stripe_slopes.py",
    Path(r"C:/Users/yudon/Documents/Codex/2026-06-05/files-mentioned-by-the-user-improved/outputs/detect_general_scene_stripe_slopes.py"),
]
SCRIPT_PATH = next((p for p in script_candidates if p.exists()), None)
if SCRIPT_PATH is None:
    raise FileNotFoundError("detect_general_scene_stripe_slopes.py が見つかりません。")

spec = importlib.util.spec_from_file_location("general_slope_detector", SCRIPT_PATH)
detector = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = detector
spec.loader.exec_module(detector)

print(SCRIPT_PATH)

cfg = detector.SlopeDetectionConfig(
    alpha_path=ALPHA_PATH,
    output_dir=OUTPUT_DIR,
    valid_mask_path=VALID_MASK_PATH,
    csv_value_column=CSV_VALUE_COLUMN,
    angle_min_deg=ANGLE_MIN_DEG,
    angle_max_deg=ANGLE_MAX_DEG,
    angle_step_deg=ANGLE_STEP_DEG,
    thin_top_k=1,
    broad_top_k=1,
    top_k_primary=2,
    min_angle_separation_deg=MIN_ANGLE_SEPARATION_DEG,
    min_slope_separation=MIN_SLOPE_SEPARATION,
    line_bin_width=BROAD_LINE_BIN_WIDTH,
    min_pixels_per_line=BROAD_MIN_PIXELS_PER_LINE,
    sample_step=BROAD_SAMPLE_STEP,
    thin_line_bin_width=THIN_LINE_BIN_WIDTH,
    thin_sample_step=THIN_SAMPLE_STEP,
    thin_high_nsigma=THIN_HIGH_NSIGMA,
    thin_lines_per_slope=THIN_LINES_PER_SLOPE,
    thin_min_high_pixels_per_line=THIN_MIN_HIGH_PIXELS_PER_LINE,
    broad_line_bin_width=BROAD_LINE_BIN_WIDTH,
    broad_min_pixels_per_line=BROAD_MIN_PIXELS_PER_LINE,
    broad_sample_step=BROAD_SAMPLE_STEP,
    statistic_method=STATISTIC_METHOD,
    exclude_high_alpha_from_offset_estimate=EXCLUDE_HIGH_ALPHA_FROM_OFFSET_ESTIMATE,
    select_local_peaks_only=SELECT_LOCAL_PEAKS_ONLY,
    peak_trend_window_deg=PEAK_TREND_WINDOW_DEG,
    peak_local_window_deg=PEAK_LOCAL_WINDOW_DEG,
    peak_edge_exclusion_deg=PEAK_EDGE_EXCLUSION_DEG,
    peak_min_prominence=PEAK_MIN_PROMINENCE,
    verbose=True,
)

result = detector.run_detection(cfg)

selected_primary = result["selected_primary"]
display_cols = ["selected_rank", "detection_type", "slope", "angle_deg", "score"]
extra_cols = [c for c in ["selection_score", "peak_prominence", "is_local_peak", "sum_selected_high_pixels", "top_high_pixels", "before_robust_std", "after_robust_std", "n_lines_scored"] if c in selected_primary.columns]
selected_primary[display_cols + extra_cols]

six_directions = result["six_directions"]
six_directions[[
    "family_rank",
    "direction_type",
    "signed_slope",
    "angle_deg_0_180",
    "direction_key_for_existing_code",
    "slope_parameter_for_existing_code",
    "equation_form",
]]

from IPython.display import Image, display

thin_curve = OUTPUT_DIR / "thin_slope_score_curve.png"
broad_curve = OUTPUT_DIR / "broad_slope_score_curve.png"
overlay = OUTPUT_DIR / "detected_slope_overlay.png"
if not overlay.exists():
    overlay = OUTPUT_DIR / "detected_six_slope_overlay.png"

if thin_curve.exists():
    display(Image(filename=str(thin_curve)))
if broad_curve.exists():
    display(Image(filename=str(broad_curve)))
if overlay.exists():
    display(Image(filename=str(overlay)))
