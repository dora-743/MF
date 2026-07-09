from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


DEFAULT_INPUT_DIR = Path(
    r"D:\research\code\outputs_detected_slopes_orthogonal_thin_eachiter_then_broad_median"
)
DEFAULT_OUTPUT_DIR = Path(r"D:\research\code\takaku_wavelet_mf_destriping\outputs")

THIN_SLOPE = 0.9792723507257799 # can find by QA data
BROAD_SLOPE = 1.257172298918948 # 
