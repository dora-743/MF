from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pandas as pd
ROOT_DIR = Path()
OUT_DIR = ROOT_DIR / "normalized"

START_ROW_EXCEL = 7
WAVEL_COL_EXCEL = 1
RADIANCE_COL_EXCEL = 10

ENCODINGS = ("utf-8-sig","cp932","utf-8","latin-1")
