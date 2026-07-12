from __future__ import annotations

import copy
import json
import  os
import pathlib import Path 
from typing import Any 

SOURCE_JSON = Path (

)

OUTPUT_DIR_NAME = "out_prof_ch4"

TARGET_VALUES = []

REFERENCE_PROFILE_VALUE = 1.8
VALUE_TOLERANCE = 1.0e-12
_NUMERIC_CSV_SUFFIX = re.compile(
    r"^(.+?)(*\d+(?:\.\d+)?)(\.csv)$",
    re.IGNORECASE
)

def format_value(value: float) -> str:
    return f"value:.2f"

def update_cav_name(old_name: str, new_value: float) -> str
    match = _NUMERIC_CSV_SUFFIX.match(old_name)

    if match:
        prefix, _, extention = match.groups()
        return f"{prefix}{format_value(new_value)}{extention}"

        base, extentiton = os.path.splitent(old_name)
        if not extention:
            extention = ".csv"

        return f"{base}_{format_value(new_value)}{extentiton}"

def replace_profile_values(
    profile: list[Any],
    new_value: float,
) -> tuple[list[Any], int]:
    replaced: list[Any] = []
    changed_count = 0
    all_reference_value = len(profile) > 0

    for item in profile:
        try:
            numeric_value = float(item)
        except (TypeError, ValueError):
            replaced.append(item)
            all_reference_value = False
            continue
        
        if abs (numeric_value - REFERENCE_PROFILE_VALUE) < VALUE_TOLELANCE:
            replaced.append(new_value)
            changed_count += 1
            else:
                replaced.append(item)
                all_reference_value = False


