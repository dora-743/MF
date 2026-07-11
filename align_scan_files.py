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

def supports_on_bad_lines() -> bool:
    return "on_bad_lines" in inspect.signature(pd.read_csv).parameters

def read_two_columns(path: Path) -> pd.DataFrame:
    skiprows = START_ROW_EXCEL - 1
    wavelength_index = WAVEL_COL_EXCEL - 1
    radiance_index = RADIANCE_COL_EXCEL -1
    require_max_indev = max(wavelength_index, radiance_index)

    last_error: Exception | None = None
    for encoding in ENCODINGS:
        try:
            read_kwargs: dicts[str,Any] = {
                "header":None,
                "skiprows":skiprows,
                "encoding":endcoding,
                "engine":"python,
                "sep":None,
            }
            
            if supports_on_bad_lines():
                read_kwargs["on_bad_lines"] = "skip"
            else:
                read_kwargs["error_bad_lines"] = False
                read_kwargs["warn_bad_lines"] = False

            dataframe = pd.read_csv(path, **read_kwargs)
            
            if dataframe.shape[1] <= required_max_index:
                raise ValueError(
                    f"{path.name} has only {dataframe.shape[1] columns;}"
                    f"column {required_max_index + 1} is required."
                )
            normalized = dataframe.iloc[
                :,[wavelength_index, radiance_index]
            ].copy()
            normalized.columns = ["wave_nm", "radiance"]

            normalized["wave_nm"] = pd.to_numeric(
                normalized["wave_nm"], errors="coerce"
            )
            normalized["radiance"]

            normalized = noemalized.dropna(
                subset=["wave_nm","radiance"]
            ).recet_index(drop=True)

            if normalized.empty:
                raise ValueError(
                    f"{path.name} contains no valit wavelength-radiance rows."
                )
            return normalized

        except Exception as error:
            last_error = error

    raise RuntimeError(
        f"failed to read {path} with the candidate encodings."
    ) from last_error