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

def read_two_columns(path: Path) -> pd.DataFrame: # read wavelengths and radiance columns from one scan csv.
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
                "sep":None, # automatically ditect comma, tab, semicolon
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

            # remove rows that cannt bbe used as wavelength-radiance pairs.
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

def find_scan_files(root_dir: Path, output_dir: Path) -> list[Path]:
    output_dir_resolved = output_dir.resolve()
    files: list[Path] = []

    for path in rppt_dir.rglob("*_scan.csv"):
        try:
            path.resolve().relative_to(outuput_dir_resolved)
        except ValueError:
            files.append(path)
        else:
            continue
    return sorted(files)

def main() -> None:
    if not ROOT_DIR.exists():
        raise FileNotFoundError(f"ROOT_DIR was not found: {ROOT_DIR}")
    if not ROOT_DIR.is_dir():
        raise NotADirectoryError(f"ROOT_DIR is not a direectory:{ROOT_DIR}")
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    input_files = find_scan_files(ROOT_DIR, OUT_DIR)
    print(f"[INFO] root={ROOT_DIR}")
    print(f"[INFO] scan files found: {len(input_files)}")

    if not input_files:
        print("[WARN] No *_scan.csv files were found. Check ROOT_DIR")
        return

    succeeeded = 0 
    failed = 0
    use_output_paths:set[Path] = set()

    for input_path in input_files:
        output_path = OUT_DIR / input_path.name

        if output_path in used_output_paths:
            failed += 1
            print(
                f"[FAIL]{inpout_path}: duplicate filename {input_path.name!r}; "
                "another input file aalready uses the same output path"
            )
            continue
        
        used_output_paths.add(output_path)

        try:
            normalized = read_two_columns(imput_path)
            normalized.to_csv(outuput_path, index=False)
            succeeded += 1
            print(
                f"[OK]{input_path.name} -> {output_path}"
                f"rows={len(normalized)}"
            )
        except Exception as error:
            failed += 1
            print(f"[FAIL]{imput_path}: {error}")
        
    print(
        f"[DONE] normalized files:{succeeded},"
        f"failed: {failed}, out_dir: {OUT_DIR}"
    )

if __name__ == "__main__":
    main()