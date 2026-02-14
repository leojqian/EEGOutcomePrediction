from __future__ import annotations

import os
import re
from typing import Iterable

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None


HEADER_FIELD_RE = re.compile(r"^([A-Za-z ]+):\s*(.*)$")

__all__ = ["get_eeg_data_by_id", "parse_header_and_data", "find_patient_files"]


def is_numeric_row(tokens: list[str]) -> bool:
    for token in tokens:
        if token == "":
            return False
        try:
            float(token)
        except ValueError:
            return False
    return True


def parse_header_and_data(
    file_path: str, load_data: bool = False
) -> tuple[dict[str, str], list[str], list[list[float]] | "np.ndarray" | None]:
    """
    Parse a NeuroGuide TDT text file.

    Returns:
      - header: dict of key/value header fields
      - columns: list of channel names (tab-delimited header row)
      - data: EEG matrix (rows = time samples, cols = channels)
              If numpy is installed, this is an ndarray of shape
              (n_samples, n_channels). Otherwise, it's a list of
              float rows.
    """
    header: dict[str, str] = {}
    columns: list[str] = []
    data_rows: list[list[float]] = []
    header_done = False

    with open(file_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped == "":
                continue

            if not header_done:
                match = HEADER_FIELD_RE.match(stripped)
                if match:
                    header[match.group(1).strip()] = match.group(2).strip()
                    continue

                tokens = stripped.split("\t")
                if len(tokens) > 1 and not is_numeric_row(tokens):
                    columns = tokens
                    header_done = True
                    continue
                continue

            tokens = stripped.split("\t")
            if not is_numeric_row(tokens):
                continue
            if load_data:
                data_rows.append([float(value) for value in tokens])
            header_done = True

    if load_data and np is not None:
        return header, columns, np.array(data_rows)
    if load_data:
        return header, columns, data_rows
    return header, columns, None


def find_patient_files(patient_id: str, folders: Iterable[str]) -> list[str]:
    matches: list[str] = []
    for folder in folders:
        if not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            if patient_id.upper() in name.upper() and name.lower().endswith(".txt"):
                matches.append(os.path.join(folder, name))
    return sorted(matches)


def get_eeg_data_by_id(
    patient_id: str,
    folders: list[str] | None = None,
) -> dict[str, dict[str, object]]:
    """
    Return EEG data for all matching TDT text files.

    Output format:
      {
        "/path/to/file.txt": {
          "header": { ... },
          "columns": ["FP1-FPz", "FP2-FPz", ...],
          "data": <matrix>
        },
        ...
      }

    Column mapping:
      - data[row_idx][col_idx] corresponds to columns[col_idx].
      - rows are time samples in order of the file.

    Example:
      results = get_eeg_data_by_id("CIPN3103")
      file_path, payload = next(iter(results.items()))
      channels = payload["columns"]
      data = payload["data"]
      print("first channel name:", channels[0])
      print("first sample for that channel:", data[0][0])
    """

    if folders is None:
        folders = [
            "data/tdt files Baseline only NFB+DUL who completed T2",
            "data/tdt files Baseline Only NFB who completed T2",
        ]
    results: dict[str, dict[str, object]] = {}
    for file_path in find_patient_files(patient_id, folders):
        header, columns, data = parse_header_and_data(file_path, load_data=True)
        results[file_path] = {
            "header": header,
            "columns": columns,
            "data": data,
        }
    return results
