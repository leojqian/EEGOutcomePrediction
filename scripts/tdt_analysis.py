#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from typing import Iterable

from functions import get_eeg_data_by_id, parse_header_and_data


def analyze_folders(folders: Iterable[str], report_dir: str) -> None:
    os.makedirs(report_dir, exist_ok=True)
    hardware_counts = Counter()
    sampling_counts = Counter()
    eyes_counts = Counter()

    per_file_rows = []

    for folder in folders:
        if not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            if not name.lower().endswith(".txt"):
                continue
            file_path = os.path.join(folder, name)
            header, columns, _ = parse_header_and_data(file_path, load_data=False)
            hardware = header.get("Collection Hardware", "UNKNOWN")
            sampling_rate = header.get("Sampling Rate", "UNKNOWN")
            eyes = header.get("Eyes Condition", "UNKNOWN")
            hardware_counts[hardware] += 1
            sampling_counts[sampling_rate] += 1
            eyes_counts[eyes] += 1

            per_file_rows.append(
                {
                    "file_path": file_path,
                    "collection_hardware": hardware,
                    "sampling_rate": sampling_rate,
                    "eyes_condition": eyes,
                    "channels": len(columns),
                }
            )

    summary_path = os.path.join(report_dir, "tdt_data_analysis.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "collection_hardware_counts": dict(hardware_counts),
                "sampling_rate_counts": dict(sampling_counts),
                "eyes_condition_counts": dict(eyes_counts),
                "total_files": sum(hardware_counts.values()),
            },
            handle,
            indent=2,
        )

    csv_path = os.path.join(report_dir, "tdt_file_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "file_path",
                "collection_hardware",
                "sampling_rate",
                "eyes_condition",
                "channels",
            ],
        )
        writer.writeheader()
        writer.writerows(per_file_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze TDT text EEG files.")
    parser.add_argument(
        "--folders",
        nargs="+",
        default=[
            "data/tdt files Baseline only NFB+DUL who completed T2",
            "data/tdt files Baseline Only NFB who completed T2",
        ],
        help="Folders to scan",
    )
    parser.add_argument(
        "--report-dir",
        default="reports",
        help="Directory to write reports",
    )
    parser.add_argument("--patient", help="Patient ID to load data for")
    args = parser.parse_args()

    analyze_folders(args.folders, args.report_dir)

    if args.patient:
        results = get_eeg_data_by_id(args.patient, args.folders)
        if not results:
            print("No files found for patient.")
            return
        for path, payload in results.items():
            data = payload["data"]
            rows = len(data) if data is not None else 0
            cols = len(payload["columns"])
            print(f"{path}: rows={rows} cols={cols}")


if __name__ == "__main__":
    main()
