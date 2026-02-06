#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from xml.etree import ElementTree

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None

try:
    import mne
except ImportError:  # pragma: no cover - optional dependency
    mne = None

try:
    import pyedflib
except ImportError:  # pragma: no cover - optional dependency
    pyedflib = None


TEST_DIR_RE = re.compile(r"\b(T\d+)\b", re.IGNORECASE)
PATIENT_ID_RE = re.compile(r"(CIPN\d{4})", re.IGNORECASE)


def find_patient_id(path_name: str) -> str | None:
    match = PATIENT_ID_RE.search(path_name)
    return match.group(1).upper() if match else None


def find_test_dirs(patient_dir: str) -> dict[str, list[str]]:
    tests = defaultdict(list)
    for root, dirs, _files in os.walk(patient_dir):
        for dirname in dirs:
            match = TEST_DIR_RE.search(dirname)
            if match:
                test = match.group(1).upper()
                tests[test].append(os.path.join(root, dirname))
        # Only recurse into the first level to avoid deep duplicates
        break
    return tests


def get_ini_shape(path: str) -> tuple[int, int]:
    rows = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("["):
                continue
            if "=" in stripped:
                rows += 1
    cols = 2 if rows > 0 else 0
    return rows, cols


def get_montage_shape(path: str) -> tuple[int, int]:
    tree = ElementTree.parse(path)
    root = tree.getroot()
    channels = root.findall(".//channel")
    rows = len(channels)
    cols = 0
    for channel in channels:
        cols = max(cols, len(channel.attrib))
    return rows, cols


def get_edf_shape(path: str) -> tuple[int, int] | None:
    if mne is not None:
        raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
        return raw.n_times, len(raw.ch_names)
    if pyedflib is not None:
        reader = pyedflib.EdfReader(path)
        try:
            rows = max(reader.getNSamples())
            cols = reader.signals_in_file
            return rows, cols
        finally:
            reader.close()
    return None


def file_shape(path: str) -> tuple[int | None, int | None, str | None]:
    _, ext = os.path.splitext(path)
    ext = ext.lower()
    try:
        if ext == ".ini":
            rows, cols = get_ini_shape(path)
            return rows, cols, None
        if ext == ".mont":
            rows, cols = get_montage_shape(path)
            return rows, cols, None
        if ext == ".edf":
            shape = get_edf_shape(path)
            if shape is None:
                return None, None, "edf_reader_missing"
            return shape[0], shape[1], None
    except Exception as exc:  # pragma: no cover - defensive
        return None, None, f"parse_error:{type(exc).__name__}"
    return None, None, "unsupported_type"


def file_size_bytes(path: str) -> int | None:
    try:
        return os.path.getsize(path)
    except OSError:
        return None


def list_files_in_dir(directory: str) -> list[str]:
    return [
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, name))
    ]


def quality_check(
    root_dir: str, report_dir: str
) -> tuple[dict[str, dict[str, bool]], Counter[str]]:
    os.makedirs(report_dir, exist_ok=True)
    patient_summary: dict[str, dict[str, bool]] = {}
    issues = []
    rows = []
    test_counts: Counter[str] = Counter()
    shape_index: dict[tuple[str, str], list[tuple[int, int, str]]] = defaultdict(list)

    for patient_folder in sorted(os.listdir(root_dir)):
        patient_dir = os.path.join(root_dir, patient_folder)
        if not os.path.isdir(patient_dir):
            continue
        patient_id = find_patient_id(patient_folder)
        if not patient_id:
            continue

        tests = find_test_dirs(patient_dir)
        patient_summary[patient_id] = {}
        for test in sorted(tests):
            patient_summary[patient_id][test] = True
            test_counts[test] += 1

        for test, test_dirs in tests.items():
            for test_dir in test_dirs:
                files = list_files_in_dir(test_dir)
                for file_path in files:
                    base_name = os.path.basename(file_path)
                    if patient_id not in base_name.upper():
                        issues.append(
                            {
                                "patient_id": patient_id,
                                "test": test,
                                "file": file_path,
                                "issue": "missing_patient_id_in_filename",
                            }
                        )
                    row_count, col_count, note = file_shape(file_path)
                    size_bytes = file_size_bytes(file_path)
                    rows.append(
                        {
                            "patient_id": patient_id,
                            "patient_dir": patient_dir,
                            "test": test,
                            "file_path": file_path,
                            "extension": os.path.splitext(file_path)[1].lower(),
                            "rows": row_count,
                            "cols": col_count,
                            "note": note or "",
                            "size_bytes": size_bytes,
                        }
                    )
                    if row_count is not None and col_count is not None:
                        shape_index[(test, os.path.splitext(file_path)[1].lower())].append(
                            (row_count, col_count, file_path)
                        )

    summary_path = os.path.join(report_dir, "data_quality_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "patient_id",
                "patient_dir",
                "test",
                "file_path",
                "extension",
                "rows",
                "cols",
                "note",
                "size_bytes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    structure_stats = {}
    inconsistent = []
    for (test, extension), shapes in shape_index.items():
        counter = Counter((row, col) for row, col, _ in shapes)
        most_common_shape, occurrences = counter.most_common(1)[0]
        structure_stats[f"{test}:{extension}"] = {
            "most_common_rows": most_common_shape[0],
            "most_common_cols": most_common_shape[1],
            "occurrences": occurrences,
            "total_with_shape": len(shapes),
        }
        for row, col, file_path in shapes:
            if (row, col) != most_common_shape:
                inconsistent.append(
                    {
                        "file_path": file_path,
                        "test": test,
                        "extension": extension,
                        "rows": row,
                        "cols": col,
                        "expected_rows": most_common_shape[0],
                        "expected_cols": most_common_shape[1],
                    }
                )

    issues_path = os.path.join(report_dir, "data_quality_issues.json")
    with open(issues_path, "w", encoding="utf-8") as handle:
        json.dump(issues, handle, indent=2)

    report_path = os.path.join(report_dir, "data_quality_report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "patients_total": len(patient_summary),
                "test_counts": dict(sorted(test_counts.items())),
                "structure_stats": structure_stats,
                "inconsistent_structures": inconsistent,
                "naming_issues": issues,
            },
            handle,
            indent=2,
        )

    return patient_summary, test_counts


def compute_fft(data: "np.ndarray", sfreq: float) -> tuple["np.ndarray", "np.ndarray"]:
    n_times = data.shape[1]
    freqs = np.fft.rfftfreq(n_times, d=1.0 / sfreq)
    fft_vals = np.fft.rfft(data, axis=1)
    power = (np.abs(fft_vals) ** 2) / n_times
    mean_power = power.mean(axis=0)
    return freqs, mean_power


def load_edf_data(path: str, max_seconds: int = 60) -> tuple["np.ndarray", float]:
    if mne is not None:
        raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
        sfreq = float(raw.info["sfreq"])
        max_samples = int(max_seconds * sfreq)
        data = raw.get_data()[:, :max_samples]
        return data, sfreq
    if pyedflib is not None:
        reader = pyedflib.EdfReader(path)
        try:
            sfreq = float(reader.getSampleFrequency(0))
            max_samples = int(max_seconds * sfreq)
            signals = []
            for ch in range(reader.signals_in_file):
                signal = reader.readSignal(ch)
                signals.append(signal[:max_samples])
            return np.vstack(signals), sfreq
        finally:
            reader.close()
    raise RuntimeError("No EDF reader available. Install mne or pyedflib.")


def get_fft_results(
    patient_id: str,
    test: str,
    root_dir: str,
    max_seconds: int = 60,
) -> dict[str, dict[str, object]]:
    if np is None:
        raise RuntimeError("numpy is required to compute FFT results.")

    patient_dirs = [
        os.path.join(root_dir, name)
        for name in os.listdir(root_dir)
        if patient_id.upper() in name.upper()
    ]
    if not patient_dirs:
        raise FileNotFoundError(f"Patient {patient_id} not found under {root_dir}")

    results: dict[str, dict[str, object]] = {}
    for patient_dir in patient_dirs:
        tests = find_test_dirs(patient_dir)
        test_dirs = tests.get(test.upper(), [])
        for test_dir in test_dirs:
            edf_files = [
                os.path.join(test_dir, name)
                for name in os.listdir(test_dir)
                if name.lower().endswith(".edf")
            ]
            for edf_path in edf_files:
                data, sfreq = load_edf_data(edf_path, max_seconds=max_seconds)
                freqs, mean_power = compute_fft(data, sfreq)
                rows = [
                    {"frequency_hz": float(f), "power": float(p)}
                    for f, p in zip(freqs, mean_power)
                ]
                results[edf_path] = {
                    "fft_rows": len(rows),
                    "fft_cols": 2,
                    "data": rows,
                }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Data quality check and FFT extraction.")
    parser.add_argument(
        "--root",
        default="CIPN3_NFB+DL_BL completed T2",
        help="Root data directory",
    )
    parser.add_argument(
        "--report-dir",
        default="reports",
        help="Directory to write quality reports",
    )
    parser.add_argument("--patient", help="Patient ID (e.g. CIPN3221)")
    parser.add_argument("--test", help="Test name (e.g. T1 or T2)")
    parser.add_argument("--max-seconds", type=int, default=60)
    args = parser.parse_args()

    summary, test_counts = quality_check(args.root, args.report_dir)
    t1_count = sum(1 for value in summary.values() if value.get("T1"))
    t2_count = sum(1 for value in summary.values() if value.get("T2"))
    print(f"Patients with T1: {t1_count}")
    print(f"Patients with T2: {t2_count}")
    if test_counts:
        print("Patient counts per test:")
        for test, count in sorted(test_counts.items()):
            print(f"{test}: {count}")

    if args.patient and args.test:
        results = get_fft_results(
            args.patient, args.test, args.root, max_seconds=args.max_seconds
        )
        if not results:
            print("No EDF files found for requested patient/test.")
            return
        fft_summary_path = os.path.join(args.report_dir, "fft_summary.csv")
        os.makedirs(args.report_dir, exist_ok=True)
        with open(fft_summary_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["edf_path", "fft_rows", "fft_cols"],
            )
            writer.writeheader()
            for path, payload in results.items():
                writer.writerow(
                    {
                        "edf_path": path,
                        "fft_rows": payload["fft_rows"],
                        "fft_cols": payload["fft_cols"],
                    }
                )
        for path, payload in results.items():
            print(f"FFT results for {path} (first 10 rows):")
            for row in payload["data"][:10]:
                print(row)


if __name__ == "__main__":
    main()
