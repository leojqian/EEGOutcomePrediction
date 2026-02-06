from __future__ import annotations

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from data_quality_and_fft import get_fft_results as _get_fft_results

__all__ = ["get_fft_results"]


def get_fft_results(
    patient_id: str,
    test: str,
    root_dir: str = "CIPN3_NFB+DL_BL completed T2",
    max_seconds: int = 60,
) -> dict[str, dict[str, object]]:
    """
    Return FFT results for all EDF files for a patient/test.

    Output format matches data_quality_and_fft.get_fft_results().
    """

    return _get_fft_results(
        patient_id=patient_id,
        test=test,
        root_dir=root_dir,
        max_seconds=max_seconds,
    )
