"""Tests for evaluation report output files."""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.reporting import save_evaluation_report


def test_save_evaluation_report_writes_metrics_and_plots(tmp_path: Path):
    result = {
        "config": {"dataset": "ncvr_10k"},
        "metrics": {
            "precision": 0.8,
            "recall": 0.6,
            "f1": 0.6857142857,
            "accuracy": 0.7,
        },
        "confusion": {"tp": 6, "fp": 2, "fn": 4, "tn": 8},
        "timing_sec": {},
        "communication_mb": {},
        "memory_peak_mb": 0.0,
    }

    paths = save_evaluation_report(result, tmp_path, prefix="demo")

    for path in paths.values():
        assert path.exists()
        assert path.stat().st_size > 0
