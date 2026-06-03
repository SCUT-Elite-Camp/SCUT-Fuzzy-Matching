"""Smoke test for the real benchmark path."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.benchmark import benchmark
from scripts.evaluate_ncvr_10k import parse_args


def test_benchmark_real_path_runs_small_smoke():
    result = benchmark(
        {
            "dataset": "ncvr_10k",
            "data_path": os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data",
            ),
            "el_cluster": 200,
            "el_match": 50,
            "k": 1,
            "tau": 0.9,
            "query_limit": 1,
            "db_limit": 2,
            "use_mock": False,
            "early_stop": True,
        }
    )

    assert set(result["metrics"]) == {"precision", "recall", "f1", "accuracy"}
    assert result["timing_sec"]["offline"] >= 0.0
    assert result["timing_sec"]["online_total"] >= 0.0
    assert result["communication_mb"]["total"] > 0.0


def test_ncvr_10k_script_defaults_to_paper_k_for_10k(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["evaluate_ncvr_10k.py"])

    args = parse_args()

    assert args.k == 50
