"""Tests for query chunking, context reuse, and batch size validation in evaluation/."""

import importlib
import os

import numpy as np
import pytest

import scripts.evaluate_ncvr_10k as evaluate_script
from evaluation.benchmark import benchmark
from minhash.encoder import batch_encode
from preprocessing.normalizer import l2_normalize
from scripts.evaluate_ncvr_10k import parse_args

_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)
benchmark_module = importlib.import_module("evaluation.benchmark")


def test_batch_chunking_execution_and_order(monkeypatch):
    """Test running real benchmark with he_batch_size=3 over 7 queries processes 3 chunks and preserves order."""
    # First, run 7 queries with serial path (he_batch_size=0)
    serial_result = benchmark(
        {
            "dataset": "ncvr_10k",
            "data_path": _DATA_PATH,
            "el_cluster": 200,
            "el_match": 50,
            "k": 2,
            "tau": 0.9,
            "query_limit": 7,
            "db_limit": 10,
            "use_mock": False,
            "he_batch_size": 0,
        }
    )

    original_compare = benchmark_module.compare_tiled_batch_to_centroids
    original_matching = benchmark_module.tiled_batch_matching

    def checked_compare(request, *args, **kwargs):
        assert all(isinstance(item, bytes) for item in request.encrypted_query_200)
        return original_compare(request, *args, **kwargs)

    def checked_matching(cluster_matrix, request, *args, **kwargs):
        assert all(isinstance(item, bytes) for item in request.encrypted_query_50)
        assert all(isinstance(item, bytes) for item in request.encrypted_selectors)
        return original_matching(cluster_matrix, request, *args, **kwargs)

    monkeypatch.setattr(
        benchmark_module, "compare_tiled_batch_to_centroids", checked_compare
    )
    monkeypatch.setattr(benchmark_module, "tiled_batch_matching", checked_matching)

    # Next, run same 7 queries with he_batch_size=3 (chunks of size 3, 3, 1)
    batch_result = benchmark(
        {
            "dataset": "ncvr_10k",
            "data_path": _DATA_PATH,
            "el_cluster": 200,
            "el_match": 50,
            "k": 2,
            "tau": 0.9,
            "query_limit": 7,
            "db_limit": 10,
            "use_mock": False,
            "he_batch_size": 3,
        }
    )

    # Metrics and communication results should be recorded
    assert "precision" in batch_result["metrics"]
    assert "communication_per_query_kb" in batch_result
    assert batch_result["communication_per_query_kb"]["total"] > 0.0

    # Exact per-query order and values must match, not merely aggregate accuracy.
    assert batch_result["predictions"] == serial_result["predictions"]


def test_illegal_he_batch_size_validation():
    """Test validation of he_batch_size (> 4096, negative, non-int)."""
    with pytest.raises(ValueError, match="he_batch_size must be between 0 and 4096"):
        benchmark(
            {
                "dataset": "ncvr_10k",
                "data_path": _DATA_PATH,
                "he_batch_size": 4097,
                "use_mock": False,
            }
        )

    with pytest.raises(ValueError, match="he_batch_size must be between 0 and 4096"):
        benchmark(
            {
                "dataset": "ncvr_10k",
                "data_path": _DATA_PATH,
                "he_batch_size": -1,
                "use_mock": False,
            }
        )

    with pytest.raises(ValueError, match="Invalid he_batch_size"):
        benchmark(
            {
                "dataset": "ncvr_10k",
                "data_path": _DATA_PATH,
                "he_batch_size": True,
                "use_mock": False,
            }
        )


def test_ncvr_10k_script_parse_batch_size_arg(monkeypatch):
    """Test evaluate_ncvr_10k.py argument parser with --batch-size."""
    monkeypatch.setattr("sys.argv", ["evaluate_ncvr_10k.py", "--batch-size", "200"])
    args = parse_args()
    assert args.batch_size == 200


def test_ncvr_10k_main_forwards_evaluation_options(monkeypatch, tmp_path):
    captured = {}

    def fake_benchmark(config):
        captured.update(config)
        return {"metrics": {}, "confusion": {}}

    monkeypatch.setattr(evaluate_script, "benchmark", fake_benchmark)
    monkeypatch.setattr(
        evaluate_script, "save_evaluation_report", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_ncvr_10k.py",
            "--tau",
            "0.87",
            "--batch-size",
            "200",
            "--fuzzy-ratio",
            "0.4",
            "--fuzzy-seed",
            "9",
            "--projection-dim",
            "128",
            "--projection-seed",
            "43",
            "--probes",
            "2",
            "--skip-zero-blocks",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert evaluate_script.main() == 0
    assert captured["tau"] == pytest.approx(0.87)
    assert captured["he_batch_size"] == 200
    assert captured["fuzzy_ratio"] == pytest.approx(0.4)
    assert captured["fuzzy_seed"] == 9
    assert captured["projection_dim"] == 128
    assert captured["projection_seed"] == 43
    assert captured["probes"] == 2
    assert captured["skip_zero_blocks"] is True


def test_batch_benchmark_single_tile_early_stop_preserves_results():
    base_config = {
        "dataset": "ncvr_10k",
        "data_path": _DATA_PATH,
        "el_cluster": 200,
        "el_match": 50,
        "k": 1,
        "tau": 0.9,
        "query_limit": 1,
        "db_limit": 2,
        "use_mock": False,
        "he_batch_size": 1,
    }
    early_result = benchmark({**base_config, "early_stop": True})
    full_result = benchmark({**base_config, "early_stop": False})

    assert early_result["predictions"] == full_result["predictions"]
    # m=1 gives T=4096, so both database columns are carried by one tile.
    # Early-stop cannot reduce communication below one ciphertext. Serialized
    # CKKS byte sizes may differ slightly because ciphertexts are randomized.
    early_recv = early_result["communication_mb"]["recv"]
    full_recv = full_result["communication_mb"]["recv"]
    assert early_recv > 0
    assert full_recv > 0
    assert abs(early_recv - full_recv) < 0.01


def test_benchmark_records_effective_probe_count_when_request_exceeds_k():
    result = benchmark(
        {
            "dataset": "ncvr_10k",
            "data_path": _DATA_PATH,
            "el_cluster": 200,
            "el_match": 50,
            "k": 1,
            "tau": 0.9,
            "query_limit": 1,
            "db_limit": 2,
            "use_mock": False,
            "he_batch_size": 1,
            "probes": 2,
            "early_stop": False,
        }
    )
    assert result["protocol"]["probes"] == 1
    assert result["protocol"]["requested_probes"] == 2


def test_mock_benchmark_records_nonzero_peak_memory():
    result = benchmark(
        {
            "dataset": "ncvr_10k",
            "data_path": _DATA_PATH,
            "el_cluster": 200,
            "el_match": 50,
            "k": 2,
            "tau": 0.9,
            "query_limit": 1,
            "db_limit": 2,
            "use_mock": True,
            "he_batch_size": 0,
        }
    )
    assert result["memory_peak_mb"] > 0.0


def test_projected_multi_probe_chunks_match_full_database_reference(monkeypatch):
    database = ["JOHN SMITH", "MARY JONES", "ALICE BROWN", "BOB WILSON"]
    queries = [database[0], "ZXQV QZXV", database[2], "VVZX QQXZ", database[3]]
    monkeypatch.setattr(
        benchmark_module, "_load_benchmark_dataset",
        lambda config: (queries, database, [True, False, True, False, True], None),
    )
    expanded_sizes = []
    original_matching = benchmark_module.tiled_batch_matching

    def checked_matching(matrix, request, public_context, **kwargs):
        expanded_sizes.append(request.layout.batch_size)
        assert all(isinstance(item, bytes) for item in request.encrypted_query_50)
        assert all(isinstance(item, bytes) for item in request.encrypted_selectors)
        return original_matching(matrix, request, public_context, **kwargs)

    monkeypatch.setattr(benchmark_module, "tiled_batch_matching", checked_matching)
    result = benchmark({
        "use_mock": False, "k": 2, "he_batch_size": 3, "probes": 3,
        "projection_dim": 4, "skip_zero_blocks": True, "early_stop": False,
    })
    q50 = l2_normalize(batch_encode(queries, 200)[:, :50])
    b50 = l2_normalize(batch_encode(database, 200)[:, :50])
    scores = q50 @ b50.T
    assert np.all(np.abs(scores - 0.9) > 1e-3)
    expected = np.any(scores >= 0.9, axis=1).tolist()
    assert len(set(expected)) == 2
    assert result["predictions"] == expected
    assert expanded_sizes == [6, 4]
    assert result["protocol"]["probes"] == 2
    assert result["protocol"]["requested_probes"] == 3


@pytest.mark.parametrize("reuse_context", [True, False])
def test_serial_context_communication_accounting(monkeypatch, reuse_context):
    names = ["JOHN SMITH", "MARY JONES"]
    monkeypatch.setattr(
        benchmark_module, "_load_benchmark_dataset",
        lambda config: (names, names, [True, True], None),
    )
    result = benchmark({
        "use_mock": False, "k": 1, "he_batch_size": 0,
        "reuse_context": reuse_context, "early_stop": False,
    })
    comm = result["communication_bytes"]
    assert comm["public_context_setup"] > 0
    if reuse_context:
        assert comm["public_context_transmitted"] == 2 * comm["public_context_setup"]
    else:
        assert comm["public_context_transmitted"] == comm["public_context_setup"]
    assert result["communication_mb"]["total_without_public_context"] == pytest.approx(
        (comm["sent"] + comm["received"] - comm["public_context_transmitted"]) / 1024**2
    )
