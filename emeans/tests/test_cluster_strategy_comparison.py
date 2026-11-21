"""Guard the provenance and capacity of strategy comparison summaries."""

import copy

import pytest

from scripts.compare_ncvr_cluster_strategies import (
    paired_changes, parse_args, validate_reused_run,
)


@pytest.mark.parametrize("extra", [
    ["--batch-size", "4096", "--strategies", "rp128_k100_p2"],
    ["--batch-size", "4097", "--strategies", "baseline_k50"],
])
def test_capacity_is_checked_before_plaintext_audit(monkeypatch, extra):
    monkeypatch.setattr("sys.argv", ["compare", *extra])
    with pytest.raises(ValueError, match="batch_size"):
        parse_args()


def test_unknown_strategy_is_not_silently_ignored(monkeypatch):
    monkeypatch.setattr("sys.argv", ["compare", "--strategies", "baseline_k50,typo"])
    with pytest.raises(SystemExit):
        parse_args()


def test_reuse_validates_config_manifest_and_prediction_order_length():
    config = {"he_batch_size": 200, "probes": 1, "skip_zero_blocks": False}
    manifest = {"output_query_rows": 2}
    run = {"config": config, "dataset_manifest": manifest, "predictions": [True, False]}
    validate_reused_run(run, config, manifest)
    legacy = copy.deepcopy(run)
    del legacy["config"]["skip_zero_blocks"]
    validate_reused_run(legacy, config, manifest)
    with pytest.raises(ValueError, match="config mismatch"):
        validate_reused_run(run, {**config, "probes": 2}, manifest)
    with pytest.raises(ValueError, match="manifest"):
        validate_reused_run(run, config, {"output_query_rows": 3})
    with pytest.raises(ValueError, match="one boolean"):
        validate_reused_run({**run, "predictions": [True]}, config, manifest)
    with pytest.raises(ValueError, match="one boolean"):
        validate_reused_run({**run, "predictions": [1, 0]}, config, manifest)


def test_paired_changes_keep_rescues_and_regressions_separate():
    result = paired_changes(
        [True, False, False, True], [False, True, True, False],
        [True, True, False, False], range(4), ["q0", "q1", "q2", "q3"],
    )
    assert result == {
        "recovered_true_positives": ["q0"], "new_false_negatives": ["q1"],
        "removed_false_positives": ["q2"], "new_false_positives": ["q3"],
    }
