"""Evaluation report writers for metrics, confusion counts, and plots."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def save_evaluation_report(
    result: dict,
    output_dir: str | Path,
    prefix: str = "evaluation",
) -> dict[str, Path]:
    """Persist benchmark metrics as JSON, CSV, and PNG plots."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    paths = {
        "json": output_path / f"{prefix}_result.json",
        "metrics_csv": output_path / f"{prefix}_metrics.csv",
        "confusion_csv": output_path / f"{prefix}_confusion.csv",
        "metrics_png": output_path / f"{prefix}_metrics.png",
        "confusion_png": output_path / f"{prefix}_confusion.png",
    }

    paths["json"].write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_metrics_csv(result["metrics"], paths["metrics_csv"])
    _write_confusion_csv(result.get("confusion", {}), paths["confusion_csv"])
    _plot_metrics(result["metrics"], paths["metrics_png"])
    _plot_confusion(result.get("confusion", {}), paths["confusion_png"])
    return paths


def _write_metrics_csv(metrics: dict, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for metric_name in ["precision", "recall", "f1", "accuracy"]:
            writer.writerow([metric_name, metrics.get(metric_name, 0.0)])


def _write_confusion_csv(confusion: dict, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["count", "value"])
        for count_name in ["tp", "fp", "fn", "tn"]:
            writer.writerow([count_name, confusion.get(count_name, 0)])


def _plot_metrics(metrics: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["precision", "recall", "f1", "accuracy"]
    values = [float(metrics.get(name, 0.0)) for name in names]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, values, color=["#3b82f6", "#10b981", "#f59e0b", "#6366f1"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("score")
    ax.set_title("NCVR 10K Evaluation Metrics")
    ax.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_confusion(confusion: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["tp", "fp", "fn", "tn"]
    values = [int(confusion.get(name, 0)) for name in names]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, values, color=["#059669", "#dc2626", "#ea580c", "#2563eb"])
    ax.set_ylabel("count")
    ax.set_title("NCVR 10K Confusion Counts")
    ax.bar_label(bars, labels=[str(value) for value in values], padding=3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
