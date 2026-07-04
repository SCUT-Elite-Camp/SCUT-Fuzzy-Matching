#!/usr/bin/env python3
"""GUI server for privacy-preserving fuzzy name matching demo.

启动方式:
    cd baseline_Integration
    pip install flask
    python gui_server.py
    # 在浏览器打开 http://localhost:5000
"""

from __future__ import annotations
from config.params import SIMILARITY_THRESHOLD
from evaluation.dataset_loader import load_dataset
from scripts.demo_streaming import (
    StreamingDemoEngineV2,
    QueryState,
    build_demo_queries,
)
import numpy as np
from flask import Flask, request, jsonify, Response, send_from_directory

import json
import sys
import threading
import time
import queue
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Flask app setup
# ---------------------------------------------------------------------------

app = Flask(__name__, static_folder=None)

# In-memory store for pending/finished runs
_runs: dict[str, dict] = {}
_runs_lock = threading.Lock()


def _parse_indices(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    """Serve the main HTML page."""
    gui_dir = PROJECT_ROOT / "gui" / "templates"
    return send_from_directory(str(gui_dir), "index.html")


@app.route("/api/run", methods=["POST"])
def api_run():
    """Run the protocol and stream SSE events back.

    Accepts JSON body:
        query: str | None        — single custom query string
        query_indices: [int]     — indices into the query dataset (default [2,7,26,49,100])
        db_limit: int            — max database records (default 100)
        k: int                   — number of clusters (default 10)
        tau: float               — similarity threshold (default 0.9)
        early_stop: bool         — early stopping (default true)
    """
    data = request.get_json(silent=True) or {}

    # query_mode: "demo"（默认演示集，与 CLI 一致）| "indices" | "text"
    query_mode = data.get("query_mode", "demo")
    query_text = data.get("query", None)
    query_indices = data.get("query_indices", [])
    db_limit = data.get("db_limit", 100)
    k = data.get("k", 10)
    tau = data.get("tau", SIMILARITY_THRESHOLD)
    early_stop = data.get("early_stop", True)

    def generate():
        """Generator that yields SSE-formatted events."""
        try:
            # Load dataset
            data_path = str(PROJECT_ROOT / "data")
            names_a, names_b_all, labels = load_dataset("ncvr_10k", data_path)
            names_b = list(names_b_all[:db_limit])

            # Build queries
            if query_mode == "text" and query_text:
                queries = [
                    QueryState(
                        query_name=query_text,
                        query_index=None,
                        expected_label=None,
                    )
                ]
            elif query_mode == "indices" and query_indices:
                queries = [
                    QueryState(
                        query_name=names_a[i],
                        query_index=i,
                        expected_label=bool(labels[i]),
                    )
                    for i in query_indices
                    if 0 <= i < len(names_a)
                ]
            else:
                # 默认：与 CLI `python scripts/demo_streaming.py` 完全一致的演示集
                queries = build_demo_queries(names_a, list(labels))

            if not queries:
                yield _sse("error", {"message": "No valid queries."})
                return

            # Run protocol, streaming each step
            engine = StreamingDemoEngineV2(
                names_b=names_b,
                k_mode=k,
                tau=tau,
                early_stop=early_stop,
            )

            for event in engine.run_all_queries_with_events(
                queries, verbose=False
            ):
                yield _sse(event["event"], event)

        except Exception as e:
            import traceback

            yield _sse(
                "error",
                {"message": str(e), "traceback": traceback.format_exc()},
            )

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.route("/api/dataset-info", methods=["GET"])
def dataset_info():
    """Return basic info about the available dataset."""
    data_path = str(PROJECT_ROOT / "data")
    names_a, names_b_all, labels = load_dataset("ncvr_10k", data_path)
    return jsonify(
        {
            "dataset": "ncvr_10k",
            "total_records_b": len(names_b_all),
            "total_queries_a": len(names_a),
            "positive_queries": int(sum(1 for l in labels if l)),
            "negative_queries": int(sum(1 for l in labels if not l)),
            "sample_names_b": names_b_all[:10],
            "sample_queries_a": [
                {"index": i, "name": names_a[i], "label": bool(labels[i])}
                for i in range(min(10, len(names_a)))
            ],
        }
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sse(event: str, data: dict) -> str:
    """Format a Server-Sent Event line."""
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="GUI server for fuzzy matching demo")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=5000, help="Bind port")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Flask debug mode")
    args = parser.parse_args()

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║   Privacy-Preserving Fuzzy Name Matching — GUI Server       ║
║                                                            ║
║   Open in browser: http://{args.host}:{args.port}                  ║
║   Press Ctrl+C to stop                                      ║
╚══════════════════════════════════════════════════════════════╝
""")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
