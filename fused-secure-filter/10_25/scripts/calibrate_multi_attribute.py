"""从带标签样本反推多维属性 schema 的权重与判定阈值。

输入一个 job 文件（和 demo 用的是同一种），输出：

* 终端上的标定说明（逐属性 AUC / 新旧权重 / 分数区间 / 新阈值下的召回与假正例率）；
* 一份**可以直接当 job 文件用**的校准结果 JSON —— schema 块被换成标定后的权重与
  tau，其余字段原样保留。把它喂给 ``demo_multi_attribute_dataset.py`` 即可复现。

    python scripts/calibrate_multi_attribute.py --config config/examples/febrl_multi_attribute.json --output config/examples/febrl_multi_attribute.calibrated.json

权重默认用 ``separability``：在**组合分**上最大化软化 AUC。不要换成 ``auc``（按
单属性 AUC 线性分配）—— 那条规则在 FEBRL 上实测更差，理由写在
``multi_attribute/calibration.py`` 的模块 docstring 里。

本脚本不 import tenseal：标定全部在明文上完成，一个没装 TenSEAL 的环境也能跑。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_attribute.calibration import (  # noqa: E402
    THRESHOLD_CRITERIA,
    WEIGHT_POLICIES,
    calibrate_schema,
)
from multi_attribute.jobfile import load_data, load_job  # noqa: E402


def _jsonable(value):
    """把 numpy 标量/数组递归转成原生类型。

    不用 ``json.dumps(..., default=float)``：那会把 ``np.bool_(False)`` 也变成
    ``0.0``，写进 job 文件里的 "separated" 就成了浮点，读回来是另一回事。
    """

    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to a job / schema JSON file")
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="max labelled queries to calibrate on (default 500; 0 = unlimited)",
    )
    parser.add_argument(
        "--db-limit",
        type=int,
        default=0,
        help="max database rows to load (default 0 = unlimited; the full database is "
        "the right call here -- the impostor distribution depends on it)",
    )
    parser.add_argument("--weight-policy", default="separability", choices=list(WEIGHT_POLICIES))
    parser.add_argument("--criterion", default="recall_first", choices=list(THRESHOLD_CRITERIA))
    parser.add_argument(
        "--recall-target",
        type=float,
        default=1.0,
        help="only used by --criterion target_recall (default 1.0)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.05,
        help="soft-AUC temperature; smaller cares only about the hardest pairs "
        "(default 0.05)",
    )
    parser.add_argument("--restarts", type=int, default=12, help="coordinate-ascent restarts")
    parser.add_argument("--candidate-top-k", type=int, default=64)
    parser.add_argument(
        "--power",
        type=float,
        default=1.0,
        help="exponent for --weight-policy auc (ignored otherwise)",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--output",
        default=None,
        help="where to write the calibrated job JSON (default: <config>.calibrated.json)",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="print the report only; do not write the calibrated job file",
    )
    args = parser.parse_args()

    job = load_job(args.config)
    schema = job["schema_object"]
    limit = None if args.limit <= 0 else args.limit
    database_limit = None if args.db_limit <= 0 else args.db_limit
    database, queries = load_data(job, limit, database_limit)

    labelled = queries
    if not labelled:
        raise SystemExit("no queries loaded; nothing to calibrate on")

    print(
        f"calibrating on {len(labelled)} quer{'y' if len(labelled) == 1 else 'ies'} "
        f"against {len(database)} database record(s); "
        f"cluster_dim={schema.cluster_dim} match_dim={schema.match_dim}"
    )
    print(f"previous tau={schema.similarity_threshold:.4f}\n")

    calibrated, report = calibrate_schema(
        database,
        labelled,
        schema,
        weight_policy=args.weight_policy,
        criterion=args.criterion,
        recall_target=args.recall_target,
        power=args.power,
        temperature=args.temperature,
        restarts=args.restarts,
        candidate_top_k=args.candidate_top_k,
        random_state=args.random_state,
    )
    print(report.explain())

    if args.no_write:
        return 0

    output_path = (
        Path(args.output) if args.output else Path(f"{args.config}.calibrated.json")
    )
    payload = dict(job)
    # 去掉加载期塞进来的派生键，别把它们写回文件。
    payload.pop("schema_object", None)
    payload["schema"] = calibrated.to_dict()
    payload["_calibration"] = report.summary()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_jsonable(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote calibrated job: {output_path}")
    print(
        "  feed it straight back in:  python scripts/demo_multi_attribute_dataset.py "
        f"--config {output_path}"
    )
    if report.overlap:
        print(
            f"\nWARNING: {report.overlap} positive(s) lose to their OWN strongest impostor. "
            "No tau fixes\n  those; more attributes or better field quality is the fix."
        )
    elif report.ranges_interleave:
        print(
            f"\nNOTE: the weakest true match ({report.lowest_true_match:.4f}) sits below the\n"
            f"  strongest impostor ({report.highest_impostor:.4f}). Every query beats its own\n"
            f"  impostors, but no single global tau gets both recall and a clean boundary:\n"
            f"  the chosen criterion ({args.criterion}) trades "
            f"{report.false_positive_rate_at():.1%} false positives for "
            f"{report.recall_at():.1%} recall."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
