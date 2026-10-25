#!/usr/bin/env python3
"""把多属性记录编码成 ApproxPSI 需要的二进制向量，并**量化它到底能不能用**。

背景（为什么这个脚本不能直接产出可跑的 ApproxPSI 实验）
--------------------------------------------------------
``approx_psi_plain`` 的 CLI 只有 ``--n/--ell/--d/--t/--lambda/--matches/
--trials/--seed/--k``，数据由 ``generateDataset()`` 在进程内合成，**没有
``--input``**（见 ``src/main.cpp`` 的 ``printHelp``）。所以本脚本产出的 ``.bits``
文件当前没有任何消费方；它存在的价值是：把"编码"这一步先做完并**用真实数据量出
可达性**，将来给 ``main.cpp`` 加 20 行读取就能直接接上。

ApproxPSI 对输入的要求不只是"128 位"，而是**真匹配对的 Hamming 距离 ≤ d**（论文
默认 d=4）。这条约束决定了编码方式：

* ``hash`` 模式：``sha256(值)`` 取前 N 位。确定性、均匀，但**毫无局部性** —— 两个
  只差一个字母的名字会在 ~50% 的位上不同，距离期望 N/2。用它喂 ApproxPSI 必然
  全漏。本脚本保留这个模式，正是为了用 ``--measure`` 把它证伪。
* ``simhash`` 模式：对每个属性的字符 3-gram 做加权 SimHash，再按属性块拼接。
  相似字符串只差少数位，是**唯一有机会满足 d 很小**的做法。

``--measure`` 会读 ground truth，统计真匹配对的 Hamming 距离分布，并给出
"距离 ≤ d 的比例"——这就是 ApproxPSI 在该数据集上**理论上限**的直接证据。

用法::

    # 编码（默认 simhash，128 位：name 64 + dob 24 + address 40）
    python3 scripts/encode_multi_attr_to_binary.py \
        --a data/multi_attr/febrl4_A.csv --b data/multi_attr/febrl4_B.csv \
        --gt data/multi_attr/febrl4_ground_truth.csv \
        --id-a record_id --id-b query_id \
        --attrs name=given_name+surname,dob=date_of_birth,address=address_1 \
        --out-dir data/multi_attr/febrl4_bits --measure --d 4

    # 对照：sha256 模式（预期 d≤4 比例 ≈ 0）
    python3 scripts/encode_multi_attr_to_binary.py ... --mode hash --measure
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

BLOCK_DEFAULT = {"name": 64, "dob": 24, "address": 40}
_WS = re.compile(r"\s+")
_DIGITS = re.compile(r"[^0-9]")


def normalize(value: str, kind: str | None = None) -> str:
    text = _WS.sub(" ", str(value or "")).strip().casefold()
    if kind == "dob":
        return _DIGITS.sub("", text)
    if kind == "address":
        return re.sub(r"[^0-9a-z\u4e00-\u9fff ]", "", text)
    return re.sub(r"[^0-9a-z\u4e00-\u9fff ]", "", text)


def shingles(text: str, size: int = 3) -> list[str]:
    padded = f"  {text}  "
    if len(padded) <= size:
        return [padded]
    return [padded[i : i + size] for i in range(len(padded) - size + 1)]


def simhash_bits(text: str, n_bits: int, salt: str) -> list[int]:
    """字符 3-gram 的加权 SimHash（权重 = tf）。"""

    counts = Counter(shingles(text))
    acc = [0] * n_bits
    for token, weight in counts.items():
        digest = hashlib.sha256(f"{salt}:{token}".encode("utf-8")).digest()
        value = int.from_bytes(digest, "big")
        for bit in range(n_bits):
            acc[bit] += weight if (value >> (255 - bit)) & 1 else -weight
    return [1 if v >= 0 else 0 for v in acc]


def hash_bits(text: str, n_bits: int, salt: str) -> list[int]:
    if not text:
        return [0] * n_bits
    digest = hashlib.sha256(f"{salt}:{text}".encode("utf-8")).digest()
    value = bin(int.from_bytes(digest, "big"))[2:].zfill(256)[:n_bits]
    return [int(b) for b in value]


def encode_row(row: dict, attrs: dict[str, list[str]], blocks: dict[str, int],
               mode: str) -> list[int]:
    bits: list[int] = []
    fn = simhash_bits if mode == "simhash" else hash_bits
    for attr, columns in attrs.items():
        n = blocks[attr]
        text = " ".join(normalize(row.get(col, ""), attr) for col in columns)
        bits.extend(fn(text, n, attr))
    return bits


def hamming(a: list[int], b: list[int]) -> int:
    return sum(x != y for x, y in zip(a, b))


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", required=True, help="A 侧 CSV（库）")
    parser.add_argument("--b", required=True, help="B 侧 CSV（查询）")
    parser.add_argument("--gt", required=True, help="ground truth CSV（A_id,B_id）")
    parser.add_argument("--id-a", default="record_id")
    parser.add_argument("--id-b", default="query_id")
    parser.add_argument("--attrs", required=True,
                        help="name=given_name+surname,dob=date_of_birth,address=address_1")
    parser.add_argument("--blocks", default=None, help="覆盖块宽，如 name=64,dob=24,address=40")
    parser.add_argument("--mode", choices=("simhash", "hash"), default="simhash")
    parser.add_argument("--out-dir", default=None, help="写 .bits 的目录（省略则只测量）")
    parser.add_argument("--measure", action="store_true", help="统计真匹配对的 Hamming 距离")
    parser.add_argument("--d", type=int, default=4, help="ApproxPSI 的 Hamming 阈值")
    parser.add_argument("--limit-pairs", type=int, default=0, help="测量时最多用多少对")
    args = parser.parse_args()

    attrs = {}
    for spec in args.attrs.split(","):
        name, columns = spec.split("=", 1)
        attrs[name.strip()] = [c.strip() for c in columns.split("+")]

    blocks = dict(BLOCK_DEFAULT)
    if args.blocks:
        for spec in args.blocks.split(","):
            name, width = spec.split("=", 1)
            blocks[name.strip()] = int(width)
    for name in attrs:
        blocks.setdefault(name, 32)
    ell = sum(blocks[name] for name in attrs)
    print(f"mode={args.mode} ell={ell} blocks={ {n: blocks[n] for n in attrs} }")

    a_rows = read_csv(Path(args.a))
    b_rows = read_csv(Path(args.b))
    a_bits = {r[args.id_a]: encode_row(r, attrs, blocks, args.mode) for r in a_rows}
    b_bits = {r[args.id_b]: encode_row(r, attrs, blocks, args.mode) for r in b_rows}
    print(f"A={len(a_bits)} B={len(b_bits)} encoded")

    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, table in (("A", a_bits), ("B", b_bits)):
            path = out / f"{args.mode}_{name}.bits"
            with path.open("w", encoding="utf-8") as handle:
                for key, bits in table.items():
                    handle.write(f"{key}\t{''.join(str(b) for b in bits)}\n")
            print(f"wrote {path}")

    if args.measure:
        truth = read_csv(Path(args.gt))
        pairs = []
        for row in truth:
            aid, bid, label = row.get("A_id", ""), row.get("B_id", ""), row.get("label", "true")
            if str(label).lower() != "true" or aid not in a_bits or bid not in b_bits:
                continue
            pairs.append((aid, bid))
            if args.limit_pairs and len(pairs) >= args.limit_pairs:
                break
        distances = sorted(hamming(a_bits[aid], b_bits[bid]) for aid, bid in pairs)
        n = len(distances)
        within = sum(1 for x in distances if x <= args.d)
        stats = {
            "mode": args.mode,
            "ell": ell,
            "pairs": n,
            "d": args.d,
            "within_d": within,
            "within_d_ratio": within / n if n else None,
            "min": distances[0] if n else None,
            "p25": distances[n // 4] if n else None,
            "median": distances[n // 2] if n else None,
            "p75": distances[3 * n // 4] if n else None,
            "max": distances[-1] if n else None,
            "mean": sum(distances) / n if n else None,
        }
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        if args.out_dir:
            (Path(args.out_dir) / f"{args.mode}_hamming_stats.json").write_text(
                json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
