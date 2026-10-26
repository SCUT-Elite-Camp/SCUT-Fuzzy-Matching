"""从 SAGE 姓名库 + 合成属性构建多语言融合数据集（多属性协议演示用）。

**目的**：阶段性展示用的"融合数据集" —— 多语言姓名（SAGE 真实数据）+ 多维度
合成属性（DOB / 城市 / 电话）。数据库记录同时携带两个姓名列：

* ``name_local``     —— 本地文字形式（SAGE 原文：拉丁重音 / 阿拉伯 / 缅甸 / 汉字…）
* ``name_romanized`` —— 罗马化 / 去重音拉丁形式

查询只携带其中一种形式（另一种留空）。管线对空单元格的语义是"缺失 → 该属性块
全零向量"（``multi_attribute/kinds.py``），对两列拼接的属性则跳过空列
（``multi_attribute/dataset.py::_project_row``）—— 所以**不需要改任何管线代码**
即可模拟真实的跨文字系统 / 跨形式查询。

罗马化形式完全复用管线既有逻辑，不引入新的转换规则：

* 拉丁重音名（匈牙利 / 意大利）：``data_pipeline/normalizers.py`` 的
  ``_fold_latin``（NFKD 去组合变音记号）；
* 阿拉伯 / 缅甸 / 纯汉字名：``anyascii`` 上下文无关转写（与 ``unicode_v1``
  normalizer 生成 ``latin_transliterated`` 变体用的是同一个库）；
* 台湾 "汉字+拉丁" 混合行（如 ``瓦歷斯‧貝林 Walis‧Perin``）：按文字系统拆成
  两列，拉丁部分即真实罗马化形式。

产出三个文件（默认 ``data/fused_multilingual/``，该目录**不在** .gitignore 内）::

    database.csv   库侧记录（record_id, name_local, name_romanized, dob, city, phone, country）
    queries.csv    查询侧（query_id, 同上的属性列；正例为扰动副本，只带一种姓名形式）
    labels.csv     query_id,true_record_id,label（true/false）

可重复运行：固定随机种子（默认 20261014），同名同参数输出逐字节一致。

用法::

    python scripts/build_fused_multilingual_dataset.py \\
        --records 300 --positives 40 --negatives 10 --seed 20261014 \\
        --out data/fused_multilingual
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anyascii import anyascii  # noqa: E402

from data_pipeline.normalizers import _fold_latin  # noqa: E402

SAGE_CSV = ROOT / "data" / "sage" / "sage_names_all.csv"

DB_FIELDS = ["record_id", "name_local", "name_romanized", "dob", "city", "phone", "country"]
QUERY_FIELDS = ["query_id", *DB_FIELDS[1:]]

# ---------------------------------------------------------------------------
# 国家元数据：真实城市池 + 电话区号（dob / 电话数字本体由 RNG 合成）。
# 只覆盖 SAGE 实际出现的国家。
# ---------------------------------------------------------------------------

_COUNTRY_META: dict[str, dict] = {
    "Afghanistan": {"dial": "93", "cities": ["Kabul", "Herat", "Kandahar", "Mazar-i-Sharif", "Kunduz"]},
    "Hungary": {"dial": "36", "cities": ["Budapest", "Debrecen", "Szeged", "Pécs", "Győr", "Miskolc"]},
    "India": {"dial": "91", "cities": ["Mumbai", "Delhi", "Bengaluru", "Chennai", "Kolkata", "Hyderabad", "Pune"]},
    "Italy": {"dial": "39", "cities": ["Roma", "Milano", "Napoli", "Torino", "Firenze", "Bologna", "Venezia"]},
    "Taiwan": {"dial": "886", "cities": ["Taipei", "Kaohsiung", "Taichung", "Tainan", "Hsinchu"]},
    "Lesotho": {"dial": "266", "cities": ["Maseru", "Teyateyaneng", "Mafeteng", "Leribe"]},
    "Pakistan": {"dial": "92", "cities": ["Karachi", "Lahore", "Islamabad", "Peshawar", "Quetta"]},
    "Iraq": {"dial": "964", "cities": ["Baghdad", "Basra", "Mosul", "Erbil", "Karbala"]},
    "Malaysia": {"dial": "60", "cities": ["Kuala Lumpur", "George Town", "Ipoh", "Johor Bahru", "Kota Kinabalu"]},
    "Ghana": {"dial": "233", "cities": ["Accra", "Kumasi", "Tamale", "Takoradi", "Cape Coast"]},
    "Myanmar": {"dial": "95", "cities": ["Yangon", "Mandalay", "Naypyidaw", "Mawlamyine", "Taunggyi"]},
    "Ivory Coast": {"dial": "225", "cities": ["Abidjan", "Yamoussoukro", "Bouaké", "Daloa", "San-Pédro"]},
    "Belize": {"dial": "501", "cities": ["Belize City", "San Ignacio", "Orange Walk", "Corozal"]},
    "Botswana": {"dial": "267", "cities": ["Gaborone", "Francistown", "Molepolole", "Maun"]},
    "Samoa": {"dial": "685", "cities": ["Apia", "Vaitele", "Faleula"]},
    "Liberia": {"dial": "231", "cities": ["Monrovia", "Gbarnga", "Kakata", "Buchanan"]},
    "Barbados": {"dial": "1246", "cities": ["Bridgetown", "Speightstown", "Oistins"]},
    "Dominica": {"dial": "1767", "cities": ["Roseau", "Portsmouth", "Marigot"]},
    "Saint Lucia": {"dial": "1758", "cities": ["Castries", "Vieux Fort", "Gros Islet"]},
    "Grenada": {"dial": "1473", "cities": ["Saint George's", "Gouyave", "Grenville"]},
    "Gambia": {"dial": "220", "cities": ["Banjul", "Serekunda", "Brikama"]},
    "Nicaragua": {"dial": "505", "cities": ["Managua", "León", "Granada", "Masaya"]},
    "Seychelles": {"dial": "248", "cities": ["Victoria", "Anse Boileau", "Beau Vallon"]},
}

_MONTH_DAYS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)

# 数据库抽样配额：按文字系统分组（而非按国家均匀分），保证多语言覆盖面。
# SAGE 实际可用的文字系统：拉丁重音 / 阿拉伯 / 缅甸 / 汉字 / 汉字+拉丁混合 / 纯拉丁。
_SCRIPT_QUOTAS = {
    "latin_accented": 90,   # 匈牙利、意大利（Árgyelán János 等）
    "arabic": 45,           # 伊拉克（阿拉伯文字）
    "myanmar": 40,          # 缅甸文
    "han": 45,              # 台湾纯汉字
    "han_mixed": 25,        # 台湾 "汉字 拉丁" 混合行（自带真实罗马化）
    "latin_plain": 55,      # 加纳 / 印度 / 马来西亚 / 阿富汗罗马化等
}


# ---------------------------------------------------------------------------
# 文字系统识别与姓名形式拆分
# ---------------------------------------------------------------------------


def _script_of(char: str) -> str:
    codepoint = ord(char)
    if "a" <= char.lower() <= "z" or "LATIN" in unicodedata.name(char, ""):
        return "Latn"
    if 0x4E00 <= codepoint <= 0x9FFF:
        return "Han"
    if 0x0600 <= codepoint <= 0x06FF:
        return "Arab"
    if 0x1000 <= codepoint <= 0x109F or 0xA9E0 <= codepoint <= 0xA9FF or 0xAA60 <= codepoint <= 0xAA7F:
        return "Mymr"
    return "Other"


def classify_name(name: str) -> str:
    """把一条 SAGE 姓名分到 ``_SCRIPT_QUOTAS`` 的某个组。"""
    scripts = {_script_of(ch) for ch in name if ch.isalpha()}
    if scripts == {"Han"}:
        return "han"
    if "Han" in scripts and "Latn" in scripts:
        return "han_mixed"
    if scripts == {"Arab"}:
        return "arabic"
    if scripts == {"Mymr"}:
        return "myanmar"
    # 拉丁（含萨摩亚 ʻokina 等修饰字母归并到拉丁组看 folded 结果）
    folded = _fold_latin(name)
    if folded and folded != " ".join(name.split()).casefold():
        return "latin_accented"
    return "latin_plain"


def split_forms(name: str, group: str) -> tuple[str, str]:
    """返回 ``(name_local, name_romanized)``。

    与 ``unicode_v1`` normalizer 的变体生成规则保持一致：不发明新的转换。
    两种形式相同（去大小写）时 romanized 留空 —— 该记录只有一种真实形式。
    """
    if group == "latin_accented":
        local = " ".join(name.split())
        romanized = _fold_latin(name)
    elif group == "han_mixed":
        # 混合行形如 "瓦歷斯‧貝林 Walis‧Perin"：拉丁文字 run 即真实罗马化。
        parts = re.findall(r"[A-Za-z][A-Za-z .'ʼ-]*|[\u4e00-\u9fff]+", name)
        cjk = " ".join(p for p in parts if any("\u4e00" <= ch <= "\u9fff" for ch in p))
        latn = " ".join(p for p in parts if p not in cjk.split() and not any("\u4e00" <= ch <= "\u9fff" for ch in p))
        local, romanized = cjk or name, " ".join(latn.split())
    elif group in {"arabic", "myanmar", "han"}:
        local = " ".join(name.split())
        romanized = _fold_latin(anyascii(local))
    else:  # latin_plain
        local = " ".join(name.split())
        romanized = _fold_latin(local)
    if romanized.casefold() == local.casefold():
        romanized = ""
    return local, romanized


# ---------------------------------------------------------------------------
# 合成属性
# ---------------------------------------------------------------------------


def _iso_date(rng: random.Random, year_lo: int = 1940, year_hi: int = 2005) -> str:
    year = rng.randint(year_lo, year_hi)
    month = rng.randint(1, 12)
    day = rng.randint(1, _MONTH_DAYS[month - 1])
    return f"{year:04d}-{month:02d}-{day:02d}"


def _phone(rng: random.Random, dial: str) -> str:
    length = 9 if len(dial) <= 3 else 8
    national = rng.choice("123456789") + "".join(rng.choice("0123456789") for _ in range(length - 1))
    if rng.random() < 0.6:  # E.164
        return f"+{dial}{national}"
    # 本地格式：空格 / 连字符分隔，national significant number 不变
    sep = rng.choice([" ", "-"])
    return f"0{national[:1]}{sep}{national[1:4]}{sep}{national[4:]}" if len(dial) <= 3 else f"{national[:3]}{sep}{national[3:5]}{sep}{national[5:]}"


def _record_id(name: str, country: str, ordinal: int) -> str:
    digest = hashlib.sha256(f"{name}|{country}|{ordinal}".encode("utf-8")).hexdigest()
    return f"fused-{digest[:8]}"


def _make_db_record(rng: random.Random, name: str, country: str, group: str, ordinal: int) -> dict:
    local, romanized = split_forms(name, group)
    meta = _COUNTRY_META[country]
    return {
        "record_id": _record_id(name, country, ordinal),
        "name_local": local,
        "name_romanized": romanized,
        "dob": _iso_date(rng),
        "city": rng.choice(meta["cities"]),
        "phone": _phone(rng, meta["dial"]),
        "country": country,
        # 内部分组标记，写 CSV 时被 extrasaction="ignore" 丢弃，只用于配额/统计。
        "_group": group,
    }


# ---------------------------------------------------------------------------
# 查询扰动
# ---------------------------------------------------------------------------


def _typo(rng: random.Random, text: str) -> str:
    """相邻字符换位 —— 对任何文字系统都成立的拼写 typo。"""
    chars = list(text)
    alpha_idx = [i for i, ch in enumerate(chars) if ch.isalpha()]
    if len(alpha_idx) < 2:
        return text
    i = rng.choice(alpha_idx[:-1])
    chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


def _transpose_year(dob: str) -> str:
    """年份内两位数字对调 —— 仍是合法日期但是不同的日期（真实录入错误）。"""
    year, month, day = dob.split("-")
    digits = list(year)
    i = rng_global.randrange(3)
    if digits[i] != digits[i + 1]:
        digits[i], digits[i + 1] = digits[i + 1], digits[i]
    return "-".join(["".join(digits), month, day])


rng_global = random.Random(0)  # _transpose_year 的辅助；主流程会先 seed


def perturb_query(rng: random.Random, record: dict, carry: str) -> dict:
    """生成一条正例查询：只带一种姓名形式 + 至多两处属性扰动。

    ``carry`` ∈ {"local", "romanized"}。扰动组合刻意保持「难但公平」：
    与 ``generate_synthetic_attributes.py`` 一样每条最多动 2 个属性，
    且姓名 typo 与 dob 扰动不同时上 —— 两个高权重属性一起错会被压到阈值下，
    不构成演示价值。
    """
    query = {k: record[k] for k in DB_FIELDS if k != "record_id"}
    query["name_local"] = record["name_local"] if carry == "local" else ""
    query["name_romanized"] = record["name_romanized"] if carry == "romanized" else ""

    global rng_global
    rng_global = rng
    modes = ["typo", "dob_transpose", "dob_format", "dob_missing", "phone_missing", "city_case"]
    rng.shuffle(modes)
    applied = 0
    for mode in modes:
        if applied >= 2:
            break
        if mode == "typo":
            column = "name_local" if carry == "local" else "name_romanized"
            query[column] = _typo(rng, query[column])
            applied += 1
        elif mode == "dob_transpose":
            query["dob"] = _transpose_year(record["dob"])
            applied += 1
        elif mode == "dob_format":
            query["dob"] = record["dob"].replace("-", rng.choice(["/", "."]))
            applied += 1
        elif mode == "dob_missing":
            query["dob"] = ""
            applied += 1
        elif mode == "phone_missing":
            query["phone"] = ""
            applied += 1
        elif mode == "city_case":
            query["city"] = record["city"].upper() if rng.random() < 0.5 else record["city"].casefold()
            applied += 1
    return query


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def load_sage_groups(path: Path = SAGE_CSV) -> dict[str, list[tuple[str, str]]]:
    """读 SAGE CSV，按 ``classify_name`` 分组；组内去重 ``(name, country)``。"""
    groups: dict[str, list[tuple[str, str]]] = {key: [] for key in _SCRIPT_QUOTAS}
    seen: set[tuple[str, str]] = set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            name = " ".join((row.get("candidate_name") or "").split())
            country = (row.get("country") or "").strip()
            if not name or country not in _COUNTRY_META or (name, country) in seen:
                continue
            seen.add((name, country))
            group = classify_name(name)
            if group in groups:
                groups[group].append((name, country))
    for key, rows in groups.items():
        if not rows:
            raise SystemExit(f"no SAGE names classified into group {key!r}")
    return groups


def build(records: int, positives: int, negatives: int, seed: int, out_dir: Path) -> dict:
    rng = random.Random(seed)
    groups = load_sage_groups()

    # 配额按 records 等比缩放（最小组至少保留 5 条，避免小样本下某文字系统消失）。
    total_quota = sum(_SCRIPT_QUOTAS.values())
    scaled = {k: max(5, round(v * records / total_quota)) for k, v in _SCRIPT_QUOTAS.items()}
    # 修正四舍五入误差，让总和精确等于 records。
    drift = records - sum(scaled.values())
    for key in sorted(scaled, key=lambda k: -_SCRIPT_QUOTAS[k]):
        if drift == 0:
            break
        step = 1 if drift > 0 else -1
        if scaled[key] + step >= 5:
            scaled[key] += step
            drift -= step

    database: list[dict] = []
    used: set[tuple[str, str]] = set()
    ordinal = 0
    shortfall = 0
    for group, quota in scaled.items():
        pool = [entry for entry in groups[group] if entry not in used]
        rng.shuffle(pool)
        # 源数据去重后某些组（han_mixed 只有 22 条唯一行）可能不够配额 —— 全取，
        # 缺额从富余组回补。
        shortfall += quota - min(quota, len(pool))
        for name, country in pool[:quota]:
            used.add((name, country))
            ordinal += 1
            database.append(_make_db_record(rng, name, country, group, ordinal))
    if shortfall:
        leftover = [
            entry
            for key, rows in groups.items()
            for entry in rows
            if entry not in used
        ]
        rng.shuffle(leftover)
        for name, country in leftover[:shortfall]:
            used.add((name, country))
            ordinal += 1
            database.append(_make_db_record(rng, name, country, classify_name(name), ordinal))
    assert len(database) == records and len({r["record_id"] for r in database}) == records

    # ---- 正例：数据库记录的扰动副本，姓名只带一种形式 ----
    positives_plan: list[tuple[dict, str]] = []
    planned_ids: set[str] = set()
    per_group = max(1, round(positives / len(scaled)))
    by_group: dict[str, list[dict]] = {key: [] for key in scaled}
    for record in database:
        by_group[record["_group"]].append(record)
    for group, rows in by_group.items():
        rng.shuffle(rows)
        for record in rows[:per_group]:
            carry = "romanized" if record["name_romanized"] and rng.random() < 0.5 else "local"
            positives_plan.append((record, carry))
            planned_ids.add(record["record_id"])

    # 分组配额向上取整可能略超目标数 —— 先截断再补特例。
    positives_plan = positives_plan[:positives]
    planned_ids = {record["record_id"] for record, _ in positives_plan}

    # 去重音拉丁查询（"argyelan janos" 对库里 "Árgyelán János"）：即重音记录的
    # romanized-only 查询，保证至少 5 条；它们天然就是"去重音拉丁查询"。
    accented = [
        r
        for r in database
        if r["_group"] == "latin_accented" and r["name_romanized"]
    ]
    have_folded = sum(
        1 for rec, carry in positives_plan if carry == "romanized" and rec["_group"] == "latin_accented"
    )
    for record in accented:
        if have_folded >= 5 or len(positives_plan) >= positives:
            break
        if record["record_id"] not in planned_ids:
            positives_plan.append((record, "romanized"))
            planned_ids.add(record["record_id"])
            have_folded += 1

    # 分组配额可能凑不满目标数（小组记录数 < 配额）—— 从其余记录补足。
    remainder = [r for r in database if r["record_id"] not in planned_ids]
    rng.shuffle(remainder)
    for record in remainder:
        if len(positives_plan) >= positives:
            break
        carry = "romanized" if record["name_romanized"] and rng.random() < 0.5 else "local"
        positives_plan.append((record, carry))
        planned_ids.add(record["record_id"])
    rng.shuffle(positives_plan)

    positive_rows: list[tuple[dict, str]] = []
    for index, (record, carry) in enumerate(positives_plan, start=1):
        query = perturb_query(rng, record, carry)
        positive_rows.append(({"query_id": f"q{index:02d}", **query}, record["record_id"]))

    # ---- 负例：库里不存在的不相干 SAGE 记录（同姓名池、未被使用过） ----
    negative_rows: list[tuple[dict, str]] = []
    pool_all = [entry for key, rows in groups.items() for entry in rows if entry not in used]
    rng.shuffle(pool_all)
    for offset in range(negatives):
        name, country = pool_all[offset]
        record = _make_db_record(rng, name, country, classify_name(name), ordinal + offset + 1)
        query = {k: record[k] for k in DB_FIELDS if k != "record_id"}
        negative_rows.append(({"query_id": "", **query}, ""))

    # 负例均摊混入正例之间（同 generate_synthetic_attributes.py 的理由：
    # --limit 取前缀时正负比例不失真），query_id 按最终顺序编号。
    combined: list[tuple[dict, str]] = list(positive_rows)
    step = len(positive_rows) / negatives if negatives else 0.0
    inserted = 0
    final: list[tuple[dict, str]] = []
    for position, (query, truth) in enumerate(combined):
        if negatives and step and position >= (inserted + 1) * step and inserted < negatives:
            final.append(negative_rows[inserted])
            inserted += 1
        final.append((query, truth))
    final.extend(negative_rows[inserted:])

    queries: list[dict] = []
    labels: list[dict] = []
    for index, (query, truth) in enumerate(final, start=1):
        query = dict(query)
        query["query_id"] = f"q{index:02d}"
        queries.append(query)
        labels.append(
            {
                "query_id": query["query_id"],
                # 列名沿用 generate_synthetic_attributes.py 的约定：
                # multi_attribute.dataset.apply_labels 默认读 true_entity_id。
                "true_entity_id": truth,
                "label": "true" if truth else "false",
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    _write(out_dir / "database.csv", DB_FIELDS, database)
    _write(out_dir / "queries.csv", QUERY_FIELDS, queries)
    _write(out_dir / "labels.csv", ["query_id", "true_entity_id", "label"], labels)

    script_counts: dict[str, int] = {}
    for record in database:
        key = record["_group"]
        script_counts[key] = script_counts.get(key, 0) + 1
    return {
        "database": len(database),
        "queries": len(queries),
        "positives": len(positive_rows),
        "negatives": negatives,
        "script_counts": script_counts,
    }


def _write(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, default=300, help="database records (default 300)")
    parser.add_argument("--positives", type=int, default=40, help="positive queries (default 40)")
    parser.add_argument("--negatives", type=int, default=10, help="negative queries (default 10)")
    parser.add_argument("--seed", type=int, default=20261014, help="random seed (default 20261014)")
    parser.add_argument(
        "--out",
        default=str(ROOT / "data" / "fused_multilingual"),
        help="output directory (default data/fused_multilingual; NOT gitignored)",
    )
    args = parser.parse_args()

    if args.positives > args.records:
        raise SystemExit("--positives must be <= --records")
    stats = build(args.records, args.positives, args.negatives, args.seed, Path(args.out))
    print(f"written to {args.out}")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
