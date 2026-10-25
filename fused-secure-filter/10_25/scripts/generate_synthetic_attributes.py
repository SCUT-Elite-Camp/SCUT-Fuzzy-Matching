"""生成带姓名/出生日期/电话/邮箱/地址/性别的合成测试数据。

**为什么需要它**：公开可下载、含真实电话号码且规模上千的干净数据集基本不存在
（隐私原因，见 ``docs/脚本汇总.md`` 第 3.5 节）。所以电话号码这一路只能靠合成数据。

**姓名池来自真实数据**：直接复用仓库已有的 ``data/common-forenames-by-country.csv``
（2480 行，自带 ``Gender`` 字段，所以性别属性不是编的）。姓氏池是内置的合成列表。

产出三个文件（默认在 ``dataset/synthetic/``，该目录已 gitignore）::

    entities.csv   库侧干净记录
    queries.csv    查询侧扰动副本（含 query_id）
    labels.csv     query_id,true_entity_id,label 真值

用法::

    python scripts/generate_synthetic_attributes.py --records 5000 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORENAMES_CSV = ROOT / "data" / "common-forenames-by-country.csv"

ENTITY_FIELDS = [
    "entity_id",
    "given_name",
    "surname",
    "full_name",
    "date_of_birth",
    "phone",
    "email",
    "address",
    "gender",
]

# 内置合成姓氏池（forenames 文件里没有姓氏）。
_SURNAMES = (
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
    "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
    "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Walker",
    "Young", "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores",
    "Green", "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell",
    "Carter", "Roberts", "Chen", "Wang", "Li", "Zhang", "Liu", "Yang", "Huang",
    "Patel", "Singh", "Kumar", "Sharma", "Ali", "Khan", "Ahmed", "Kim", "Park",
    "Sato", "Suzuki", "Takahashi", "Muller", "Schmidt", "Fischer", "Weber", "Rossi",
    "Ferrari", "Russo", "Dubois", "Moreau", "Laurent", "Ross", "Novak", "Horvath",
)

_STREETS = (
    "Main", "Oak", "Pine", "Maple", "Cedar", "Elm", "Washington", "Lake", "Hill",
    "Park", "River", "Church", "High", "Mill", "Walnut", "Spring", "Sunset",
    "Ridge", "Meadow", "Franklin",
)
_STREET_TYPES = (("Street", "St"), ("Avenue", "Ave"), ("Road", "Rd"), ("Drive", "Dr"))
_SUBURBS = (
    "Springfield", "Riverton", "Fairview", "Georgetown", "Clayton", "Ashfield",
    "Kingston", "Brookside", "Northgate", "Westport", "Eastwood", "Hillcrest",
)
_STATES = ("NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT")
_DIAL_CODES = ("1", "44", "61", "86", "91", "49", "33", "81", "65", "60")
_EMAIL_DOMAINS = ("example.com", "mail.example.net", "webmail.example.org")

# 姓名扰动里用到的中间名，制造 "J. Smith" / "John A Smith" 这类真实变体。
_MIDDLE_INITIALS = "ABCDEFGHJKLMNPRSTW"

_MONTH_DAYS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


# ---------------------------------------------------------------------------
# 姓名池
# ---------------------------------------------------------------------------


def load_given_names(path: Path = FORENAMES_CSV) -> list[tuple[str, str]]:
    """读 ``(Romanized Name, Gender)`` 列表。

    罗马化名为空时退回本地名；两者都空的行走掉。
    """

    if not path.exists():
        raise SystemExit(f"forenames file not found: {path}")
    pool: list[tuple[str, str]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            name = (row.get("Romanized Name") or "").strip() or (
                row.get("Localized Name") or ""
            ).strip()
            gender = (row.get("Gender") or "").strip().upper()
            if name and gender in {"F", "M"}:
                pool.append((name, gender))
    if not pool:
        raise SystemExit(f"no usable names in {path}")
    return pool


# ---------------------------------------------------------------------------
# 单条记录的生成
# ---------------------------------------------------------------------------


def _iso_date(rng: random.Random, year_lo: int = 1940, year_hi: int = 2005) -> str:
    year = rng.randint(year_lo, year_hi)
    month = rng.randint(1, 12)
    day = rng.randint(1, _MONTH_DAYS[month - 1])
    return f"{year:04d}-{month:02d}-{day:02d}"


def _digits(rng: random.Random, length: int) -> str:
    return "".join(rng.choice("0123456789") for _ in range(length))


def _make_entity(rng: random.Random, pool: list[tuple[str, str]], index: int) -> dict[str, str]:
    given, gender = rng.choice(pool)
    surname = rng.choice(_SURNAMES)
    dial = rng.choice(_DIAL_CODES)
    national = _digits(rng, 9 if dial != "1" else 10)
    email = f"{given}.{surname}".casefold().replace(" ", "") + _digits(rng, 2)
    return {
        "entity_id": f"ent-{index:06d}",
        "given_name": given,
        "surname": surname,
        "full_name": f"{given} {surname}",
        "date_of_birth": _iso_date(rng),
        "phone": f"+{dial}{national}",
        "email": f"{email}@{rng.choice(_EMAIL_DOMAINS)}",
        "address": (
            f"{rng.randint(1, 9999)} {rng.choice(_STREETS)} {rng.choice(_STREET_TYPES)[0]}, "
            f"{rng.choice(_SUBURBS)} {rng.choice(_STATES)} {_digits(rng, 4)}"
        ),
        "gender": gender,
    }


def _make_negative(rng: random.Random, pool: list[tuple[str, str]], index: int) -> dict[str, str]:
    """独立生成的另一个人 —— 用作真值为「不匹配」的查询。"""

    return _make_entity(rng, pool, index)


# ---------------------------------------------------------------------------
# 扰动
# ---------------------------------------------------------------------------


def _perturb_name(rng: random.Random, record: dict[str, str]) -> None:
    given, surname = record["given_name"], record["surname"]
    mode = rng.choice(("swap", "typo", "initial", "case", "drop", "spacing"))
    if mode == "swap" and len(given) > 2:
        i = rng.randrange(len(given) - 1)
        given = given[:i] + given[i + 1] + given[i] + given[i + 2 :]
    elif mode == "typo" and len(surname) > 3:
        i = rng.randrange(len(surname) - 1)
        surname = surname[:i] + surname[i + 1] + surname[i] + surname[i + 2 :]
    elif mode == "initial":
        given = f"{given[0]}. {rng.choice(_MIDDLE_INITIALS)}"
    elif mode == "case":
        given, surname = given.upper(), surname.casefold()
    elif mode == "drop":
        surname = ""
    elif mode == "spacing":
        given = f"  {given} "
    record["given_name"], record["surname"] = given, surname
    record["full_name"] = f"{given} {surname}".strip()


def _perturb_dob(rng: random.Random, record: dict[str, str]) -> None:
    iso = record["date_of_birth"]
    year, month, day = iso.split("-")
    mode = rng.choice(("ymd", "dmy", "compact", "dotted", "transpose_year"))
    if mode == "ymd":
        record["date_of_birth"] = f"{year}/{month}/{day}"
    elif mode == "dmy":
        record["date_of_birth"] = f"{day}/{month}/{year}"
    elif mode == "compact":
        record["date_of_birth"] = f"{year}{month}{day}"
    elif mode == "dotted":
        record["date_of_birth"] = f"{year}.{month}.{day}"
    else:
        # 年份两位数字对调 —— 仍是合法日期，但是不同的日期（真实录入错误）。
        a, b = 0, 1
        if year[2] != year[3]:
            a, b = 2, 3
        y = list(year)
        y[a], y[b] = y[b], y[a]
        record["date_of_birth"] = "".join(y) + month + day


def _perturb_phone(rng: random.Random, record: dict[str, str]) -> None:
    digits = "".join(c for c in record["phone"] if c.isdigit())
    mode = rng.choice(("plain", "dashed", "dotted", "spaced", "parens"))
    if mode == "plain":
        record["phone"] = digits
    elif mode == "dashed":
        record["phone"] = f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
    elif mode == "dotted":
        record["phone"] = f"{digits[:3]}.{digits[3:6]}.{digits[6:]}"
    elif mode == "spaced":
        record["phone"] = f"+{digits[0]} {digits[1:4]} {digits[4:7]} {digits[7:]}"
    else:
        record["phone"] = f"+{digits[0]} ({digits[1:4]}) {digits[4:7]}-{digits[7:]}"


def _perturb_email(rng: random.Random, record: dict[str, str]) -> None:
    local, _, domain = record["email"].partition("@")
    mode = rng.choice(("upper", "dot", "none"))
    if mode == "upper":
        local = local.upper()
    elif mode == "dot" and len(local) > 3:
        local = f"{local[:2]}.{local[2:]}"
    elif mode == "none":
        domain = "mail.example.com"
    record["email"] = f"{local}@{domain}"


def _perturb_address(rng: random.Random, record: dict[str, str]) -> None:
    text = record["address"]
    mode = rng.choice(("abbrev", "expand", "case", "none"))
    if mode == "abbrev":
        for full, short in _STREET_TYPES:
            text = text.replace(f" {full},", f" {short},")
    elif mode == "expand":
        for full, short in _STREET_TYPES:
            text = text.replace(f" {short},", f" {full},")
    elif mode == "case":
        text = text.upper()
    else:
        text = text.replace(",", "")
    record["address"] = text


def _perturb_gender(rng: random.Random, record: dict[str, str]) -> None:
    # 精确属性很脆（一个字符错就 0 分），所以只以很低概率扰动。
    record["gender"] = "F" if record["gender"] == "M" else "M"


_PERTURBERS = {
    "name": _perturb_name,
    "date_of_birth": _perturb_dob,
    "phone": _perturb_phone,
    "email": _perturb_email,
    "address": _perturb_address,
    "gender": _perturb_gender,
}

# 扰动顺序固定，保证同种子可复现。
_PERTURB_ORDER = ("name", "date_of_birth", "phone", "email", "address", "gender")

# 每条副本最多扰动几个属性 —— 扰太多就低于阈值了，不构成「难但公平」的用例。
_MAX_PERTURBED = 2
_GENDER_PERTURB_RATE = 0.08


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def _duplicate(rng: random.Random, entity: dict[str, str], query_id: str) -> dict[str, str]:
    record = dict(entity)
    record["entity_id"] = ""  # 查询侧不应带真值 id
    candidates = [k for k in _PERTURB_ORDER if k != "gender"]
    rng.shuffle(candidates)
    applied = 0
    for key in candidates:
        if applied >= _MAX_PERTURBED:
            break
        if rng.random() < 0.5:
            _PERTURBERS[key](rng, record)
            applied += 1
    if rng.random() < _GENDER_PERTURB_RATE:
        _perturb_gender(rng, record)
    record["query_id"] = query_id
    return record


def generate(
    records: int,
    seed: int,
    negative_ratio: float,
    out_dir: Path,
    pool: list[tuple[str, str]] | None = None,
) -> dict[str, int]:
    rng = random.Random(seed)
    pool = pool if pool is not None else load_given_names()

    entities = [_make_entity(rng, pool, i + 1) for i in range(records)]

    # 正例（每条库记录一个扰动副本）。
    positive_rows: list[tuple[dict[str, str], str]] = [
        (_duplicate(rng, entity, ""), entity["entity_id"]) for entity in entities
    ]

    # 负例：独立生成的另一个人，库内没有对应记录，真值留空。
    negative_count = round(records * negative_ratio)
    negative_rows: list[dict[str, str]] = []
    for offset in range(negative_count):
        record = _make_negative(rng, pool, records + offset + 1)
        record["entity_id"] = ""
        negative_rows.append(record)

    # 负例**均摊混入**正例之间，而不是全部堆在末尾 —— 否则 --limit 取前缀样本时
    # 只会拿到正例，一致率虚高。query_id 按混入后的最终顺序编号。
    combined: list[tuple[dict[str, str], str]] = []
    step = (len(positive_rows) / negative_count) if negative_count else 0.0
    next_negative = 0
    for position, (record, truth) in enumerate(positive_rows):
        if negative_count and step and position >= next_negative * step:
            combined.append((negative_rows[next_negative], ""))
            next_negative += 1
        combined.append((record, truth))
    combined.extend((row, "") for row in negative_rows[next_negative:])

    queries: list[dict[str, str]] = []
    labels: list[dict[str, str]] = []
    for index, (record, truth) in enumerate(combined, start=1):
        query_id = f"q-{index:06d}"
        record["query_id"] = query_id
        queries.append(record)
        labels.append(
            {
                "query_id": query_id,
                "true_entity_id": truth,
                "label": "true" if truth else "false",
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    _write(out_dir / "entities.csv", ENTITY_FIELDS, entities)
    _write(out_dir / "queries.csv", ["query_id", *ENTITY_FIELDS], queries)
    _write(out_dir / "labels.csv", ["query_id", "true_entity_id", "label"], labels)

    return {
        "entities": len(entities),
        "queries": len(queries),
        "positives": len(entities),
        "negatives": negative_count,
    }


def _write(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, default=5000, help="number of database records (default 5000)")
    parser.add_argument("--seed", type=int, default=42, help="random seed (default 42)")
    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=0.25,
        help="extra negative queries as a fraction of the database (default 0.25)",
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "dataset" / "synthetic"),
        help="output directory (default baseline_Integration/dataset/synthetic, gitignored)",
    )
    args = parser.parse_args()

    if args.records < 1:
        raise SystemExit("--records must be >= 1")
    if args.negative_ratio < 0:
        raise SystemExit("--negative-ratio must be >= 0")

    out_dir = Path(args.out)
    stats = generate(args.records, args.seed, args.negative_ratio, out_dir)
    print(f"written to {out_dir}")
    for key in ("entities", "queries", "positives", "negatives"):
        print(f"  {key:>10}: {stats[key]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
