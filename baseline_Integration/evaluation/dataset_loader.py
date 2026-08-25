"""
数据集加载：NCVR、图书馆目录、US Census、以及从常见名字合成数据。
支持多语言文本标准化、音译和双通道编码。
"""

from __future__ import annotations

import csv
import os
import random
import re
from pathlib import Path

import numpy as np

# 导入音译模块
from utils.transliteration import transliterate_text, detect_language, is_chinese_script


# ============================================================
# 多语言文本标准化
# ============================================================

def normalize_text(text: str, lang: str = "en") -> str:
    """
    根据语言对姓名进行标准化（清洗），用于加载真实数据时统一格式。
    注意：这里不做噪声添加，只做规范化。

    Args:
        text: 原始姓名
        lang: 语言类型，支持 "en", "zh", "pinyin", "es", "fr", "de", "ar" 等

    Returns:
        标准化后的姓名
    """
    if not text:
        return text

    import unicodedata

    # 1. Unicode 规范化（统一全角/半角、组合字符等）
    text = unicodedata.normalize("NFKC", text)

    # 2. 去除首尾空格、压缩多余空格
    text = " ".join(text.split())

    # 3. 根据语言做特定清洗
    if lang in ["en", "es", "fr", "de", "it", "pt"]:
        # 拉丁字母：转小写、去除重音
        text = text.lower()
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")

    elif lang in ["zh", "ja", "ko"]:
        # 中日韩：保持原样（不做大小写转换）
        pass

    elif lang == "pinyin":
        # 拼音：转小写，保留空格
        text = text.lower()

    elif lang == "ar":
        # 阿拉伯语：保持原样或标准化转写（暂不处理）
        pass

    return text


# ============================================================
# 数据集加载接口
# ============================================================

def load_dataset(name, path, lang="en", dual_channel=False, **kwargs):
    """
    Args:
        name: 数据集名称，可选 "ncvr", "ncvr_10k", "libcat", "census", "forenames"
        path: 数据集文件路径
        lang: 语言类型，用于文本标准化（默认 "en"）
        dual_channel: 是否启用双通道编码（原文 + 音译），仅对中文等有效
        **kwargs: 其他参数，传递给具体加载函数

    Returns:
        tuple: (names_A, names_B, labels)
            names_A: list of str, 查询方名单
            names_B: list of str, 响应方名单
            labels: list of bool, 对于每个查询，是否存在真实匹配
    """
    dataset_name = name.lower()
    if dataset_name == "ncvr":
        return _load_ncvr(path, lang=lang, dual_channel=dual_channel)
    elif dataset_name == "ncvr_10k":
        return _load_ncvr_10k(path, lang=lang, dual_channel=dual_channel)
    elif dataset_name == "libcat":
        return _load_libcat(path, lang=lang, dual_channel=dual_channel)
    elif dataset_name == "census":
        return _load_census(path, lang=lang, dual_channel=dual_channel)
    elif dataset_name == "forenames":
        return _load_forenames(path, lang=lang, dual_channel=dual_channel, **kwargs)
    else:
        raise ValueError(
            f"Unknown dataset: {name}. Choose from 'ncvr', 'ncvr_10k', "
            "'libcat', 'census', 'forenames'."
        )


# ============================================================
# NCVR 10K 数据加载
# ============================================================

def _load_ncvr_10k(path, lang="en", dual_channel=False):
    """Load the curated two-file NCVR 10K subset."""
    database_path, queries_path = _resolve_ncvr_10k_paths(path)
    database_rows = _read_csv_records(database_path)
    query_rows = _read_csv_records(queries_path)

    _require_columns(database_rows, database_path, {"ncid", "full_name"})
    _require_columns(query_rows, queries_path, {"query_ncid", "query_name", "label"})

    names_B = [
        normalize_text(row["full_name"].strip(), lang=lang)
        for row in database_rows
        if row["full_name"].strip()
    ]
    names_A = [
        normalize_text(row["query_name"].strip(), lang=lang)
        for row in query_rows
        if row["query_name"].strip()
    ]
    labels = [
        _parse_bool(row["label"])
        for row in query_rows
        if row["query_name"].strip()
    ]

    if not names_B:
        raise ValueError(f"NCVR 10K database is empty: {database_path}")
    if not names_A:
        raise ValueError(f"NCVR 10K queries are empty: {queries_path}")
    if len(names_A) != len(labels):
        raise ValueError("NCVR 10K query names and labels are misaligned")

    # 双通道编码
    if dual_channel and lang == "zh":
        # 注意：NCVR 数据主要是英文，这里仅作为示例框架
        # 实际使用时，需要根据数据内容决定是否生成音译
        print("⚠️ NCVR 数据集通常不含中文，双通道模式可能不适用。")
        return names_A, names_B, labels

    return names_A, names_B, labels


def _resolve_ncvr_10k_paths(path):
    root = Path(path)
    if root.is_file():
        raise ValueError(
            "ncvr_10k expects a directory containing both "
            "ncvr_10k_database.csv and ncvr_10k_queries.csv"
        )

    candidates = [root, root / "ncvr_10k"]
    for candidate in candidates:
        database_path = candidate / "ncvr_10k_database.csv"
        queries_path = candidate / "ncvr_10k_queries.csv"
        if database_path.exists() and queries_path.exists():
            return database_path, queries_path

    raise FileNotFoundError(
        "Could not find NCVR 10K files. Expected either "
        f"{root / 'ncvr_10k_database.csv'} and {root / 'ncvr_10k_queries.csv'}, "
        f"or {root / 'ncvr_10k' / 'ncvr_10k_database.csv'} and "
        f"{root / 'ncvr_10k' / 'ncvr_10k_queries.csv'}."
    )


def _read_csv_records(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _require_columns(rows, path, expected_columns):
    if not rows:
        raise ValueError(f"CSV file has no data rows: {path}")
    actual_columns = set(rows[0].keys())
    missing = expected_columns - actual_columns
    if missing:
        raise ValueError(
            f"CSV file {path} is missing required columns: {sorted(missing)}"
        )


def _parse_bool(value):
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Invalid boolean label: {value!r}")


# ============================================================
# 其他数据集（模拟）
# ============================================================

def _load_ncvr(path, lang="en", dual_channel=False):
    """模拟 NCVR 数据加载。实际使用时根据 NCVR CSV 文件格式修改。"""
    if os.path.exists(path):
        import pandas as pd
        df = pd.read_csv(path)
        names_A = [normalize_text(x, lang=lang) for x in df['name_A'].tolist()]
        names_B = [normalize_text(x, lang=lang) for x in df['name_B'].tolist()]
        labels = df['match_label'].astype(bool).tolist()
        if dual_channel and lang == "zh":
            # 示例：生成音译版本
            names_A_trans = [transliterate_text(x, lang="zh") for x in names_A]
            names_B_trans = [transliterate_text(x, lang="zh") for x in names_B]
            names_A = names_A + names_A_trans
            names_B = names_B + names_B_trans
            labels = labels + labels
        return names_A, names_B, labels
    else:
        print(f"NCVR data not found at {path}, generating mock data for testing.")
        np.random.seed(42)
        n_queries = 100
        n_db = 1000
        names_A = [normalize_text(f"Person_{i}", lang=lang) for i in range(n_queries)]
        names_B = [normalize_text(f"Person_{j}", lang=lang) for j in range(n_db)]
        labels = [i < 10 for i in range(n_queries)]
        if dual_channel and lang == "zh":
            # mock 数据不产生实际音译效果，仅示意
            names_A_trans = names_A
            names_B_trans = names_B
            names_A = names_A + names_A_trans
            names_B = names_B + names_B_trans
            labels = labels + labels
        return names_A, names_B, labels


def _load_libcat(path, lang="en", dual_channel=False):
    """模拟图书馆目录数据加载"""
    if os.path.exists(path):
        import pandas as pd
        df = pd.read_csv(path)
        names_A = [normalize_text(x, lang=lang) for x in df['title_A'].tolist()]
        names_B = [normalize_text(x, lang=lang) for x in df['title_B'].tolist()]
        labels = df['match_label'].astype(bool).tolist()
        if dual_channel and lang == "zh":
            # 示例：生成音译版本
            names_A_trans = [transliterate_text(x, lang="zh") for x in names_A]
            names_B_trans = [transliterate_text(x, lang="zh") for x in names_B]
            names_A = names_A + names_A_trans
            names_B = names_B + names_B_trans
            labels = labels + labels
        return names_A, names_B, labels
    else:
        print(f"LibCat data not found at {path}, generating mock data.")
        np.random.seed(123)
        n_queries = 80
        n_db = 2000
        names_A = [normalize_text(f"BookTitle_{i}", lang=lang) for i in range(n_queries)]
        names_B = [normalize_text(f"BookTitle_{j}", lang=lang) for j in range(n_db)]
        labels = [i < 8 for i in range(n_queries)]
        if dual_channel and lang == "zh":
            names_A_trans = names_A
            names_B_trans = names_B
            names_A = names_A + names_A_trans
            names_B = names_B + names_B_trans
            labels = labels + labels
        return names_A, names_B, labels


def _load_census(path, lang="en", dual_channel=False):
    """模拟美国普查数据加载"""
    if os.path.exists(path):
        import pandas as pd
        df = pd.read_csv(path)
        names_A = [normalize_text(x, lang=lang) for x in df['name_A'].tolist()]
        names_B = [normalize_text(x, lang=lang) for x in df['name_B'].tolist()]
        labels = df['match_label'].astype(bool).tolist()
        if dual_channel and lang == "zh":
            names_A_trans = [transliterate_text(x, lang="zh") for x in names_A]
            names_B_trans = [transliterate_text(x, lang="zh") for x in names_B]
            names_A = names_A + names_A_trans
            names_B = names_B + names_B_trans
            labels = labels + labels
        return names_A, names_B, labels
    else:
        print(f"Census data not found at {path}, generating mock data with typos.")
        np.random.seed(456)
        common_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis"]
        n_queries = 50
        n_db = 500
        names_A = []
        names_B = []
        labels = []
        for i in range(n_queries):
            if i < 20:
                orig = common_names[i % len(common_names)]
                names_A.append(normalize_text(orig, lang=lang))
                names_B.append(normalize_text(orig, lang=lang))
                labels.append(True)
            else:
                names_A.append(normalize_text(f"Random_{i}", lang=lang))
                names_B.append(normalize_text(f"Random_{i+100}", lang=lang))
                labels.append(False)
        while len(names_B) < n_db:
            names_B.append(normalize_text("Extra_" + str(len(names_B)), lang=lang))
        if dual_channel and lang == "zh":
            # mock 数据不产生实际音译效果，仅示意
            names_A_trans = names_A
            names_B_trans = names_B
            names_A = names_A + names_A_trans
            names_B = names_B + names_B_trans
            labels = labels + labels
        return names_A, names_B, labels


# ============================================================
# Forenames 合成数据（支持语言标准化和双通道）
# ============================================================

def _load_forenames(
    csv_path,
    lang="en",
    n_B=1000,
    n_match=100,
    n_nonmatch=100,
    fuzzy_ratio=0.3,
    random_seed=42,
    dual_channel=False,
    **kwargs
):
    """
    从 common-forenames-by-country.csv 合成模糊姓名匹配数据集。

    Args:
        csv_path: CSV 文件路径
        lang: 语言类型，用于标准化
        n_B: 响应方 B 的数据库大小
        n_match: 匹配查询的数量（标签为 True）
        n_nonmatch: 不匹配查询的数量（标签为 False）
        fuzzy_ratio: 匹配查询中被故意引入拼写错误的比例
        random_seed: 随机种子
        dual_channel: 是否启用双通道（原文 + 音译），仅对中文有效
        **kwargs: 额外参数

    Returns:
        (names_A, names_B, labels)
    """
    import pandas as pd

    # 读取 CSV
    if not os.path.exists(csv_path):
        print(f"Forenames CSV not found at {csv_path}, falling back to mock data.")
        return _load_ncvr(csv_path, lang=lang, dual_channel=dual_channel)

    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    name_pool = df['Romanized Name'].dropna().unique().tolist()
    # 标准化
    name_pool = [normalize_text(n, lang=lang) for n in name_pool if isinstance(n, str) and n.strip() != '']

    if len(name_pool) < n_B + n_match + n_nonmatch:
        print(f"Forenames pool too small ({len(name_pool)}), allowing duplicates.")

    np.random.seed(random_seed)
    random.seed(random_seed)

    # 1. 构建 B 的数据库
    if len(name_pool) >= n_B:
        names_B = list(np.random.choice(name_pool, n_B, replace=False))
    else:
        names_B = list(np.random.choice(name_pool, n_B, replace=True))

    # 2. 构建匹配查询（从 B 中选，引入模糊变体）
    matched_originals = list(np.random.choice(names_B, n_match, replace=False))
    names_A_matched = []
    for name in matched_originals:
        if random.random() < fuzzy_ratio and len(name) > 2:
            pos = random.randint(0, len(name)-1)
            new_char = chr(random.randint(97, 122))
            fuzzy_name = name[:pos] + new_char + name[pos+1:]
            names_A_matched.append(fuzzy_name)
        else:
            names_A_matched.append(name)

    # 3. 构建不匹配查询
    candidate_nonmatch = [n for n in name_pool if n not in names_B]
    if len(candidate_nonmatch) < n_nonmatch:
        candidate_nonmatch = name_pool
    names_A_nonmatch = list(np.random.choice(candidate_nonmatch, n_nonmatch, replace=False))

    # 4. 合并查询集
    names_A = names_A_matched + names_A_nonmatch
    labels = [True] * n_match + [False] * n_nonmatch

    # 5. 标准化输出（保证一致性）
    names_A = [normalize_text(n, lang=lang) for n in names_A]
    names_B = [normalize_text(n, lang=lang) for n in names_B]

    # 6. 双通道编码
    if dual_channel and (lang == "zh" or any(is_chinese_script(n) for n in names_A)):
        # 只有数据中包含中文时才启用音译
        print("🔄 双通道模式启用：生成音译版本")
        names_A_trans = [transliterate_text(n, lang="zh") for n in names_A]
        names_B_trans = [transliterate_text(n, lang="zh") for n in names_B]
        names_A = names_A + names_A_trans
        names_B = names_B + names_B_trans
        labels = labels + labels
        print(f"  查询数量翻倍: {len(names_A)} 条")

    # 注意：这里不主动添加扰动，扰动请使用独立的 add_noise.py
    if kwargs.get('perturbation_ratio', 0) > 0:
        print(f"⚠️ 警告: perturbation_ratio 参数已弃用，请使用 scripts/add_noise.py 生成扰动数据。")

    return names_A, names_B, labels