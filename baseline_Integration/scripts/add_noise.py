#!/usr/bin/env python3
"""
独立噪声添加工具 - 用于开发测试阶段生成模糊数据集

用法：
    python scripts/add_noise.py --input data/ncvr_10k/ncvr_10k_database.csv \
                                --output data/ncvr_10k/ncvr_10k_database_noisy.csv \
                                --ratio 0.2 \
                                --lang en

    python scripts/add_noise.py --input data/common-forenames-by-country.csv \
                                --output data/common-forenames-by-country_noisy.csv \
                                --ratio 0.25 \
                                --lang pinyin

功能：
    1. 读取原始 CSV 文件
    2. 对 full_name 列中的姓名施加随机扰动
    3. 输出新的 CSV 文件（保留原始其他列不变）
    4. 支持多种语言和扰动类型
"""

import csv
import random
import argparse
import os
import sys
from pathlib import Path

# ============================================================
# 多语言扰动配置
# ============================================================

# 中文相似字符映射（可扩展）
SIMILAR_CHINESE_CHARS = {
    "张": ["章", "彰", "樟"],
    "伟": ["玮", "炜", "纬"],
    "明": ["铭", "鸣", "洺"],
    "华": ["桦", "骅", "晔"],
    "强": ["疆", "犟"],
    "丽": ["俪", "郦"],
    "芳": ["方", "放"],
    "敏": ["闵", "悯"],
    "静": ["靖", "婧"],
    "涛": ["滔", "韬"],
    "军": ["君", "珺"],
    "勇": ["涌", "踊"],
    "刚": ["纲", "钢"],
    "杰": ["桀", "洁"],
    "峰": ["锋", "烽"],
    "辉": ["晖", "珲"],
    "玲": ["灵", "铃"],
    "秀": ["绣", "琇"],
    "英": ["瑛", "莺"],
    "兰": ["栏", "拦"],
}

# 键盘邻近映射（英文）
KEYBOARD_MAP = {
    'a': ['s', 'q', 'w', 'z'],
    'b': ['v', 'g', 'h', 'n'],
    'c': ['x', 'd', 'f', 'v'],
    'd': ['s', 'f', 'e', 'c'],
    'e': ['w', 'r', 'd', 's'],
    'f': ['d', 'g', 'r', 'v'],
    'g': ['f', 'h', 't', 'b'],
    'h': ['g', 'j', 'y', 'n'],
    'i': ['u', 'o', 'k', 'j'],
    'j': ['h', 'k', 'u', 'm'],
    'k': ['j', 'l', 'i', 'o'],
    'l': ['k', 'o', 'p'],
    'm': ['n', 'j', 'k'],
    'n': ['b', 'm', 'h', 'j'],
    'o': ['i', 'p', 'l', 'k'],
    'p': ['o', 'l'],
    'q': ['w', 'a'],
    'r': ['e', 't', 'f', 'd'],
    's': ['a', 'd', 'w', 'x'],
    't': ['r', 'y', 'g', 'f'],
    'u': ['y', 'i', 'h', 'j'],
    'v': ['c', 'b', 'f', 'g'],
    'w': ['q', 'e', 'a', 's'],
    'x': ['z', 'c', 's', 'd'],
    'y': ['t', 'u', 'h', 'g'],
    'z': ['x', 'a', 's'],
}

# 拼音常见错误
PINYIN_ERRORS = {
    "zh": ["z", "zi", "zhi"],
    "ch": ["c", "ci", "chi"],
    "sh": ["s", "si", "shi"],
    "ang": ["an", "eng"],
    "eng": ["en", "ang"],
    "ing": ["in", "eng"],
    "an": ["ang", "en"],
    "en": ["eng", "an"],
    "ai": ["ei", "an"],
    "ei": ["ai", "en"],
}

# 重音符号替换（欧洲语言）
ACCENT_MAP = {
    "é": ["è", "ê", "e"],
    "è": ["é", "ê", "e"],
    "ê": ["é", "è", "e"],
    "à": ["á", "â", "a"],
    "á": ["à", "â", "a"],
    "â": ["á", "à", "a"],
    "ô": ["ó", "ò", "o"],
    "ó": ["ô", "ò", "o"],
    "ò": ["ô", "ó", "o"],
    "û": ["ú", "ù", "u"],
    "ú": ["û", "ù", "u"],
    "ù": ["û", "ú", "u"],
    "î": ["ï", "i"],
    "ï": ["î", "i"],
    "ç": ["c"],
    "ñ": ["n"],
}


# ============================================================
# 核心扰动函数
# ============================================================

def get_available_noise_types(lang: str) -> list:
    """根据语言返回可用的扰动类型列表"""
    if lang in ["zh", "ja", "ko"]:
        return ["replace_similar", "delete", "insert", "swap", "random_case"]
    elif lang == "pinyin":
        return ["insert", "delete", "replace", "swap", "tone_error", "random_case"]
    elif lang in ["ar"]:
        return ["delete", "insert", "swap", "transliteration_error"]
    elif lang in ["es", "fr", "de", "it", "pt"]:
        return ["insert", "delete", "replace", "swap", "accent_error", "random_case"]
    else:  # 默认英文
        return ["insert", "delete", "replace", "swap", "abbreviate_middle", "reverse_order", "double_space", "random_case"]


def apply_noise(name: str, lang: str = "en", noise_type: str = None) -> str:
    """对单个姓名施加随机扰动"""
    if not name or len(name) < 2:
        return name

    available = get_available_noise_types(lang)
    if noise_type is None:
        noise_type = random.choice(available)

    parts = name.split()
    is_multi_word = len(parts) > 1

    # ---- 英文/拉丁字母扰动 ----

    if noise_type == "insert":
        if len(name) < 3:
            return name
        pos = random.randint(0, len(name) - 1)
        char = random.choice("abcdefghijklmnopqrstuvwxyz")
        return name[:pos] + char + name[pos:]

    if noise_type == "delete":
        if len(name) <= 2:
            return name
        pos = random.randint(0, len(name) - 1)
        if pos == 0 and len(name) > 2:
            pos = random.randint(1, len(name) - 1)
        return name[:pos] + name[pos + 1:]

    if noise_type == "replace":
        if len(name) < 2:
            return name
        pos = random.randint(0, len(name) - 1)
        char_lower = name[pos].lower()
        if char_lower in KEYBOARD_MAP:
            replacement = random.choice(KEYBOARD_MAP[char_lower])
            if name[pos].isupper():
                replacement = replacement.upper()
            return name[:pos] + replacement + name[pos + 1:]
        return name

    if noise_type == "swap":
        if len(name) < 3:
            return name
        pos = random.randint(0, len(name) - 2)
        return name[:pos] + name[pos + 1] + name[pos] + name[pos + 2:]

    if noise_type == "abbreviate_middle":
        if not is_multi_word or len(parts) < 3:
            if len(parts) == 2 and len(parts[1]) > 1:
                return parts[0] + " " + parts[1][0] + "."
            return name
        abbreviated = [parts[0]]
        for i in range(1, len(parts) - 1):
            abbreviated.append(parts[i][0] + ".")
        abbreviated.append(parts[-1])
        return " ".join(abbreviated)

    if noise_type == "reverse_order":
        if not is_multi_word or len(parts) < 2:
            return name
        return ", ".join(reversed(parts))

    if noise_type == "double_space":
        if len(name) < 3:
            return name
        pos = random.randint(1, len(name) - 1)
        if name[pos - 1] != ' ' and name[pos] != ' ':
            return name[:pos] + "  " + name[pos:]
        return name

    if noise_type == "random_case":
        chars = list(name)
        flipped = 0
        for i in range(len(chars)):
            if chars[i].isalpha() and random.random() < 0.4:
                chars[i] = chars[i].swapcase()
                flipped += 1
        if flipped == 0 and len(name) > 1:
            for i in range(len(chars)):
                if chars[i].isalpha():
                    chars[i] = chars[i].swapcase()
                    break
        return "".join(chars)

    # ---- 中文扰动 ----

    if noise_type == "replace_similar":
        chars = list(name)
        for i, ch in enumerate(chars):
            if ch in SIMILAR_CHINESE_CHARS:
                chars[i] = random.choice(SIMILAR_CHINESE_CHARS[ch])
                break
        return "".join(chars)

    # ---- 拼音扰动 ----

    if noise_type == "tone_error":
        if len(name) < 2:
            return name
        # 针对拼音：替换声母或韵母
        for pattern, replacements in PINYIN_ERRORS.items():
            if pattern in name.lower():
                idx = name.lower().find(pattern)
                replacement = random.choice(replacements)
                if idx >= 0:
                    return name[:idx] + replacement + name[idx + len(pattern):]
        # 如果没有匹配到，退化为替换
        pos = random.randint(0, len(name) - 1)
        return name[:pos] + random.choice("abcdefghijklmnopqrstuvwxyz") + name[pos + 1:]

    # ---- 欧洲语言重音扰动 ----

    if noise_type == "accent_error":
        chars = list(name)
        for i, ch in enumerate(chars):
            if ch in ACCENT_MAP:
                chars[i] = random.choice(ACCENT_MAP[ch])
                break
        return "".join(chars)

    # ---- 阿拉伯语音译扰动 ----

    if noise_type == "transliteration_error":
        # 简化版：随机替换一个字母
        if len(name) < 2:
            return name
        pos = random.randint(0, len(name) - 1)
        return name[:pos] + random.choice("abcdefghijklmnopqrstuvwxyz") + name[pos + 1:]

    return name


def add_noise_to_csv(
    input_path: str,
    output_path: str,
    column: str = "full_name",
    ratio: float = 0.2,
    lang: str = "en",
    seed: int = None
) -> dict:
    """
    为 CSV 文件中的指定列添加噪声

    Args:
        input_path: 输入 CSV 文件路径
        output_path: 输出 CSV 文件路径
        column: 需要添加噪声的列名
        ratio: 扰动比例 (0-1)
        lang: 语言类型
        seed: 随机种子

    Returns:
        dict: 统计信息
    """
    if seed is not None:
        random.seed(seed)

    # 读取 CSV
    rows = []
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if column not in fieldnames:
            raise ValueError(f"列 '{column}' 不存在于 CSV 文件中。可用列: {fieldnames}")

        for row in reader:
            if random.random() < ratio:
                row[column] = apply_noise(row[column], lang=lang)
            rows.append(row)

    # 写入 CSV
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return {
        "total_rows": len(rows),
        "ratio": ratio,
        "lang": lang,
        "output_path": output_path
    }


# ============================================================
# 命令行入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="为 CSV 数据集添加姓名噪声（开发测试用）")
    parser.add_argument("--input", "-i", required=True, help="输入 CSV 文件路径")
    parser.add_argument("--output", "-o", required=True, help="输出 CSV 文件路径")
    parser.add_argument("--column", "-c", default="full_name", help="需要添加噪声的列名 (默认: full_name)")
    parser.add_argument("--ratio", "-r", type=float, default=0.2, help="扰动比例 (0-1), 默认: 0.2")
    parser.add_argument("--lang", "-l", default="en", choices=["en", "zh", "pinyin", "es", "fr", "de", "ar"],
                        help="语言类型 (默认: en)")
    parser.add_argument("--seed", type=int, default=None, help="随机种子 (用于可复现)")
    parser.add_argument("--info", action="store_true", help="显示可用扰动类型")

    args = parser.parse_args()

    if args.info:
        print("可用扰动类型（按语言）:")
        for lang in ["en", "zh", "pinyin", "es", "fr", "ar"]:
            types = get_available_noise_types(lang)
            print(f"  {lang}: {', '.join(types)}")
        return

    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        print(f"❌ 错误: 输入文件 '{args.input}' 不存在")
        sys.exit(1)

    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"📁 输入文件: {args.input}")
    print(f"📁 输出文件: {args.output}")
    print(f"📊 扰动比例: {args.ratio * 100}%")
    print(f"🌐 语言类型: {args.lang}")
    if args.seed is not None:
        print(f"🔢 随机种子: {args.seed}")

    stats = add_noise_to_csv(
        input_path=args.input,
        output_path=args.output,
        column=args.column,
        ratio=args.ratio,
        lang=args.lang,
        seed=args.seed
    )

    print(f"\n✅ 完成！共处理 {stats['total_rows']} 行")
    print(f"📄 输出文件: {stats['output_path']}")


if __name__ == "__main__":
    main()