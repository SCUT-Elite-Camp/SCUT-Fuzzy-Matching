"""名字预处理模块

对原始名字执行：
  1. 转换为小写
  2. 去除首尾空格
  3. 统一化多余空格与特殊字符
"""

import re


def clean_name(name: str) -> str:
    """清洗单个名字。

    Args:
        name: 原始名字字符串。

    Returns:
        清洗后的名字：小写、strip、连续空格/下划线/连字符压缩为单空格。
    """
    if not isinstance(name, str):
        name = str(name)

    # 小写化
    cleaned = name.lower()

    # 去除首尾空格
    cleaned = cleaned.strip()

    # 将下划线、连字符、制表符等统一替换为空格
    cleaned = re.sub(r'[_\-\t]+', ' ', cleaned)

    # 将连续多个空格压缩为单个空格
    cleaned = re.sub(r'\s+', ' ', cleaned)

    # 去除首尾标点（保留内部标点，如 O'Brien 中的撇号）
    cleaned = cleaned.strip(' .,;:!?"\'')

    return cleaned
