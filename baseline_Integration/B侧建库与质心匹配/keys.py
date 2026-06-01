# ckks/keys.py
"""
密钥管理工具（辅助模块，主要由成员二调用）。
成员一只需使用 public context，不持有 secret key。
"""
from __future__ import annotations
from typing import Any


def has_secret_key(context: Any) -> bool:
    """
    检查 context 是否包含 secret key。
    成员五必须用此函数验证 public context 不含 secret key。
    """
    try:
        return context.is_private()
    except Exception:
        return False
