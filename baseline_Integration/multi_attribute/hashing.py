"""低层编码原语：日期归一化与精确类属性的哈希 one-hot 块。

这两个函数原本位于 ``multi_attribute.encoder``。抽出到独立模块是为了避免
``kinds -> encoder -> kinds`` 的循环导入：``encoder`` 会 import ``kinds`` 来触发
内置 kind 注册，而 ``kinds`` 又需要这两个原语。

``encoder`` 仍然按原名重新导出它们，因此既有的 import 路径不受影响。
"""

from __future__ import annotations

import hashlib
import math
from datetime import datetime

import numpy as np

_DOB_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%Y.%m.%d",
    "%Y%m%d",
)

# 日在前（DD/MM/YYYY）的写法。只在调用方显式声明 day_first=True 时启用 ——
# "05/03/2001" 到底是 5 月 3 日还是 3 月 5 日无法从值本身判断，猜错会得到一个
# 语法合法但语义错误的日期，比直接报错更糟。所以这里做成显式策略，不做启发式。
_DOB_FORMATS_DAY_FIRST = (
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%d.%m.%Y",
)


def normalize_dob(value: str | None, *, day_first: bool = False) -> str | None:
    """Normalize common DOB spellings to ``YYYY-MM-DD``.

    Missing/blank DOB returns ``None``. Invalid non-empty values fail closed so
    accidental garbage does not silently become a matching feature.

    Args:
        day_first: 额外接受 ``DD/MM/YYYY`` / ``DD-MM-YYYY`` / ``DD.MM.YYYY``。
            默认 ``False``：这几种写法与月在前版本有歧义，必须由调用方声明。
    """

    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    formats = _DOB_FORMATS + (_DOB_FORMATS_DAY_FIRST if day_first else ())
    for fmt in formats:
        try:
            parsed = datetime.strptime(text, fmt)
            return parsed.strftime("%Y-%m-%d")
        except ValueError:
            continue
    expected = "YYYY-MM-DD, YYYY/MM/DD, YYYY.MM.DD or YYYYMMDD"
    if day_first:
        expected += " (plus DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY)"
    raise ValueError(f"unsupported DOB format {value!r}; expected {expected}")


def exact_hash_vector(
    value: str | None,
    *,
    blocks: int,
    buckets_per_block: int,
    seed: int,
) -> np.ndarray:
    """Encode an exact categorical value into independent one-hot hash blocks.

    The vector is L2-normalized by construction. Missing value => all zeros.
    """

    dim = blocks * buckets_per_block
    out = np.zeros(dim, dtype=np.float64)
    if value is None:
        return out

    amplitude = 1.0 / math.sqrt(blocks)
    for block in range(blocks):
        payload = f"{seed}:{block}:{value}".encode("utf-8")
        digest = hashlib.sha256(payload).digest()
        bucket = int.from_bytes(digest[:8], "big") % buckets_per_block
        out[block * buckets_per_block + bucket] = amplitude
    return out


# 保留旧名，供既有代码与测试继续按原私有名引用。
_exact_hash_vector = exact_hash_vector

__all__ = ["normalize_dob", "exact_hash_vector", "_exact_hash_vector"]
