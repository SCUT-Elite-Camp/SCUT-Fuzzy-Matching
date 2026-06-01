# preprocessing/text_cleaner.py
import re
import unicodedata


def clean_name(name: str) -> str:
    """
    清洗姓名字符串：
    - Unicode NFKC 规范化
    - 转小写
    - 去除多余空白（保留单个空格）
    - 去除非字母、非空格字符
    """
    if not isinstance(name, str):
        name = str(name)
    name = unicodedata.normalize("NFKC", name)
    name = name.lower().strip()
    name = re.sub(r"[^a-z\s]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name
