"""
音译模块：支持中文 → 拼音，阿拉伯语 → 拉丁转写
"""

import os
import json
from typing import Dict, List, Optional

# ============================================================
# 中文 → 拼音映射表（常用汉字，可扩展）
# ============================================================

# 这里只列了常用姓和名，实际使用需要扩展到数千个汉字
# 建议从开源库加载完整映射，或使用 pypinyin 库
PINYIN_MAP: Dict[str, str] = {
    # 常见姓氏
    "张": "Zhang", "王": "Wang", "李": "Li", "刘": "Liu", "陈": "Chen",
    "杨": "Yang", "黄": "Huang", "赵": "Zhao", "吴": "Wu", "周": "Zhou",
    "徐": "Xu", "孙": "Sun", "马": "Ma", "朱": "Zhu", "胡": "Hu",
    "郭": "Guo", "林": "Lin", "何": "He", "高": "Gao", "罗": "Luo",
    "郑": "Zheng", "梁": "Liang", "谢": "Xie", "宋": "Song", "唐": "Tang",
    "许": "Xu", "韩": "Han", "冯": "Feng", "邓": "Deng", "曹": "Cao",
    "彭": "Peng", "曾": "Zeng", "肖": "Xiao", "田": "Tian", "董": "Dong",
    "潘": "Pan", "袁": "Yuan", "蔡": "Cai", "蒋": "Jiang", "余": "Yu",
    "于": "Yu", "叶": "Ye", "杜": "Du", "苏": "Su", "魏": "Wei",
    "吕": "Lv", "丁": "Ding", "任": "Ren", "姚": "Yao", "沈": "Shen",
    "钟": "Zhong", "姜": "Jiang", "崔": "Cui", "谭": "Tan", "陆": "Lu",
    "范": "Fan", "汪": "Wang", "廖": "Liao", "石": "Shi", "金": "Jin",
    "韦": "Wei", "贾": "Jia", "夏": "Xia", "付": "Fu", "方": "Fang",
    "白": "Bai", "邹": "Zou", "孟": "Meng", "熊": "Xiong", "秦": "Qin",
    "邱": "Qiu", "江": "Jiang", "尹": "Yin", "薛": "Xue", "闫": "Yan",
    "段": "Duan", "雷": "Lei", "侯": "Hou", "龙": "Long", "史": "Shi",
    "陶": "Tao", "贺": "He", "顾": "Gu", "毛": "Mao", "郝": "Hao",
    "龚": "Gong", "邵": "Shao", "万": "Wan", "钱": "Qian", "严": "Yan",
    "覃": "Qin", "武": "Wu", "戴": "Dai", "莫": "Mo", "孔": "Kong",
    "向": "Xiang", "汤": "Tang", "温": "Wen", "康": "Kang", "施": "Shi",
    "文": "Wen", "牛": "Niu", "樊": "Fan", "葛": "Ge", "邢": "Xing",
    # 常见名字用字
    "伟": "Wei", "明": "Ming", "华": "Hua", "强": "Qiang", "丽": "Li",
    "芳": "Fang", "敏": "Min", "静": "Jing", "涛": "Tao", "军": "Jun",
    "勇": "Yong", "刚": "Gang", "杰": "Jie", "峰": "Feng", "辉": "Hui",
    "玲": "Ling", "秀": "Xiu", "英": "Ying", "兰": "Lan", "文": "Wen",
    "武": "Wu", "国": "Guo", "建": "Jian", "平": "Ping", "海": "Hai",
    "山": "Shan", "红": "Hong", "梅": "Mei", "云": "Yun", "龙": "Long",
    "凤": "Feng", "玉": "Yu", "花": "Hua", "琴": "Qin", "香": "Xiang",
    "瑞": "Rui", "安": "An", "宁": "Ning", "欣": "Xin", "怡": "Yi",
    "婷": "Ting", "慧": "Hui", "洁": "Jie", "琳": "Lin", "瑶": "Yao",
    "璐": "Lu", "颖": "Ying", "心": "Xin", "雅": "Ya", "梦": "Meng",
    "晨": "Chen", "阳": "Yang", "宇": "Yu", "轩": "Xuan", "浩": "Hao",
    "然": "Ran", "博": "Bo", "毅": "Yi", "恒": "Heng", "哲": "Zhe",
    "瀚": "Han", "霖": "Lin", "泽": "Ze", "睿": "Rui", "涵": "Han",
    "桐": "Tong", "萱": "Xuan", "琪": "Qi", "瑶": "Yao", "璇": "Xuan",
    "依": "Yi", "诺": "Nuo", "伊": "Yi", "梦": "Meng", "雨": "Yu",
}

# 多音字需要特殊处理
POLYPHONIC_MAP: Dict[str, List[str]] = {
    "重": ["Zhong", "Chong"],
    "长": ["Chang", "Zhang"],
    "乐": ["Le", "Yue"],
    "行": ["Xing", "Hang"],
    # ... 可以扩展
}

# ============================================================
# 阿拉伯语 → 拉丁转写（简化版）
# ============================================================

ARABIC_TO_LATIN: Dict[str, str] = {
    "ا": "a", "ب": "b", "ت": "t", "ث": "th", "ج": "j",
    "ح": "h", "خ": "kh", "د": "d", "ذ": "dh", "ر": "r",
    "ز": "z", "س": "s", "ش": "sh", "ص": "s", "ض": "d",
    "ط": "t", "ظ": "z", "ع": "a", "غ": "gh", "ف": "f",
    "ق": "q", "ك": "k", "ل": "l", "م": "m", "ن": "n",
    "ه": "h", "و": "w", "ي": "y",
}


# ============================================================
# 核心接口
# ============================================================

def transliterate_chinese(text: str, polyphonic: bool = True) -> str:
    """
    将中文姓名转写为拼音

    Args:
        text: 中文姓名
        polyphonic: 是否处理多音字（默认True，取第一个）

    Returns:
        拼音字符串（如 "Zhang Wei"）
    """
    result = []
    for ch in text:
        if ch in PINYIN_MAP:
            result.append(PINYIN_MAP[ch])
        else:
            result.append(ch)
    return " ".join(result)


def transliterate_arabic(text: str) -> str:
    """
    将阿拉伯语姓名转写为拉丁字母

    Args:
        text: 阿拉伯语姓名

    Returns:
        拉丁转写字符串
    """
    result = []
    for ch in text:
        if ch in ARABIC_TO_LATIN:
            result.append(ARABIC_TO_LATIN[ch])
        else:
            result.append(ch)
    return "".join(result)


def transliterate_text(text: str, lang: str = "zh") -> str:
    """
    根据语言自动选择音译方式

    Args:
        text: 原始文本
        lang: 语言类型 ("zh", "ar", "en")

    Returns:
        音译后的文本
    """
    if lang == "zh":
        return transliterate_chinese(text)
    elif lang == "ar":
        return transliterate_arabic(text)
    else:
        return text


def detect_language(text: str) -> str:
    """
    自动检测文本的语言/文字类型

    Args:
        text: 待检测文本

    Returns:
        "zh", "ar", "ja", "ko", "en", "european"
    """
    if not text:
        return "en"

    for ch in text:
        # 中日韩统一表意文字
        if '\u4e00' <= ch <= '\u9fff':
            return "zh"
        # 日文假名
        if '\u3040' <= ch <= '\u30ff':
            return "ja"
        # 韩文
        if '\uac00' <= ch <= '\ud7af' or '\u1100' <= ch <= '\u11ff':
            return "ko"
        # 阿拉伯语
        if '\u0600' <= ch <= '\u06ff':
            return "ar"
        # 带重音符号的拉丁字符
        if ch.isalpha() and ord(ch) > 127:
            return "european"
    return "en"


def is_chinese_script(text: str) -> bool:
    """判断文本是否包含中文字符"""
    return any('\u4e00' <= ch <= '\u9fff' for ch in text)
