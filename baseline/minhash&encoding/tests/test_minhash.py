"""MinHash 与预处理模块的单元测试。"""

import numpy as np
import sys
from pathlib import Path

# 将项目根目录加入路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.text_cleaner import clean_name
from preprocessing.normalizer import l2_normalize
from minhash.encoder import build_shingles, generate_signature, batch_encode
from config.params import MAX_HASH, NUM_PERMUTATIONS_CLUSTER, NUM_PERMUTATIONS_MATCH


def test_clean_name_basic():
    assert clean_name("  John Doe  ") == "john doe"
    assert clean_name("JOHN") == "john"
    assert clean_name("O'Brien") == "o'brien"


def test_clean_name_punctuation_and_whitespace():
    assert clean_name("John  Doe") == "john doe"
    assert clean_name("John_Doe") == "john doe"
    assert clean_name("John-Doe") == "john doe"
    assert clean_name("  ,John.  ") == "john"


def test_build_shingles_example():
    """规范示例：'John' → 含边界空格的 trigrams（经 clean_name 后为小写）。"""
    shingles = build_shingles("John", size=3)
    # clean_name 会将 'John' 转为 'john'，故 shingle 为小写
    expected = {" jo", "joh", "ohn", "hn "}
    assert shingles == expected, f"Expected {expected}, got {shingles}"


def test_build_shingles_with_space():
    shingles = build_shingles("ab", size=3)
    # " ab " → " ab", "ab "
    expected = {" ab", "ab "}
    assert shingles == expected


def test_build_shingles_too_short():
    # 空名字（或仅空格）经 clean_name 后为空串，加边界空格为 "  "，长度 2 < 3
    assert build_shingles("   ", size=3) == set()
    # size=4 时，3 字符的边界名字无法生成 shingle
    assert build_shingles("a", size=4) == set()


def test_generate_signature_shape_and_range():
    sig = generate_signature("John", NUM_PERMUTATIONS_CLUSTER)
    assert sig.shape == (NUM_PERMUTATIONS_CLUSTER,)
    assert sig.dtype == np.float32
    # MinHash 值应在 [0, MAX_HASH) 范围内
    assert np.all(sig >= 0)
    assert np.all(sig < MAX_HASH)


def test_generate_signature_consistency():
    """同一名字、同一参数，两次生成结果必须一致。"""
    sig1 = generate_signature("Alice Smith", NUM_PERMUTATIONS_CLUSTER)
    sig2 = generate_signature("Alice Smith", NUM_PERMUTATIONS_CLUSTER)
    assert np.array_equal(sig1, sig2)


def test_generate_signature_different_names():
    """不同名字应产生不同签名（概率极高）。"""
    sig1 = generate_signature("Alice", NUM_PERMUTATIONS_CLUSTER)
    sig2 = generate_signature("Bob", NUM_PERMUTATIONS_CLUSTER)
    assert not np.array_equal(sig1, sig2)


def test_generate_signature_el50_is_prefix():
    """EL=50 签名必须是 EL=200 签名的前 50 维。"""
    sig_200 = generate_signature("Charlie", NUM_PERMUTATIONS_CLUSTER)
    sig_50 = generate_signature("Charlie", NUM_PERMUTATIONS_MATCH)
    assert np.array_equal(sig_200[:NUM_PERMUTATIONS_MATCH], sig_50)


def test_generate_signature_el50_prefix_reverse_order():
    """回归测试：先调 EL=50 再调 EL=200，前缀一致性也必须成立。

    历史 bug：当置换缓存按传入 num_perms 大小生成时，b_vals 在 rng 中
    的起点会随之偏移，导致先 50 后 200 的调用顺序破坏前缀一致性。
    """
    import minhash.encoder as enc
    enc._permutation_params = None  # 清空缓存模拟全新进程
    sig_50 = generate_signature("Charlie", NUM_PERMUTATIONS_MATCH)
    sig_200 = generate_signature("Charlie", NUM_PERMUTATIONS_CLUSTER)
    assert np.array_equal(sig_200[:NUM_PERMUTATIONS_MATCH], sig_50)


def test_batch_encode_consistency():
    """批量编码结果应与逐个编码后 vstack 一致。"""
    names = ["Alice", "Bob", "Charlie"]
    batch = batch_encode(names, NUM_PERMUTATIONS_CLUSTER)
    individual = np.vstack([generate_signature(n, NUM_PERMUTATIONS_CLUSTER) for n in names])
    assert np.array_equal(batch, individual)


def test_batch_encode_shape():
    names = ["A", "B", "C", "D"]
    mat = batch_encode(names, NUM_PERMUTATIONS_MATCH)
    assert mat.shape == (4, NUM_PERMUTATIONS_MATCH)


def test_l2_normalize():
    mat = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
    normed = l2_normalize(mat)
    # 第一行范数应为 1
    assert np.isclose(np.linalg.norm(normed[0]), 1.0)
    # 第二行范数应为 1
    assert np.isclose(np.linalg.norm(normed[1]), 1.0)
    # 方向应保持
    assert np.isclose(normed[0, 1] / normed[0, 0], 4.0 / 3.0)


def test_l2_normalize_zero_row():
    """零向量不应触发除零错误。"""
    mat = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32)
    normed = l2_normalize(mat)
    assert np.array_equal(normed[0], np.array([0.0, 0.0]))
    assert np.isclose(np.linalg.norm(normed[1]), 1.0)


if __name__ == "__main__":
    test_clean_name_basic()
    test_clean_name_punctuation_and_whitespace()
    test_build_shingles_example()
    test_build_shingles_with_space()
    test_build_shingles_too_short()
    test_generate_signature_shape_and_range()
    test_generate_signature_consistency()
    test_generate_signature_different_names()
    test_generate_signature_el50_is_prefix()
    test_generate_signature_el50_prefix_reverse_order()
    test_batch_encode_consistency()
    test_batch_encode_shape()
    test_l2_normalize()
    test_l2_normalize_zero_row()
    print("All tests passed!")
