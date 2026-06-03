"""
性能基准测试：运行完整协议并收集指标。
"""

import time
import tracemalloc
import numpy as np
from .metrics import compute_confusion_counts, compute_metrics
from .communication_cost import measure_ct_size
from config.params import SIMILARITY_THRESHOLD
from party_a.local_prep import (
    create_ckks_context,
    prepare_encrypted_query,
    prepare_encrypted_query_with_context,
)
from party_a.online_querier import (
    check_encrypted_scores_debug,
    choose_cluster_and_build_request,
)
from party_b.offline_prep import prepare_party_b_offline
from party_b.online_responder import compare_to_centroids, column_wise_matching

def benchmark(config):
    """
    运行一次实验配置，返回各项指标。

    Args:
        config: dict，包含以下字段（示例）：
            {
                "dataset": "ncvr",            # 数据集名称
                "data_path": "./data/",       # 数据文件路径
                "el_cluster": 200,            # 聚类用编码长度
                "el_match": 50,               # 匹配用编码长度
                "k": 50,                      # 聚类数（若为0则线性搜索）
                "tau": 0.9,                   # 相似度阈值
                "query_limit": 100,           # 最多使用多少个查询（-1表示全部）
                "db_limit": 1000,             # 最多使用多少个 B 侧名字（-1表示全部）
                "use_mock": True              # 若为True，使用mock模拟其他模块；False则调用真实接口
            }

    Returns:
        dict: {
            "config": config,
            "metrics": {"precision":..., "recall":..., ...},
            "timing_sec": {"offline":..., "online_total":..., "round1":..., "round2":...},
            "communication_mb": {"sent":..., "recv":..., "total":...},
            "memory_peak_mb": float,
        }
    """
    # 1. 加载数据
    from .dataset_loader import load_dataset
    names_A, names_B, labels = load_dataset(config["dataset"], config["data_path"])
    if config.get("query_limit", -1) > 0:
        names_A = names_A[:config["query_limit"]]
        labels = labels[:config["query_limit"]]
    if config.get("db_limit", -1) > 0:
        names_B = names_B[:config["db_limit"]]

    # 2. 离线阶段（Party B 预处理）
    print(
        "Running offline preprocessing "
        f"({'simulated' if config.get('use_mock', True) else 'real'})..."
    )
    tracemalloc.start()
    offline_start = time.perf_counter()
    use_mock = config.get("use_mock", True)
    if use_mock:
        # 模拟 B 的离线产物
        centroids = np.random.randn(config["k"], config["el_cluster"]).astype(np.float64)
        cluster_matrix = np.random.randn(config["k"], 300, config["el_match"]).astype(np.float64)
        scaler_mean = np.zeros(config["el_cluster"])
        scaler_scale = np.ones(config["el_cluster"])
        offline_time = time.perf_counter() - offline_start
        # 记录内存峰值
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory_peak = peak / 1024**2
        artifacts = None
    else:
        k_config = config.get("k", 0)
        k_mode = config.get("k_mode", "sqrt" if not k_config else k_config)
        artifacts = prepare_party_b_offline(names_B, k_mode=k_mode)
        centroids = artifacts.centroids
        cluster_matrix = artifacts.cluster_matrix
        scaler_mean = artifacts.scaler_mean
        scaler_scale = artifacts.scaler_scale
        offline_time = time.perf_counter() - offline_start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory_peak = peak / 1024**2

    # 3. 在线阶段（对每个查询执行协议）
    print(f"Running online phase for {len(names_A)} queries...")
    # 通信量统计
    total_sent_bytes = 0
    total_recv_bytes = 0
    online_start = time.perf_counter()
    round1_times = []
    round2_times = []
    predictions = []
    reusable_secret_context = None
    reusable_public_context_bytes = None
    reuse_context = (
        not use_mock and config.get("reuse_context", True)
    )
    if reuse_context:
        reusable_secret_context = create_ckks_context()
        reusable_secret_context.generate_relin_keys()

    for idx, query_name in enumerate(names_A):
        # 模拟 A 预处理：生成 MinHash 签名、归一化、标准化（mock）
        if use_mock:
            # 随机生成查询签名（mock）
            query_sig_cluster = np.random.randn(config["el_cluster"])
            query_sig_match = np.random.randn(config["el_match"])
            # 标准化（使用 B 的 scaler 参数）
            query_norm_scaled = (query_sig_cluster - scaler_mean) / (scaler_scale + 1e-8)
            # 加密（模拟）
            # 实际应调用成员2的加密函数
            # 这里用字节大小模拟通信
            mock_ciphertext_cluster = b"mock_cipher_cluster" * (config["el_cluster"] // 10)
            mock_ciphertext_match = b"mock_cipher_match" * (config["el_match"] // 10)
            total_sent_bytes += len(mock_ciphertext_cluster) + len(mock_ciphertext_match)

            # 第一轮：与质心比较
            round1_start = time.perf_counter()
            # 模拟 B 计算点积并返回加密分数（每个质心一个密文）
            # 实际应调用成员4的 compare_to_centroids
            sim_scores_cipher = [b"fake_cipher"] * config["k"]
            total_recv_bytes += sum(len(c) for c in sim_scores_cipher)
            # 模拟 A 解密并构造 one-hot（这里直接构造）
            best_idx = np.random.randint(0, config["k"])  # mock
            one_hot = np.zeros(config["k"])
            one_hot[best_idx] = 1.0
            # 加密 one-hot 并发送
            one_hot_cipher = b"fake_one_hot_cipher"
            total_sent_bytes += len(one_hot_cipher)
            round1_time = time.perf_counter() - round1_start
            round1_times.append(round1_time)

            # 第二轮：列式匹配
            round2_start = time.perf_counter()
            # 模拟 B 逐列计算并返回分数
            scores_cipher = []
            # 假设最大列数 300
            for col in range(300):
                scores_cipher.append(b"fake_score_cipher")
                total_recv_bytes += len(scores_cipher[-1])
            # A 解密后判断是否有正分数
            # 模拟解密结果
            decrypted_scores = np.random.randn(300)  # 随机值，可能正可能负
            predicted_match = any(s > config["tau"] for s in decrypted_scores)
            predictions.append(predicted_match)
            round2_time = time.perf_counter() - round2_start
            round2_times.append(round2_time)
        else:
            round1_start = time.perf_counter()
            if reuse_context:
                (
                    first_round_request,
                    party_a_state,
                    reusable_public_context_bytes,
                ) = prepare_encrypted_query_with_context(
                    query_name,
                    scaler_mean,
                    scaler_scale,
                    reusable_secret_context,
                    reusable_public_context_bytes,
                )
            else:
                first_round_request, party_a_state = prepare_encrypted_query(
                    query_name,
                    scaler_mean,
                    scaler_scale,
                )
            total_sent_bytes += len(first_round_request.public_context_bytes)
            total_sent_bytes += measure_ct_size(
                first_round_request.encrypted_query_200
            )

            sim_scores_cipher = compare_to_centroids(
                first_round_request,
                centroids,
            )
            total_recv_bytes += sum(
                measure_ct_size(score) for score in sim_scores_cipher
            )
            second_round_request, _ = choose_cluster_and_build_request(
                sim_scores_cipher,
                party_a_state,
                k=centroids.shape[0],
            )
            total_sent_bytes += measure_ct_size(
                second_round_request.encrypted_query_50
            )
            total_sent_bytes += measure_ct_size(
                second_round_request.encrypted_selector
            )
            round1_time = time.perf_counter() - round1_start
            round1_times.append(round1_time)

            round2_start = time.perf_counter()
            raw_scores = column_wise_matching(
                cluster_matrix,
                second_round_request,
                first_round_request.public_context_bytes,
                tau=config.get("tau", SIMILARITY_THRESHOLD),
            )

            def counted_scores():
                nonlocal total_recv_bytes
                for score in raw_scores:
                    total_recv_bytes += measure_ct_size(score)
                    yield score

            result, _ = check_encrypted_scores_debug(
                counted_scores(),
                party_a_state.secret_context,
                early_stop=config.get("early_stop", True),
            )
            predictions.append(result.catch)
            round2_time = time.perf_counter() - round2_start
            round2_times.append(round2_time)

    online_time = time.perf_counter() - online_start
    avg_round1 = float(np.mean(round1_times)) if round1_times else 0.0
    avg_round2 = float(np.mean(round2_times)) if round2_times else 0.0

    # 4. 计算指标
    metrics = compute_metrics(predictions, labels)
    confusion = compute_confusion_counts(predictions, labels)

    # 5. 组织返回结果
    result = {
        "config": config,
        "metrics": metrics,
        "confusion": confusion,
        "timing_sec": {
            "offline": float(offline_time),
            "online_total": float(online_time),
            "avg_round1": avg_round1,
            "avg_round2": avg_round2,
        },
        "communication_mb": {
            "sent": total_sent_bytes / 1024**2,
            "recv": total_recv_bytes / 1024**2,
            "total": (total_sent_bytes + total_recv_bytes) / 1024**2,
        },
        "memory_peak_mb": memory_peak,
    }
    return result

# 示例：单独测试此模块
if __name__ == "__main__":
    test_config = {
        "dataset": "ncvr",
        "data_path": "./data/",
        "el_cluster": 200,
        "el_match": 50,
        "k": 50,
        "tau": 0.9,
        "query_limit": 10,
        "use_mock": True
    }
    res = benchmark(test_config)
    print("Benchmark result:")
    for k, v in res.items():
        print(f"{k}: {v}")

# ========== 性能测量工具（供测试使用）==========
import time
from functools import wraps

def timer(func):
    """装饰器：打印函数执行时间"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        elapsed = end - start
        print(f"[TIMER] {func.__name__} took {elapsed:.4f} seconds")
        if isinstance(result, dict):
            result['_time_sec'] = elapsed
        return result
    return wrapper


class StageTimer:
    """分段计时器"""
    def __init__(self):
        self.timings = {}
        self._current_stage = None
        self._start_time = None

    def start(self, stage_name: str):
        self._current_stage = stage_name
        self._start_time = time.perf_counter()

    def stop(self):
        if self._current_stage and self._start_time:
            elapsed = time.perf_counter() - self._start_time
            self.timings[self._current_stage] = elapsed
            self._current_stage = None
            self._start_time = None

    def get(self):
        return self.timings.copy()
