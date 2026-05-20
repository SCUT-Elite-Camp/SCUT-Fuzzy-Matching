"""
示例脚本：运行实验并打印结果
"""

import sys
sys.path.insert(0, ".")   # 确保能导入 evaluation 包

from evaluation import load_dataset, compute_metrics, measure_ct_size, benchmark

# 1. 测试数据集加载
print("=== Testing dataset loader ===")
names_A, names_B, labels = load_dataset("ncvr", "./data/ncvr_sample.csv")
print(f"Loaded {len(names_A)} queries, {len(names_B)} DB entries, {sum(labels)} matches")

# 2. 测试指标计算
print("\n=== Testing metrics ===")
preds = [True, False, True, True, False]
truth = [True, False, False, True, True]
metrics = compute_metrics(preds, truth)
print(metrics)

# 3. 测试密文大小测量（mock）
print("\n=== Testing ciphertext size measurement ===")
class MockCiphertext:
    def serialize(self):
        return b"x" * 1024
ct = MockCiphertext()
size = measure_ct_size(ct)
print(f"Mock ciphertext size: {size} bytes")

# 4. 运行 benchmark
print("\n=== Running benchmark (mock mode) ===")
config = {
    "dataset": "ncvr",
    "data_path": "./data/",
    "el_cluster": 200,
    "el_match": 50,
    "k": 50,
    "tau": 0.9,
    "query_limit": 20,
    "use_mock": True
}
result = benchmark(config)
print("\nBenchmark result:")
for key, value in result.items():
    print(f"{key}: {value}")