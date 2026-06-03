"""
示例脚本：运行实验并打印结果
支持使用 forenames 数据集（从常见名字合成匹配/不匹配样本）
"""
import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

from evaluation import load_dataset, compute_metrics, measure_ct_size, benchmark

# 1. 测试数据集加载（使用 forenames 合成数据）
print("=== Testing dataset loader with forenames ===")
# 请确保 common-forenames-by-country.csv 放在 ./data/ 目录下
csv_path = project_root / "data" / "common-forenames-by-country.csv"
names_A, names_B, labels = load_dataset("forenames", csv_path)
print(f"Loaded {len(names_A)} queries, {len(names_B)} DB entries, {sum(labels)} matches")
print("Sample query names (first 5):", names_A[:5])
print("Sample labels (first 5):", labels[:5])
print()

# 2. 测试指标计算（使用上面加载的 labels 和模拟预测）
print("=== Testing metrics ===")
# 模拟预测结果：简单地将前一半预测为 True，后一半预测为 False
preds = [True] * (len(labels)//2) + [False] * (len(labels) - len(labels)//2)
metrics = compute_metrics(preds, labels)
print(metrics)
print()

# 3. 测试密文大小测量
print("=== Testing ciphertext size measurement ===")
class MockCiphertext:
    def serialize(self):
        return b"x" * 1024
ct = MockCiphertext()
size = measure_ct_size(ct)
print(f"Mock ciphertext size: {size} bytes")
print()

# 4. 运行 benchmark（mock 模式，使用 forenames 数据）
print("=== Running benchmark (real protocol smoke) with forenames dataset ===")
config = {
    "dataset": "forenames",                # 使用新数据集
    "data_path": str(csv_path),            # CSV 文件路径
    "el_cluster": 200,
    "el_match": 50,
    "k": 1,                                # 小规模 smoke 固定 k
    "tau": 0.9,
    "query_limit": 1,
    "db_limit": 2,
    "use_mock": False,
    "early_stop": True,
}
result = benchmark(config)
print("\nBenchmark result:")
for key, value in result.items():
    # 对于字典中的子字典，缩进打印
    if isinstance(value, dict):
        print(f"{key}:")
        for subkey, subvalue in value.items():
            print(f"    {subkey}: {subvalue}")
    else:
        print(f"{key}: {value}")
