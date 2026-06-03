# Privacy-Preserving Fuzzy Name Matching Baseline

这是一个基于论文设想整理出的隐私保护模糊姓名匹配工程基线。当前代码重点是把五人分工中的模块边界、真实数据子集、端到端链路和评估脚本整理到可运行状态；它还不是论文性能的完整复现。

## 当前完成度

已完成：

- 代码已整理为正式模块：`preprocessing/`、`minhash/`、`clustering/`、`party_a/`、`party_b/`、`protocol/`、`evaluation/`、`tests/`。
- 单查询协议链路已跑通：B 侧建库、MinHash、归一化、聚类、A 侧查询加密、质心匹配、cluster 选择、列式匹配、最终判断。
- NCVR 10K 子集已接入：`data/ncvr_10k/` 包含 10000 条 B 侧库记录和 200 条查询。
- 评估脚本已支持输出 `precision`、`recall`、`f1`、`accuracy`、混淆计数和 PNG 可视化图。
- 当前测试通过：`39 passed`。

未对齐论文性能的两个核心点：

1. **多查询 HE batching 尚未实现。** 当前评估是 single-query serial baseline，200 条查询会串行跑 200 次。论文性能依赖 TenSEAL/CKKS batching，把多个 query 打包到密文 slots 中，因此 query 数增加时成本只小幅上升。
2. **论文级 packed 通信/计算布局尚未实现。** 当前主要使用同进程 TenSEAL 对象传递和 correctness-first 的列式匹配；还没有严格复现论文中面向通信量统计的 packed ciphertext layout、bytes 序列化边界和 column-wise batch 返回格式。

因此，当前结果说明“真实数据 + 协议链路 + 指标评估已完成”，但不能复现论文的 1000 queries / 10K records 约 100 秒性能。

## 目录结构

```text
baseline_Integration/
├── ckks/                  # TenSEAL CKKS 上下文、密钥与运算封装
├── clustering/            # cosine/spherical K-Means
├── config/                # 全局参数
├── data/ncvr_10k/         # 可提交的 NCVR 10K 测试子集
├── dataset/               # 本地原始大数据，已被 .gitignore 忽略
├── docs/                  # 分工接口规范与流程图
├── evaluation/            # 数据加载、指标、通信量、benchmark、reporting
├── minhash/               # MinHash 编码
├── party_a/               # A 侧查询准备、cluster 选择、最终判断
├── party_b/               # B 侧离线建库、质心匹配、列式匹配
├── preprocessing/         # 姓名清洗与归一化
├── protocol/              # 协议数据结构与端到端编排
├── scripts/               # 评估脚本
├── tests/                 # 测试
├── requirements.txt
└── run_all_test.py
```

## 安装

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

依赖使用正式包名 `scikit-learn`，不要安装旧的 `sklearn` 占位包。

## 测试

```powershell
python -m pytest tests -q
```

或：

```powershell
python run_all_test.py
```

## NCVR 10K 数据

原始 NCVR 大文件放在 `dataset/`，该目录已被 `.gitignore` 忽略。

当前评估用的小型真实子集在：

```text
data/ncvr_10k/
├── ncvr_10k_database.csv  # 10000 条 B 侧库记录：ncid,full_name
├── ncvr_10k_queries.csv   # 200 条 A 侧查询：query_ncid,query_name,label
└── README.md
```

读取方式：

```python
from evaluation.dataset_loader import load_dataset

names_a, names_b, labels = load_dataset("ncvr_10k", "data")
```

## 跑 10K 指标与可视化

全量运行：

```powershell
python scripts/evaluate_ncvr_10k.py
```

默认配置：

```text
B-side records = 10000
queries        = 200
k              = 50
tau            = 0.9
HE path        = real TenSEAL path, not mock
```

输出目录：

```text
artifacts/evaluation/ncvr_10k/
```

输出文件：

```text
ncvr_10k_result.json
ncvr_10k_metrics.csv
ncvr_10k_confusion.csv
ncvr_10k_metrics.png
ncvr_10k_confusion.png
```

查看结果：

```powershell
Get-Content artifacts\evaluation\ncvr_10k\ncvr_10k_metrics.csv
Get-Content artifacts\evaluation\ncvr_10k\ncvr_10k_confusion.csv
explorer artifacts\evaluation\ncvr_10k
```

快速 smoke：

```powershell
python scripts/evaluate_ncvr_10k.py --db-limit 20 --query-limit 2 --k 1 --output-dir artifacts/evaluation/ncvr_10k_smoke
```

常用调试：

```powershell
python scripts/evaluate_ncvr_10k.py --query-limit 20 --output-dir artifacts/evaluation/ncvr_10k_20q
python scripts/evaluate_ncvr_10k.py --db-limit 100 --query-limit 10 --k 10 --output-dir artifacts/evaluation/ncvr_10k_debug
python scripts/evaluate_ncvr_10k.py --k 0 --k-mode sqrt --output-dir artifacts/evaluation/ncvr_10k_sqrt
```

注意：全量 200 查询当前是串行 HE baseline，可能需要数分钟。该耗时不代表论文 batching 性能。

## 最小端到端调用

```python
from protocol.orchestrator import run_single_query_protocol

result = run_single_query_protocol(
    names_b=["john smith", "mary jones"],
    query_name="john smith",
    k_mode="sqrt",
    random_state=1,
    early_stop=False,
)

print(result.match_result.catch)
```

## 下一步

优先级建议：

1. 实现多查询 HE batching，让多个 query 共享 packed ciphertext 计算。
2. 对齐论文 packed ciphertext layout、bytes 序列化边界和通信量统计。
3. 构造更强的模糊查询集，包括 typo、缩写、顺序变化和 nickname。
4. 扩展到更大 NCVR 子集或完整 NCVR。
