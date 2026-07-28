# Privacy-Preserving Fuzzy Name Matching Baseline

这是一个基于论文设想整理出的隐私保护模糊姓名匹配工程基线。当前代码重点是把五人分工中的模块边界、真实数据子集、端到端链路和评估脚本整理到可运行状态；它还不是论文性能的完整复现。

## 当前完成度

已完成：

- 代码已整理为正式模块：`preprocessing/`、`minhash/`、`clustering/`、`party_a/`、`party_b/`、`protocol/`、`evaluation/`、`tests/`。
- 单查询协议链路已跑通：B 侧建库、MinHash、归一化、聚类、A 侧查询加密、质心匹配、cluster 选择、列式匹配、最终判断。
- NCVR 10K 子集已接入：`data/ncvr_10k/` 包含 10000 条 B 侧库记录和 200 条查询。
- 评估脚本已支持输出 `precision`、`recall`、`f1`、`accuracy`、混淆计数和 PNG 可视化图。
- 二维 query×candidate SIMD batching 已实现：`m=200` 时每个密文同时承载 200 条查询和 20 个候选列。
- batch 生产路径强制 A→B bytes 序列化边界，B 侧密文只绑定公开 context。
- 生产批处理入口已从 483 次逐列同态核切换为 25 个 candidate tiles，全量回归通过。

当前实测（2026-07-28，本机）：

1. 合成固定尺寸 `m=200, k=50, L=483` 完整 HE 链路为 `79.134s`，R1/R2 严格输出 `3/25` 个密文。
2. 真实 NCVR 10K、200-query、关闭 early-stop 的 `online_total=144.421s`，相对旧 45min 基准约 `18.7x`。
3. 真实评估结果为 `precision=0.9804`、`recall=1.0`、`F1=0.9901`、`accuracy=0.99`。

以上是当前 Windows/TenSEAL/Python 实现的本机数据，不等同于论文运行环境的性能复现。

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

## 演示

展示协议链路，跑一个几秒级 demo：

```powershell
python scripts/demo_ncvr_matches.py
```

默认配置使用 `100` 条 B 侧记录、`5` 条查询和 `k=10`。其中 4 条正样本来自不同 cluster 和不同命中列，另有 1 条负样本；脚本会打印每条查询的真实标签、是否命中、A 侧选择的 cluster、首个命中的 B 侧姓名和检查列数。示例输出会保存到：

输出中的单条 `sec` 只统计该 query 的在线协议耗时；B 侧离线预处理和 HE context 初始化会在表格下方单独列出。

```text
artifacts/demo/ncvr_matches/demo_ncvr_matches.json
artifacts/demo/ncvr_matches/demo_ncvr_matches.csv
```

注意：`selected_cluster` 只在本地 demo/debug 输出中展示，用于汇报解释协议流程；正式第二轮请求仍发送加密 selector，不把 cluster id 明文发给 B。

## 跑 10K 指标与可视化

全量运行 (二维 query×candidate SIMD Batching 模式)：

```powershell
python scripts/evaluate_ncvr_10k.py --k 50 --query-limit 200 --db-limit 10000 --batch-size 200 --output-dir artifacts/evaluation/ncvr_10k_batch200
```

默认串行运行：

```powershell
python scripts/evaluate_ncvr_10k.py
```

默认配置：

```text
B-side records = 10000
queries        = 200
k              = 50
tau            = 0.9
batch_size     = 0 (可设 1..4096 启用二维 tiled SIMD 批处理)
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

注意：本机全量 200 查询、关闭 early-stop 的实测在线耗时为 `144.421s`；其他 CPU、TenSEAL 版本和系统负载下会变动。

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

1. 将第二轮 `k×d` 特征累加循环下沉到 C++/TenSEAL tensor 核，同时保持每个 `(query, candidate)` 独立正掩码。
2. 构造更强的模糊查询集，包括 typo、缩写、顺序变化和 nickname。
3. 扩展到更大 NCVR 子集或完整 NCVR。
