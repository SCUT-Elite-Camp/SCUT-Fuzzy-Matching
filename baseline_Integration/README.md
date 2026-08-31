# Privacy-Preserving Fuzzy Name Matching Baseline

这是一个基于论文设想整理出的隐私保护模糊姓名匹配工程基线。当前代码重点是把五人分工中的模块边界、真实数据子集、端到端链路和评估脚本整理到可运行状态；它还不是论文性能的完整复现。

## 当前完成度

已完成：

- 代码已整理为正式模块：`preprocessing/`、`minhash/`、`clustering/`、`party_a/`、`party_b/`、`protocol/`、`evaluation/`、`tests/`。
- 单查询协议链路已跑通：B 侧建库、MinHash、归一化、聚类、A 侧查询加密、质心匹配、cluster 选择、列式匹配、最终判断。
- NCVR 10K 子集已接入：`data/ncvr_10k/` 包含 10000 条 B 侧库记录和 200 条查询。
- 已加入统一数据预处理管线：不同数据库 CSV、查询 CSV 和字段名可通过 JSON 配置映射到同一套建库/查询接口。
- SAGE 多语言样本已接入：43,206 条清洗后记录会展开为 56,705 个去重搜索项，支持原生文字与拉丁转写 variant 的统一建库查询。
- `unicode_v1` 会保留阿拉伯、汉字、缅甸文字等 Unicode 字母，并通过固定版本 `anyascii==0.3.3` 生成拉丁转写 variants。
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
├── data_pipeline/         # 数据适配、标准化、评测扰动、审计清单与导出
├── data/ncvr_10k/         # 可提交的 NCVR 10K 测试子集
├── data/sage/             # SAGE 多语言源表、验证查询和清洗产物
├── dataset/               # 本地原始大数据，已被 .gitignore 忽略
├── docs/                  # 分工接口规范与流程图
├── evaluation/            # 数据加载、指标、通信量、benchmark、reporting
├── minhash/               # MinHash 编码
├── party_a/               # A 侧查询准备、cluster 选择、最终判断
├── party_b/               # B 侧离线建库、质心匹配、列式匹配
├── preprocessing/         # 姓名清洗与归一化
├── protocol/              # 协议数据结构与端到端编排
├── scripts/               # 数据准备、评估、验证与演示脚本
├── tests/                 # 测试
└── requirements.txt
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

NCVR 评测会复用协议现有的英文姓名清洗规则：Unicode NFKC、转小写、移除非英文字母并压缩空格。loader 和原始 CSV 保持不变。

正式评测脚本默认将 30% 的正样本查询替换一个字母，标签和 B 侧数据库保持不变，并使用固定种子保证可复现：

```powershell
python scripts/evaluate_ncvr_10k.py --fuzzy-ratio 0.3 --fuzzy-seed 42
```

如需运行原始精确查询基线，使用 `--fuzzy-ratio 0`。

## 统一数据预处理与匹配

新数据集不需要修改 `evaluation/dataset_loader.py`。使用 `csv_pair` adapter 分别配置数据库文件、查询文件和各自字段映射即可；NCVR 10K 则已有专用 adapter。示例配置位于：

```text
config/examples/ncvr_10k_pipeline.json
config/examples/csv_pair_pipeline.json
config/examples/ncvr_10k_evaluation.json
config/examples/sage_pipeline.json
```

标准化管线的边界如下：adapter 只负责把源字段映射成统一记录；normalizer 负责确定性标准化；perturbation 只用于评测查询的可复现模糊化，不会污染数据库或生产查询。输出仍通过 `PreparedDataset.as_legacy_tuple()` 对接现有 Party A/Party B 协议，因此 HE 建库与查询代码无需针对数据集分支。

只做清理并导出标准数据：

```powershell
python scripts/prepare_dataset.py `
  --config config/examples/ncvr_10k_pipeline.json `
  --output-dir artifacts/prepared/ncvr_10k
```

输出包含 `database.csv`、`queries.csv` 和 `manifest.json`。manifest 会记录输入/输出行数、空值拒绝、去重、模糊化数量、孤立正样本以及实际配置。

使用同一配置运行完整建库、查询和评测：

```powershell
python scripts/evaluate_dataset.py `
  --config config/examples/ncvr_10k_evaluation.json
```

自定义 CSV 的数据库部分至少配置 `path`、`name_column`，查询部分至少配置 `path`、`name_column`。可选字段包括 `id_column`、`label_column`、`match_id_column`、`expected_ids_column`、`country_column`、`language_column` 和 `metadata_columns`。完整字段示例见 `csv_pair_pipeline.json`。

`english_v1` 严格复用原有英文 NCVR 清洗规则。`unicode_v1` 支持 Unicode NFKC、跨文字大小写处理、标点清理、拉丁重音折叠，并保留原生文字、script、language/country hints 和 variants；MinHash 字符 shingling 也会保留 Unicode 字母与组合符号，不再把阿拉伯、汉字或缅甸文字压成空签名。

SAGE 原始文件包含 123,479 行、23 个国家。`sage_names` adapter 按 NFKC/casefold 后的“姓名+国家”清理成 43,206 条标准数据库记录，生成结果已放在 `data/sage/prepared/`。其中包括 36,437 条拉丁文字、3,225 条阿拉伯文字、2,052 条汉字、1,469 条缅甸文字和少量混合文字记录。加入拉丁转写 variant 后，实际用于模糊匹配建库的是 56,705 个去重搜索项。

运行真实 HE batching 多语言 smoke：

```powershell
python scripts/validate_sage_multilingual.py
```

当前验证查询共 9 条：5 条正查询与 4 条负查询；其中 `zh-romanized` 和 `ar-romanized` 直接展示跨文字拉丁转写匹配。

`unicode_v1` 使用固定版本 `anyascii==0.3.3` 生成 `latin_transliterated` 和 `latin_compact` variants，并把这些 variants 一同加入 Party B 建库输入。当前 smoke 已覆盖 `廖學廣 ↔ LiaoXueGuang` 和阿拉伯原名 ↔ 拉丁转写查询。

这仍然是上下文无关的字符级转写，不是语言学姓名模型。真实世界中的多音字、姓名顺序、阿拉伯元音补全以及不同拼音/罗马化标准仍可能产生不一致；后续应把可配置的语言专用 transliterator 接到现有 variant 注册边界，而不是改动 HE 协议。

## 演示

展示本次 SAGE 跨文字查询特性，运行一个约几秒的真实 HE batching demo：

```powershell
python scripts/demo_sage_cross_script.py
```

默认展示全部 9 条验证查询：5 条正查询与 4 条负查询。其中最直观的新跨文字功能是 `LiaoXueGuang` 命中原生汉字姓名 `廖學廣`，以及拉丁转写查询命中阿拉伯原名。终端会按 Party A / Party B 的步骤打印清洗与转写、建库聚类、查询加密、两轮密文计算、解密判断和最终准确率，并把报告保存到：

```text
artifacts/demo/sage_cross_script/demo_sage_cross_script.json
artifacts/demo/sage_cross_script/demo_sage_cross_script.csv
```

可指定其他验证查询：

```powershell
python scripts/demo_sage_cross_script.py `
  --query-ids zh-romanized,ar-romanized,negative-latin
```

终端显示的 cluster、命中 variant 和原始姓名只属于本地 demo/debug 信息；生产协议仍发送序列化密文和加密 selector。

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
