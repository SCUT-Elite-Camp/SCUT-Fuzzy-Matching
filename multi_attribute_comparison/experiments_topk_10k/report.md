# 10K 多属性数据集（fused_multilingual_10k）Top-k 与投影组合实验

> 配置：**10,000 库 / 20 查询（17 正 / 3 负）**；固定阈值 τ=0.6135671549269659（数据作者标定）。
> 另附 τ=0.9 敏感性实验：**10,000 库 / 8 查询（7 正 / 1 负）**。
> 环境：WSL2 / Ubuntu 22.04.5 · Python 3.10.12 · TenSEAL 0.3.16 · CKKS N=8192 / 2^40。

## 0. 摘要

| 结论 | 内容 |
|---|---|
| 最佳组合 | **Baseline + Top-2**（τ=0.6136）：Recall 0.7647 → **0.8824**，F1 0.8667 → **0.9375**，代价 +20.7% 时延（16.40 → 19.79 s/查询） |
| Top-3 | Recall 与 Top-2 **完全相同**，但慢 13.9%（22.55 s），**不划算** |
| 投影 | 多属性路径**无投影实现** → 全部 N/A（见 §5.3） |
| τ=0.9 | Recall 崩到 0.1429，三种 Top-k 都一样：**阈值主导，Top-k 救不回**（见 §4） |

## 1. 实验设置

### 1.1 数据

| 项 | 值 |
|---|---|
| 库 | `baseline_Integration/data/fused_multilingual_10k/database.csv`，10,000 条 |
| 评测查询 | `queries.csv`，200 条（160 正 / 40 负）；本实验取**前 20 条**（17 正 / 3 负） |
| 标签 | `labels.csv`（`query_id,true_entity_id,label`），由数据生成过程给出，不是按相似度反推 |
| 属性 | name（`name_local`+`name_romanized`）、dob、city、phone；权重 0.40 / 0.25 / 0.15 / 0.20 |
| 向量维度 | cluster_dim 3,272 / match_dim 3,122（< CKKS 4096 槽位） |
| 聚类 | k = sqrt(10000) = **100 簇**，最大簇 331，索引矩阵约 788 MiB |
| 配置文件 | `baseline_Integration/config/examples/fused_multilingual_10k_multi_attribute.json`（**本次新建**，见 §5.1） |

### 1.2 阈值为什么有两个

数据 README 第 80–83 行把阈值**预先固定**为「校准集所有查询的最高非目标分 + 0.001」=
`0.6135671549269659`，并明确要求**不得用评测查询重调**。任务书要求固定 `0.9`。两者都跑：
τ=0.6136 是数据作者口径（主实验），τ=0.9 是任务书口径（敏感性对照）。

### 1.3 Top-k 是什么

协议只回传三样东西：**是否越阈、选中的簇号、检查到第几列**，从不返回命中记录 id。所以
"取回 top-k 候选记录"在加密链路上无处可接。本仓库自己的口径也是按簇算的：`--probes`
表示第二轮按质心分降序探测前几个簇（数据 README 的"top-3 明文协议参考 TP=133"就是
probes=3）。因此 **Top-k ≡ probes**：

- Top-1：只探最可能的 1 个簇；
- Top-2 / Top-3：探前 2 / 3 个簇，命中即早停。

判定用检测口径（与上一轮 10K 无关的 5K 报告一致）：

```
TP = 标签为真 且 catch      FP = 标签为假 且 catch
FN = 标签为真 且 no-catch   TN = 标签为假 且 no-catch
```

catch 指"探测的 k 个簇里存在某个候选得分 ≥ τ"。协议不返回命中 id，**无法区分命中的是真匹配
还是同簇冒充者**，故这是检测口径而非 identification 口径。

### 1.4 为什么只跑 20 条

10K 库下单条查询 16–35 s（负例要扫满被探测的簇，最多 ~600 列），全量 200 条预计 1.5–2 小时
且中途有 `KSwitchKeys` 崩溃风险（上一轮 5K 实验已复现过）。按任务书"先控量跑通"的要求，
主实验取 20 条。**注意：17 个正例的样本量不足以支撑统计显著性结论**（15/17 的 95% Wilson
区间约 [0.66, 0.97]），下表是点估计。

## 2. 主实验：固定阈值 τ=0.6136（数据作者标定）

| 方案配置 | Top-k | 固定阈值 | Precision | Recall | F1 | 平均单次查询耗时 (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Baseline (无投影) | 1 | 0.6136 | 1.0000 | 0.7647 | 0.8667 | 16.40 |
| **Baseline (无投影)** | **2** | **0.6136** | **1.0000** | **0.8824** | **0.9375** | **19.79** |
| Baseline (无投影) | 3 | 0.6136 | 1.0000 | 0.8824 | 0.9375 | 22.55 |
| 加入投影 | 1 | 0.6136 | N/A | N/A | N/A | N/A |
| 加入投影 | 2 | 0.6136 | N/A | N/A | N/A | N/A |
| 加入投影 | 3 | 0.6136 | N/A | N/A | N/A | N/A |

原始计数（20 条查询 = 17 正 + 3 负）：

| Top-k | TP | FN | FP | TN | 明文 ≥ τ 的正例 | 在线总耗时 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 13 | 4 | 0 | 3 | 16 | 327.95 s |
| 2 | 15 | 2 | 0 | 3 | 16 | 395.75 s |
| 3 | 15 | 2 | 0 | 3 | 16 | 450.96 s |

### 2.1 FN 归因（为什么 Top-2 之后不再涨）

| Top-k | 漏掉的查询 | 明文分 | 是否本该 catch | 归因 |
|---|---|---|---|---|
| 1 | eval-0006 | 0.6680 | 是 | 真匹配簇排在第 2 位 → 第 2 个探针补回 |
| 1 | eval-0020 | 0.6287 | 是 | 真匹配簇排在第 2 位 → 第 2 个探针补回 |
| 1 / 2 / 3 | eval-0017 | 0.6149 | 是（刚过线 0.0013） | 真匹配簇排在第 4 位及以后 → 需要 k≥4 才可能救 |
| 1 / 2 / 3 | eval-0019 | 0.4898 | 否 | 明文就低于 τ，**任何 Top-k 都不该 catch**，属数据信息量上限 |

所以 Top-1→Top-2 的收益全部来自"真簇排第 2"的两条；Top-3 多探一个簇没有新增命中，
而剩余 2 条 FN 里一条是阈值问题、一条需要 k≥4。

## 3. 敏感性：固定阈值 τ=0.9（任务书指定）

| 方案配置 | Top-k | 固定阈值 | Precision | Recall | F1 | 平均单次查询耗时 (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Baseline (无投影) | 1 | 0.9 | 1.0000 | 0.1429 | 0.2500 | 16.12 |
| Baseline (无投影) | 2 | 0.9 | 1.0000 | 0.1429 | 0.2500 | 30.07 |
| Baseline (无投影) | 3 | 0.9 | 1.0000 | 0.1429 | 0.2500 | 35.48 |
| 加入投影 | 1 / 2 / 3 | 0.9 | N/A | N/A | N/A | N/A |

8 条查询（7 正 + 1 负）里只有 **1 条**正例的明文分 ≥ 0.9，其余 6 条即使分数高于协议本来
能判定的水平（0.61）也被阈值挡掉。Top-k 从 1 加到 3 完全不改变结果，只把耗时从 16.1 s
推到 35.5 s（负例要扫满 3 个簇）。

**结论**：τ=0.9 与这批数据的分数分布不匹配（正例分布在 0.40–0.99，冒充者最高 0.67），
固定成 0.9 等于把召回锁死在 1/7。若要按任务书用 0.9，必须先换分数分布更接近 1.0 的编码
或数据集。

## 4. 结论：最佳组合

- **表现最佳的组合是：Baseline + Top-2（τ=0.6136）**。
- **原因分析**：
  1. **召回**：Top-1 有 3 条本该 catch 的正例漏检，其中 2 条的真匹配簇就排在第 2 位；把探针
     从 1 提到 2 把这两条捞回来，Recall 0.7647 → **0.8824**（+11.8 pp），F1 0.8667 → **0.9375**。
     剩下 1 条（eval-0017）要 k≥4，另 1 条（eval-0019）明文就 0.4898 < τ，不是协议的锅。
  2. **时延**：Top-2 平均 19.79 s/查询，比 Top-1 多 3.39 s（+20.7%）；但 Top-3 要 22.55 s
     （+13.9%），**却没换来任何召回**——多探的那个簇在这批查询里没有贡献。
  3. **误差**：三种配置 FP 都是 0，Precision 恒 1.0000；说明放宽 Top-k 在本样本上没有引入误报，
     但这与 τ 偏保守（0.6136 已经高于最高冒充者 0.586）有关。
- **一句话**：**Top-2 是这条曲线上唯一的拐点**；Top-3 只买延迟不买召回。

## 5. 已知问题与限制

### 5.1 配置文件的补建

数据 README 第 122 行引用了 `baseline_Integration/config/examples/fused_multilingual_10k_multi_attribute.json`，
但该文件**没有随数据一起提交**（`git ls-files` 里只有 10k 的 CSV/JSON 数据本身）。本次按 300 条
版本的 schema 原样补齐，只改数据路径与阈值，已一并入库。

### 5.2 首次冒烟"全漏"是调用方式错误（已修复）

第一版 runner 把 `AttributeQuery` 包装对象直接传给 `run_multi_attribute_protocol`，而编码器
只认底层 `record`（`record_values` 不做解包），于是查询向量全零、所有得分 0、必然全漏，
表现为"3 条查询全部 FN 且都扫满全部列"。demo 脚本传的是 `query.record`，改正后同 3 条查询
变成 TP=2/FN=0。该坑已写进 runner 的注释与 §5.2 这条记录。

### 5.3 投影为什么是 N/A

代码库里**有**随机投影实现（`baseline_Integration/clustering/projection.py`，Rademacher 矩阵），
但它只服务**单属性 V1 管线**：`protocol/orchestrator.py` → `party_b/offline_prep.py`
（`projection_dim` / `projection_seed` 参数）→ `clustering/projection.py`，作用对象是
MinHash 200 维签名。**多属性路径**（`multi_attribute/protocol.py::prepare_party_b_multi_offline`）
是 `StandardScaler` + 余弦 KMeans，**没有任何投影开关**，`AttributeSchema` 里也没有该参数。
按上报规则 1（找不到该路径的投影实现即跳过配置 B），本实验只跑 Baseline，投影行填 N/A。

> 若要做投影消融，需要先给多属性 offline 阶段加 `projection_dim`（约 20 行 Python：对
> cluster 向量乘 Rademacher 矩阵后再 KMeans，match 向量不变），属**新增功能**而非开关，
> 会改变聚类结构与 recall 口径，建议单独开一轮实验。

### 5.4 其他限制

| # | 限制 | 影响 |
|---|---|---|
| 1 | 只跑 20 条查询（17 正） | 点估计，17 个正例的 95% CI 约 ±0.15；Top-1 与 Top-2 的差异未经显著性检验 |
| 2 | 检测口径而非 identification | 无法区分"命中真匹配"与"同簇冒充者得分越阈" |
| 3 | 10K 规模下负例极慢 | 负例扫描满 3 个簇，单条最长 52.8 s；全量 200 条预计 1.5–2 小时 |
| 4 | 未做多 seed / 多次重复 | KMeans 只有 random_state=42 一个种子，聚类边界抖动未评估 |
| 5 | τ=0.9 实验只有 8 条 | 只用于说明阈值主导效应，不代表总体指标 |
| 6 | 长跑崩溃风险仍在 | tenseal `KSwitchKeys data is invalid` 是间歇性的；本次 6 个配置未触发 |

### 5.5 与明文参考的对照

数据自带的 200 条查询明文结果：全库扫描 TP=135 / FN=25 / FP=0 / TN=40，P=1.0000、R=0.84375、
F1=0.9153；top-3 协议参考 TP=133 / FN=27，R=0.83125。本次 20 条样本上 Top-2/Top-3 的
R=0.8824 略高于该参考，属小样本波动，**不可据此宣称优于数据作者的 200 条结果**。

## 6. 复现命令

```bash
cd ~/projects/SCUT-Fuzzy-Matching

# 主实验（τ=0.6136）
python3 multi_attribute_comparison/experiments_topk_10k/run_topk.py \
  --config baseline_Integration/config/examples/fused_multilingual_10k_multi_attribute.json \
  --limit 20 --db-limit 0 --top-k 2 \
  --fixed-threshold 0.6135671549269659 \
  --out multi_attribute_comparison/experiments_topk_10k/results/k2_tau0614.json

# 敏感性（τ=0.9）
python3 multi_attribute_comparison/experiments_topk_10k/run_topk.py \
  --config baseline_Integration/config/examples/fused_multilingual_10k_multi_attribute.json \
  --limit 8 --db-limit 0 --top-k 2 \
  --fixed-threshold 0.9 \
  --out multi_attribute_comparison/experiments_topk_10k/results/k2_tau090.json
```

## 7. 产物

| 文件 | 内容 |
|---|---|
| `results/k{1,2,3}_tau0614.json` | 主实验逐查询明细（含明文分、列数、耗时） |
| `results/k{1,2,3}_tau090.json` | 敏感性实验逐查询明细 |
| `results/summary.json` | 六个配置的汇总行 |
| `logs/k{1,2,3}_tau0614.log` | 主实验原始终端日志 |
| `logs/k{1,2,3}_tau090.log` | 敏感性实验原始终端日志 |
| `logs/smoke_fixed_k3.log` | 修复调用方式后的 3 条查询验证 |
| `run_topk.py` | 实验 runner（Top-k→probes、固定 τ、P/R/F1 与逐查询耗时） |
