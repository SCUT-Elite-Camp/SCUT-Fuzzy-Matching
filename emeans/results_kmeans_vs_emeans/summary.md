# K-Means vs E-Means 控制变量实验

> **唯一变量**：聚类器，通过环境变量 `SCUT_CLUSTER_ALGORITHM` 切换（`kmeans` = 原 Lloyd 实现，`emeans` = FAISS 1.16 `NumpyEmeans`，spherical=True，20 epoch，seed 42）。
> 数据读取、编码、归一化、阈值、下游 CKKS 协议与入口脚本均未改动；`clustering/` 之下只换参数不换实现。
> 环境：WSL2 / Ubuntu 22.04 · Python 3.10.12 · faiss-cpu 1.16.0 · TenSEAL 0.3.16 · numpy 1.26.4

| 数据集 | 聚类器 | 离线建库 (s) | 第一轮 (s/查询) | 第二轮 (s/查询) | 总时延 (s) | 通信 (MB) | TP/FP/FN |
|---|---|---|---|---|---|---|---|
| NCVR 10K（200 查询，batch 200） | K-Means | 1.22 | 2.71 | 76.34 | 79.79 | 132.27 | 82/2/18 |
| NCVR 10K（200 查询，batch 200） | E-Means | 1.30 | 2.61 | 127.12 | 130.35 | 133.16 | 83/2/17 |
| fused_multilingual_10k（20 查询） | K-Means | 5.52 | N/A | N/A | 450.45 | N/A | 15/0/2 |
| fused_multilingual_10k（20 查询） | E-Means | 6.28 | N/A | N/A | 556.92 | N/A | 16/0/1 |

**口径说明**

- NCVR 的「第一轮/第二轮」是 `evaluation/benchmark.py` 统计的 `avg_round1` / `avg_round2`（秒/查询）；总时延是 `online_total`，与 `avg_round1 + avg_round2 × 批次数` 不完全相等（批内并行）。
- fused_multilingual_10k 走 `demo_multi_attribute_dataset.py`（逐查询串行），该入口**只报总在线耗时**，不拆分轮次、也没有字节计数器，故这两列为 N/A。
- 多属性的 TP/FP/FN 按检测口径给：TP = should-catch 正例中被 catch 的条数，FP = 负例被 catch 的条数，FN = 正例总数 − TP。

## 关键对照

- **NCVR 10K 时延**：E-Means 130.35s vs K-Means 79.79s（+63.4%）；第二轮 127.12s vs 76.34s（+66.5%）。
- **NCVR 10K 簇结构**：`max_cluster_size` 323 → 478（+48.0%），k=50 不变。
- **NCVR 10K 精度**：TP 82 → 83，Recall 0.8200 → 0.8300，F1 0.8913 → 0.8973。
- **NCVR 10K 通信**：132.27 MB → 133.16 MB（+0.7%）。
- **fused_10k 时延**：E-Means 556.92s vs K-Means 450.45s（+23.6%），离线 5.52s → 6.28s。
- **fused_10k 簇结构**：max_cluster 331 → 305（-7.9%）。
- **fused_10k 召回**：should-catch 15/16 → 16/16；cluster recall 15/17 → 17/17。

## 结论

### 1. NCVR 10K：E-Means 更慢，而且慢在第二轮

- 总在线 **130.35 s vs 79.79 s（+63.4%）**，其中第二轮 **127.12 s vs 76.34 s（+66.5%）**，
  第一轮反而略快（2.61 s vs 2.71 s）。
- 原因是簇结构：`max_cluster_size` **323 → 478（+48%）**。batch 路径的第二轮按簇宽扫描，
  最宽簇直接决定每批要跑多少列密文运算；第一轮只比簇心、与簇宽无关，所以几乎不变。
- 换来的收益只有 **Recall 0.8200 → 0.8300（TP 82 → 83）**，F1 0.8913 → 0.8973，通信 +0.7%。
  用 +63% 时延换 1 个百分点召回，不划算。

### 2. fused_multilingual_10k：召回确实更好，但慢的原因和 NCVR 相反

- 20 条查询（17 正 / 3 负）：**should-catch 命中 16/16 vs 15/16**，
  **cluster recall 17/17 = 100% vs 15/17 = 88.2%**，整体 94.1% vs 88.2%；负例误报均为 0/3。
- 时延 **556.92 s vs 450.45 s（+23.6%）**，离线建库 6.28 s vs 5.52 s。
- 但 `max_cluster` 反而更小（**305 vs 331**）。慢的真实原因是**平均扫描列数 +29%**：
  219.4 vs 170.2（正例 181.2 vs 138.7，负例 436.3 vs 348.3）——
  E-Means 的簇更均衡，单簇更深，命中之前要多扫约 30% 的列。
- 注意样本量：只有 17 个正例，+1 TP 的差异**不足以谈统计显著**，只能作为方向性证据。

### 3. 最终建议

| 场景 | 建议 |
|---|---|
| 默认配置 | **保留 K-Means**。NCVR 这类"簇大小长尾"的数据上，E-Means 会把最宽簇放大近一半，正打在最贵的第二轮上 |
| 召回优先 | E-Means 可作为**可选档**：多语言融合集上 cluster recall 从 88.2% 到 100%，代价约 +25% 时延 |
| 若要默认启用 E-Means | 必须先动 k 或加簇宽上限（例如 k=50→64、或按 max_cluster 截断），否则长尾簇会把时延吃掉 |

## 与组员既有结果的对照

| 项 | 组员记录 | 本次实测 | 说明 |
|---|---|---|---|
| NCVR 总时延（K→E） | 94 s → 143 s | 79.79 s → 130.35 s | 同向；比例 1.52× vs 1.63×，差异来自机器与 faiss 版本 |
| NCVR max_cluster（K→E） | 323 → 476 | 323 → 478 | 基本一致，确认不是实现问题 |
| fused_10k 召回（K→E） | TP 39 → 41 | 15/17 → 16/17（20 条子集） | 规模不同（60 条 vs 20 条），方向一致 |
| fused_10k 时延（K→E） | +5.6% | +23.6% | 本次只跑 20 条，负例占比 3/20 更高，负例要扫满 3 个簇，占比被放大 |

## 复现命令

```bash
cd ~/projects/SCUT-Fuzzy-Matching/emeans
mkdir -p results_kmeans_vs_emeans artifacts/ab

# NCVR 10K（batch 200）
SCUT_CLUSTER_ALGORITHM=kmeans python3 scripts/evaluate_ncvr_10k.py \
  --batch-size 200 --k 50 --probes 1 --no-early-stop \
  --output-dir artifacts/ab/ncvr_kmeans > results_kmeans_vs_emeans/ncvr_kmeans.log 2>&1
SCUT_CLUSTER_ALGORITHM=emeans python3 scripts/evaluate_ncvr_10k.py \
  --batch-size 200 --k 50 --probes 1 --no-early-stop \
  --output-dir artifacts/ab/ncvr_emeans > results_kmeans_vs_emeans/ncvr_emeans.log 2>&1

# fused_multilingual_10k（20 条查询）
SCUT_CLUSTER_ALGORITHM=kmeans python3 scripts/demo_multi_attribute_dataset.py \
  --config config/examples/fused_multilingual_10k_multi_attribute.json \
  --limit 20 --db-limit 0 --no-color \
  --output-dir artifacts/ab/multi20_kmeans > results_kmeans_vs_emeans/multi20_kmeans.log 2>&1
SCUT_CLUSTER_ALGORITHM=emeans python3 scripts/demo_multi_attribute_dataset.py \
  --config config/examples/fused_multilingual_10k_multi_attribute.json \
  --limit 20 --db-limit 0 --no-color \
  --output-dir artifacts/ab/multi20_emeans > results_kmeans_vs_emeans/multi20_emeans.log 2>&1
```

## 执行说明与偏差

1. **切换方式不是 CLI 参数**：仓库里没有 `--clusterer`，唯一开关是环境变量 `SCUT_CLUSTER_ALGORITHM`；
   **默认值已经是 `emeans`**，所以控制组必须显式写 `SCUT_CLUSTER_ALGORITHM=kmeans`。
2. **fused_10k 从 60 条降到 20 条**：60 条那一轮 K-Means 跑了 24 分钟只完成 36/60（约 0.67 min/查询），
   全量 60 条约 40 分钟、E-Means 更久，触发"单组超过 30 分钟"的中断规则。
   已保留半程日志 `multi_kmeans_limit60_aborted.log` 作为耗时证据。两组统一用 20 条，保证可比。
3. **多属性路径没有通信量计数器**：`demo_multi_attribute_dataset.py` 只统计耗时与召回，
   字节数只有单属性的 `evaluation/benchmark.py` 才有，故该两格为 N/A。
4. **未找到 `REPORT_K_EMeans.md`**：仓库（含 main / pm_w / codex 三个分支）里都没有这个文件，
   组员结果只能以本次任务描述中的数字为参照（见上一节对照表）。
5. 依赖：只装了 `faiss-cpu==1.16.0`（cp310 abi3 wheel）；`torch` 既不在 `requirements.txt` 里、
   代码也没有 import，未安装；未触碰 libOTe / volepsi / EMP / SEAL。
