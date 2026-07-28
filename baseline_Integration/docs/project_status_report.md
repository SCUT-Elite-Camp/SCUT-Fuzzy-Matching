# 项目状态报告：HE Batching 优化与隐私保护模糊姓名匹配

日期：2026-07-28
分支：`codex/he-batching`（基于 main，已全部合入最新主分支内容）
测试状态：`pytest tests -q` → **133 passed**（302 秒全绿）
运行环境：本机 Windows 11 + Python 3.x（miniconda3）+ TenSEAL（真实 CKKS，非 mock）

---

## 一、这个项目在做什么

项目是一个**隐私保护的模糊姓名匹配（fuzzy matching）协议工程基线**，模拟两个互不信任的参与方：

- **Party A（查询方）**：手上有 200 条待查询的人名，想知道它们是否在 B 的数据库中出现，但不想把明文查询交给 B。
- **Party B（数据方）**：持有 10,000 条 NCVR 选民登记姓名，愿意帮 A 做匹配，但不想暴露任何一条库内记录。

协议用 **CKKS 同态加密**（TenSEAL 实现）+ **MinHash 局部敏感哈希** + **余弦相似度阈值**完成。核心安全性是：A 发出的所有请求都是密文，B 看到的所有响应也都是密文，任何一方的明文数据都不离开本地。

### 协议两轮结构

**Round 1（质心匹配）**：B 提前把 10,000 条姓名编码成 200 维 MinHash 向量并做 k-means 聚类（k=50），得到 50 个簇质心。A 把每条查询的 200 维 MinHash 向量加密发给 B；B 在密文上做"查询向量 × 每个质心"点积，返回加密得分。A 本地解密后选出最相似的那个簇，生成 one-hot **selector 向量**并加密。

**Round 2（列式匹配）**：A 把查询的 50 维匹配向量（另一套 MinHash 参数）和加密 selector 发给 B。B 在密文上用 selector 从 50 个簇中"选中"对应簇的候选列（每列是一条库记录的 50 维向量），与查询向量做密文×密文点积，再减去 `mask × τ`（τ=0.9 为相似度阈值，mask 是 B 侧每对 [query, 候选] 独立采样的正随机掩码，防止 A 从分数反推库内容）。A 解密后只要看到正数就知道"命中"，看到负数就是"未命中"。

**最终判定**：A 解密所有候选列的得分，任一列为正则 `catch = true`。

---

## 二、我们解决了什么问题：Batching 瓶颈

### 原始实现的瓶颈

原协议是**逐查询、逐候选列**处理：

- Round 1：每条查询发 1 个请求，B 返回 50 个密文（k=50 个质心得分）。
- Round 2：B 对最大簇内的 **483 个候选列**，每列做 50 次密文×明文乘加（selector 选择）+ 50 次密文×密文乘加（相似度），输出 483 个密文。

200 条查询 × 每条约 14 秒在线耗时 ≈ **45 分钟**全量跑完，且通信量高达 **15.2 GB**（每个密文都是一次完整序列化传输）。瓶颈本质：CKKS 一个密文有 4096 个 slot（`POLY_MODULUS_DEGREE=8192`），但旧实现每次只用其中的几十个，硬件算力被大量浪费。

### 我们的方案：二维 query×candidate SIMD Tiling

CKKS 密文本质是 4096 槽的向量寄存器，slot 间运算天然 SIMD（单指令多数据）。我们把"批处理"从直觉的"一次装多条查询"（V1 纵向 SoA）升级为**二维平铺（V2 tiling）**：

```
slot(q, t) = q × T + t
q ∈ [0, m)   ← 批内第 q 条查询
t ∈ [0, T)   ← 第 t 个候选项（质心或候选列）
S = 4096     ← 总 slot 容量
T = ⌊S / m⌋  ← 每个 query 分到的候选槽数
active = m × T ≤ S
```

以 `m = 200` 为例：`T = 20`，`active = 4000`，**一个密文同时承载 200 条查询 × 20 个候选项 = 4000 个独立分数**。利用率 4000/4096 ≈ 97.7%。

- **Round 1**：50 个质心切成 `⌈50/20⌉ = 3` 个 tile，B 只算 3 次加密点积（而非 50 次 ×200 条查询），返回 **3 个密文**覆盖全部 200×50 个得分。
- **Round 2**：483 个候选列切成 `⌈483/20⌉ = 25` 个 tile，B 算 25 个 tile 核（而非 483 列 ×200 查询），返回 **25 个密文**覆盖全部 200×483 个得分。

**安全性保持不变**：每个 `(query, candidate)` 对仍然采样独立的正随机掩码 mask（`_sample_positive_mask_matrix(m, T)`），padding 槽置零且 mask=1，unpack 时按 `valid_width` 丢弃尾部；判定逻辑（正→命中）与串行版逐位一致。

### 关键实现细节

- `ckks/tiling.py`：`make_slot_tile_layout`（由 batch_size 推出 T 和 active_slots）、`repeat_query_rows`（q-major 复制 T 次）、`expand_plain_tile`（把 T 长度明文扩成 active_slots）、`pack_centroid_tile` / `pack_candidate_tile`（零填充 tile 构造）、`unpack_score_tile`（解密后还原 (m, valid_width) 矩阵）。
- `party_b/online_responder.py` 的 `compare_tiled_batch_to_centroids` 与 `tiled_batch_matching`：外层只循环 tile 数，内层仍是逐特征乘加，但每次操作的是 4000 槽密文；padding 列零填充保证不污染有效槽。
- 传输边界强制 **bytes 序列化**：A 侧加密→`serialize()`→B 侧 `ts.ckks_vector_from(public_context, ...)`，B 侧只持有**公钥 context**（无私钥、有 relinearization key），从代码层面保证 B 无法解密任何中间结果。
- 数值容差：批量路径第二轮要累加 50 个 ct-ct 乘积，误差高于单查询路径，解密阈值灰区取 `BATCH_DECRYPT_EPS = 1e-4`（实测最大绝对误差 9.6e-5 以内）。

---

## 三、我们达成了什么指标

### 1. 正确性指标（真实 NCVR 10K 数据集）

| 指标 | 数值 |
|---|---|
| Precision | **0.9804** |
| Recall | **1.0000** |
| F1 | **0.9901** |
| Accuracy | **0.9900** |
| 混淆矩阵 | TP=100, FP=2, FN=0, TN=98 |

数据：B 侧 10,000 条真实选民姓名，A 侧 200 条查询（100 正例 + 100 负例），阈值 τ=0.9，k=50，关闭 early-stop（跑满全部 483 个候选列）。零漏报（recall=1），仅 2 例误报。

### 2. 性能指标

**合成固定尺寸基准**（`m=200, k=50, L=483`，完整真实 HE 链路，`artifacts/benchmark_he_batching_v2_m200.json`）：

| 阶段 | 耗时 |
|---|---|
| HE context + 公钥生成 | 0.82 s |
| A 侧查询加密（200×200 维 + 50×50 维 + 50 selector） | 0.79 s |
| Round 1 请求序列化 | 0.37 s |
| Round 1 B 侧质心核（3 个 tile） | 1.03 s |
| A 侧解密 + selector 加密 | 0.16 s |
| Round 2 请求序列化 | 0.18 s |
| **Round 2 B 侧候选列核（25 个 tile）** | **75.76 s** |
| A 侧解密判定 | 0.01 s |
| **在线总耗时** | **79.13 s** |

输出规模严格符合理论值：Round 1 返回 3 个密文，Round 2 返回 25 个密文。

**真实 NCVR 10K 全量评估**（200 查询 × 10,000 库，`artifacts/ncvr_10k_v2_m200/ncvr_10k_result.json`）：

| 指标 | 新（batch m=200） | 旧（串行逐查询） |
|---|---|---|
| 在线总耗时 | **144.42 s（≈2.4 分钟）** | **2722.17 s（≈45.4 分钟）** |
| 通信总量 | **133.3 MB** | 15,181.5 MB（≈14.8 GB） |
| 每查询通信 | 682 KB | 75.9 MB |
| 峰值内存 | 73.3 MB | 73.3 MB |

**在线速度提升约 18.7 倍，通信量下降约 114 倍**，且判定结果与串行版完全一致（precision/recall/F1/accuracy 完全相同）。

### 3. 数值精度与测试保障

- 微基准全链路最大绝对误差 `9.57e-05`，p99 误差 `3.61e-05`，**符号一致率 100%**（每个 score 的正负判定与明文参考完全一致）。
- 新增 10 个专项测试文件覆盖：slot tiling 布局合法性、A/B 两侧批处理形状、数值灰区、端到端等价性、chunking 边界、安全边界（B 侧 context 必须无私钥）等。
- 全量回归：`133 passed`，无失败。

---

## 四、当前边界与下一步

**已实现**：真实 TenSEAL CKKS 全链路、二维 SIMD batching、生产路径强制 bytes 序列化边界、完整指标与基准脚本（`scripts/benchmark_he_batching.py`、`scripts/evaluate_ncvr_10k.py --batch-size`）。

**已知边界**：

- 第二轮 75 秒的 B 侧核仍是 Python 层 50 次 ct-ct 乘加循环（每 tile 内 `k×d` selector 选择 + `d` 次点积累加），下沉到 TenSEAL C++ tensor 核后还有显著余量。
- 数据仍是 NCVR 10K 子集；完整 NCVR 或更强模糊查询集（typo、缩写、nickname）未覆盖。
- `pytest` 缓存目录有 Windows 权限警告（`.pytest_cache` 拒绝访问），不影响测试结果。

**建议优先级**：

1. Round 2 内核下沉 C++/TenSEAL tensor API，保持每对独立掩码语义。
2. 构造模糊查询增强集验证鲁棒性。
3. 扩展至更大规模 NCVR 子集。

---

## 五、复现方式

```powershell
# 全量指标 + 可视化（batch 模式）
python scripts/evaluate_ncvr_10k.py --k 50 --query-limit 200 --db-limit 10000 --batch-size 200 --output-dir artifacts/evaluation/ncvr_10k_batch200

# 微基准（合成固定尺寸，输出每阶段耗时/通信/误差）
python scripts/benchmark_he_batching.py

# 回归测试
python -m pytest tests -q
```

详细施工文档见 `docs/he_batching_execution_plan.md`，接口规范见 `docs/五人分工接口规范_agent版.md`。
