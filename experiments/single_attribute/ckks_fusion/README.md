# CKKS 同态隐私匹配

> 目录整理（2026-10-25）：原 `fused-secure-filter/10_25/` 已迁至本目录。
> 这是**单姓名 CKKS 研究快照**，不是 C++ EMP 代码，也不替换根目录的
> `baseline_Integration/`。以下命令均须先在仓库根目录执行
> `cd experiments/single_attribute/ckks_fusion`。
> 交付中的 31 份结果已保存在 `results/received_2026-10-25/`；新的运行仍写入
> 被忽略的 `artifacts/`。`verify_claims.py` 默认读取前者，也可用
> `--reference-dir` 指定其他归档。下文性能数字是原交付的记录，不是此次整理重新测得的加速比。

两轮 CKKS 同态匹配协议的可运行实现。A 持查询与私钥，B 持库明文，双方不同任何一方
暴露明文查询、明文库或连续相似度分数。

> 完整的改动方案、已排除的尝试、PSI 侧的未完成工作与可行性分析，见 **[REPORT.md](REPORT.md)**。

```
Round 1  A 加密 200 维查询 → B 算 50 个簇心得分（3 tiles）→ A 解密选簇
Round 2  A 加密 one-hot 选择器 + 50 维查询 → B 做 tiled 匹配（20 tiles）→ A 解密判定
```

Round 2 每 tile 做 `k·d = 2,500` 次「密文选择器 × 明文候选」乘法 + 50 次密文乘密文。

---

## 快速开始

需要 **Python 3.10+**（已在 3.12 与 3.14 上验证）与 4 GB 内存。

```bash
git clone <this-repo> && cd <this-repo>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

首次安装约 2–3 分钟（tenseal 轮子 5 MB）。`data/` 已包含全部所需数据（3.8 MB），无需额外下载。

### 三步验证

```bash
# 1. 数字复核：重算并比对 16 条关键结论（约 2 秒）
python scripts/verify_claims.py --level quick
#    期望：16/16 通过

# 2. 单元测试（约 3 分钟）
python -m pytest tests -q
#    期望：281 passed, 3 skipped

# 3. 组合收益基准（约 6 分钟）
python fusion/bench_combined_ckks.py --repeats 2
#    期望：tiles 25 / 25 / 20，乘法 62,500 / 62,500 / 50,000，符号不一致 0
```

第 3 步同时写入 `artifacts/combined_ckks/report.json`。

---

## 与 baseline 的对比

### 确定性部分（任何环境都应一致）

| 指标 | baseline | 现在 | 变化 |
|---|---|---|---|
| tiles | 25 | **20** | −20% |
| selector 乘法 | 62,500 | **50,000** | −20% |
| 输出一致性 | — | 前 400 列 **80,000 槽位零差异** | — |
| test 召回 | 0.805 | **0.883** | **+7.8 点** |
| 查询级 FPR | 0/600 | 1/600 = 0.167% | +0.167% |

### 计时部分（**随环境波动，请以本机实测为准**）

| 环境 | v2 单独 | 组合（v2 + 截断） |
|---|---|---|
| Python 3.12 / tenseal cp312 | 1.106× | **1.381×**（73.57 → 53.26 s） |
| Python 3.14 / tenseal cp314 | 1.000× | **1.230×** |

**为什么不同**：`plain_tensor` 优化的收益取决于 tenseal 轮子的实现细节；
在 cp314 轮子上几乎为零，而截断（25→20 tiles）的收益在所有环境下都成立。

**结论**：**tiles 与乘法次数的减少是确定的；计时加速比在 1.23×–1.38× 之间，取决于环境。**
请勿把这两个数字混用。

---

## 已采纳的三项改动

| # | 改动 | 文件 | 收益 | 代价 |
|---|---|---|---|---|
| 1 | kernel：`plain_tensor` + 就地累加 | `party_b/online_responder.py` | 取决于环境 1.00–1.11× | 零，输出逐位一致 |
| 2 | 候选截断 c=483→400 | `config/params.py` | **tiles −20%**（确定） | 位置覆盖 0.995 |
| 3 | 阈值 τ=0.90→0.84 | `config/params.py` | **召回 +7.8 点** | FPR 0.167% |

改动 1 的代码：

```python
# 改前
term = enc_selectors[i] * plain_i.tolist()
acc_f = term if acc_f is None else acc_f + term   # 每次分配新密文

# 改后
term = enc_selectors[i] * ts.plain_tensor(plain_i, dtype="float")
if acc_f is None: acc_f = term
else: acc_f += term                                # in-place
```

---

## 有数据但**未采纳**的方案

| 方案 | 收益 | 代价 | 未采纳原因 |
|---|---|---|---|
| 特征维 d=50→25 | kernel **2.02×** | 召回 **−8.6 点** | 需业务方拍板 |
| 两级分带 cut=250 | 乘法 **1.786×**；与 v2 组合 **2.42×** | A 侧 0.6 bit + B 侧 0.4 bit 泄漏 | 有可量化泄漏 |
| 均衡聚类压 max_size | 483→200 | 召回 **0.880→0.750**（13 新漏报） | 质量损失过大 |

`2.42×` 与 `1.381×` 是**两种不同配置**：前者两级分带（有泄漏），后者固定截断（零泄漏）。

---

## 尝试过并排除的

| # | 尝试 | 实际结果 | 排除了什么 |
|---|---|---|---|
| 1 | 均衡聚类压 max_size | 召回 0.880→0.750 | 形状压缩的无损空间 |
| 2 | 候选组织（k/L 搭配） | 三组差 **2.4%** | 组织方式作为成本变量 |
| 3 | 槽位打包 | 成本与槽位数无关，利用率已 97.7% | 打包作为优化轴 |
| 4 | 代数外提掩码 R | **`scale out of bounds`** | 深度只有 2 层 |
| 5 | 组级共享形状 | 小组被 tile 粒度拖死 | 「中间折中」的存在性 |
| 6 | R3 份额同态重构 | 数值完美但**不减少任何计算** | 该性能路线 |
| 7 | 两阶段维度拼接 | 算术上零节省 | 该方向 |
| 8 | 多分辨率 | 各维 iid，退化为选 d | 该方向 |
| 9 | 自适应精度 | 阈值校准控 FPR，补不回判别力 | 该方向 |
| 10 | **平衡树归约**（Wallace 式） | **0.889×，慢 12.5%** | 树归约在 CKKS 上不适用 |
| 11 | 明文预计算向量化 | **1.002×** | 该方向 |

第 10 条的原因值得记住：CKKS 的密文-密文加法**不消耗模数层级**，
树归约的「降低深度」收益为零，却要在归约前同时持有 50 个密文（串行累加只需 2 个）。
**更浅的计算图在这里是负优化。**

---

## PSI / 混淆电路侧：未做的部分

### 代码不在此仓库

本仓库**不含** PSI/GC 实现（EMP + volePSI，C++）。它独立于 `../approx_psi_merged/`
与 `../approx_psi_opt/`，**未被最终模型采纳**。

### 那边已取得的成果（独立系统）

| 指标 | 原始 | 优化后 | 变化 |
|---|---|---|---|
| n=1000 端到端 | 213.14 s | 85.66 s | 2.51× |
| EMP 网络往返 | 729,303 | 576 | 1,266× |

**这个 2.51× 不能与本仓库的 1.23–1.38× 相加或比较**：匹配语义不同
（Hamming vs 余弦），且两者从未串成一个可运行系统。

### 为什么没做融合

| 方向 | 阻塞点 |
|---|---|
| PSI 候选 → CKKS 精排 | PSI 给出约 2 条候选，但隐藏「选了哪 2 条」仍需对全部 483 列加权求和；选择维度从簇(50)变成记录(483)，乘法 ×9.7 |
| CKKS + GC 阈值判定 | CKKS 分数用 A 的单密钥加密；GC 要判定，A 必须先解密——那时 A 已看到分数，GC 失去意义 |
| 不透明候选编码 | 随机 ID/哈希只是工程匿名，不构成密码学保护；且当前流程本就不向 B 暴露候选身份 |

真正的融合需要**门限 CKKS** 或**统一匹配语义**，属于新的协议设计任务。

其他已排除：跨轮 PSI 批处理（无效 bin 的 `my_id` 恒为 0，双方无法就轮次达成一致）、
ℓ=64（误报风险未被审计排除，需 10⁷ 量级样本）。

---

## 已知问题

### 安全缺陷（未修复）

接口声称只返回 `bool`，实现却让 A 解密**连续的掩码分数** `y = R·(sim − τ)`。
重复查询可恢复分数幅度（T=500 时约 0.5%）。当前**没有**去重、限流或掩码复用防护。

### 性能边界

- 乘法次数 `k·d = 2,500` 是「对 d 个特征各做一次 1-of-k 隐藏选择」的下界；
- 单次乘法 0.883 ms，已贴近 NTT 下限；
- 槽位利用率 97.7%；
- 乘法深度只有 **2 层**，任何更深的重写都会 `scale out of bounds`。

三者同时顶到边界，短期内没有进一步的算法级空间。

### 未验证

- 端到端（含密钥生成/解密）的加速比未实测；
- 无形式化安全证明；半诚实模型；未评估恶意参与方与共谋。

---

## 目录结构

```
ckks/            CKKS 上下文、密钥、tiling、算子
config/          参数（CKKS、阈值、截断列数）
party_a/         A 侧：加密、解密、选簇
party_b/         B 侧：簇心得分、tiled 匹配、离线聚类
minhash/         MinHash 签名
preprocessing/   姓名规范化
protocol/        消息类型与序列化边界
fusion/          实验与基准（含 bench_combined_ckks.py）
scripts/         编排入口与 verify_claims.py
tests/           281 个单元测试
data/            ncvr_10k 子集（304 KB）+ SAGE 数据
artifacts/       基准输出（含 combined_ckks/report.json）
```

### 生产路径（缺一不可）

| 文件 | 作用 |
|---|---|
| `config/params.py` | CKKS 参数、阈值、截断列数、掩码范围 |
| `ckks/context.py` / `keys.py` | 上下文与密钥 |
| `ckks/tiling.py` | 槽位布局 `slot=q*T+t`、tile 切分、候选打包 |
| `ckks/batching.py` / `operations.py` | 分批加密、密文算子 |
| `minhash/encoder.py` | MinHash 签名（200 维聚类 / 50 维匹配） |
| `protocol/types.py` / `transport.py` | 消息类型与序列化边界 |
| `party_a/online_querier.py` | A 侧全流程 |
| `party_b/online_responder.py` | B 侧两轮响应 |
| `party_b/offline_prep.py` | 离线聚类与候选矩阵 |
| `scripts/benchmark_he_batching.py` | 端到端编排入口 |

---

## 复现说明

`artifacts/combined_ckks/report.json` 保存了基准的原始输出。
在新机器上重跑时，**tiles 与乘法次数必须完全一致**；
计时会因 CPU、tenseal 轮子、后台负载而不同，属正常现象。

若只想验证逻辑正确性而不计时间，运行 `python -m pytest tests -q`（约 3 分钟）。
