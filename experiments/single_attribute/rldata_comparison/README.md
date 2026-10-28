# RLdata10000 单姓名对比实验

> 本目录比较两条技术路线在 RLdata10000 单姓名任务上的表现：
>
> - **方案一**：KMeans 选簇 + CKKS 同态加密（主线 `baseline_Integration/`）
> - **方案二**：随机投影候选 + 安全汉明验证（C++ `fused-secure-filter/` 的应用层适配）
>
> 原独立目录 `comparison/` 已于 2026-10-27 合并到本目录；这里现在是该对比实验的
> 唯一入口，不再保留平行的 `comparison/` 目录。

## 结论与口径

| 文件 | 内容 |
|---|---|
| `results/scheme_comparison.png` | 两方案全密文管线对比图（时延、通信量等） |
| `results/baseline_he_test.json` | 方案一（真实 TenSEAL CKKS）三次重复的测量结果 |
| `results/secure_e2e.json` | 方案二（volePSI + EMP 安全两方）端到端结果 |
| `results/reference_table1.png` | 主线 Table 1 参考图（零块跳过），不是本目录跑出来的 |

`results/baseline_he_test.json`、`results/secure_e2e.json` 与上面两张图是
2026-10-27 目录合并时的既有记录；2026-10-28 的 Top-k 扩展实验（见下文）重新
真实运行了 CKKS 与 EMP/volePSI 两条后端，其结果单独存放在 `results/topk/`。

原始交付只有两个裸姓名 CSV 和一张结果图，缺少记录 ID、身份真值、源码和逐次日志。
本目录后来补齐了可复现的数据构造与两方案 runner，但历史结果仍需按上述边界解读。

## 目录结构

```text
rldata_comparison/
├── README.md
├── requirements.txt
├── data/
│   ├── database.csv               # 10000 条 B 侧记录（含身份、生日）
│   ├── queries.csv                # 200 条评测查询（100 正 / 100 负，含模糊姓名）
│   ├── queries_exact.csv          # 200 条精确姓名查询（100 正 / 100 负）
│   ├── calibration_queries.csv    # 200 条不重叠的标定查询
│   └── manifest.json              # 组成与生成参数
├── scripts/
│   ├── names.py                   # 多语言姓名归一化（两条路线共用）
│   ├── encoding.py                # b-bit MinHash 二值编码与 Hamming
│   ├── prepare_rldata.py          # 从 RLdata10000.rda 生成 data/ 下的数据文件
│   ├── prepare_exact_queries.py   # 生成精确姓名查询集 queries_exact.csv
│   ├── baseline_top1.py           # 方案一：KMeans + CKKS 的 Top-1 runner
│   ├── baseline_topk.py           # 方案一：KMeans + CKKS 的 Top-k runner
│   ├── fused_top1.py              # 方案二：随机投影 + Hamming 的明文实现
│   ├── secure_export.py           # 导出两方输入与明文参考（安全路径）
│   ├── secure_runner.py           # 通过 WSL 跑 EMP/volePSI 端到端管线
│   ├── aggregate_topk.py          # 汇总 Top-k 12 行结果表
│   └── make_figure.py             # 由两份 JSON 结果渲染对比图
└── results/
    ├── baseline_he_test.json
    ├── secure_e2e.json
    ├── secure_e2e_seed20261023.json
    ├── secure/                    # 每 seed 的两方输入、参考预测与日志
    ├── topk/                      # Top-k 扩展实验：12 行汇总表与逐次结果
    ├── scheme_comparison.png
    └── reference_table1.png
```

## 脚本各自做什么

| 脚本 | 作用 | 依赖 |
|---|---|---|
| `names.py` | Unicode 归一化、德语变音转写、非拉丁文字转 ASCII | `anyascii` |
| `encoding.py` | 把姓名变成 b-bit MinHash 二值向量，并计算打包 Hamming 距离 | `numpy` |
| `prepare_rldata.py` | 解析 `RLdata10000.rda`，在 `data/` 下生成库、评测查询、标定查询和 manifest | `rdata` |
| `baseline_top1.py` | 方案一的 Top-1 扩展：每查询返回一条记录，而不是全候选二值结果 | 主线 CKKS 依赖（`tenseal`、`scikit-learn`） |
| `fused_top1.py` | 方案二的明文模拟：随机投影分桶 + Hamming 阈值 Top-1 | `numpy`、`pandas` |
| `secure_export.py` | 把两方输入编码成二进制，并生成同口径的明文参考 | `numpy`、`pandas` |
| `secure_runner.py` | 启动 C++ `name_secure_pipeline` 两方进程，逐查询核对结果 | WSL + EMP + volePSI |
| `make_figure.py` | 读取 `baseline_he_test.json` 与 `secure_e2e.json`，输出对比图 | `matplotlib` |

## 快速开始

以下命令都在**仓库根目录**执行，使用项目已有的 `.venv`：

```powershell
# 0) 安装本实验依赖（主线的 CKKS 依赖另见 baseline_Integration/requirements.txt）
.venv\Scripts\python.exe -m pip install -r experiments\single_attribute\rldata_comparison\requirements.txt

# 1) 生成可复现数据；需要 baseline_Integration/dataset/rldata10000/RLdata10000.rda
.venv\Scripts\python.exe experiments\single_attribute\rldata_comparison\scripts\prepare_rldata.py

# 2) 方案二：明文模拟跑一遍
.venv\Scripts\python.exe experiments\single_attribute\rldata_comparison\scripts\fused_top1.py `
    --rounds 32 --output experiments\single_attribute\rldata_comparison\results\fused_plaintext.json

# 3) 方案一：真实 CKKS Top-1
.venv\Scripts\python.exe experiments\single_attribute\rldata_comparison\scripts\baseline_top1.py `
    --output experiments\single_attribute\rldata_comparison\results\baseline_he_test.json

# 4) 画对比图（默认读取 results/baseline_he_test.json 与 results/secure_e2e.json）
.venv\Scripts\python.exe experiments\single_attribute\rldata_comparison\scripts\make_figure.py
```

安全路径（方案二端到端）需要先按 `fused-secure-filter/patches/README.md` 构建
`name_secure_pipeline`，然后：

```powershell
.venv\Scripts\python.exe experiments\single_attribute\rldata_comparison\scripts\secure_runner.py --seeds 20261023,20261040,20261057
```

它会先把两方输入写到 `results/secure/`，再通过 WSL 启动 server/client，
逐查询比对明文参考，最后写 `results/secure_e2e*.json`。

## Top-k 扩展实验（2026-10-28）

在原有「模糊姓名查询 + Top-1」之外，新增 5 个场景 × 2 个方案的完整密文测试：

| 查询集 | Top-k |
|---|---|
| `data/queries.csv`（模糊姓名，100 正 / 100 负） | Top-1 / Top-2 / Top-3 |
| `data/queries_exact.csv`（精确姓名，100 正 / 100 负） | Top-1 / Top-2 / Top-3 |

`queries_exact.csv` 由 `scripts/prepare_exact_queries.py` 生成：100 条逐字
正查询（查询名与库记录完全相同）与 100 条重组负查询（取一条记录的名 + 另一条
记录的姓，且不在库中）；源记录与 `queries.csv`、`calibration_queries.csv`
不重叠。每个正/负查询仍带 DOB。

**Top-k 口径（两方案不同，需分开读）。**

- **方案 1**：按第一轮解密的簇心相似度排序，第二轮依次探测前 k 个簇；某个簇内
  有候选过接受线（`cosine > tau`）就早停，返回该簇的最优身份。k=1 即原 Top-1。
  探针按「查询数 × k」展开成一个批量一次性加密、匹配，早停只是选结果时的规则，
  不减少密文计算，因此二轮时延、请求与通信量随 k 增长。
- **方案 2**：PSI 把 80 轮投影桶 × 每桶 8 槽的候选并成每个查询一个组，混淆电路
  在组内返回距离最小的 k 个身份。候选集已包含全部桶，没有可逐级加深的簇结构，
  所以 k 只改变返回条数。

评估在查询级：正查询当且仅当返回身份与期望身份一致记为命中（方案 2 为 k 个中
任一）；负查询只要返回任意身份即记为 FP。

**两条方案都没有明文捷径。** 方案 1 仍执行两轮 CKKS 密文匹配（密钥生成、第一轮
簇心探测、第二轮 tiled 匹配、逐 tile 解密）；方案 2 仍执行 volePSI ApproxPSI +
EMP 混淆电路两方管线，明文参考与安全输出逐查询比对。

新增/修改的文件：

| 文件 | 变化 |
|---|---|
| `data/queries_exact.csv` | 新的精确姓名查询集（200 条） |
| `scripts/prepare_exact_queries.py` | 生成上述查询集 |
| `scripts/baseline_topk.py` | 方案 1 的 Top-k runner（真实 CKKS） |
| `scripts/secure_export.py` | 增加 `--top-k`、`--queries-file`；参考实现输出 `identities` 列表 |
| `scripts/secure_runner.py` | 增加 `--top-k`、`--queries-file`、`--artifact-dir`；按身份集合计算指标 |
| `scripts/aggregate_topk.py` | 汇总 12 行结果表（CSV / Markdown / PNG） |

安全后端的接口改动保持在最小范围：`name_secure_pipeline` 新增可选参数
`--top-k K`（两端必须一致），客户端输出在原有 `identity` 之外增加
`identities` 数组；k=1 仍走原 Top-1 电路。`topk_group` 与 Top-1 电路共用
`top1_argmin.h`，明文回归见 `tests/test_name_top1_clear.cpp`。

```powershell
# 查询集
.venv\Scripts\python.exe experiments\single_attribute\rldata_comparison\scripts\prepare_exact_queries.py

# 方案 1：真实 CKKS Top-k（每个场景一次完整运行）
.venv\Scripts\python.exe experiments\single_attribute\rldata_comparison\scripts\baseline_topk.py `
    --top-k 2 --output experiments\single_attribute\rldata_comparison\results\topk\baseline_fuzzy_top2.json

# 方案 2：安全两方 Top-k（3 个种子）
.venv\Scripts\python.exe experiments\single_attribute\rldata_comparison\scripts\secure_runner.py `
    --top-k 2 --artifact-dir experiments\single_attribute\rldata_comparison\results\topk\secure_fuzzy_top2

# 汇总表
.venv\Scripts\python.exe experiments\single_attribute\rldata_comparison\scripts\aggregate_topk.py
```

结果布局：

```text
results/topk/
├── baseline_fuzzy_top{1,2,3}.json      # 方案 1：真实 CKKS，每次完整运行
├── baseline_exact_top{1,2,3}.json
├── secure_fuzzy_top{2,3}/              # 方案 2：每目录 3 个投影种子的两方产物
├── secure_exact_top{1,2,3}/
├── topk_summary.csv / .md / .png       # 12 行汇总表
└── *.log                               # 各 runner 的逐种子/逐次日志
```

**阈值与参数来源（方案 1）。** `shingle=2, k=150, tau=0.926` 取自原交付对比包
（见 `docs/reviews/2026-10-25/压缩包审阅与整合建议_2026-10-25.md`），本目录沿用
以便与既有图可比；主线姓名基线用的是 `shingle=3, k=50, tau=0.9`。用仓库自带的
标定模式在独立标定集 `data/calibration_queries.csv`（200 条，与评测集不重叠）上
复算：`tau` 从 0.500 扫到 1.000（步长 0.002，取 F1 最优），最优点正是 **0.926**
（0.926–0.932 平台，F1 0.7937；`tau=0.90` 时 F1 0.7358，`tau=0.94` 时 0.7845）。
所以它不是随手取的常数。`eps=1e-4`（`BATCH_DECRYPT_EPS`）只是 CKKS 解密噪声
余量，不是语义阈值，真正的接受下限是 `cosine > tau`。方案 2 的汉明阈值 32/256
位是 `secure_export.py` 的 `DEFAULT_DISTANCE`，作用于 b-bit MinHash 签名。

**资源开销口径。** 汇总表中的「资源开销」为双向通信量（发送 + 接收，MB）：
方案 1 取 `sent_bytes + recv_bytes`（含一次性公开评估上下文），方案 2 取
`psi_bytes_sent + psi_bytes_received + emp_bytes_sent + emp_bytes_received`。
「总时延」方案 1 为两轮在线协议耗时，方案 2 为 PSI + 全部候选组电路的在线耗时。

### 汇总结果

方案 1 每行一次完整 CKKS 运行；方案 2 每行 3 个投影种子取均值（有离散度时附样本标准差）。

| 方案 | 查询集 | Top-k | Accuracy | Precision | Recall | F1 | 总时延 (s) | 资源开销 (MB) |
|---|---|---|---|---|---|---|---|---|
| 方案 1 | 模糊姓名 | Top-1 | 0.7950 | 0.8391 | 0.7300 | 0.7807 | 169.76 | 172.2 |
| 方案 2 | 模糊姓名 | Top-1 | 0.8217 ± 0.0058 | 0.8206 ± 0.0021 | 0.8233 ± 0.0115 | 0.8219 ± 0.0068 | 7.70 | 299.5 |
| 方案 1 | 模糊姓名 | Top-2 | 0.7900 | 0.8222 | 0.7400 | 0.7789 | 590.70 | 173.4 |
| 方案 2 | 模糊姓名 | Top-2 | 0.8217 ± 0.0058 | 0.8206 ± 0.0021 | 0.8233 ± 0.0115 | 0.8219 ± 0.0068 | 7.71 | 309.5 |
| 方案 1 | 模糊姓名 | Top-3 | 0.7900 | 0.8152 | 0.7500 | 0.7813 | 712.94 | 175.0 |
| 方案 2 | 模糊姓名 | Top-3 | 0.8217 ± 0.0058 | 0.8206 ± 0.0021 | 0.8233 ± 0.0115 | 0.8219 ± 0.0068 | 7.26 | 317.7 |
| 方案 1 | 精确姓名 | Top-1 | 0.8650 | 0.7874 | 1.0000 | 0.8811 | 212.32 | 172.3 |
| 方案 2 | 精确姓名 | Top-1 | 0.8767 ± 0.0058 | 0.8022 ± 0.0074 | 1.0000 | 0.8902 ± 0.0046 | 7.78 | 323.8 |
| 方案 1 | 精确姓名 | Top-2 | 0.8550 | 0.7752 | 1.0000 | 0.8734 | 327.00 | 173.4 |
| 方案 2 | 精确姓名 | Top-2 | 0.8767 ± 0.0058 | 0.8022 ± 0.0074 | 1.0000 | 0.8902 ± 0.0046 | 7.66 | 335.3 |
| 方案 1 | 精确姓名 | Top-3 | 0.8550 | 0.7752 | 1.0000 | 0.8734 | 544.74 | 175.0 |
| 方案 2 | 精确姓名 | Top-3 | 0.8767 ± 0.0058 | 0.8022 ± 0.0074 | 1.0000 | 0.8902 ± 0.0046 | 7.27 | 344.9 |

观察：

1. **方案 1 的 Top-k 确实提高召回，但每多探一个簇只多 1 个 TP。** 模糊集
   TP 73 → 74 → 75（召回 0.730 → 0.740 → 0.750），精确集 TP 恒为 100。原因是
   Top-1 的 27 个模糊未命中里，**12 个的期望记录根本不在第一轮选中的簇**，15 个
   在簇内但明文余弦 0.731–0.915 低于 `tau=0.926`。跨簇的 12 个里只有 2 个余弦
   高于 `tau`（0.990、0.976），即多探簇最多只能救回这 2 个——实测 k=1→k=3 正好
   多召回 2 个，与明文镜像预测完全一致。代价是 FP 同时增加（14 → 16 → 17）：
   负查询每多探一个簇就多一次「某个候选越过接受线」的机会，所以精确率下降
   （0.839 → 0.822 → 0.815），F1 基本持平（0.7807 / 0.7789 / 0.7813）。**真正的
   瓶颈是阈值和聚类归属，不是排名深度。**
2. **方案 2 的指标与 k 无关，因为它的失效点同样在排名之前。** 3 个种子共 300 次
   正查询：命中 247，期望身份从未进入 PSI 候选集 32 次，进入候选但汉明距离 > 32
   有 21 次，距离 ≤32 却排在 k 之外的为 0。32 次生成失配里只有 2 次真实距离
   ≤32，其余即使进了候选也过不了阈值。候选集已由 80 轮投影桶并成一组，k 只改
   返回条数，因此 TP/FP/Recall/Precision 三行完全相同，只有电路规模与通信量变化。
3. **k 的代价：方案 1 是真实的密文计算增长，方案 2 只是多几次排序交换。**
   方案 1 把「查询数 × k」个探针一次性加密、批量匹配，早停只决定读哪个探针的
   结果，不省计算：二轮 tile 数 8 → 15 → 25，round 2 耗时 163.5 → 585.0 →
   697.9 s（模糊），通信 172.2 → 173.4 → 175.0 MB（公开评估上下文占主导，故只
   微增）。早停确实在发生——模糊集平均只用到 1.55 个（k=2）、2.09 个（k=3）探针，
   精确集 1.37、1.72 个——但它不是省时机制。方案 2 的混淆电路每候选多 k−1 次
   排序交换，通信随 k 单调增长：模糊集 299.5 → 309.5 → 317.7 MB，精确集
   323.8 → 335.3 → 344.9 MB；电路只占 0.22–0.27 s，PSI 阶段 7.0–7.5 s 主导，
   所以方案 2 的 Top-1/2/3 时延差异被 PSI 波动淹没。
   **方案 1 的时延只看趋势、不要读精确比值**：同样的密文工作量（200 × 2 批量、
   15 个 tile）模糊行 590.7 s、精确行 327.0 s，同一次运行里 round 1 也出现过
   15.0 s 对 5.3 s 的差异，说明这台机器在跑批时被其它进程挤占过；干净的四次
   运行（模糊 Top-1 169.8 s、精确 Top-2 327.0 s、精确 Top-3 544.7 s）大致符合
   「每多一探针约 ×1.8–2」。
4. **两条方案的取舍。** 模糊集上方案 2 召回更高（0.823 对 0.750 @Top-3），精确率
   在 k=3 时已基本追平（0.821 对 0.815）；精确集上两者召回都是 1.000，方案 2
   精确率略高（0.802 对 0.775，即 24–26 个负查询误报，方案 1 为 27–29 个）。
   方案 2 时延低 23–97 倍（7.3–7.7 s 对 169.8–712.9 s），但通信量高 1.7–2.0 倍。
5. **方案 1 的六行同源。** 原 `results/baseline_he_test.json` 的三次重复是早前
   机器高负载下的记录（286–492 s），已弃用；模糊 Top-1 用
   `baseline_topk.py --top-k 1` 在同一 runner、同一机器状态下重跑，六行结果全部
   来自 `results/topk/`。

## 已知问题

1. **Top-1 helper 依赖主线版本。** `baseline_top1.py` 需要
   `party_a.online_querier.check_tiled_score_top1_debug`，并要求
   `party_b.online_responder.tiled_batch_matching` 支持
   `mask_scope="per_query"`。这两个接口已在 2026-10-27 补回主线；如果换到更早的
   分支，需要同步这两个改动，否则脚本会在导入或调用时失败。
2. **Windows 控制台编码。** 脚本把含非 ASCII 字符的 JSON 直接打印到终端时，GBK
   控制台可能报 `UnicodeEncodeError`；输出文件通常已经写好。可加
   `PYTHONIOENCODING=utf-8` 再运行。
3. **安全路径不是纯 Python。** `secure_runner.py` 依赖 WSL、EMP/volePSI 和已编译的
   C++ 二进制；没有后端时只能跑 `fused_top1.py` 的明文模拟。

## 来源与重复文件

- 原始目录 `comparison/` 的代码、数据和结果已整体移入本目录；生成数据现在平铺在
  `data/` 下，不再保留 `data/rldata10000/` 子目录。
- 原交付的两份裸姓名 CSV（`A_queries_names.csv`、`B_database_names.csv`）已删除；
  `data/database.csv` 与 `data/queries.csv` 是带记录 ID、生日和标签的可复现版本。
- 以下内容与已有文件逐字节相同，或只是原始压缩包，未重复保留，已移到被 Git 忽略的
  `deliveries/local/2026-10-27/comparison_archive/`：
  `rldata10000.rar`、中文命名交付目录及其 ZIP、重复的 `scheme_comparison.png`。
- 原 `result/scheme_comparison.png` 已并入 `results/`，统一使用复数目录名。
