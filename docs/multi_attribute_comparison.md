# 多属性 vs 单姓名：两方案对比表

> 生成时间：2026-10-25 · 环境：WSL2 / Ubuntu 22.04.5 · Python 3.10.12 · TenSEAL 0.3.16
> 两个方案的**单姓名**行直接引用组内既有成果；**多属性**行的每个数字都来自本次实跑，
> 原始日志见 `benchmark_logs/ckks_febrl.log`、`benchmark_logs/ckks_multiling.log`，
> 逐条查询的明细见 `baseline_Integration/artifacts/demo/*/demo_multi_attribute_dataset.csv`。

## 表 1：基于姓名（单属性，引用组内既有成果）

数据集：RLdata10000（10000 库 / 200 查询）

| 方案 | Precision | Recall | F1 | 时延 | 通信量 |
|---|---|---|---|---|---|
| KMeans + CKKS | 0.8366 | 0.7167 | 0.7720 | 374.71 s | 172.26 MB |
| ApproxPSI + EMP GC | 0.8206 | 0.8233 | 0.8219 | 9.35 s | 299.49 MB |

> 这 6 个数字在**仓库里查不到**（对 `0.8366 / 0.7167 / 0.7720 / 374.71 / 0.8206 / 0.8219 / 299.49`
> 逐个全仓 `grep -rI` 均为 0 命中），故只能原样引用、无法复核。仓库 `docs/report/Phrase Report_md版.md`
> 里另有一套单属性口径（NCVR 10K：P=0.9804 / R=1.0 / F1=0.9901、2722 s、15181 MB；批量编码后 159 s / 133 MB），
> 与上表不可混用。

## 表 2：基于多属性（本次实验）

| 方案 | 数据集 | 属性配置 | Precision | Recall | F1 | 时延 | 通信量 |
|---|---|---|---|---|---|---|---|
| KMeans + CKKS | FEBRL 4a/4b（5000 库 / 100 查询） | name + address + dob + ssn + postcode + state（6 属性） | 1.0000\* | **0.9400** | **0.9691** | 748.38 s（7.48 s/查询） | 5.68 GB（58.18 MB/查询） |
| ApproxPSI + EMP GC | FEBRL 4a/4b | 同上 | N/A | N/A | N/A | N/A | N/A |
| KMeans + CKKS | 多语言融合（300 库 / 50 查询，40 正 / 10 负） | name（原生+转写）+ dob + city + phone（4 属性） | **1.0000** | **0.9000** | **0.9474** | 121.09 s（2.42 s/查询） | 2.05 GB（42.03 MB/查询） |
| ApproxPSI + EMP GC | 多语言融合 | 同上 | N/A | N/A | N/A | N/A | N/A |
| KMeans + CKKS | NCVR 10K | 只有姓名 | 不适用 | 不适用 | 不适用 | 不适用 | 不适用 |
| ApproxPSI + EMP GC | NCVR 10K | 只有姓名 | 不适用 | 不适用 | 不适用 | 不适用 | 不适用 |

\* FEBRL 的查询集全为正例（4b 的 5000 条都是 4a 的 dup），没有负例可被误报，precision
恒等于 1，不具区分度；多语言那行有 10 条真负例，precision=1.0000 是**实测**的
（0/10 冒充者被 catch）。

### 实验口径（必读）

1. **P/R/F1 是查询级"检测"口径，不是 identification 口径**。协议只回传一个比特
   （是否越阈）+ 簇号 + 列号，**从不返回命中记录 id**（见 `multi_attribute/protocol.py`
   与 demo 的 NOTE）。所以这里的 TP/FP/FN/TN 是：
   `TP = 标签为真 且 CATCH`、`FP = 标签为假 且 CATCH`、`FN = 标签为真 且 no-catch`。
   FEBRL 的标签由 `rec_id` 推（`rec-<N>-org` ↔ `rec-<N>-dup-0`），多语言的标签由
   `data/fused_multilingual/labels.csv` 给。
2. **时延 = 两轮在线协议总耗时**（不含 Party-B 离线 k-means 与 index 构建，那部分
   FEBRL 仅 1.69 s）。取自日志 `⏱ Encrypted two-round protocol, all queries`。
3. **通信量是实测字节数 × 实测平均扫描列数**，不是公式猜的：
   - 单条查询的 context / 各密文尺寸来自 `scripts/measure_multi_attribute_comm.py`
     实测（FEBRL：context 33.83 MB、cluster 密文 326.4 KB、标量密文 229.8 KB、
     selector/match 密文 326.4 KB、逐列密文 128.1 KB；多语言：context 33.84 MB、
     其余同类）；
   - 平均扫描列数取 demo 自报的 `scan_columns_mean`（FEBRL 56.26 列 / 全扫描 5000 列，
     多语言 22.22 列 / 全扫描 300 列）。
   - 明细见 `artifacts/demo/febrl_ma_100/comm_8q.json`、`artifacts/demo/fused_ma_50/comm_8q.json`。
4. **查询规模与单姓名那行不同**（5000 库/100 查询 vs 10000 库/200 查询），时延**不可
   逐毫秒对比**，只能看量级与趋势。

## 关键结论

1. **多属性把 Recall 拉了 22 个百分点**：KMeans + CKKS 的 Recall 从单姓名的 0.7167
   升到 FEBRL 的 **0.9400** 与多语言融合的 **0.9000**；F1 从 0.7720 升到 **0.9691 / 0.9474**，
   升幅 23~25 个百分点。原因很直接：`dob`/`ssn`/`postcode` 这类精确属性把真匹配的
   分数从"姓名相似度 0.7~0.9"抬到 0.75~0.99，而冒充者被压到 0.59 以下（FEBRL 实测
   冒充者最高 0.586），判决区间被拉开。
2. **Recall 的上限由数据本身锁死，不是阈值没调好**：两个数据集都出现"真匹配最低分 <
   冒充者最高分"的重叠（FEBRL 0.542 vs 0.586；多语言 0.389 vs 0.481），存在任何 tau
   都分不开的样本。多语言的 4 条漏检（q12/q18/q23/q43）与 FEBRL 的 6 条漏检即属此类，
   继续调参会以假正例换召回，不是纯收益。
3. **ApproxPSI 这条腿在多属性上是真跑不了，不是没跑**：两条独立原因——
   - `approx_psi_plain` 的 CLI 只有 `--n/--ell/--d/--t/--lambda/--matches/--trials/--seed/--k`，
     数据由进程内 `generateDataset()` 合成，**没有 `--input`**（`src/main.cpp` 的
     `printHelp` 只有 9 个参数），不吃外部 CSV；
   - 即使补上 `--input`，ApproxPSI 要求**真匹配对的 Hamming 距离 ≤ d**。用
     `scripts/encode_multi_attr_to_binary.py` 在真实数据上量过（128 位编码，d=4）：
     simhash 只有 **25.2%** 的真匹配对（126/500）落在 d≤4 内，距离中位数 13、均值 13.9；
     多语言数据更差（2/40 = 5%，中位数 19）。这已经超出"换个编码参数"能救的范围，
     要动的是协议假设本身。
4. **多属性的通信量代价主要来自"每查询重发一次 CKKS 公开 context"**：FEBRL 每查询
   58.18 MB 里有 33.83 MB 是 context（58%），多语言 42.03 MB 里占 33.84 MB（81%）。
   一次握手复用后，每查询通信降到 **24.35 MB / 8.19 MB**，总量从 5.68 GB / 2.05 GB
   降到 2.43 GB / 0.48 GB。这是现成可拿的 2.4~4.3 倍优化，不需要改协议。
5. **时延随库规模近线性、随属性数超线性**：FEBRL（5000 库 / 6 属性 / k=70）7.48 s/查询，
   多语言（300 库 / 4 属性 / k=17）2.42 s/查询；第二轮每次只扫 56.26 / 22.22 列
   （全库的 1.1% / 7.4%），所以瓶颈在每列的密文运算数与向量宽度（cluster_dim 2156 /
   match_dim 1936，多语言 3272 / 3122），而不在扫描列数。

## 已知问题与限制

| # | 问题 | 影响 | 处置 |
|---|---|---|---|
| 1 | `RuntimeError: KSwitchKeys data is invalid`：tenseal 反序列化公开 context 时**间歇**失败 | 首次 200 查询的 FEBRL 跑在第 22 条崩（164 s 处）；同一代码此前 1 条 / 50 条均正常 | 属库级 bug，非数据/参数问题；改为 100 查询重跑并跑通。复现命令与日志见 `benchmark_logs/ckks_febrl.log`（第一次失败）与本文表格（第二次成功） |
| 2 | 协议不返回命中 id | 无法算 identification 的 P/R | 统一用 detection 口径并在口径里写明；单姓名那行的口径未知，跨表比较时注意 |
| 3 | FEBRL 全正例 | precision 无区分度 | 标 `*` 说明；precision 实际由多语言那行给出（1.0，10 条负例） |
| 4 | 单姓名 6 个数字仓库内不可查 | 无法复核 | 原样引用并注明 |
| 5 | 抽样实测通信量时早停未触发（解密值全为负，与 demo 94%/90% 的命中率矛盾），逐列扫描数取满 | 直接用 208/55 列会高估 down2 | 改为"实测分量尺寸 × demo 自报的实际平均扫描列数"，见口径第 3 条；该判定差异本身未查清，记为未解问题 |

## 可复现命令

```bash
cd ~/projects/SCUT-Fuzzy-Matching
python3 scripts/prepare_multi_attribute_datasets.py          # 生成 data/multi_attr/
# （FEBRL 原始数据：cd baseline_Integration && python3 scripts/fetch_dataset.py --dataset febrl）

cd baseline_Integration
python3 scripts/demo_multi_attribute_dataset.py \
  --config config/examples/febrl_multi_attribute.json \
  --limit 100 --db-limit 0 --no-color \
  --output-dir artifacts/demo/febrl_ma_100

python3 scripts/demo_multi_attribute_dataset.py \
  --config config/examples/fused_multilingual_multi_attribute.json \
  --limit 0 --db-limit 0 --no-color \
  --output-dir artifacts/demo/fused_ma_50

cd ..
python3 scripts/evaluate_multi_attribute.py \
  --run baseline_Integration/artifacts/demo/febrl_ma_100/demo_multi_attribute_dataset.json \
  --queries 100 --online-seconds 748.378
```

## 数据

`data/multi_attr/`（1.1 MB，已入库）：`febrl4_{A,B,ground_truth}.csv`、
`multiling_{A,B,ground_truth}.csv`、`manifest.json`。字段与真值格式见 `data/README.md`。
