# 脚本汇总

- 汇总`baseline_Integration/scripts/`中每个脚本的**测试目的**、**数据来源**、**输出**。
---

## 1. 总览

| 脚本 | 测试目的 | 数据来源 | 输出 | 关键 CLI 默认值 |
|---|---|---|---|---|
| `benchmark_he_batching.py` | 压测 V2「按候选分块」CKKS 协议的耗时与内存 | **纯合成**：进程内 `np.random.default_rng(42)` 造向量，不读任何文件 | 单个 JSON | `--batch-size 200 --clusters 50 --columns 483 --tau 0.9 --seed 42` |
| `evaluate_ncvr_10k.py` | NCVR 10K 正式评测，出指标 + 图 | `data/ncvr_10k/`（10000 库 / 200 查询） | `artifacts/evaluation/ncvr_10k/` 指标 JSON + 图 | `--query-limit -1 --db-limit -1 --k-mode sqrt --tau 0.9 --fuzzy-ratio 0.3 --fuzzy-seed 42` |
| `demo_ncvr_matches.py` | 小样本 NCVR 演示，逐条打印匹配与簇细节 | `data/ncvr_10k/` | `artifacts/demo/ncvr_matches/` | `--db-limit 100 --query-indices 2,7,26,49,100 --k 10 --tau SIMILARITY_THRESHOLD` |
| `demo_streaming.py` | 流式终端演示，逐步展示查询全流程 | `data/ncvr_10k/` | `artifacts/demo/`（JSON + 终端输出） | `--db-limit 100 --k 10 --tau SIMILARITY_THRESHOLD` |
| `prepare_dataset.py` | 把任意配置的库/查询对规范化成标准 CSV | **由 `--config` 指向的 JSON 决定** | `--output-dir` 下的 `database.csv` / `queries.csv` / `manifest.json` | `--config`（必填）、`--output-dir`（必填） |
| `evaluate_dataset.py` | 按一份 JSON 配置跑统一评测 | 同上，由 `--config` 决定 | 配置里指定（可被 `--output-dir` 覆盖） | `--config`（必填）、`--output-dir`（可选） |
| `validate_sage_multilingual.py` | SAGE 多语言真实 HE 小规模验证 | `config/examples/sage_pipeline.json` → `data/sage/` | 可选 `--output` JSON | `--config config/examples/sage_pipeline.json` |
| `demo_sage_cross_script.py` | 跨文字系统（拉丁/中文/阿拉伯…）姓名匹配的演示 | `config/examples/sage_pipeline.json` → `data/sage/` | `artifacts/demo/sage_cross_script/` | `--config config/examples/sage_pipeline.json`、`--query-ids DEFAULT_QUERY_IDS` |
| `demo_multi_attribute.py` | 多属性（姓名 + 出生日期）双轮协议最小可跑示例 | **纯合成**：脚本内硬编码 5 条库记录 + 3 条查询 | 仅终端 | 无 CLI 参数 |
| `demo_multi_attribute_dataset.py` | 真实 CSV 上的多属性匹配，schema 驱动 | `--config` 指定的 job JSON → 指向 `dataset/` 下的 CSV | `artifacts/demo/multi_attribute_dataset/`（JSON + CSV）+ 终端 | `--config`（必填）、`--limit 20`、`--db-limit 500`、`--k-mode sqrt`、`--probes 3`、`--no-encrypted`、`--output-dir artifacts/demo/multi_attribute_dataset` |
| `calibrate_multi_attribute.py` | 从带标签样本反推 schema 的权重与 tau，产出可直接当 job 文件用的 JSON | `--config` 指定的 job JSON → 指向 `dataset/` 下的 CSV（同 demo） | 终端标定说明 + `<config>.calibrated.json` | `--config`（必填）、`--limit 500`、`--db-limit 0`（不限制）、`--weight-policy separability`、`--criterion recall_first` |
| `fetch_dataset.py` | 下载公开数据集到 `dataset/` | jsdelivr / gcore / raw.githubusercontent 三个镜像依次回退 | `dataset/febrl/*.csv`、`dataset/fake_1000.csv` | `--dataset febrl`、`--output dataset/` |
| `generate_synthetic_attributes.py` | 生成带电话/邮箱/地址的合成数据及真值 | 姓名池复用 `data/common-forenames-by-country.csv`，其余为内置词表 | `dataset/synthetic/{entities,queries,labels}.csv` | `--records 5000 --seed 42 --negative-ratio 0.25` |

---

## 2. 按数据来源分三类

**A. 纯合成，不依赖任何数据文件**

- `benchmark_he_batching.py` —— 向量由 `rng(42)` 现造，测的是加密运算本身。
- `demo_multi_attribute.py` —— 5 条库记录 + 3 条查询，全部硬编码，只验证协议通路。

好处是离线可跑；代价是**无法反映真实数据的分布**。要看真实数据下的表现，用
`demo_multi_attribute_dataset.py`。

**B. 直接读仓库内 NCVR 10K，不走 config**

- `evaluate_ncvr_10k.py`、`demo_ncvr_matches.py`、`demo_streaming.py`
  都调用 `evaluation.dataset_loader.load_dataset("ncvr_10k", args.data_path)`，
  默认 `--data-path data`，读 `data/ncvr_10k/`。

**C. 由 config JSON 驱动**

- `prepare_dataset.py` / `evaluate_dataset.py` —— `--config` 必填，adapter 类型、
  路径、清洗规则、限额全写在 JSON 里（见 `config/examples/`）。
- `validate_sage_multilingual.py` / `demo_sage_cross_script.py` —— 默认读
  `config/examples/sage_pipeline.json`。

**D. 多维属性数据脚本**（产出物落到 `dataset/`，该目录已在 `.gitignore` 中）

- `fetch_dataset.py` 下载公开数据；
- `generate_synthetic_attributes.py` 造合成数据；
- `demo_multi_attribute_dataset.py` / `calibrate_multi_attribute.py` 读上面两者的产物
  （走 `--config` 指向的 job JSON），前者跑协议、后者反推权重与 tau。


---

## 3. 数据集清单

### 3.1 仓库内已有

| 数据集 | 规模 | 带哪些属性 | 位置 |
|---|---|---|---|
| NCVR 10K | 10000 库 / 200 查询 | `full_name`（仅姓名） | `data/ncvr_10k/` |
| SAGE multilingual | 源表 123479 行 / 23 国；规范化去重后 **43206** 条库记录 + **9** 条跨文字系统验证查询 | 源表只有 `candidate_name, country`；`prepared/database.csv` 另有 `raw_name, canonical_name, script, language_hint, variants_json, warnings_json` | `data/sage/`（源表）与 `data/sage/prepared/`（规范化产物） |
| common-forenames-by-country | 2480 行可用（`Gender` 为 F/M 且姓名非空） | 国别、`Gender`、本地名、罗马化名 | `data/common-forenames-by-country.csv` |

SAGE 的两级数字容易搞混：**123479 是源表行数，43206 才是适配器去重 + `unicode_v1`
规范化之后的库记录数**。细节见 `data/sage/README.md`。用
`python scripts/prepare_dataset.py --config config/examples/sage_pipeline.json --output-dir data/sage/prepared`
可重新生成可审计的 `prepared/`。

以上均**没有任何带出生日期 / 电话 / 地址的数据集**，以下为多维属性数据集。


### 3.2 FEBRL

| 项 | 值 |
|---|---|
| 规模 | `dataset4a.csv` / `dataset4b.csv` 各 **5000 条**（原始 / 扰动副本）；`dataset3.csv` 5000 条，单表去重，未使用 |
| 属性 | `rec_id, given_name, surname, street_number, address_1, address_2, suburb, postcode, state, date_of_birth, soc_sec_id` |
| 真值 | **自带**。`rec_id` 形如 `rec-1070-org` ↔ `rec-1070-dup-0`，同实体号即配对 |
| 许可 | MIT |
| 获取 | `python scripts/fetch_dataset.py --dataset febrl` |

三处**实测**注意点：

1. `fetch_dataset.py` 按 jsdelivr → gcore → raw.githubusercontent 顺序回退，**jsdelivr 实测最稳**。
2. **表头每个列名前有一个空格**（`rec_id, given_name, ...`）。`multi_attribute/dataset.py` 已统一 strip；自己写解析时要注意。
3. **4b 有 64/5000（1.3%）非法出生日期**（如 `19450493`），4a 为 0，dataset3 为 35。
   这是 FEBRL 刻意注入的脏值。`date` kind 默认 fail-closed 会直接抛错，所以在 `config/examples/febrl_multi_attribute.json` 里显式声明了 `"on_invalid": "missing"`。

### 3.3 合成数据
**解决找不到包含电话号码的数据集的问题**

**内置合成生成器**：
 ```
 python scripts/generate_synthetic_attributes.py --records 5000 --seed 42
 ```
 产出 `dataset/synthetic/{entities,queries,labels}.csv`，含姓名 / 出生日期 / 电话 / 邮箱 / 地址 / 性别，并带真值配对。
 姓名池来自 `data/common-forenames-by-country.csv`（自带 `Gender`，所以性别属性不是编的）；
 姓氏是内置合成词表。扰动包括数字转置、字段缺失、格式差异（`DD/MM/YYYY`、`+1 (555) 010-0199` 等），并且**负例均摊混入**正例之间，这样 `--limit` 取前缀样本时不会全是正例。

---

## 4. 多维属性输入补充

### 4.1 召回率：分母含 tau，但不含簇

FEBRL，500 条查询 / 5000 条库记录实测：

- 明文 top-1 召回（整库）：**100%**
- **召回率（分母是 `should_catch`）**：分母内漏检的原因已被多簇探测消除大半（见下）
- **簇召回（tau 无关）**：默认 `--probes 3` 下 **0.96**。曾经的 87% 缺口全部来自第二轮
  只看一个簇：13 条正例的真匹配落在别的簇里，**根本没有被检查的机会**（`boundary` 0 /
  `unexplained` 0，这就是缺口的全部来源）。top-3 修回 9 条，剩下 4 条仍是同一个原因。
- **top-1-only 簇召回（tau 无关、与 probe 数无关）**：**0.84 ~ 0.89**。这是第一轮自身
  的判别质量 —— 即"最近质心是否就是真匹配所在簇"。`--probes` 调大只会让上面的簇召回归
  1，动不了这一项，所以脚本把它作为真正的 tau 无关锚点单独打印。

**多簇探测同时改善召回和成本。** 按质心分降序逐个探测并提前停止。加密实测
（FEBRL 100 查询 / 5000 记录 / k=71 / tau=0.513）：top-1 簇召回 0.87、平均扫 41.5 列；
top-3 簇召回 **0.96**、平均扫 53.3 列、最坏 207 列。真匹配簇的平均排名只有 0.15，这是
它便宜的原因。更早的 500 查询 sweep（k-means 种子平均，部署口径）在 k=150 穷举下能到
簇召回 1.0000 / 平均 49 列，top-8 要扫 271 列才到 0.987 —— top-m 无论命不命中都要付 m
个簇的钱，穷举按列命中即停。代价是 A 方要发的 one-hot 选择子密文数正比于探测数。

**默认是 top-3 而不是穷举：early stop 只对"命得中"的查询成立。** 库中根本没有对应记录
的查询撞不到命中列，停不下来，穷举对它退化成整库线性扫描（实测：库 500 条时 20 条负
查询条条扫满 500 列）。固定 top-3 给这类查询一个硬上限——最多 3 个簇的列数——代价是簇
召回从穷举的 1.0000 让到 0.96。**这条流程只对多维属性输入开启**；姓名单属性路径
（`party_a/party_b`）仍是纯 argmax 的 top-1。脚本把命中的平均列数和空查询的平均列数
**分开打印**，不把"平均 1%" 冒充成对所有查询都成立。

**召回率的分母是 `should_catch`（真匹配那条记录的明文分 > tau），不是"标签为真"。**
被扰动到低于阈值的副本算进分母，等于让召回率背上一份协议不负责的债。代价是这个分母
自己含 tau —— **调高 tau 会自动删掉最难的查询、把数字推向 100%，而一条错误都没修**。
实测：tau 从 0.60 抬到 0.90，分母 99 → 50，召回显示 **100.0%**，而簇召回一个数都没变。

所以脚本每个比率都带绝对计数，并把 tau 无关的簇召回紧跟其后，另外打印 tau 护栏。
**判一个 run 好不好看簇召回的 n/d，不看那个百分比。**

注意"漏掉条数"本身**不是** tau 无关的 —— 它的分母就是 should-catch 集合，抬 tau 会把
第一轮漏掉的查询整批移出分母，于是屏幕上显示 0。FEBRL 就是 12 → 0，而那 12 条错误一条
没修。屏幕上的 `misses among those` 行下面明确标注了 `tau-DEPENDENT`，并在同处给出
tau 无关的那一项。

另外注意这是**检出率而非识别准确率**：协议只回传"是否catch / 簇号 / 列号"，从不回传
命中记录的 id，所以 CATCH 只意味着"选中簇里有记录超过 tau"，**不等于命中的是对的那条**。

### 4.2 阈值与权重由实测反推，不再手写

`config/examples/` 里的 `similarity_threshold` 和逐属性权重是**标定结果**：

```bash
python scripts/calibrate_multi_attribute.py \
    --config config/examples/febrl_multi_attribute.json \
    --output config/examples/febrl_multi_attribute.calibrated.json
```

输出就是一份**可以直接当 job 文件喂回 demo** 的 JSON（schema 块被换成标定后的权重和
tau，另加一节 `_calibration` 说明），所以标定一次可复现，不必每次重跑拟合。

**最直觉的那条规则是错的。** 按单属性 AUC 线性分配权重（`--weight-policy auc`）在
FEBRL 上实测更差：

| 规则 | 组合 AUC | 最弱真匹配 | 最高冒充者 |
|---|---|---|---|
| 手写（V1） | 0.9994 | 0.539 | 0.618 |
| 按单属性 AUC 分配 | 0.9825 | **0.215** | 0.609 |
| separability（默认） | 0.9997 | 0.513 | 0.598 |

`ssn` 单属性 AUC 最高（0.911），按 AUC 分配会把 44.7% 权重压在它身上 —— 但 `ssn` 接近
二值，FEBRL 一旦损坏某条副本的 SSN，那条查询的真匹配分直接塌到 0，最弱真匹配从 0.539
掉到 0.215。**AUC 高只说明"平时冗余"。**

默认策略 `separability` 直接在**组合分**上做坐标上升，最大化软化 AUC（成对 sigmoid）：
`mean σ((真匹配分 − 该查询最强冒充者分) / T)`。这正是阈值检验需要的量；`T` 小则只盯最难
的样本对，权重因此被**联合**选出，互补属性才拿得到分。负例取**每条查询自己的最强冒充
者**，而不是随机库记录 —— 后者在 FEBRL 上几乎全被真匹配打败，会报出一个毫无信息量的
AUC 1.0。

阈值判据 `--criterion`：`recall_first`（默认，贴在最弱真匹配之下）、`youden`、
`target_recall`。本脚本不 import `tenseal`，没装 TenSEAL 的环境也能跑。

**这套权重不是过拟合。** 在 250 条上拟合、另 250 条上测（seed 0）：

| 权重 | 留出 AUC | 留出最弱真匹配 | 留出最高冒充者 |
|---|---|---|---|
| 手写（V1） | 0.9994 | 0.5417 | 0.6182 |
| 拟合 `T=0.05` | 0.9999 | 0.5323 | **0.5724** |
| 拟合 `T=0.15` | 0.9922 | 0.2784 | 0.4839 |

留出集上最高冒充者从 0.618 降到 0.572，而最弱真匹配只从 0.542 退到 0.532 —— 交错带被压窄
了。`T=0.15` 那行说明温度不是随便旋的：温度一大，sigmoid 近似线性，目标函数奖励的是
平均区分度而不是最难的那批样本对，拟合会塌到 4 个属性上，留出最弱真匹配掉到 0.278。

### 4.3 读取分离性结论

报告会给出三选一的判定，中间那种最容易被误读：

- **separable** —— `最高冒充者 < 最弱真匹配`，一个 tau 就能分开。
- **overlap** —— 有查询输给**自己的**最强冒充者，任何 tau 都救不回来；要靠加属性或
  提高字段质量。
- **ranges interleave** —— 每条查询都赢过自己的冒充者，但最弱真匹配仍低于最高冒充者。
  FEBRL 正是这种（0.5132 vs 0.5982）：没有全局 tau 能同时拿到召回和干净边界，
  `recall_first` 用假正例换召回，报告把兑换率直接印出来而不是藏起来。

### 4.4 运行步骤

```bash
# 1. 拉公开数据 + 造合成数据（都落到 gitignored 的 dataset/）
python scripts/fetch_dataset.py --dataset all
python scripts/generate_synthetic_attributes.py --records 5000 --seed 42

# 2. 标定权重与 tau，产出可直接复用的 job 文件
python scripts/calibrate_multi_attribute.py \
    --config config/examples/febrl_multi_attribute.json \
    --output config/examples/febrl_multi_attribute.calibrated.json

# 3. FEBRL：真实数据上的多属性双轮协议（top-3 探测 + 实测最优 k）
python scripts/demo_multi_attribute_dataset.py \
    --config config/examples/febrl_multi_attribute.calibrated.json \
    --db-limit 0 --limit 20 --k-mode auto

# 4. 合成数据：带电话号码的那一路
python scripts/demo_multi_attribute_dataset.py --config config/examples/multi_attribute_schema.json --db-limit 500 --limit 20

# 5. 只要明文分数拆解、不要加密（快得多）
python scripts/demo_multi_attribute_dataset.py \
    --config config/examples/multi_attribute_schema.json --limit 50 --no-encrypted

# 6. 原有 name-only 生产路径的回归哨兵
python -m pytest tests -q
```

`demo_multi_attribute_dataset.py` 的 `--limit` 是**查询条数**，`--db-limit` 是**库规模**。
每条查询都是一次完整的双轮 HE 运行，所以 `--limit` 默认只有 20；
B 方离线阶段（k-means + 加密整个 cluster 矩阵）只做一次，N 条查询共用。

`--k-mode` 取 `sqrt` / `log2` / `auto` / 整数 / `fixed:<k>`：`log2` 实测完败（n=5000 时
k=12，每簇约 417 条，平均要扫 621 列，而 sqrt 只要 105 —— 选择子少发 58 个完全抵不过列数
暴涨）；k 越大列数按 ~1/k 降、选择子载荷按 k 涨，最优落在 1.4·√n 附近，k=70~150 之间
曲线很平，所以 `auto`（=⌊1.4·√n⌋）是默认推荐值，而 demo 的 `--k-mode` 默认仍是 `sqrt`
以保持与 V1 可比。

上面第 2/3/4 步的产物落在 `artifacts/demo/multi_attribute_dataset/`：

```text
demo_multi_attribute_dataset.json   # schema + 数据 + 汇总指标 + 每条查询的完整记录
demo_multi_attribute_dataset.csv    # 每条查询一行，含 expected_score/tau/should_catch/冒充者分/逐属性相似度
```

CSV 里的 `sim_<属性名>` 列跟着 schema 走，换 schema 就换列名；标定 `tau` 时拿
**`expected_score`**（真匹配那条记录的明文分）和 `impostor_score`（冒充者分）这两列
画分布，比看终端滚屏靠谱。

注意不要拿 `top1_score` 当标定依据：它是明文 argmax 的分，实测数据上恰好与真匹配是同一条记录，但一旦 argmax 落到冒充者身上，那个数就是冒充者分，窗口会算反。
`expected_score` 是查真值 id 得来的，不会退化成 argmax。配套的 `expected_score_margin`
（= `expected_score` - `tau`）符号即"按协议判据应不应该 catch"，绝对值是离边界多远。
`enc_miss_cause` 则给漏掉的查询归因（`round1_cluster` / `boundary` / `unexplained`）。

`--no-encrypted` 跑出来的 CSV 里 `enc_*` 列留空，明文部分照常完整。
