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
| `demo_multi_attribute_dataset.py` | 真实 CSV 上的多属性匹配，schema 驱动 | `--config` 指定的 job JSON → 指向 `dataset/` 下的 CSV | `artifacts/demo/multi_attribute_dataset/`（JSON + CSV）+ 终端 | `--config`（必填）、`--limit 20`、`--db-limit 500`、`--no-encrypted`、`--output-dir artifacts/demo/multi_attribute_dataset` |
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
- `generate_synthetic_attributes.py` 造合成数据。


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

### 4.1 FEBRL 上「调不出完美阈值」，这是数据本身的限制

`demo_multi_attribute_dataset.py` 在 FEBRL 上跑 40 条查询的实测结果：

- 明文 top-1 召回（整库）：**40/40 = 100%**
- 加密路径准确率（catch与真值的一致率）：**33/40 = 82.5%**
- **簇召回**：**34/40 = 85.0%** ← 第二轮只看**一个**簇，真匹配落在别的簇里就**根本没有被检查的机会**。
- **真匹配分最低 0.542 / 冒充者分最高 0.579 —— 两者重叠**。也就是说被扰动得最狠的那批副本（`dob` / `postcode` / `state` 全错，如 `rec-4618-dup-0`）在任何阈值下都无法与冒充者分开。这要靠**增加属性或调权重**解决，不是调阈值能解决的。

所以示例 config 里的 `similarity_threshold` 是**标定结果而非最优解**：
FEBRL 取 0.60（略高于实测冒充者最高分），合成数据取 0.60（5000 条实测区间 0.501 ~ 0.841 之内）。
**换数据集必须重新标定。**

### 4.2 运行步骤

```bash
# 1. 拉公开数据 + 造合成数据（都落到 gitignored 的 dataset/）
python scripts/fetch_dataset.py --dataset all
python scripts/generate_synthetic_attributes.py --records 5000 --seed 42

# 2. FEBRL：真实数据上的多属性双轮协议
python scripts/demo_multi_attribute_dataset.py --config config/examples/febrl_multi_attribute.json --db-limit 0 --limit 20

# 3. 合成数据：带电话号码的那一路
python scripts/demo_multi_attribute_dataset.py --config config/examples/multi_attribute_schema.json --db-limit 500 --limit 20

# 4. 只要明文分数拆解、不要加密（快得多）
python scripts/demo_multi_attribute_dataset.py \
    --config config/examples/multi_attribute_schema.json --limit 50 --no-encrypted

# 5. 原有 name-only 生产路径的回归哨兵
python -m pytest tests -q
```

`demo_multi_attribute_dataset.py` 的 `--limit` 是**查询条数**，`--db-limit` 是**库规模**。
每条查询都是一次完整的双轮 HE 运行，所以 `--limit` 默认只有 20；
B 方离线阶段（k-means + 加密整个 cluster 矩阵）只做一次，N 条查询共用。

上面第 2/3/4 步的产物落在 `artifacts/demo/multi_attribute_dataset/`：

```text
demo_multi_attribute_dataset.json   # schema + 数据 + 汇总指标 + 每条查询的完整记录
demo_multi_attribute_dataset.csv    # 每条查询一行，含真匹配分/冒充者分/逐属性相似度
```

CSV 里的 `sim_<属性名>` 列跟着 schema 走，换 schema 就换列名；标定 `tau` 时直接拿
`top1_score`（真匹配分）和 `impostor_score`（冒充者分）这两列画分布，比看终端滚屏靠谱。`--no-encrypted` 跑出来的 CSV 里 `enc_*` 列留空，明文部分照常完整。
