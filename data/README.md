# 多属性数据集说明

本目录存放**多属性**模糊匹配实验用的 A/B/ground-truth 三件套，供
`baseline_Integration/multi_attribute/` 的 schema 驱动管线使用。

- 生成脚本：`scripts/prepare_multi_attribute_datasets.py`
- 生成命令（在仓库根目录）：`python3 scripts/prepare_multi_attribute_datasets.py`
- 数据集性质：**仓库既有真实数据**，不是本次新造；脚本只做字段整理与真值导出，不重新采样。

```
data/multi_attr/
├── febrl4_A.csv                 5000 行   库（database）侧
├── febrl4_B.csv                 5000 行   查询（queries）侧
├── febrl4_ground_truth.csv      5000 行   真匹配对
├── multiling_A.csv               300 行   库侧（SAGE 真实姓名 + 合成属性）
├── multiling_B.csv                50 行   查询侧（40 正 / 10 负）
├── multiling_ground_truth.csv     50 行   真值（含 label）
└── manifest.json                         生成清单与 NCVR 的排除理由
```

## 1. FEBRL 4a / 4b（英文姓名 + 地址 + 生日等多属性）

| 文件 | 行数 | 大小 | 说明 |
|---|---|---|---|
| `febrl4_A.csv` | 5000 | 459 KB | 原始 `dataset4a.csv`，`rec_id` 改名为 `record_id` |
| `febrl4_B.csv` | 5000 | 460 KB | 原始 `dataset4b.csv`，`rec_id` 改名为 `query_id` |
| `febrl4_ground_truth.csv` | 5000 | 140 KB | `rec-<N>-org` ↔ `rec-<N>-dup-0` 一一对应 |

字段（两侧相同）：

| 列 | 含义 | schema 属性 |
|---|---|---|
| `record_id` / `query_id` | 记录号，`rec-<N>-org` / `rec-<N>-dup-<M>` | — |
| `given_name`, `surname` | 名、姓（含拼写扰动） | `name`（fuzzy_text） |
| `street_number`, `address_1`, `address_2` | 门牌与街道 | `address`（fuzzy_text） |
| `suburb` | 城区 | —（未进 schema） |
| `postcode` | 邮编 | `postcode`（exact） |
| `state` | 州缩写 | `state`（exact） |
| `date_of_birth` | 生日，`YYYYMMDD`，含非法值（如 `19450493`） | `dob`（date，`on_invalid="missing"`） |
| `soc_sec_id` | 社保号 | `ssn`（exact） |

> 原始数据的表头是 `rec_id, given_name, surname, ...`（逗号后带空格）。本目录里的文件
> **已去掉字段名两侧空格**——带空格的键会让 `row["given_name"]` 取到空串，进而把不同
> 记录编码成同一个零向量。

## 2. fused_multilingual（多语言 + 多属性）

| 文件 | 行数 | 大小 | 说明 |
|---|---|---|---|
| `multiling_A.csv` | 300 | 28 KB | 库侧，`record_id` |
| `multiling_B.csv` | 50 | 3.1 KB | 查询侧，`query_id` |
| `multiling_ground_truth.csv` | 50 | 1.2 KB | 40 正 / 10 负 |

字段：`record_id|query_id, name_local, name_romanized, dob, city, phone, country`

- `name_local`：原生文字（汉字 / 阿拉伯 / 缅甸文 / 拉丁重音）；
- `name_romanized`：拉丁转写；查询可能只带其中一种，空列按缺失零向量编码；
- `dob` / `city` / `phone`：合成属性（`phone` 同时存在 E.164 与本地两种写法）。

## 3. ground truth 格式

统一两列，**一行一条真匹配**：

| 列 | 含义 |
|---|---|
| `A_id` | 库侧记录 id（FEBRL 为 `rec-<N>-org`；多语言为 `record_id`） |
| `B_id` | 查询侧记录 id（FEBRL 为 `rec-<N>-dup-<M>`；多语言为 `query_id`） |
| `label` | 仅多语言有：`true` / `false`。负例行的 `A_id` 为空 |

## 4. 用途

- 驱动 `baseline_Integration/scripts/demo_multi_attribute_dataset.py`（需在
  `baseline_Integration/` 下用 job 文件指向这些 CSV）；
- 作为 `scripts/evaluate_multi_attribute.py` 计算 P/R/F1 的真值来源；
- 作为 `scripts/encode_multi_attr_to_binary.py` 测量"编码后真匹配对 Hamming 距离"的输入。

## 5. NCVR 10K 为什么不在本目录

`baseline_Integration/data/ncvr_10k/ncvr_10k_database.csv` 只有 `ncid,full_name`
两列（10000 行），查询只有 `query_ncid,query_name,label`（200 行）——**没有
dob / address / postcode**，是单属性数据。要凑"NCVR 多属性"只能凭空合成属性，
那已经不叫 NCVR 了，故不纳入多属性对比表，只在对比文档里注明原因。

## 6. 体量说明

全部文件合计 **1.1 MB**，均小于 GitHub 单文件 10 MB 限制，已随仓库提交；
原始 `dataset/`（fetch_dataset.py 下载的 1.5 MB）仍按仓库既有 `.gitignore` 排除。
