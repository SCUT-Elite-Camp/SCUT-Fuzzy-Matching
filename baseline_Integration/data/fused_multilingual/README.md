# 融合多语言数据集（fused_multilingual）

阶段性展示用的小型融合数据集：**SAGE 真实多语言姓名** + **合成多维属性**
（DOB / 城市 / 电话），配套 `config/examples/fused_multilingual_multi_attribute.json`
直接喂给 `scripts/demo_multi_attribute_dataset.py` 跑真实 CKKS 双轮协议。

## 来源与生成

- **姓名**：`data/sage/sage_names_all.csv`（SAGE 姓名库，123,479 行，23 国）。
  按文字系统分组抽样 300 条，覆盖 6 种文字系统情况：

  | 组 | 条数 | 说明 |
  |---|---|---|
  | latin_accented | 91 | 匈牙利 / 意大利，带变音记号（Árgyelán János 等） |
  | arabic | 45 | 伊拉克，阿拉伯文字 |
  | myanmar | 40 | 缅甸文 |
  | han | 45 | 台湾，纯汉字（罗马化列由 anyascii 转写生成） |
  | han_mixed | 22 | 台湾，"汉字+拉丁"混合行（拉丁部分即真实罗马化，源数据去重后仅 22 条唯一行） |
  | latin_plain | 57 | 加纳 / 印度 / 马来西亚 / 阿富汗（已罗马化）等纯拉丁 |

  > SAGE 源数据不含西里尔 / 韩文 / 日文，故未覆盖这两类文字系统。

- **罗马化列的生成规则**（全部复用管线既有逻辑，无新转换）：
  - 拉丁重音名：`data_pipeline/normalizers.py` 的 `_fold_latin`（NFKD 去重音）；
  - 阿拉伯 / 缅甸 / 纯汉字：`anyascii` 上下文无关转写（与 `unicode_v1` normalizer
    生成 `latin_transliterated` 变体同一库）；
  - 汉字+拉丁混合行：按文字系统拆分，拉丁 run 即真实罗马化；
  - 罗马化结果与本地形式仅差大小写时，romanized 列留空（该记录只有一种真实形式）。

- **合成属性**：DOB（1940–2005 合法日期，ISO 格式）、城市（国家对应的真实城市名）、
  电话（E.164 与本地格式混用，国家对应真实区号）。合成逻辑参考
  `scripts/generate_synthetic_attributes.py`。

- **重新生成**（固定种子，逐字节可复现）：

  ```bash
  python scripts/build_fused_multilingual_dataset.py \
      --records 300 --positives 40 --negatives 10 --seed 20261014 \
      --out data/fused_multilingual
  ```

## 文件结构

- `database.csv`（300 条）：`record_id`（`fused-xxxxxxxx`）、`name_local`、
  `name_romanized`、`dob`、`city`、`phone`、`country`。每条记录**同时携带两种姓名形式**。
- `queries.csv`（50 条）：40 正例 + 10 负例。正例是数据库记录的扰动副本，
  **姓名只带一种形式**（本地 / 罗马化约各占一半；重音记录的 romanized-only 查询
  即"去重音拉丁查询"，≥5 条），扰动包括相邻字符 typo、DOB 年份数字对调、
  DOB 格式变换、DOB / 电话留空、城市大小写；每条至多 2 处。负例为库里不存在
  的不相干 SAGE 记录。
- `labels.csv`：`query_id,true_entity_id,label`（true/false；负例的
  true_entity_id 为空；列名沿用 `apply_labels` 的默认约定）。

## 关键管线事实（为什么不改代码就能跑）

- 空单元格：`multi_attribute/dataset.py::_project_row` 跳过空列；整属性为空时
  不写入 values，`multi_attribute/kinds.py` 把缺失值编码为**全零块**（零贡献），
  不会崩溃。
- `name` 属性映射到 `["name_local", "name_romanized"]` 两列，以空格拼接 —
  查询只带一种形式时即与库里双形式拼接串做 MinHash 字符 shingle 模糊匹配。
- 编码器保留所有文字系统（NFKC + casefold，不去重音、不罗马化），
  见 `preprocessing/text_cleaner.py::clean_name_unicode`。

## 配套 schema 与阈值

`config/examples/fused_multilingual_multi_attribute.json`：name 0.4
（fuzzy_text 200/50 维，映射两列）/ dob 0.25（date）/ city 0.15（exact
casefold）/ phone 0.2（exact digits），cluster_dim=3272 / match_dim=3122。
city/dob/phone 用 512 桶（仿 FEBRL 的 ssn 惯例）压低哈希块碰撞给冒充者送的
部分分。similarity_threshold=0.485 由 `scripts/calibrate_multi_attribute.py`
明文标定：负例冒充者最高 0.452、全局冒充者上限 0.481；被双重扰动的 4 条最难
正例（q18/q43/q23/q12，真匹配分 0.39–0.45）与负例 q30（0.452）交叠，
任何 tau 都无法同时分开，0.485 是"零假正例"一侧的最优取值。换数据必须重新标定。

## 运行演示

```bash
python scripts/demo_multi_attribute_dataset.py \
    --config config/examples/fused_multilingual_multi_attribute.json
```

## demo10 演示子集（demo10/）

`demo10/queries.csv` + `demo10/labels.csv`：从 50 条查询里策划出的 10 条（7 正 3 负），
配套 `config/examples/fused_multilingual_demo10.json`（数据库仍指父集 300 条），
全量 HE 双轮约几十秒，是现场演示的推荐入口。

选样原则（`scripts/build_fused_multilingual_demo10.py` 生成，清单显式人工策划并带
expect 断言，父集重建导致不再满足原则时脚本直接报错退出）：

- **文字系统多样性**：拉丁重音（q02 为去重音罗马化查询）、汉字（q27/q06）、
  汉罗混合（q34）、阿拉伯（q47 本地文字 + neg-arabic-01 负例）、缅甸（q16 +
  neg-myanmar-01 负例）、拉丁负例 q15 —— 3 条负例分属拉丁/阿拉伯/缅甸。父集的
  10 条负例恰好全是拉丁文，凑不出三个文字系统，故阿拉伯/缅甸负例由脚本用构建
  管线同款逻辑从 SAGE 姓名池新造（固定种子、逐字节可复现、逐个校验冒充者分
  低于安全线）。
- **扰动多样性**：姓名 typo（q34/q16/q06）、dob 留空（q02/q27）、phone 留空
  （q06）、dob 年份对调（q47）、dob 格式变换（q13/q16）、城市大小写（q13/q47）。
- **true 分有讲头**：q13 满分（1.000）基线；q27（0.573）/q47（0.636）是
  0.5–0.7 区间"姓名单科不到线、靠 dob/city/phone 互补拉回阈值上"的讲解案例；
  q16（0.899）/q34（0.837）展示 typo + 跨文字下的稳定命中。

重新生成（`--report` 先看全部 50 条统计）：

```bash
python scripts/build_fused_multilingual_demo10.py --report
python scripts/build_fused_multilingual_demo10.py
```
