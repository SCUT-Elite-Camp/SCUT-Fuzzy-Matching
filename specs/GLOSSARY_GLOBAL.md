# 全局统一领域术语表 (GLOSSARY_GLOBAL)

> **适用范围**：`specs/` 下全部需求与技术规约。所有术语唯一定义于此（SSOT），
> 各需求文档仅引用，不得重复定义或另造同义词。
> **版本**：v1.0.0（2026-10-14）

## 一、协议角色与实体

| 术语 (中文) | 英文标识 / 符号 | 定义 |
| :--- | :--- | :--- |
| 查询方 | `Party A` | 持有待查询姓名、持有 CKKS 私钥的一方；发起两轮请求并在本地完成解密判定 |
| 数据方 | `Party B` | 持有姓名数据库的一方；仅持有公钥 context（无私钥），在密文上执行质心比较与列式匹配 |
| 命中判定 | `catch` | 单查询二值结论：所选簇内存在至少一列解密掩码得分通过阈值判定即为 true；不等于"找到了正确记录"，仅为检测（detection） |
| 簇 / 聚类 | `cluster` | B 侧对库记录签名做 K-Means 聚类得到的候选分组 |
| 质心 | `centroid` | 各 cluster 的中心向量，Round 1 与查询向量做密文点积 |
| 选择器 | `selector` | A 侧构造的 one-hot 向量，加密后发给 B，用于在密文上选中目标簇的候选列 |

## 二、编码与同态加密

| 术语 (中文) | 英文标识 / 符号 | 定义 |
| :--- | :--- | :--- |
| CKKS | `CKKS` | 近似数值全同态加密方案（Cheon-Kim-Kim-Song），本实现经 TenSEAL 调用；参数 `POLY_MODULUS_DEGREE=8192`、`COEFF_MOD_BIT_SIZES=[60,40,40,60]`、`SCALE=2^40` |
| MinHash | `MinHash` | 局部敏感哈希签名；姓名经 trigram shingle 后以 20-bit 哈希空间（`MAX_HASH=2^20`）、固定种子 `HASH_SEED=42` 生成签名，保证 A/B 两侧置换一致 |
| 字符 shingles | `shingle` | 姓名的连续字符片段，长度 `SHINGLE_SIZE=3`（trigram，含空格），为 MinHash 的输入特征 |
| 签名长度 | `EL` | 单条签名的维度：聚类签名 `EL=200`（`NUM_PERMUTATIONS_CLUSTER`），匹配签名 `EL=50`（`NUM_PERMUTATIONS_MATCH`） |
| slot | `slot` | CKKS 明文槽位；单密文可用 `S = 4096` 个（`CKKS_SLOT_LIMIT = POLY_MODULUS_DEGREE // 2`） |
| 批大小 | `batch_size (m)` | 单个密文批内并行承载的查询条数 |
| 候选槽宽 | `tile_width (T)` | 批内每条查询分到的候选槽数，`T = floor(S / m)` |
| tile | `tile` | 二维 query x candidate 平铺单元；槽位编址 `slot(q, t) = q * T + t` |
| 有效槽数 | `active_slots` | `m * T <= S`；余下为置零 padding 槽 |
| 解密灰区宽度 | `BATCH_DECRYPT_EPS` | 批处理路径阈值判定的数值灰区半宽，取值 `1e-4`；绝对值得分不超过该值时按未命中保守判负 |
| 随机掩码 | `mask` | B 侧为每个 (query, candidate) 对独立采样的正随机数，范围 `[1.0, 10.0]`；得分减去 `mask * tau` 后 A 只能获知符号，无法反推原始相似度 |

## 三、判定与评估口径

| 术语 (中文) | 英文标识 / 符号 | 定义 |
| :--- | :--- | :--- |
| 相似度阈值 | `tau (τ)` | 加权/余弦相似度阈值；取值以 PRD 5.4 阈值表为 SSOT（姓名单属性 0.9；FEBRL 0.6；fused_multilingual 0.485） |
| 应命中集合 | `should_catch` | 判定口径：真匹配的明文得分 > tau（严格大于）。召回率分母只含 should_catch 为真的查询，而非"标签为正"的全部查询 |
| 召回率（should_catch 口径） | `recall` | `#{catch 且 should_catch} / #{should_catch}` |
| 簇召回 | `cluster recall` | 真匹配所在簇被第二轮探测覆盖的正例比例；`top1-only` 变体仅计首轮 argmax 簇，刻画 Round 1 自身决策质量，与 tau 无关 |
| 冒充者 | `impostor` | 某查询的非匹配库记录；`strongest/highest impostor` 指其中得分最高者，是 tau 必须拒绝的对象 |
| 标定窗口 | `calibration window` | 区间 `[最低真匹配得分, 最高冒充者得分]`；两值交叠（interleave）时无单一 tau 兼得满召回与零误报 |
| 提前停止 | `early stop` | 当批内全部查询均已命中时停止消费剩余密文流；只能为"有命中"的查询节省成本 |
| 第二轮探测数 | `probes` | 多属性路径 Round 2 探测的簇数（按质心得分降序）；demo 默认 3，库默认 1，禁止无成本上限的穷举用于生产 |
| 聚类数策略 | `k-mode` | `sqrt`（k = floor(√n)，默认）/ `log2` / `auto`（k = floor(1.4√n)）/ 整数 / `fixed:<k>` |
| 模糊化比例 | `fuzzy_ratio` | 评测时对正样本查询施加单字母扰动的比例（默认 0.3），必须配固定种子，仅作用于评测查询副本 |

## 四、多属性 schema

| 术语 (中文) | 英文标识 / 符号 | 定义 |
| :--- | :--- | :--- |
| 属性 schema | `AttributeSchema` | 声明式多属性配置：属性列表（name / kind / weight / params）+ `similarity_threshold` + `name_attribute`；权重和必须为 1.0，维度不得超过 4096 |
| 属性类型 | `kind` | 内置三类：`fuzzy_text`（MinHash 模糊文本）、`exact`（分块哈希精确值）、`date`（日期，经 `normalize_dob`）；别名 `text`/`name`/`categorical`/`dob` 归一化到上述三类 |
| 权重 | `weight (wᵢ)` | 属性在组合得分 `S = Σ wᵢ · simᵢ` 中的权重；各属性块 L2 归一化后按 `sqrt(wᵢ)` 缩放再拼接 |
| schema 指纹 | `fingerprint` | schema 布局的确定性摘要；A/B 两侧指纹不一致必须 fail-closed 抛错，防止同宽异布局产生貌似合理的错误得分 |
| 非法日期策略 | `on_invalid` | `raise`（默认，fail-closed）或 `missing`（编码全零块且计数审计） |
| 日在前解析 | `day_first` | 默认 `false`；`DD/MM/YYYY` 等有歧义格式仅在显式声明 `day_first=true` 时解析 |
| 逐位容错编码 | `tolerate` | `exact`/`date` 的可选逐位对齐编码；默认关闭（FEBRL 实测净负收益），开启前必须实测 median impostor |

## 五、多语言归一化

| 术语 (中文) | 英文标识 / 符号 | 定义 |
| :--- | :--- | :--- |
| 多语言归一化器 | `unicode_v1` | 归一化管线：NFKC + casefold + 标点清理 + 拉丁重音折叠；保留原生 Unicode 文字并生成可检索变体 |
| 英文归一化器 | `english_v1` | NCVR 原始英文清洗规则：NFKC、转小写、移除非英文字母并压缩空格 |
| 可检索变体 | `variant` | 同一记录的可选检索形式，随原生姓名一并入库；含 `latin_folded`、`latin_transliterated`、`latin_compact` |
| 拉丁转写变体 | `latin_transliterated` / `latin_compact` | 由固定版本 `anyascii==0.3.3` 上下文无关转写生成的拉丁形式变体 |
| 文字系统 | `script` | 姓名所属书写系统类别（latin / arabic / han / myanmar 等），由归一化管线识别并计入 manifest 统计 |
| NFKC / casefold | `NFKC` / `casefold` | Unicode 兼容性归一化与跨文字大小写折叠，为全部归一化路径的前置步骤 |

## 六、数据集标识

| 术语 (中文) | 英文标识 | 定义 |
| :--- | :--- | :--- |
| NCVR 10K | `ncvr_10k` | 北卡罗来纳选民登记 10,000 条 B 侧记录 + 200 条查询的可提交子集（`data/ncvr_10k/`） |
| SAGE | `sage` / `sage_names` | 多语言姓名源表（123,479 行、23 国），清洗为 43,206 条记录、56,705 个去重搜索项；不含西里尔/韩文/日文 |
| FEBRL | `febrl` | 合成记录链接基准数据集（4a 库 / 4b 查询，含注入脏值）；经 `scripts/fetch_dataset.py` 获取，`dataset/` 不入库 |
| 融合多语言数据集 | `fused_multilingual` | SAGE 真实多语言姓名 300 条 + 合成 DOB/城市/电话属性；40 正例 + 10 负例；`--seed 20261014` 逐字节可复现 |
