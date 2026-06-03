# 五人分工接口规范 Agent 版

> 本文档基于 `新分工.md`，用于 5 名成员分别使用 AI 辅助实现模块时统一边界、接口和数据流。
> 分工方式为：`13 / 2 / 45 / 6789 / 测试`。
> 本文档强调实现约束，不重新解释论文背景。

---

## 0. 总分工

| 成员 | 负责阶段 | 模块定位 | 主要交付 |
|---|---|---|---|
| 成员一 | Step 1 + Step 3 | B 侧离线建库与第一轮质心匹配 | `party_b/offline_prep.py`、`clustering/kmeans_cosine.py`、`party_b/online_responder.py` 中的质心匹配 |
| 成员二 | Step 2 | A 侧查询向量生成、标准化、CKKS 加密 | `party_a/local_prep.py`、`ckks/context.py`、`ckks/keys.py` |
| 成员三 | Step 4 + Step 5 | A 侧解密质心结果、构造 one-hot、发送第二轮密文 | `party_a/online_querier.py` 中的 cluster 选择与第二轮请求 |
| 成员四 | Step 6 + Step 7 + Step 8 + Step 9 | B 侧列式匹配与 A 侧最终判断 | `party_b/online_responder.py` 中的列式匹配、`party_a/online_querier.py` 中的最终判断 |
| 成员五 | 测试 | 单元测试、端到端测试、评估和 benchmark | `tests/`、`evaluation/` |

---

## 1. 全局命名与数据约定

### 1.1 统一命名

流程图和旧文档中存在 `C`、`M` 混用。实现中统一使用：

| 名称 | 含义 | 形状 | 持有方 |
|---|---|---|---|
| `centroids` | 聚类质心矩阵 | `(k, 200)` | B |
| `cluster_matrix` | 列式名称矩阵 | `(k, max_size, 50)` | B |
| `scaler_mean` | B 侧 StandardScaler 均值 | `(200,)` | B 生成，公开给 A |
| `scaler_scale` | B 侧 StandardScaler 标准差 | `(200,)` | B 生成，公开给 A |
| `query_200_std` | A 侧标准化后的 EL=200 查询向量 | `(200,)` | A |
| `query_50_norm` | A 侧 L2 归一化后的 EL=50 查询向量 | `(50,)` | A |
| `encrypted_query_200` | 加密后的 `query_200_std` | TenSEAL ciphertext | A 发给 B |
| `encrypted_query_50` | 加密后的 `query_50_norm` | TenSEAL ciphertext | A 第二轮发给 B |
| `selector` | 最匹配 cluster 的 one-hot 向量 | `(k,)` | A |
| `encrypted_selector` | 加密后的 one-hot 向量 | TenSEAL ciphertext | A 发给 B |
| `encrypted_scores` | B 返回的列式匹配密文分数流 | iterator/list | B 发给 A |
| `catch` | 最终是否匹配 | `bool` | A |

### 1.2 全局参数来源

所有模块必须从 `config/params.py` 导入参数，不允许在业务代码里硬编码。

```python
SHINGLE_SIZE = 3
NUM_PERMUTATIONS_CLUSTER = 200
NUM_PERMUTATIONS_MATCH = 50
MAX_HASH = 2 ** 20
HASH_SEED = 42
K_CLUSTERS_FUNC = lambda n: int(n ** 0.5)
KMEANS_ITERATIONS = 20
POLY_MODULUS_DEGREE = 8192
COEFF_MOD_BIT_SIZES = [60, 40, 40, 60]
SCALE = 2 ** 40
SIMILARITY_THRESHOLD = 0.9
DECRYPT_EPS = 1e-6
RANDOM_MASK_MIN = 1.0
RANDOM_MASK_MAX = 10.0
```

`K_CLUSTERS_FUNC = sqrt(n)` 是当前 baseline 的工程默认。若要复刻论文实验表，应新增 `choose_k(n, mode)`，用 `paper` 模式按实验设置映射固定 k，用 `sqrt` 模式保留当前默认策略。实现者不得把 `sqrt(n)` 误写成论文所有实验的唯一 k 规则。

`DECRYPT_EPS` 是当前 baseline 的经验默认值，不是理论常数。若 CKKS 参数、乘法层数、序列化方式或随机 mask 范围变化，必须通过 `test_ckks_ops.py` 和端到端样例重新校准。

`RANDOM_MASK_MIN` 和 `RANDOM_MASK_MAX` 是 Step 8 随机混淆系数的工程范围。`r_j` 必须为正数，并在该范围内采样，避免过度放大 CKKS 近似误差。

### 1.3 必须复用的基础模块和库

已有基础模块：

| 能力 | 优先复用 |
|---|---|
| 名字清洗 | `preprocessing.text_cleaner.clean_name` |
| MinHash 签名 | `minhash.encoder.generate_signature`、`minhash.encoder.batch_encode` |
| L2 归一化 | `preprocessing.normalizer.l2_normalize`，也可内部改为封装 `sklearn.preprocessing.normalize` |
| 标准化 | `sklearn.preprocessing.StandardScaler` |
| 聚类 | cosine/spherical K-Means；`sklearn.cluster.KMeans` 只能作为初始化或底层循环辅助 |
| 同态加密 | `TenSEAL` |
| 数值计算 | `numpy` |
| 测试 | `pytest` |

### 1.4 禁止越界规则

| 规则 | 原因 |
|---|---|
| A 侧不得重新 fit scaler | A 必须使用 B 公开的 `scaler_mean` 和 `scaler_scale` |
| 下游模块不得重新实现 MinHash | 保证 `EL=50` 是 `EL=200` 的前 50 维 |
| B 侧不得拿到 A 的私钥 | 协议安全边界 |
| 列式矩阵 padding 必须使用零向量 | dummy 元素不能产生正匹配 |
| HE 模块不得用明文 cosine 替代密文运算 | 明文 cosine 只能出现在测试或 debug |
| 成员四不得重新选择 cluster | cluster 选择属于成员三 |
| `selected_cluster` 不得进入 B 侧接口 | B 只能收到加密后的 `encrypted_selector` |
| 成员五不得修改业务逻辑来绕过测试 | 测试只暴露问题，修复回到对应模块成员 |

---

## 1.5 当前范围：单查询基线

当前 baseline 明确只支持单查询 `m=1`。所有 A 侧查询向量在协议接口中均按一维向量处理：

| 数据 | 形状 |
|---|---|
| `query_200_std` | `(200,)` |
| `query_50_norm` | `(50,)` |
| `sim_scores` | `(k,)` |
| `selector` | `(k,)` |

批处理是后续扩展，不得在当前接口中半实现。若后续支持 batching，必须整体改为 `selected_clusters: (m,)`、`selectors: (m, k)`，并重新定义 TenSEAL packing layout。

## 1.6 CKKS 表示约定

TenSEAL 的 `CKKSVector` 是 SIMD packed ciphertext，不能在业务模块中假设可以像普通数组一样自由访问 `E(x)[i]`。所有密文向量操作必须封装在 `ckks/operations.py` 内，成员二和成员四必须使用同一套表示。

当前 baseline 采用 correctness-first 表示：

| 名称 | 表示 |
|---|---|
| `encrypted_query_200` | `EncryptedVector200`，具体类型由 `protocol/types.py` 统一定义 |
| `encrypted_query_50` | `EncryptedVector50`，必须与 Step 7 的 CT-CT dot 实现兼容 |
| `encrypted_selector` | `EncryptedVectorK`，必须与 Step 6 的 CT-PT selection 实现兼容 |
| `encrypted_selected_name_j` | 默认表示为 50 个 encrypted scalar 的列表，除非 `ckks/operations.py` 明确实现 packed vector 版本 |

如果采用 packed CKKSVector 优化，必须同时更新成员二的加密输出、成员四的 Step 6/7、通信量评估和 `test_ckks_ops.py`。不得让成员二输出 packed vector，而成员四按 scalar list 读取。

`protocol/types.py` 必须定义以下类型别名，所有成员按这些别名写接口，不得自行发明密文类型名：

```python
from typing import Any, TypeAlias

CipherBytes: TypeAlias = bytes
CipherObject: TypeAlias = Any
CipherLike: TypeAlias = CipherBytes | CipherObject

EncryptedVector200: TypeAlias = CipherLike
EncryptedVector50: TypeAlias = CipherLike
EncryptedVectorK: TypeAlias = CipherLike
EncryptedScalar: TypeAlias = CipherLike
```

网络通信和通信量评估必须使用 `bytes`。同进程端到端测试可以使用 TenSEAL 对象，但必须在测试名或 fixture 中明确这是 in-process 模式。

## 1.7 Public Context 边界

`public_context_bytes` 必须包含 B 侧密文计算需要的 public key、relinearization keys，以及需要旋转时的 galois keys。

`public_context_bytes` 不得包含 secret key。成员五必须增加测试确认 public context 不含 secret key。

---

## 2. 推荐数据结构

实现时可放在 `protocol/types.py` 或各模块本地，字段名需保持一致。

```python
from dataclasses import dataclass
from typing import Iterator
import numpy as np


@dataclass
class OfflineArtifacts:
    centroids: np.ndarray
    cluster_matrix: np.ndarray
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    cluster_assignments: np.ndarray
    max_size: int


@dataclass
class FirstRoundRequest:
    public_context_bytes: bytes
    encrypted_query_200: EncryptedVector200


@dataclass
class PartyALocalState:
    secret_context: object
    encrypted_query_50: EncryptedVector50


@dataclass
class SecondRoundRequest:
    encrypted_query_50: EncryptedVector50
    encrypted_selector: EncryptedVectorK


@dataclass
class ClusterSelectionDebug:
    selected_cluster: int


@dataclass
class MatchResult:
    catch: bool


@dataclass
class MatchDebug:
    checked_columns: int
    first_positive_column: int | None
```

`FirstRoundRequest` 是第一轮唯一允许发给 B 侧的网络请求，只包含 public context 和 `encrypted_query_200`。

`PartyALocalState` 留在 A 侧本地，供成员三和成员四的 A 侧判断逻辑使用，不得交给 B 侧。它可以保存 `secret_context` 和第二轮要用的 `encrypted_query_50`。

`SecondRoundRequest` 是唯一允许交给 B 侧的第二轮请求。`ClusterSelectionDebug` 只能留在 A 侧日志、debug 或测试中，不得作为 `column_wise_matching` 的参数。

`MatchResult` 是 production API 的对外结果，只包含 `catch`。`MatchDebug.checked_columns` 和 `MatchDebug.first_positive_column` 只能用于 A 侧本地 benchmark、debug 或测试；若对外暴露，会泄露 early stop 命中的列位置信息。

序列化边界建议：

| 场景 | 内部计算 | 跨模块传输 |
|---|---|---|
| 同一进程端到端测试 | TenSEAL 对象可直接传递 | 不强制序列化 |
| 模拟网络通信 | 计算完成后 `.serialize()` | `bytes` |
| 通信开销评估 | 必须使用序列化后的 `bytes` | 统计字节长度 |

当前 correctness-first 版本的密文表示可能不复现论文中的通信量估计。`evaluation/communication_cost.py` 必须按实际 `.serialize()` 结果统计。若要对齐论文通信量，需要切换到统一 packed layout，并重新跑通信评估。

---

## 3. 成员一：Step 1 + Step 3

### 3.1 模块职责

成员一负责 B 侧的两个环节。

Step 1 是离线建库。输入是 B 的原始姓名列表，输出是后续在线阶段需要的 `centroids`、`cluster_matrix`、`scaler_mean` 和 `scaler_scale`。

Step 3 是第一轮在线质心匹配。输入是 A 发来的 `FirstRoundRequest` 和 B 本地的 `centroids`，输出是加密的质心相似度分数。

### 3.2 上游输入

Step 1 输入：

| 输入 | 类型 | 来源 |
|---|---|---|
| `names_b` | `Iterable[str]` | B 侧原始数据库 |
| 全局参数 | `config/params.py` | 配置模块 |

Step 3 输入：

| 输入 | 类型 | 来源 |
|---|---|---|
| `first_round_request` | `FirstRoundRequest` | 成员二 |
| `centroids` | `np.ndarray`，形状 `(k, 200)` | Step 1 产物 |

### 3.3 下游输出

Step 1 输出：

| 输出 | 类型 | 下游 |
|---|---|---|
| `centroids` | `np.ndarray`，形状 `(k, 200)` | Step 3 |
| `cluster_matrix` | `np.ndarray`，形状 `(k, max_size, 50)` | 成员四 |
| `scaler_mean` | `np.ndarray`，形状 `(200,)` | 成员二 |
| `scaler_scale` | `np.ndarray`，形状 `(200,)` | 成员二 |
| `cluster_assignments` | `np.ndarray`，形状 `(n,)` | 测试与调试 |
| `max_size` | `int` | 成员四与测试 |

Step 3 输出：

| 输出 | 类型 | 下游 |
|---|---|---|
| `encrypted_sim_scores` | list of TenSEAL ciphertext 或 list of `bytes` | 成员三 |

### 3.4 对外接口

建议实现：

```python
def prepare_party_b_offline(names_b: Iterable[str]) -> OfflineArtifacts:
    ...


def compare_to_centroids(
    first_round_request: FirstRoundRequest,
    centroids: np.ndarray,
) -> list:
    ...
```

如果 orchestrator 为了适配旧接口需要拆包，只能在 orchestrator 层临时拆出 `public_context_bytes` 和 `encrypted_query_200`。B 侧 Step 3 函数不得接收或访问 `encrypted_query_50`。

若拆分更细，可使用：

```python
def fit_scaler(normalized_200: np.ndarray) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    ...


def choose_k(n: int, mode: str = "sqrt") -> int:
    ...


def run_cosine_kmeans(standardized_200: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    ...


def build_cluster_matrix(
    normalized_50: np.ndarray,
    cluster_assignments: np.ndarray,
    k: int,
) -> tuple[np.ndarray, int]:
    ...
```

### 3.5 内部实现逻辑

离线建库流程：

1. 调用 `batch_encode(names_b, NUM_PERMUTATIONS_CLUSTER)` 得到 `signatures_200`。
2. 使用 `signatures_200[:, :NUM_PERMUTATIONS_MATCH]` 得到 `signatures_50`，不得单独用不同种子生成。
3. 对 `signatures_200` 和 `signatures_50` 分别做逐行 L2 归一化。
4. 使用 `StandardScaler().fit(normalized_200)` 拟合 B 侧 scaler。
5. 使用同一个 scaler 生成 `standardized_200`。
6. 在 `standardized_200` 上运行 cosine/spherical K-Means，得到 `cluster_assignments` 和 `centroids`。
7. 使用 `cluster_assignments` 对 `normalized_50` 分组，构建 `cluster_matrix`。
8. 每个 cluster 不足 `max_size` 的位置用全零向量补齐。
9. 保存 artifacts 时使用 `.npy`，字段名和 `OfflineArtifacts` 保持一致。

cosine/spherical K-Means 强约束：

1. 不得直接用 `sklearn.cluster.KMeans` 的欧氏距离结果替代最终聚类语义。
2. 每轮分配必须按 cosine similarity 选择最近质心。
3. 每轮更新质心时先取 cluster 内向量均值，再做 L2 normalize。
4. 若复用 `sklearn.cluster.KMeans`，只能用于初始化质心或辅助迭代，最终 `cluster_assignments` 和 `centroids` 必须满足上述 cosine 语义。

empty cluster 处理策略必须固定，且不得生成 NaN 质心。当前 baseline 默认策略是保留该 cluster 的上一轮质心；如果没有上一轮质心，则用当前误差最大的样本重置该质心。成员五必须覆盖 empty cluster 测试。

质心匹配流程：

1. 从 `first_round_request` 读取 `public_context_bytes` 和 `encrypted_query_200`。
2. 用 public context 反序列化 `encrypted_query_200`。
3. 遍历每个 centroid。
4. 对每个 centroid 调用 `DotProduct_CT_PT`。
5. 返回加密相似度列表。
6. 若走网络模拟，在返回前序列化为 `bytes`。

### 3.6 使用标准库和第三方库

| 类别 | 使用项 |
|---|---|
| 标准库 | `typing`、`pathlib` |
| 第三方 | `numpy`、`sklearn.preprocessing.StandardScaler`、`sklearn.metrics.pairwise.cosine_similarity`、`TenSEAL` |
| 项目模块 | `config.params`、`minhash.encoder`、`preprocessing.normalizer`、`ckks.operations` |

### 3.7 验收条件

| 项目 | 条件 |
|---|---|
| `centroids` | 形状为 `(k, 200)` |
| `centroids` L2 | 每个非空质心应完成 L2 normalize |
| `cluster_matrix` | 形状为 `(k, max_size, 50)` |
| padding | 所有 dummy 行必须是全零向量 |
| scaler | `mean_` 和 `scale_` 均为长度 200 |
| Step 3 | 解密后的质心分数与明文点积近似一致 |

---

## 4. 成员二：Step 2

### 4.1 模块职责

成员二负责 A 侧查询预处理和查询加密。输入是查询姓名和 B 公开的 scaler 参数，输出是第一轮要发送给 B 的加密 EL=200 查询向量，并提前准备第二轮会用到的加密 EL=50 查询向量。

### 4.2 上游输入

| 输入 | 类型 | 来源 |
|---|---|---|
| `query_name` | `str` | 查询方 |
| `scaler_mean` | `np.ndarray`，形状 `(200,)` | 成员一 |
| `scaler_scale` | `np.ndarray`，形状 `(200,)` | 成员一 |
| CKKS 参数 | `config/params.py` | 配置模块 |

### 4.3 下游输出

| 输出 | 类型 | 下游 |
|---|---|---|
| `public_context_bytes` | `bytes` | 成员一和成员四 |
| `FirstRoundRequest.encrypted_query_200` | TenSEAL ciphertext 或 `bytes` | 成员一 |
| `PartyALocalState.encrypted_query_50` | TenSEAL ciphertext 或 `bytes` | 成员三 |
| `PartyALocalState.secret_context` | TenSEAL context | 只留在 A 侧，供成员三和成员四的 A 侧逻辑使用 |

### 4.4 对外接口

建议实现：

```python
def prepare_encrypted_query(
    query_name: str,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
) -> tuple[FirstRoundRequest, PartyALocalState]:
    ...
```

拆分接口：

```python
def encode_query_vectors(
    query_name: str,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ...


def create_ckks_context():
    ...


def encrypt_query_vectors(
    query_200_std: np.ndarray,
    query_50_norm: np.ndarray,
    context,
) -> tuple[FirstRoundRequest, PartyALocalState]:
    ...
```

### 4.5 内部实现逻辑

1. 调用 `batch_encode([query_name], NUM_PERMUTATIONS_CLUSTER)` 得到二维矩阵 `query_200_raw_matrix`，形状为 `(1, 200)`。
2. 用 `query_200_raw_matrix[:, :NUM_PERMUTATIONS_MATCH]` 得到 `query_50_raw_matrix`，形状为 `(1, 50)`。
3. 先在二维矩阵形状 `(1, 200)`、`(1, 50)` 上调用与 B 侧相同的 `l2_normalize`。
4. 得到 `query_200_norm_matrix` 和 `query_50_norm_matrix` 后，再 squeeze 成 `(200,)` 和 `(50,)` 交给下游。
5. 使用成员一提供的 `scaler_mean` 和 `scaler_scale` 对 `query_200_norm` 做标准化，得到 `query_200_std`。
6. `query_50_norm` 只做 L2 归一化，不做 StandardScaler。
7. 创建 TenSEAL CKKS context，设置 `poly_modulus_degree`、`coeff_mod_bit_sizes` 和 `global_scale`。
8. 生成必要的 Galois keys 和 relin keys。
9. 用 secret context 加密 `query_200_std` 和 `query_50_norm`。
10. 构造 `FirstRoundRequest(public_context_bytes, encrypted_query_200)` 作为第一轮网络 payload。
11. 构造 `PartyALocalState(secret_context, encrypted_query_50)` 留在 A 侧本地。
12. 第一轮不得把 `encrypted_query_50` 发给 B。

### 4.6 使用标准库和第三方库

| 类别 | 使用项 |
|---|---|
| 标准库 | `typing` |
| 第三方 | `numpy`、`TenSEAL`、`sklearn.preprocessing.normalize` 或项目封装 |
| 项目模块 | `config.params`、`minhash.encoder`、`preprocessing.normalizer`、`ckks.context`、`ckks.keys` |

### 4.7 验收条件

| 项目 | 条件 |
|---|---|
| `query_200_std` | 形状为 `(200,)` |
| `query_50_norm` | 形状为 `(50,)` |
| normalize 顺序 | 先按二维矩阵调用 B 侧同款 normalizer，再 squeeze |
| scaler | 未在 A 侧 fit，只使用 B 传入参数 |
| `query_50_norm` | 不做 StandardScaler |
| public context | 包含 public key、relinearization keys、必要时包含 galois keys，不包含 secret key |
| 第一轮 payload | 只包含 `public_context_bytes` 和 `encrypted_query_200` |
| 私钥 | 不写入 B 侧 artifacts，不参与 B 侧函数参数 |

---

## 5. 成员三：Step 4 + Step 5

### 5.1 模块职责

成员三负责 A 侧收到质心匹配结果后的选择逻辑。输入是成员一返回的加密质心分数和成员二保留的 A 侧 secret context，输出是第二轮发送给 B 的 `encrypted_selector` 和 `encrypted_query_50`。`selected_cluster` 只能作为 A 侧 debug 信息保存，不得进入 B 侧请求对象。

### 5.2 上游输入

| 输入 | 类型 | 来源 |
|---|---|---|
| `encrypted_sim_scores` | list of TenSEAL ciphertext 或 list of `bytes` | 成员一 |
| `PartyALocalState.secret_context` | TenSEAL context | 成员二 |
| `PartyALocalState.encrypted_query_50` | TenSEAL ciphertext 或 `bytes` | 成员二 |
| `k` | `int` | 成员一的 artifacts |

### 5.3 下游输出

| 输出 | 类型 | 下游 |
|---|---|---|
| `SecondRoundRequest.encrypted_selector` | TenSEAL ciphertext 或 `bytes` | 成员四 |
| `SecondRoundRequest.encrypted_query_50` | TenSEAL ciphertext 或 `bytes` | 成员四 |
| `ClusterSelectionDebug.selected_cluster` | `int` | 仅 A 侧测试、日志、debug |

### 5.4 对外接口

建议实现：

```python
def choose_cluster_and_build_request(
    encrypted_sim_scores,
    party_a_state: PartyALocalState,
    k: int,
) -> tuple[SecondRoundRequest, ClusterSelectionDebug]:
    ...
```

拆分接口：

```python
def decrypt_sim_scores(encrypted_sim_scores, secret_context) -> np.ndarray:
    ...


def build_selector(sim_scores: np.ndarray, k: int) -> tuple[int, np.ndarray]:
    ...


def encrypt_selector(selector: np.ndarray, secret_context):
    ...
```

### 5.5 内部实现逻辑

1. 使用 A 的 secret context 解密 `encrypted_sim_scores`。
2. 将结果整理为长度为 `k` 的 `np.ndarray`。
3. 使用 `np.argmax(sim_scores)` 得到 `selected_cluster`。
4. 构造长度为 `k` 的 one-hot 向量。
5. 使用 A 的 context 加密 one-hot，得到 `encrypted_selector`。
6. 从 `PartyALocalState` 取出 `encrypted_query_50`，构造 `SecondRoundRequest(encrypted_query_50, encrypted_selector)` 交给成员四。
7. 构造 `ClusterSelectionDebug(selected_cluster)` 留在 A 侧测试或日志中。
8. 将 `encrypted_query_50` 原样放入 `SecondRoundRequest`，不重新编码、不重新加密。

### 5.6 使用标准库和第三方库

| 类别 | 使用项 |
|---|---|
| 标准库 | `typing` |
| 第三方 | `numpy`、`TenSEAL` |
| 项目模块 | `config.params`、`party_a.local_prep` |

### 5.7 验收条件

| 项目 | 条件 |
|---|---|
| `sim_scores` | 解密后长度为 `k` |
| `selector` | 只有一个位置为 1，其余为 0 |
| `selected_cluster` | 等于 `np.argmax(sim_scores)` |
| `encrypted_query_50` | 不被修改 |
| 输出边界 | 不访问 B 的 `cluster_matrix` |
| 隐私边界 | 交给 B 的 `SecondRoundRequest` 不含 `selected_cluster` |

---

## 6. 成员四：Step 6 + Step 7 + Step 8 + Step 9

### 6.1 模块职责

成员四负责第二轮列式匹配和最终判断。该成员同时实现 B 侧密文列式计算和 A 侧解密判断，但必须保持两个角色的数据边界。

B 侧只接收 `SecondRoundRequest`、public context 和 B 本地 `cluster_matrix`。A 侧只接收 B 返回的加密分数，用 secret context 判断是否存在匹配。

### 6.2 上游输入

B 侧 Step 6 到 Step 8 输入：

| 输入 | 类型 | 来源 |
|---|---|---|
| `cluster_matrix` | `np.ndarray`，形状 `(k, max_size, 50)` | 成员一 |
| `second_round_request.encrypted_selector` | TenSEAL ciphertext 或 `bytes` | 成员三 |
| `second_round_request.encrypted_query_50` | TenSEAL ciphertext 或 `bytes` | 成员三 |
| `public_context` | TenSEAL context 或 `bytes` | 成员二 |
| `tau` | `float` | `SIMILARITY_THRESHOLD` |

A 侧 Step 9 输入：

| 输入 | 类型 | 来源 |
|---|---|---|
| `encrypted_scores` | iterator/list of TenSEAL ciphertext 或 `bytes` | B 侧 Step 8 |
| `secret_context` | TenSEAL context | 成员二 |
| `early_stop` | `bool` | 协议配置 |

### 6.3 下游输出

| 输出 | 类型 | 下游 |
|---|---|---|
| `encrypted_scores` | iterator/list | A 侧 Step 9 |
| `MatchResult` | dataclass | protocol、tests、evaluation |

### 6.4 对外接口

B 侧建议实现：

```python
def column_wise_matching(
    cluster_matrix: np.ndarray,
    second_round_request: SecondRoundRequest,
    public_context,
    tau: float,
) -> Iterator:
    ...
```

A 侧建议实现：

```python
def check_encrypted_scores(
    encrypted_scores,
    secret_context,
    early_stop: bool = True,
    eps: float = DECRYPT_EPS,
) -> MatchResult:
    ...


def check_encrypted_scores_debug(
    encrypted_scores,
    secret_context,
    early_stop: bool = True,
    eps: float = DECRYPT_EPS,
) -> tuple[MatchResult, MatchDebug]:
    ...
```

底层 HE 操作建议统一放在 `ckks/operations.py`：

```python
def dot_ct_pt(encrypted_vector, plain_vector: np.ndarray):
    ...


def dot_ct_ct(encrypted_left, encrypted_right):
    ...


def add_plain(encrypted_value, plain_value: float):
    ...


def mul_plain(encrypted_value, plain_value: float):
    ...
```

### 6.5 内部实现逻辑

Step 6：

1. 遍历 `j`，范围是 `0` 到 `max_size - 1`。
2. 取 `column_j = cluster_matrix[:, j, :]`，形状为 `(k, 50)`。
3. 使用 `encrypted_selector` 与 `column_j` 做 CT-PT 选择。
4. 得到 `encrypted_selected_name_j`，语义上是最匹配 cluster 中第 `j` 个名字的 EL=50 签名。

Step 7：

1. 使用 `encrypted_query_50` 和 `encrypted_selected_name_j` 做 CT-CT 点积。
2. 得到 `encrypted_cos_score_j`。
3. 该结果是加密标量，明文语义是余弦相似度。

Step 8：

1. 计算 `encrypted_temp_j = encrypted_cos_score_j - tau`。
2. 为每一列生成一个新的正随机数 `r_j`，采样范围为 `[RANDOM_MASK_MIN, RANDOM_MASK_MAX]`。
3. 计算 `encrypted_score_j = encrypted_temp_j * r_j`。
4. 逐列 yield 或 append `encrypted_score_j`。
5. 不得使用过大的随机数，以免放大 CKKS 近似误差并击穿 `DECRYPT_EPS`。
6. 若需要通信评估，输出前序列化。

Step 9：

1. A 侧逐个解密 `encrypted_score_j`。
2. 如果任意分数大于 `DECRYPT_EPS`，返回 `catch=True`。
3. 开启 early stop 时，一旦发现正分数立即停止消费后续分数。
4. 如果全部分数都不大于 `DECRYPT_EPS`，返回 `catch=False`。
5. 对外语义统一为“相似度超过阈值”，不要写成严格的“相似度大于等于阈值”。
6. `checked_columns` 和 `first_positive_column` 只能写入 A 侧 `MatchDebug`，不得放进 production `MatchResult`。

### 6.6 使用标准库和第三方库

| 类别 | 使用项 |
|---|---|
| 标准库 | `typing`、`secrets` 或 `random.SystemRandom` |
| 第三方 | `numpy`、`TenSEAL` |
| 项目模块 | `config.params`、`ckks.operations`、`protocol.types` |

### 6.7 A/B 侧实现隔离

成员四虽然负责 Step 6 到 Step 9，但必须把 B 侧密文计算和 A 侧解密判断放在不同文件中：

| 角色 | 文件 | 允许持有 |
|---|---|---|
| B 侧 | `party_b/online_responder.py` | `cluster_matrix`、`public_context`、`SecondRoundRequest` |
| A 侧 | `party_a/online_querier.py` | `secret_context`、`encrypted_scores` |

硬性限制：

- B 侧函数参数不得出现 `secret_context`。
- A 侧最终判断函数参数不得出现 `cluster_matrix`。
- 不得用一个类同时保存 `cluster_matrix` 和 `secret_context`。
- `first_positive_column` 与 `checked_columns` 只允许保存在 A 侧 debug 结果中。

### 6.8 验收条件

| 项目 | 条件 |
|---|---|
| Step 6 | 不能解密 `encrypted_selector` |
| Step 7 | 解密后的结果应近似明文余弦相似度 |
| Step 8 | 每列使用不同正随机数，且范围受 `RANDOM_MASK_MIN` 和 `RANDOM_MASK_MAX` 约束 |
| Step 9 | `catch=True` 当且仅当存在分数大于 `DECRYPT_EPS` |
| early stop | `checked_columns` 只出现在 A 侧 debug，不出现在 production result |
| production result | 对外只暴露 `catch` |
| A/B 隔离 | B 侧函数无 `secret_context`，A 侧函数无 `cluster_matrix` |

---

## 7. 成员五：测试与评估

### 7.1 模块职责

成员五负责证明各模块可独立工作，也能端到端跑通。测试不得改业务逻辑，只能通过断言暴露接口不一致、形状错误、协议语义错误和性能退化。

### 7.2 上游输入

| 输入 | 来源 |
|---|---|
| 成员一到成员四的公开接口 | 对应模块 |
| 小规模固定数据集 | `tests/fixtures/` |
| artifacts | B 侧离线输出 |
| protocol logs | 端到端调度输出 |

### 7.3 下游输出

| 输出 | 用途 |
|---|---|
| 单元测试结果 | 验证模块正确性 |
| 端到端测试结果 | 验证协议链路 |
| 指标报告 | Precision、Recall、F1 |
| 通信量报告 | 各步骤密文字节数 |
| benchmark 报告 | 时间、内存、规模扩展 |

### 7.4 测试文件建议

| 文件 | 覆盖范围 |
|---|---|
| `tests/test_minhash.py` | 清洗、shingle、MinHash、EL 前缀一致性 |
| `tests/test_normalizer.py` | L2 normalize、StandardScaler 参数复用 |
| `tests/test_clustering.py` | K-Means 输出、cluster matrix shape、zero padding |
| `tests/test_ckks_ops.py` | CT-PT、CT-CT、add plain、mul plain |
| `tests/test_party_a.py` | 查询编码、加密、one-hot 构造 |
| `tests/test_party_b.py` | 离线建库、质心匹配、列式匹配 |
| `tests/test_end_to_end.py` | 从 B 离线建库到 A 最终 catch |
| `evaluation/metrics.py` | Accuracy、Precision、Recall、F1 |
| `evaluation/communication_cost.py` | `.serialize()` 后的通信量 |
| `evaluation/benchmark.py` | latency、memory、scaling |

### 7.5 核心测试用例

MinHash：

```python
def test_el50_is_prefix_of_el200():
    ...
```

Normalization：

```python
def test_party_a_uses_party_b_scaler():
    ...
```

Clustering：

```python
def test_cluster_matrix_shape_and_zero_padding():
    ...


def test_cosine_kmeans_centroids_are_l2_normalized():
    ...


def test_empty_cluster_does_not_create_nan_centroid():
    ...
```

CKKS：

```python
def test_ct_pt_dot_matches_plain_dot_after_decrypt():
    ...


def test_ct_ct_dot_matches_plain_dot_after_decrypt():
    ...


def test_public_context_has_no_secret_key():
    ...


def test_ckks_layout_is_consistent_between_query_and_column_matching():
    ...


def test_random_mask_preserves_sign_without_amplifying_noise():
    ...


def test_first_round_request_does_not_include_query_50():
    ...


def test_second_round_request_does_not_include_selected_cluster():
    ...


def test_party_b_functions_do_not_accept_secret_context():
    ...
```

End to end：

```python
def test_protocol_returns_catch_for_known_similar_name():
    ...


def test_protocol_returns_no_catch_for_known_different_name():
    ...
```

### 7.6 验收条件

| 项目 | 条件 |
|---|---|
| 单测 | 每个成员模块至少有独立测试 |
| 端到端 | 小规模样例能完整跑通 |
| 通信评估 | 能输出 Step 2、Step 3、Step 5、Step 6-8 的字节数 |
| 指标评估 | 能计算 Precision、Recall、F1 |
| benchmark | 至少支持 10k 前的 smoke benchmark 或小规模替代 |

---

## 8. 端到端数据流

```text
成员一 Step 1:
names_b
→ signatures_200, signatures_50
→ normalized_200, normalized_50
→ scaler_mean, scaler_scale
→ centroids, cluster_matrix

成员二 Step 2:
query_name + scaler_mean + scaler_scale
→ query_200_std, query_50_norm
→ FirstRoundRequest(public_context, encrypted_query_200)
→ PartyALocalState(secret_context, encrypted_query_50)

成员一 Step 3:
encrypted_query_200 + centroids
→ encrypted_sim_scores

成员三 Step 4/5:
encrypted_sim_scores + PartyALocalState
→ selected_cluster 仅留在 A 侧 debug
→ SecondRoundRequest(encrypted_query_50, encrypted_selector)

成员四 Step 6/7/8:
cluster_matrix + SecondRoundRequest
→ encrypted_scores

成员四 Step 9:
encrypted_scores + PartyALocalState.secret_context
→ MatchResult(catch)
→ MatchDebug 仅留在 A 侧本地

成员五:
all public interfaces + artifacts + protocol outputs
→ tests + metrics + benchmark
```

---

## 9. 集成顺序

1. 先固定 `config/params.py` 和 MinHash 接口。
2. 成员一完成 B 侧离线 artifacts。
3. 成员二用成员一的 scaler 参数完成 A 侧加密查询。
4. 成员一完成 Step 3 质心匹配。
5. 成员三完成 cluster 选择和第二轮请求。
6. 成员四完成列式匹配和最终判断。
7. 成员五补齐单测和端到端测试。
8. 最后由 `protocol/orchestrator.py` 串联全流程。

---

## 10. Agent 实现提示词模板

各成员使用 AI 辅助实现时，建议把对应段落和以下约束一起提供给 agent：

```text
请只实现我负责的模块，不改其他成员模块。
必须复用现有 MinHash、normalizer、config 参数。
数据归一化和标准化优先使用 scikit-learn.preprocessing。
CKKS/FHE 优先使用 TenSEAL。
当前 baseline 只实现单查询 m=1。
第一轮只能发送 FirstRoundRequest，不能提前发送 encrypted_query_50。
不得把 selected_cluster 传给 B 侧接口。
不得用 sklearn 欧氏 KMeans 结果替代 cosine/spherical KMeans。
必须遵守 ckks/operations.py 中的统一密文表示。
成员四必须把 B 侧 online_responder 和 A 侧 online_querier 放在不同文件，B 侧函数不得接收 secret_context。
不得改变协议语义和数据形状。
如果发现上游接口缺失，只定义最小兼容适配层，并在注释或文档中说明。
```
