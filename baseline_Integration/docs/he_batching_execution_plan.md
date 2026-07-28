# HE 二维 SIMD Batching 施工、测试与收紧计划（V2）

状态：**V2 已完成生产切换和全量验收（2026-07-28）**。本计划替代旧的“只把 query 放进 slots、仍逐列计算”方案。

已验证结果：

```text
生产尺寸                   m=200, k=50, L=483
布局                       T=20, active_slots=4000
R1 / R2 输出密文          3 / 25
合成完整 HE 墙钟             79.134s
真实 NCVR 10K online_total  144.421s
真实 NCVR 10K offline       1.248s
相对旧 45min                 18.7x
数值 max_abs_error          9.57e-5
阈值外符号一致率             100%
```

固定执行环境：

```powershell
Set-Location D:\Projects\SCUT-Fuzzy-Matching\baseline_Integration
C:\Users\Dinking\miniconda3\python.exe -m pytest tests -q
```

当前工作分支为 `codex/he-batching`，工作区已有未提交改动。施工期间禁止 `git reset --hard`、`git clean` 或删除上级目录未跟踪文件。任何阶段开始前先查看 `git status --short` 和本阶段文件的 diff，只修改本计划授权的文件。

---

## 1. 问题复盘与施工目标

旧方案采用单轴 SoA 布局：

```text
一个 feature 一个 ciphertext
slot 只表示 query
200 queries 只使用 200 / 4096 slots
```

它确实消除了外层 per-query 完整协议调用，但第二轮仍然执行：

```python
for column_index in range(max_size):
    # selector + 50 维点积 + mask + 返回一个密文
```

NCVR 10K 的实际参数是：

```text
queries m          = 200
clusters k         = 50
max_size L         = 483
CKKS slot capacity = 4096
```

因此旧 batch 仍需 483 次第二轮同态核，实测估计约 38 分钟，未满足“查询向量整体批处理并显著降低全量 wall time”的需求。

V2 固定采用二维 slots 布局：

```text
slot = query × logical-item
logical-item 在第一轮表示 centroid
logical-item 在第二轮表示 candidate column
```

当前参数下：

```text
tile_width T = floor(4096 / 200) = 20
active_slots = 200 × 20 = 4000
slot utilization = 97.66%

第一轮：50 centroids -> ceil(50 / 20) = 3 个 tile ciphertext
第二轮：483 columns -> ceil(483 / 20) = 25 个 tile ciphertext
```

本机临时原型已经证明该布局可行，但尚未进入生产源码：

```text
Round 1: 50 个逐项输出 7.160s -> 3 tiles 1.349s，5.3×
Round 2: 1 column 2.495s -> 20-column tile 4.798s
Round 2 等效每列 0.240s，10.4×
Round 2 的 483 列核时间预测约 120s
最大绝对误差：Round 1 8.68e-6；Round 2 3.11e-6
```

这些数字只作为路线可行性证据。最终性能只能由真实 200-query 全链路运行确认。

---

## 2. 完成定义

只有同时满足以下条件，才允许声明 V2 batching 完成：

1. 生产 batch 路径不调用 `run_single_query_protocol()`。
2. 第二轮不再逐 candidate column 调用同态核；只允许按 tile 循环。
3. `m=200, k=50, L=483` 时第一轮输出严格为 3 个密文，第二轮输出严格为 25 个密文。
4. 每个第二轮密文同时承载最多 `200 × 20` 个 query-column 分数。
5. batch 与 serial 的 cluster 选择、catch、首命中列逐 query 一致；阈值灰区单独处理。
6. 每个 `(query, column)` 使用独立正随机 mask，禁止 batch 共享 mask。
7. A→B 只传 bytes 密文和公开 context；B 不持有 secret context、不解密、不接收明文 query 或 cluster id。
8. 全量测试通过，不删除旧单查询 API，不改变 MinHash、聚类、阈值或 CKKS 参数。
9. 真实 NCVR 10K / 200 queries / k=50 全量运行成功并生成指标、通信量和分阶段时间。
10. 同机热启动端到端硬门禁为 `<= 5 分钟`，目标为 `<= 3 分钟`。未达到 5 分钟必须停工分析，不得用“每 query 摊销”冒充达标。

pytest 不加入易抖动的绝对耗时断言；绝对性能门禁由独立 benchmark 脚本和 JSON 报告执行。

---

## 3. 固定约束与非目标

本轮固定保持：

```text
POLY_MODULUS_DEGREE = 8192
slot_capacity S     = 4096
COEFF_MOD_BIT_SIZES = [60, 40, 40, 60]
SCALE               = 2**40
EL cluster          = 200
EL match            = 50
tau                 = 0.9（CLI 传入值优先）
```

禁止通过以下方式伪造性能提升：

- 不减少数据库记录、查询数、cluster 数或 MinHash 维度。
- 不改变阈值、CKKS 参数或精度容差来隐藏错误。
- 不用线程池、多进程或并发调用 serial protocol 冒充 batching。
- 不把 selected cluster 明文发给 B。
- 不复用同一随机 mask 覆盖多个 query 或 column。
- 不只跑单 tile 后乘法外推并宣称完成；最终必须跑真实全量。
- 不先迁移 OpenFHE、HELayers 或自定义 C++。TenSEAL 0.3.16 内先完成正确闭环；只有 V2 Python/TenSEAL 仍未过 5 分钟门禁时才评估迁移。

旧单查询路径暂时保留作为 correctness oracle。旧单轴 batch 路径只可作为迁移期间对照，不能继续作为生产 `--batch-size` 路径。

---

## 4. 二维 slot 布局

设：

```text
S = POLY_MODULUS_DEGREE // 2 = 4096
m = 当前 query chunk 大小
T = floor(S / m)
A = m × T              # active slots
```

生产布局固定为 query-major：

```text
slot(q, t) = q × T + t

[q0-item0, q0-item1, ..., q0-item(T-1),
 q1-item0, q1-item1, ..., q1-item(T-1),
 ...]
```

`m` 必须位于 `[1, 4096]`，`T >= 1`，`A <= 4096`。最后一个 tile 不重复真实数据，使用零填充；解密端必须按 `valid_width` 忽略 padding。

对于任意逻辑宽度 `W`：

```text
tile_count(W) = ceil(W / T)
tile_start(t) = t × T
valid_width(t, W) = min(T, W - tile_start(t))
```

第一轮信息论输出下界为 `ceil(m*k/S)`；第二轮为 `ceil(m*L/S)`。当前矩形布局的实际输出分别为 `ceil(k/T)` 和 `ceil(L/T)`：

```text
R1 lower bound = ceil(200×50/4096)  = 3，实际 3
R2 lower bound = ceil(200×483/4096) = 24，实际 25
```

第二轮 25 个输出已接近 slot 容量下界，禁止退回 483 个输出。

---

## 5. 协议代数

### 5.1 A 侧 query 加密

原始查询矩阵：

```text
Q200.shape = (m, 200)
Q50.shape  = (m, 50)
```

对每个 query 重复 T 次：

```text
Q200_tile = repeat_rows(Q200, repeats=T) -> (m*T, 200)
Q50_tile  = repeat_rows(Q50,  repeats=T) -> (m*T, 50)
```

再按 feature 纵向加密：

```text
EncQ200 数量 = 200，每个 size() == m*T
EncQ50  数量 = 50，每个 size() == m*T
```

不允许为每个 tile 重复加密 query。相同 EncQ200/EncQ50 在所有 centroid/column tiles 上复用。

### 5.2 第一轮：同时计算多个 centroids

对 centroid tile `C_t.shape == (valid_width, 200)` 零填充到 `(T, 200)`。对每个 feature `f` 构造明文 slots：

```text
P_f = tile(C_t[:, f], reps=m) -> (m*T,)
```

计算：

```text
score_tile = Σ_f EncQ200[f] ⊙ P_f
```

输出一个 `m*T` slots 密文。A 解密并 reshape 为 `(m, T)`，拼接有效列得到 `(m, k)`，再执行：

```python
selected_clusters = np.argmax(scores, axis=1)
```

禁止每个 centroid 单独生成一个输出密文。

### 5.3 第二轮 selector

A 构造：

```text
selector.shape = (m, k)
selector[q, selected_clusters[q]] = 1
selector_tile = repeat_rows(selector, repeats=T) -> (m*T, k)
```

按 cluster 特征加密为 k 个密文：

```text
EncSelector 数量 = k
EncSelector[i].size() == m*T
```

B 只能看到加密 selector，不能看到 `selected_clusters`。

### 5.4 第二轮：同时计算多个 candidate columns

对 column tile：

```text
B_tile.shape = (k, T, 50)       # 尾块零填充
R.shape      = (m, T)           # 每个 q,j 独立采样，严格为正
```

对 feature `f`：

```text
selected_f = Σ_i EncSelector[i] ⊙ flatten_qmajor(
                 R[q,t] × B_tile[i,t,f]
             )
```

再计算：

```text
score_tile = Σ_f EncQ50[f] ⊙ selected_f - tau × flatten_qmajor(R)
```

解密后：

```text
score[q,t] = R[q,t] × (
    dot(Q50[q], B[selected_cluster[q], tile_start+t]) - tau
)
```

每个 tile 只产生一个密文。允许 Python 循环遍历 tile、feature 和 cluster；禁止遍历 query 或单个 column 去调用完整同态核。

乘法深度与当前 batch 相同：selector 只做 ct-pt，最终只有一层 query/selected-feature 的 ct-ct 乘法。不得增加 bootstrapping。

### 5.5 A 侧 tile 解密与命中

A 每次解密一个 tile，reshape 为 `(m,T)`，裁掉尾部 padding。命中判断必须使用 NumPy 行向量化：

```python
positive = scores[:, :valid_width] > eps
has_hit = positive.any(axis=1)
local_first = positive.argmax(axis=1)
new_hit = (~catches) & has_hit
first_positive_columns[new_hit] = tile_start + local_first[new_hit]
catches |= has_hit
```

`early_stop=True` 只有在全部 query 命中后才能停止。存在负查询时将处理全部 25 tiles，但不再处理 483 个单列密文。

---

## 6. 最终接口和数据类型

采用“旁路构建、验证后切换”的策略，避免低成本模型在中间阶段破坏当前可运行路径。

在 `protocol/types.py` 临时新增以下 V2 类型：

```python
@dataclass(frozen=True)
class SlotTileLayout:
    batch_size: int
    tile_width: int
    active_slots: int
    slot_capacity: int

@dataclass
class TiledFirstRoundRequest:
    public_context_bytes: bytes
    encrypted_query_200: list[CipherLike]   # 200 cts, size active_slots
    layout: SlotTileLayout

@dataclass
class TiledPartyALocalState:
    secret_context: object
    encrypted_query_50: list[CipherLike]    # 50 cts, size active_slots
    layout: SlotTileLayout

@dataclass
class TiledSecondRoundRequest:
    encrypted_query_50: list[CipherLike]    # 50 cts
    encrypted_selectors: list[CipherLike]   # k cts
    layout: SlotTileLayout

@dataclass
class TiledMatchDebug:
    checked_tiles: int
    logical_columns_checked: int
    first_positive_columns: np.ndarray
```

V2 新接口固定为：

```python
# ckks/tiling.py
def make_slot_tile_layout(batch_size: int) -> SlotTileLayout: ...
def iter_tile_slices(logical_width: int, layout: SlotTileLayout): ...
def repeat_query_rows(matrix: np.ndarray, layout: SlotTileLayout) -> np.ndarray: ...
def expand_plain_tile(values: np.ndarray, layout: SlotTileLayout) -> np.ndarray: ...
def unpack_score_tile(values, layout, valid_width) -> np.ndarray: ...

# party_a/local_prep.py
def prepare_tiled_query_batch(...) -> tuple[
    TiledFirstRoundRequest, TiledPartyALocalState
]: ...

# party_b/online_responder.py
def compare_tiled_batch_to_centroids(
    request, centroids, *, serialize_output=False
) -> list[CipherLike]: ...

def tiled_batch_matching(
    cluster_matrix, request, public_context, tau,
    *, serialize_output=False
) -> Iterator[CipherLike]: ...

# party_a/online_querier.py
def choose_clusters_and_build_tiled_request(
    encrypted_score_tiles, party_a_state, k
) -> tuple[TiledSecondRoundRequest, BatchClusterSelectionDebug]: ...

def check_tiled_score_batch_debug(
    encrypted_score_tiles, secret_context, *,
    layout, logical_width, early_stop=True, eps=BATCH_DECRYPT_EPS
) -> tuple[BatchMatchResult, TiledMatchDebug]: ...

# protocol/orchestrator.py
def run_tiled_batch_query_protocol(...) -> BatchProtocolRun: ...
```

集成测试和全量性能通过后执行切换：

- `run_batch_query_protocol()` 改为调用 V2 tiled 链路。
- `--batch-size > 0` 只走 V2。
- README 中 batching 默认指 V2。
- 旧单轴函数标记 deprecated；确认没有生产调用后删除，或保留一个版本周期但不得从 evaluator/orchestrator 导入。
- 不删除 `run_single_query_protocol()` 和单查询 Party A/B API。

---

## 7. 分阶段施工

### 阶段 0：冻结现场和基线证据

不改源码。执行：

```powershell
git status --short --branch
git diff --stat
C:\Users\Dinking\miniconda3\python.exe -m pytest tests -q
C:\Users\Dinking\miniconda3\python.exe scripts\benchmark_he_batching.py --help
```

记录：

```text
当前 commit
修改/未跟踪文件列表
pytest 通过数
Python 和 TenSEAL 版本
NCVR 查询数、正负标签数、k、max_size
旧单轴 R1、R2 单列和全量预测时间
```

门禁：基线测试失败时先报告，不得修改测试掩盖。不得触碰仓库上级目录未跟踪文件。

### 阶段 1：纯布局与 NumPy packing 原语

允许修改：

```text
ckks/tiling.py                 # 新增
protocol/types.py
tests/test_tiled_layout.py     # 新增
```

实现第 6 节的 `SlotTileLayout` 和纯 NumPy helper。此阶段不创建 CKKS context、不修改 Party A/B。

测试必须覆盖：

```text
m=1    -> T=4096, active=4096
m=200  -> T=20,   active=4000
m=4096 -> T=1,    active=4096
m=0、4097、bool -> ValueError
repeat_query_rows 保持 q-major 顺序
expand_plain_tile 对每个 query 重复同一 logical tile
尾块 zero padding 和 valid_width 正确
unpack 后恢复 (m, valid_width)
R1 当前参数 tile_count=3
R2 当前参数 tile_count=25
```

命令：

```powershell
C:\Users\Dinking\miniconda3\python.exe -m pytest tests\test_tiled_layout.py -q
C:\Users\Dinking\miniconda3\python.exe -m pytest tests -q
```

### 阶段 2：A 侧二维 query 加密和 bytes 边界

允许修改：

```text
party_a/local_prep.py
protocol/transport.py
protocol/types.py
tests/test_tiled_party_a_prep.py     # 新增
tests/test_security_boundary.py
```

实现 `prepare_tiled_query_batch()`：

1. `encode_query_batch()` 仍只执行一次 MinHash batch encode。
2. 计算 layout。
3. Q200/Q50 按 q-major 重复 T 次。
4. 使用同一个 secret context 和 relin keys 加密。
5. 第一轮请求只含 Q200；Q50 留在 A 本地状态。
6. transport 序列化后每个密文均为 bytes，并保留 layout 元数据。

硬断言：

```text
m=200 时 Q200 密文数=200，Q50 密文数=50
每个 ciphertext.size()=4000，而不是 200
public context 无法解密任何 query ciphertext
B 侧类型不包含 secret context
```

旧 `prepare_encrypted_query_batch()` 此阶段不改，确保全量测试仍绿。

### 阶段 3：第一轮 centroid tiling

允许修改：

```text
ckks/tiling.py
party_b/online_responder.py
party_a/online_querier.py
tests/test_tiled_round1.py       # 新增
tests/test_tiled_party_a_prep.py
```

实现：

```text
compare_tiled_batch_to_centroids()
decrypt/unpack centroid score tiles
choose_clusters_and_build_tiled_request()
```

要求：

- B 加载 200 个 `active_slots` 长度的 feature ciphertext。
- 每个 centroid tile 只返回一个密文。
- A 拼接成 `(m,k)` 后 `argmax(axis=1)`。
- selector `(m,k)` 重复 T 次后加密为 k 个 `active_slots` 密文。
- 不按 centroid 输出 k 个密文。

测试：

```text
小矩阵明文参考 Q @ C.T，max_abs_error < 1e-4
k<T、k=T、k=T+1、k 非 T 整数倍
m=200,k=50 的结构测试严格输出 3 cts
bytes 和 object 路径结果一致
不同 query 可选择不同 cluster
尾 tile padding 不参与 argmax
```

性能 smoke 只记录，不作 pytest 门禁：当前参数 R1 tile 核应明显低于旧 50-output 核。

### 阶段 4：第二轮 query-column tiling

允许修改：

```text
ckks/tiling.py
party_b/online_responder.py
tests/test_tiled_round2.py       # 新增
tests/test_batch_numerics.py
```

实现 `tiled_batch_matching()`。必须：

- 外层只遍历 `ceil(L/T)` tiles。
- 每 tile 采样 `R.shape=(m,T)`，每个有效 `(q,t)` 独立。
- selector 阶段只循环 cluster 和 feature，不循环 query。
- query-selected-feature 点积产生一个 `m*T` 分数密文。
- 尾块 padding 得分保持负值并在 A 侧被忽略。
- `serialize_output=True` 每 tile yield 一份 bytes。

测试使用可控 mask helper 或 monkeypatch 随机采样器，不得在生产接口暴露固定 mask 参数。

必须覆盖：

```text
m=3,T=4,k=2,L=7,d=5 的明文参考
不同 query 选择不同 cluster
每个 q,j 的 mask 不同且位于 [RANDOM_MASK_MIN,RANDOM_MASK_MAX]
正、负、阈值灰区分数
max_abs_error < 1e-4
|plain_margin| >= 5e-4 时符号 100% 一致
L=1、T、T+1、483
m=200,L=483 时核调用/输出严格为 25，不是 483
```

结构门禁通过 spy/call-count 验证 tile kernel 调用数；不要只靠搜索变量名。

### 阶段 5：A 侧 tile 解密、端到端 V2 和安全边界

允许修改：

```text
party_a/online_querier.py
protocol/orchestrator.py
protocol/transport.py
tests/test_tiled_party_a_result.py    # 新增
tests/test_tiled_end_to_end.py        # 新增
tests/test_security_boundary.py
tests/test_interface_shapes.py
```

实现：

```text
check_tiled_score_batch_debug()
run_tiled_batch_query_protocol()
完整两轮 bytes 往返
```

测试矩阵：

```text
m=1：V2 与 single query cluster/catch/first column 一致
m=2/5：exact、fuzzy、non-match 混合
全部正样本：early stop 在命中 tile 后停止
包含负样本：处理所有 tiles，不漏掉后续正样本
object 与 bytes 路径一致
重复运行 cluster 和 catch 稳定
生产结果只暴露 catches
```

源码安全门禁：

```text
B 侧 V2 函数不得出现 .decrypt()
B 侧签名不得接受 secret context、query name、selected cluster
V2 orchestrator 不得调用 run_single_query_protocol
A→B 请求密文必须全部是 bytes
```

### 阶段 6：生产 evaluator 切换与分块

允许修改：

```text
evaluation/benchmark.py
scripts/evaluate_ncvr_10k.py
protocol/orchestrator.py
README.md
tests/test_batch_chunking.py
tests/test_benchmark_real.py
```

切换规则：

```text
batch_size=0/None -> 保持旧 serial oracle
1..4096          -> V2 tiled batch
>4096            -> ValueError
```

查询超过 batch size 时按输入顺序分块。每块按实际 `m` 重新计算 T；不填充重复 query。整个评估会话只创建一次 secret context、relin keys 和 public context。

通信统计按真实 bytes：

```text
public context 整个会话只计一次
A→B R1：200 cts / chunk
B→A R1：ceil(k/T) cts / chunk
A→B R2：50+k cts / chunk
B→A R2：ceil(L/T) cts / chunk
```

结果 JSON 新增：

```json
{
  "batching": {
    "mode": "2d_query_item_tiling",
    "slot_capacity": 4096,
    "batch_size": 200,
    "tile_width": 20,
    "active_slots": 4000,
    "slot_utilization": 0.9765625,
    "round1_tiles": 3,
    "round2_tiles": 25
  }
}
```

预测顺序必须与输入 query 顺序一致。README 只写已经实际跑出的结果。

### 阶段 7：性能、数值与通信验收

允许修改：

```text
scripts/benchmark_he_2d_batching.py   # 新增
tests/test_tiled_performance_contract.py
README.md
```

benchmark 固定输出原始 JSON，记录：

```text
Python/TenSEAL/CPU 环境
m,k,L,T,active_slots,slot utilization
context/key 时间（单列）
Q200/Q50/selector 加密时间
R1 全部 tiles 核时间和输出数
R2 单 tile 中位数、全部 tiles 核时间和输出数
序列化/反序列化时间
A→B/B→A 总 bytes
最大、p99 误差和符号一致率
端到端 online wall time
serial 或旧 45min 基准来源
```

微基准预热一次，正式至少三次取中位数。全量 NCVR 可以只跑一次，但必须是完整真实运行，不得外推。

执行：

```powershell
C:\Users\Dinking\miniconda3\python.exe scripts\benchmark_he_2d_batching.py `
  --queries 200 --clusters 50 --columns 483 `
  --output artifacts\benchmarks\he_2d_batch200.json

C:\Users\Dinking\miniconda3\python.exe scripts\evaluate_ncvr_10k.py `
  --k 50 --query-limit 200 --db-limit 10000 --batch-size 200 `
  --output-dir artifacts\evaluation\ncvr_10k_batch200_2d
```

性能门禁：

```text
结构硬门禁：R1=3 cts，R2=25 cts
slot 利用率：4000/4096 >= 97%
数值硬门禁：max_abs_error < 1e-4
符号硬门禁：灰区外 100%
性能目标：online <= 180s
性能硬门禁：online <= 300s
相对硬门禁：相对 45min 至少 9×
```

若 R2 超过 180s，先按 primitive profile 分解：

```text
ct-pt selector 时间
ct-ct feature dot 时间
plaintext vector 构造时间
serialize/deserialize 时间
decrypt/unpack 时间
```

只有确认 Python/TenSEAL binding 成为主瓶颈后，才进入 C++/SEAL、OpenFHE 或 HElayers 迁移评估。

### 阶段 8：切换、清理和最终回归

允许修改所有本计划涉及的 batch 文件，但禁止无关重构。

完成：

1. `run_batch_query_protocol()` 和 `--batch-size` 指向 V2。
2. 移除生产代码对旧单轴 batch responder 的导入。
3. 旧单轴函数若保留，显式加 deprecated 注释并仅用于对照测试；最好在确认无调用后删除。
4. 删除错误性能描述，README 写入真实全量 JSON 数据。
5. 跑全量测试和静态搜索。

命令：

```powershell
C:\Users\Dinking\miniconda3\python.exe -m pytest tests -q
rg -n "run_single_query_protocol" evaluation protocol scripts
rg -n "column_wise_batch_matching|tiled_batch_matching" party_b protocol evaluation
git diff --check
git status --short
```

最终人工检查：

```text
[x] 没有每 query 调用完整 HE protocol
[x] V2 第二轮只有 tile loop，没有 column kernel loop
[x] m=200 时 active_slots=4000
[x] R1 输出=3，R2 输出=25
[x] bytes 边界和公开 context 安全测试通过
[x] serial parity 通过
[x] 真实 200-query 全量 <=5min
[x] 结果指标与原输入顺序一致
[x] 未提交数据副本、密钥、pytest cache 或大 benchmark 二进制
```

---

## 8. 测试分层

### 8.1 纯布局测试

不启动 HE，毫秒级完成。验证索引、padding、tile 数和 shape，是发现轴错误的第一道门禁。

### 8.2 小矩阵 HE 代数测试

使用小 `m,T,k,d,L`，直接与 NumPy reference 比较。测试用 layout 可显式设置较小 T；生产 layout 始终由 4096 自动计算。

### 8.3 固定协议维度测试

使用真实 `d=200/50`，验证密文数量、slot size、bytes roundtrip 和阈值符号。

### 8.4 端到端小数据测试

用 exact/fuzzy/non-match 验证 serial parity、cluster 选择和首命中列。

### 8.5 NCVR 全量验收

使用 10K B 数据、200 queries、k=50、真实 bytes 通信链路。只有这一层可以确认 45 分钟问题是否解决。

---

## 9. 数值与随机 mask 收紧

V2 暂时沿用：

```text
BATCH_DECRYPT_EPS = 1e-4
灰区边界          = 5e-4
```

不得因为某个阈值样本翻转而直接放大 eps。先输出：

```text
plain margin
decrypted margin
absolute error
mask value
tile/query/column index
```

随机 mask 必须满足：

```text
shape == (m,T)
finite
RANDOM_MASK_MIN <= value <= RANDOM_MASK_MAX
value > 0
每 tile 重新采样
同 tile 不共享单一标量
```

测试中可固定随机源；生产代码必须使用现有 `SystemRandom` 或等价密码学安全来源。

---

## 10. 安全边界与威胁模型

本项目保持当前 honest-but-curious B 模型：B 按协议计算但尝试从请求推断 query。V2 不新增 malicious-server 证明、ZK 或结果完整性验证。

必须保持：

- query 和 selector 仅以 CKKS ciphertext 发送。
- public context 不含 secret key。
- B 不解密。
- selected cluster 只存在 A 本地 debug。
- 每个 query-column 分数独立正 mask，B 不向 A 发送 mask。
- query 在 slots 中重复不改变协议明文语义，不能因此改成明文缓存。
- 通信测试必须经过 object → bytes → public-context-bound object，而不是同进程直接传私钥绑定对象。

如果未来要求防恶意 B，需另立协议设计；不得在本次性能施工中顺手宣称已支持。

---

## 11. 低成本模型执行契约

每次只发送一个阶段。提示词首部固定为：

```text
你只执行 docs/he_batching_execution_plan.md 的“阶段 N”。
这是 V2 二维 query×logical-item slot tiling，旧单轴 batching 不是目标。
先读取本阶段授权文件、依赖接口和当前 git diff，再修改。
不得执行 git reset/clean，不得触碰上级目录未跟踪文件。
不得修改阶段范围外文件，不得改变 CKKS/MinHash/聚类/阈值参数。
严格遵守文档函数名、shape、slot index、密文数量和禁止项。
先跑本阶段定向测试，再跑全量测试；禁止删除测试、放宽误差或用 mock 代替真实 HE。
完成后只报告：修改文件、关键 shape、密文数量、测试命令和结果、剩余风险。
若文档与源码冲突，停止并报告文件、行号和冲突，不自行改协议。
```

每阶段交付检查：

```text
[ ] git diff 只包含授权文件
[ ] 当前阶段目标接口存在且 shape 正确
[ ] 没有 per-query 完整协议循环
[ ] V2 没有 per-column 同态核循环
[ ] bytes 安全边界未退化
[ ] 定向测试通过
[ ] 全量测试通过，或明确说明尚未切换的临时原因
[ ] 未写入密钥、数据副本或大二进制
```

低成本模型不得根据测试名猜实现；必须读取 `ckks/batching.py`、相关 Party A/B 源码和本阶段 reference algebra。

---

## 12. 停工与回滚条件

出现以下任一情况立即停止当前阶段：

- `m=200` 时 V2 第二轮输出超过 25 个密文。
- V2 第二轮仍按 483 个 column 调用同态核。
- 为获得性能而共享 mask、发送明文 cluster 或让 B 解密。
- 需要改变 CKKS 参数、MinHash 维度、k、tau 或数据库规模才能通过。
- 灰区外符号一致率不是 100%。
- 完整 bytes 路径与 object 路径不一致。
- batch 与 serial 的逐 query 结果不一致且无法由灰区解释。
- 需要修改本阶段未授权文件。
- 全量端到端超过 5 分钟且没有 primitive profile 证据。

回滚只回滚当前阶段产生的文件改动。禁止 `git clean`，禁止删除未知未跟踪文件，禁止回滚用户或其他阶段的已验证改动。

---

## 13. 预期最终复杂度

令 `d1=200`、`d2=50`：

旧单轴：

```text
R1 HE kernel count ~ k
R2 HE kernel count ~ L
R2 output cts       = L
```

V2：

```text
T = floor(S/m)
R1 tile count = ceil(k/T)
R2 tile count = ceil(L/T)
R1 output cts = ceil(k/T)
R2 output cts = ceil(L/T)
```

当前数据：

```text
R1: 50  -> 3
R2: 483 -> 25
```

V2 不减少每个 tile 内的 feature/cluster 代数，但把原先闲置的 CKKS slots 用于同时计算多个 candidate/centroid。它解决的是错误 SIMD 轴导致的 95% slot 浪费和 483 次密文往返，而不是用并发掩盖串行协议。
