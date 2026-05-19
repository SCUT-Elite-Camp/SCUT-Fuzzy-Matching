# Pipeline 规范文档
## Privacy-Preserving Fuzzy Name Matching via MinHash + CKKS FHE + Clustering

> 本文档依据论文原文逐步骤还原算法流程，不含任何失真或简化。
> 先理流程与规范，暂不涉及 Python 函数名定义。

---

## 一、系统角色与威胁模型

### 1.1 两方角色

| 角色 | 代号 | 职责描述 |
|---|---|---|
| 查询方 | **Party A (Querier)** | 持有待查询名单，发起搜索请求，持有 CKKS 密钥对，最终解密得知是否有潜在匹配 |
| 响应方 | **Party B (Responder)** | 持有私有名单数据库，执行离线预处理与聚类，接收加密查询并返回加密响应，不学习任何关于查询的信息 |

### 1.2 威胁模型

- **半诚实（Semi-honest / Honest-but-curious）**：双方严格按照协议步骤执行，不篡改输入，但会尝试从对方的通信中推断信息。
- **隐私保证**：
  - Party B 仅收到加密密文，无法区分不同查询（语义安全）。
  - Party A 仅能得知是否存在潜在匹配（catch = 0 或 1），无法得知具体的余弦相似度数值（被随机数 r 混淆）。
- **CKKS 语义安全**：基于 Ring Learning With Errors (RLWE) 难题，密文在计算上不可区分。

---

## 二、共享公开参数（协议开始前双方约定）

以下参数在协议执行前公开约定或由 Party B 单向共享给 Party A：

### 2.1 MinHash 编码参数

| 参数 | 符号 | 值 | 说明 |
|---|---|---|---|
| Shingle 大小 | `shingle_size` | **3** | 使用 trigram（3-gram），含空格 |
| 聚类用编码长度 | `num_permutations_cluster` (EL_cluster) | **200** | 用于质心匹配阶段 |
| 匹配用编码长度 | `num_permutations_match` (EL_match) | **50** | 用于列式名字匹配阶段 |
| 最大哈希值 | `max_hash` | **2^20**（20-bit hash space） | 限制哈希值范围，降低碰撞概率 |
| 哈希函数 | H | SHA-256（或约定函数） | 对每个 n-gram 计算哈希值 |
| 置换函数集 | π₁, π₂, ..., π₂₀₀ | 由约定随机种子生成 | 200 个随机置换，前 50 个同时用于 EL=50 |

> **重要**：EL=50 的签名使用与 EL=200 相同的置换函数集的前 50 个（同一套种子），以保证两者之间的一致性。

### 2.2 聚类参数

| 参数 | 符号 | 值 | 说明 |
|---|---|---|---|
| 聚类数 | `k` | ≈ √\|N_B\| | 由 Party B 根据数据集大小决定（如 10k→50，100k→100，1000k→500） |
| 迭代次数 | `iterations` | **20** | K-Means 最大迭代次数 |
| 距离度量 | — | **余弦相似度** | K-Means 使用 cosine similarity 而非欧氏距离 |

### 2.3 CKKS 同态加密参数

| 参数 | 符号 | 值 | 说明 |
|---|---|---|---|
| 多项式模数次数 | `poly_modulus_degree` | **8192** | N=8192，每个密文可 batch N/2=4096 个槽位 |
| 缩放因子 | `scale` | **2^40** | 控制明文编码精度 |
| 系数模大小列表 | `coeff_mod_bit_sizes` | **[60, 40, 40, 60]** | 决定支持的乘法层数 |
| 实现库 | — | **TenSEAL** | Python HE 张量库，支持 CKKS 加法、乘法、批处理、序列化 |

### 2.4 相似度阈值

| 参数 | 符号 | 值 | 说明 |
|---|---|---|---|
| 余弦相似度阈值 | `τ` (tau) | **0.9** | 基于 NCVR 数据集实验选定，精度与召回率在此处最优 |

### 2.5 Party B 向 Party A 共享的 Scaler 参数（单向公开）

Party B 在离线阶段对 EL=200 的归一化签名拟合 StandardScaler，将以下参数作为**公开参数**共享给 A：

- `scaler.mean_`：长度为 200 的均值向量（每维度的均值）
- `scaler.scale_`：长度为 200 的标准差向量（每维度的标准差）

> **原因**：质心匹配阶段要求 A 发送的标准化查询与 B 的质心处于同一向量空间。B 的质心是在 B 自己数据的标准化空间内计算的，因此 A 必须使用同一组 Scaler 参数对自己的查询进行标准化，才能使点积（余弦相似度）有意义。

---

## 三、PHASE 0：Party B 离线预处理（Algorithm 2 + Algorithm 3）

> 此阶段完全在 Party B 本地执行，结果缓存供在线阶段使用。时间开销不计入在线搜索时间。

### Step B.1：名字预处理

对 Party B 数据集中每个名字执行：
1. 转换为小写
2. 去除首尾空格
3. 统一化特殊字符（如多余空格、标点）

输出：清洗后的名字列表 `names_B`，大小 |N_B|。

---

### Step B.2：MinHash 签名生成（Algorithm 2，两种编码长度）

对每个名字，分别生成 EL=200 和 EL=50 的 MinHash 签名。

**详细步骤（对单个名字 name）**：

1. **生成 n-gram 集合（Shingles）**：
   - 对名字（含边界空格）按 `shingle_size=3` 滑窗切分
   - 例如："John" → {" Jo", "Joh", "ohn", "hn "}
   - 生成集合 `shingles = {g₁, g₂, ..., gₙ}`

2. **计算每个 n-gram 的哈希值**：
   - `h_i = H(g_i) mod max_hash`
   - H 为 SHA-256 或约定哈希函数，结果取模 `max_hash = 2^20`

3. **应用置换函数集，取最小值**：
   - 对每个置换函数 πⱼ（j = 1 to 200）：
     - `minhash_j = min{ πⱼ(h_i) | h_i ∈ {h(g₁),...,h(gₙ)} }`
   - 实践中，置换函数通过线性变换模拟：`πⱼ(x) = (aⱼ * x + bⱼ) mod max_hash`
     - 其中 aⱼ, bⱼ 由约定随机种子生成

4. **输出 MinHash 签名**：
   - EL=200 签名：`sig_200 = [minhash_1, ..., minhash_200]`，形状 (200,)
   - EL=50 签名：`sig_50 = [minhash_1, ..., minhash_50]`，即前 50 个，形状 (50,)

**输出矩阵**：
- `N_B_200`：形状 (|N_B|, 200)，每行为一个名字的 EL=200 MinHash 签名
- `N_B_50`：形状 (|N_B|, 50)，每行为一个名字的 EL=50 MinHash 签名

---

### Step B.3：L2 归一化（两种编码长度）

对两个签名矩阵分别进行 L2 归一化（逐行）：

```
N̂_B_200[i] = N_B_200[i] / ‖N_B_200[i]‖₂
N̂_B_50[i]  = N_B_50[i]  / ‖N_B_50[i]‖₂
```

> **原因**：归一化后，两个向量的内积等于余弦相似度，即：
> `⟨N̂_B[i], N̂_A⟩ = cos(N̂_B[i], N̂_A)`

**输出矩阵**：
- `N̂_B_200`：形状 (|N_B|, 200)，L2 归一化后的 EL=200 签名
- `N̂_B_50`：形状 (|N_B|, 50)，L2 归一化后的 EL=50 签名

---

### Step B.4：StandardScaler 标准化（仅对 EL=200）

> 此步骤仅用于聚类阶段（EL=200），EL=50 的签名不做此步骤。

```
scaler.fit(N̂_B_200)
N̂_Bs_200 = scaler.transform(N̂_B_200)
```

操作含义（对每一维度 d）：
```
N̂_Bs_200[:, d] = (N̂_B_200[:, d] - mean_d) / std_d
```

- `mean_d`：N̂_B_200 第 d 列的均值
- `std_d`：N̂_B_200 第 d 列的标准差

**输出**：
- `N̂_Bs_200`：形状 (|N_B|, 200)，标准化+归一化的 EL=200 签名
- `scaler.mean_`：长度 200，**公开共享给 Party A**
- `scaler.scale_`：长度 200，**公开共享给 Party A**

---

### Step B.5：K-Means 聚类（Algorithm 3，在 N̂_Bs_200 上执行）

使用余弦相似度距离的 K-Means 对标准化签名进行聚类：

1. **初始化** k 个质心 `C_k = {c₁, c₂, ..., cₖ}`（随机初始化）
2. **迭代**（最多 `iterations=20` 次）：
   - 将每个 `N̂_Bs_200[i]` 分配到余弦相似度最高的质心所属的 cluster
   - 更新每个 cluster 的质心（取 cluster 内所有向量的均值，再 L2 归一化）
3. **输出**：
   - `C_k`：质心矩阵，形状 (k, 200)，每行为一个质心向量（在标准化空间内）
   - `cluster_assignments`：长度 |N_B|，每个名字的 cluster 编号（0 到 k-1）

> **注意**：质心 C_k 是在 **标准化+归一化** 的 EL=200 空间（`N̂_Bs_200`）内计算得到的。
> 在第一轮通信中，A 发送的也是标准化后的 `E(N̂_As_200)`，与质心处于同一空间。

---

### Step B.6：构建列式聚类矩阵 C（使用 EL=50 签名）

> 聚类结构由 EL=200 决定，但矩阵 C 中存储的是 EL=50 的归一化签名，用于实际的名字匹配。

**矩阵 C 的结构**：

1. 按 `cluster_assignments` 将 `N̂_B_50` 中的签名分组为 k 个 cluster：
   ```
   cluster_0: [N̂_B_50[i] for i where cluster_assignments[i] == 0]
   cluster_1: [N̂_B_50[i] for i where cluster_assignments[i] == 1]
   ...
   cluster_{k-1}: [...]
   ```

2. 计算最大 cluster 大小：
   ```
   max_size = max(len(cluster_j) for j in 0..k-1)
   ```

3. 对每个 cluster，将不足 max_size 的部分用**零向量（dummy elements）**填充至 max_size。

4. **矩阵 C 的布局**（概念上）：
   - 形状：k 行 × max_size 列，每个元素是长度 50 的向量
   - 即 C 是一个三维张量，形状 **(k, max_size, 50)**
   - **每一列（column j）** 包含来自每个 cluster 的第 j 个名字签名，共 k 个

   ```
   C[row=cluster_idx, col=position_in_cluster, dim=50]
   
   Column j 的内容：
   [cluster_0 的第 j 个名字的 N̂_B_50,
    cluster_1 的第 j 个名字的 N̂_B_50,
    ...
    cluster_{k-1} 的第 j 个名字的 N̂_B_50]
   ```

   > **视觉化**：列式操作中，column j 是一个 (k × 50) 的明文子矩阵。
   > 用 E(S)（k 维 one-hot 向量）与 column j 做点积，相当于从 k 行中"选择"出与最匹配 cluster 对应的那一行（50 维签名向量），形成 E(selected_name_j)。

**总结 Party B 离线输出**：

| 产物 | 形状 | 用途 | 是否公开 |
|---|---|---|---|
| `C_k`（质心） | (k, 200) | 第一轮：A 的标准化查询与之做 CT-PT 点积 | 不公开（明文，仅 B 持有） |
| `C`（列式矩阵） | (k, max_size, 50) | 第二轮：列式 CT-PT 操作 | 不公开（明文，仅 B 持有） |
| `scaler.mean_` | (200,) | A 用于对自己的查询做标准化 | **公开共享给 A** |
| `scaler.scale_` | (200,) | 同上 | **公开共享给 A** |

---

## 四、PHASE 1：Party A 本地预处理（在线阶段开始）

### Step A.1：名字预处理

与 B 侧一致：小写、去空格、统一化，输出清洗后的查询名字列表 `names_A`（m 个查询）。

### Step A.2：MinHash 签名生成（EL=200 和 EL=50）

与 Step B.2 完全相同的 MinHash 流程（使用相同的置换函数集）：

**输出**：
- `N_A_200`：形状 (m, 200)，A 的 EL=200 MinHash 签名
- `N_A_50`：形状 (m, 50)，A 的 EL=50 MinHash 签名

### Step A.3：L2 归一化（两种编码长度）

```
N̂_A_200[i] = N_A_200[i] / ‖N_A_200[i]‖₂
N̂_A_50[i]  = N_A_50[i]  / ‖N_A_50[i]‖₂
```

### Step A.4：StandardScaler 标准化（仅 EL=200，使用 B 共享的 Scaler 参数）

```
N̂_As_200[i] = (N̂_A_200[i] - scaler.mean_) / scaler.scale_
```

> 注意：A 使用的是 B 共享的 `scaler.mean_` 和 `scaler.scale_`，而不是在 A 自己数据上 fit 的 Scaler。

### Step A.5：CKKS 密钥对生成

1. 生成密钥对：
   - **公钥 pk**（加密用，共享给 B）
   - **私钥 sk**（解密用，仅 A 持有，永不离开 A）

2. 生成 **重线性化密钥（relinearization keys）**：用于 CT-CT 乘法后的度数降低（也共享给 B，或在 B 侧不需要时由 TenSEAL 自动处理）

> 按 CKKS 参数：poly_modulus_degree=8192, scale=2^40, coeff_mod=[60,40,40,60]

### Step A.6：加密查询向量（两种）

**加密 EL=200 的标准化查询**（用于第一轮质心匹配）：
```
E(N̂_As_200) = CKKS_Encrypt(N̂_As_200, pk)
```
- 单查询：形状 (200,)，加密后密文大小 **≈89.1 MB**（EL=200 决定）
- 批处理 m 个查询时：将 m 个 200-dim 向量 pack 进若干密文（每密文最多 4096/200≈20 个查询）

**加密 EL=50 的归一化查询**（用于第二轮列式匹配）：
```
E(N̂_A_50) = CKKS_Encrypt(N̂_A_50, pk)
```
- 单查询：形状 (50,)，加密后密文大小 **≈22.3 MB**（EL=50 决定）

> **两个密文在第一轮与第二轮分开发送**，时机见下方通信流程。

---

## 五、PHASE 2：第一轮通信——质心匹配（Step 2 → Step 3 → Step 4）

### Step 2（A → B）：A 发送加密查询

**传输内容**：
- A 的公钥 pk（首次，一次性）
- `E(N̂_As_200)`：加密的 EL=200 标准化查询，**≈89.1 MB**

### Step 3（B 本地计算 → B → A）：B 计算与各质心的点积

Party B 执行 **CT-PT 内积**（Algorithm 4 的第一变体）：

对每个质心 `C_k[j]`（j = 0 to k-1，明文 200-dim 向量）：

```
E(sim_j) = DotProduct_CT_PT(E(N̂_As_200), C_k[j])
```

其中 DotProduct_CT_PT 执行：
```
E(dotp) = Σᵢ₌₀ᵢ₌₁₉₉ CKKSMultConst(E(N̂_As_200)[i], CKKSEncode(C_k[j][i]))
         = CKKS 逐元素相乘后累加
```

结果：
```
E(sim_scores) = [E(sim_0), E(sim_1), ..., E(sim_{k-1})]
```

**传输内容（B → A）**：
- `E(sim_scores)`：k 个加密相似度分数，大小 **≈15.7 MB**（k=50 时）

### Step 4（A 本地）：A 解密质心分数，构造 one-hot 向量并加密

1. 解密：
   ```
   sim_scores = CKKS_Decrypt(E(sim_scores), sk)
   ```
   得到 k 个实数值（近似余弦相似度，有 CKKS 噪声）

2. 找最匹配质心：
   ```
   j* = argmax(sim_scores)
   ```

3. 构造 one-hot 指示向量 `s`（大小 k）：
   ```
   s[j] = 1 if j == j* else 0
   ```
   即：`s = [0, 0, ..., 1, ..., 0]`（仅位置 j* 为 1）

4. 加密 one-hot 向量：
   ```
   E(S) = CKKS_Encrypt(s, pk)
   ```
   大小 **≈22.3 MB**（k=50 时，k 维向量）

---

## 六、PHASE 3：第二轮通信——A 发送列式匹配所需密文（Step 5）

### Step 5（A → B）：A 发送两个密文

**传输内容**：
- `E(N̂_A_50)`：加密的 EL=50 归一化查询，**≈22.3 MB**
- `E(S)`：加密的 one-hot 指示向量（k 维），**≈22.3 MB**

> 总计此轮 A → B 传输：**≈44.6 MB**

---

## 七、PHASE 4：列式匹配——Party B 逐列处理（Steps 6-8，重复 max_size 次）

> 这是协议的核心计算阶段。B 对矩阵 C 的每一列（j = 0 to max_size-1）独立执行以下操作。
> 各列操作相互独立，可并行化。

### Step 6（B 本地）：从最匹配 Cluster 中选出名字签名

对列 j：
- 取矩阵 C 的第 j 列：`col_j = C[:, j, :]`，形状 (k, 50)（k 个名字各 50 维）
- 执行 **CT-PT 向量-矩阵内积**（Algorithm 4 第一变体的扩展）：

```
E(selected_name_j) = Σᵢ₌₀ᵢ₌ₖ₋₁  E(S)[i] * col_j[i]
```

详细展开（对结果向量的每一维 d）：
```
E(selected_name_j[d]) = Σᵢ  CKKSMultConst(E(S)[i], CKKSEncode(col_j[i][d]))
```

> **语义**：E(S) 是 one-hot 加密的，所以此操作在密文域中"选择"出最匹配 cluster 中第 j 个位置的名字签名，而 B 无法知道哪个 cluster 被选中（因为 S 是加密的）。

**示例**（论文原图）：
```
E([0,0,1,0,0]) × [N̂_B11, N̂_B12, N̂_B13, N̂_B14, N̂_B15]ᵀ = E([N̂_B13])
```
其中 N̂_B13 是第 1 列中 cluster 3 的名字签名（50维向量）。

**输出**：
- `E(selected_name_j)`：50 维加密向量，用 A 的公钥加密

### Step 7（B 本地）：计算加密余弦相似度

执行 **CT-CT 内积**（Algorithm 4 第二变体）：

```
E(cos_score_j) = DotProduct_CT_CT(E(N̂_A_50), E(selected_name_j))
```

展开：
```
E(cos_score_j) = Σᵈ₌₀ᵈ₌₄₉  CKKSMult(E(N̂_A_50)[d], E(selected_name_j)[d])
```

> **语义**：由于两个向量均为 L2 归一化，它们的内积即为余弦相似度。
> 结果 `E(cos_score_j)` 是一个加密的标量，理论值 ∈ [0, 1]。

> **密文操作细节**（来自 Appendix A）：
> - CKKSMult：对两个密文执行分量相乘，产生 degree-3 密文，再经 relinearization 降回 degree-2
> - 每次 Mult 后执行 rescaling，消除额外的缩放因子
> - noise 上界由 Theorem 3/4 给出，保证结果仍可解密得到近似值

### Step 8（B 本地 → B → A）：阈值减法、随机混淆、发送

B 执行以下两步混淆操作：

1. **减去阈值**：
   ```
   E(temp_j) = CKKSAddConst(E(cos_score_j), -τ)
   ```
   语义：`temp_j = cos_score_j - τ`
   - 若 cos_score_j > τ，则 temp_j > 0（潜在匹配）
   - 若 cos_score_j < τ，则 temp_j < 0（非匹配）

2. **乘以随机数**：
   ```
   rⱼ ← 随机采样自 Z_p*（非零正整数群）
   E(score_j) = CKKSMultConst(E(temp_j), rⱼ)
   ```
   语义：`score_j = rⱼ × (cos_score_j - τ)`
   - 符号保持：正仍为正，负仍为负
   - 但数值被随机化：A 无法从 score_j 反推出 cos_score_j 的具体值
   - **每列使用不同的 rⱼ**，防止跨列比较

**传输内容（B → A，每列）**：
- `E(score_j)`：每列 1 个加密标量，大小 **≈175 KB**

> Step 6-8 对每列重复 max_size 次。B 可以**边计算边发送**，A 也可以**边接收边解密**。
> 一旦 A 发现某个 score_j > 0，即可提前终止等待（early stopping），无需等待所有列的结果。

---

## 八、PHASE 5：Party A 接收并做最终判断（Step 9）

### Step 9（A 本地）：解密并判断

对每一列 j 收到的 `E(score_j)`：

```
score_j = CKKS_Decrypt(E(score_j), sk)
catch = 1 if any(score_j > 0 for j in 0..max_size-1) else 0
```

> **语义**：
> - `catch = 1`：Party B 的数据库中存在至少一个名字与查询名字的余弦相似度 ≥ τ（潜在匹配）
> - `catch = 0`：未找到潜在匹配
> - A **仅知道** 是否存在匹配，**不知道** 匹配的是哪个名字，也不知道实际相似度数值

---

## 九、批处理扩展（m-to-n Batching，m 个查询同时搜索）

CKKS 支持 **batching**（SIMD packing），将多个明文值 pack 进同一密文的不同槽位（最多 N/2 = 4096 槽）。

**批处理策略**：

| 阶段 | EL | 每个密文可 pack 查询数 | 说明 |
|---|---|---|---|
| 第一轮：质心匹配 | 200 | ≤ ⌊4096/200⌋ ≈ 20 | m 个查询需 ⌈m/20⌉ 个密文 |
| 第二轮：列式匹配 | 50 | ≤ ⌊4096/50⌋ ≈ 81 | m ≤ 81 时仅需 1 个密文 |

**批处理效果**：
- 第一轮通信量随 m 线性增长（微小）；
- 列式计算和通信开销**与 m 无关**（Table 4 验证），因为 B 在同一次列式操作中同时处理所有批处理的查询；
- 实验中 m=1000 时批处理因内存限制在当前模拟环境中失败（需要更大内存）。

---

## 十、CKKS DotProduct 算法细节（Algorithm 4）

### 10.1 CT-PT 内积（密文 × 明文）

输入：`ctm1`（密文向量，长度 L），`ptm2`（明文向量，长度 L）

```
dotp = 0（零密文）
for i = 0 to L-1:
    m = CKKSEncode(ptm2[i])          # 明文编码为多项式
    ctmult = CKKSMultConst(ctm1[i], m)  # 密文 × 明文
    dotp = CKKSAdd(dotp, ctmult)        # 累加
return dotp
```

噪声上界（Theorem 3）：`B_dotp ≤ (N/2) × ‖a‖^can_∞ × B`

### 10.2 CT-CT 内积（密文 × 密文）

输入：`ctm1`（密文向量），`ctm2`（密文向量），长度相同

```
dotp = 0（零密文）
for i = 0 to L-1:
    ctmult = CKKSMult(ctm1[i], ctm2[i])  # 密文 × 密文（需 relinearization）
    dotp = CKKSAdd(dotp, ctmult)
return dotp
```

噪声上界（Theorem 4）：`B_dotp ≤ (N/2) × B_mu + (N/2) × B_mult(l)`

其中 `B_mu = ν₁B₂ + ν₂B₁ + B₁B₂`，`B_mult(l) = P⁻¹·qₗ·B_ks + B_scale`

> **关键**：CT-CT 乘法比 CT-PT 乘法噪声更大、计算更重。
> 因此协议设计中，质心匹配（Step 3）用 CT-PT，列式选择（Step 6）也用 CT-PT，
> 只有最后的余弦相似度计算（Step 7）用 CT-CT，最小化了 CT-CT 操作次数。

---

## 十一、完整通信流程总览

```
Party A                                        Party B
─────────────────────────────────────────────────────────
[OFFLINE]                      ←─ 共享 scaler.mean_, scaler.scale_ ─

[OFFLINE]                                    执行 Steps B.1-B.6
                                             生成 C_k, C, Scaler

[ONLINE - Step A.1-A.6]
MinHash → 归一化 → 标准化
CKKS 密钥对生成
生成 E(N̂_As_200), E(N̂_A_50)

─── pk, E(N̂_As_200) [89.1 MB] ──────────────→
                                    [Step 3] CT-PT DotProduct
                                    E(sim_scores) = [E(sim_0)..E(sim_{k-1})]
←── E(sim_scores) [15.7 MB] ────────────────

[Step 4] 解密 sim_scores
argmax → j*
构造 one-hot s
E(S) = Encrypt(s)

─── E(N̂_A_50) [22.3 MB] ─────────────────────→
─── E(S) [22.3 MB] ───────────────────────────→

                                    [Step 6-8，重复 max_size 次]
                              for each column j:
                                    CT-PT: E(S) × col_j → E(selected_j)
                                    CT-CT: E(N̂_A_50) · E(selected_j) → E(cos_j)
                                    E(score_j) = rⱼ × (E(cos_j) - τ)

←── E(score_j) [175 KB × max_size] ─────────

[Step 9] 逐列解密 score_j
catch = 1 if any(score_j > 0)
─────────────────────────────────────────────────────────
```

---

## 十二、项目目录结构（规范）

```
project_root/
│
├── config/
│   └── params.py                  # 所有超参数、路径常量
│
├── data/
│   ├── raw/                       # 原始数据集（NCVR, LibCat, USCensus）
│   │   ├── ncvr_2014.csv
│   │   ├── ncvr_2017.csv
│   │   ├── libgen_books.csv
│   │   ├── bookdep_books.csv
│   │   └── us_census_names.csv
│   ├── processed/                 # 清洗后的数据
│   └── ground_truth/              # 匹配真值标签
│
├── minhash/
│   └── encoder.py                 # MinHash 签名生成（Algorithm 2）
│
├── preprocessing/
│   ├── text_cleaner.py            # 名字预处理（小写、去空格）
│   └── normalizer.py              # L2 归一化 + StandardScaler 标准化
│
├── clustering/
│   └── kmeans_cosine.py           # 余弦相似度 K-Means（Algorithm 3）+ 列式矩阵构建
│
├── ckks/
│   ├── context.py                 # TenSEAL CKKS 上下文初始化（参数配置）
│   ├── keys.py                    # 密钥对生成、序列化
│   └── operations.py              # DotProduct CT-PT 和 CT-CT（Algorithm 4）
│
├── party_b/
│   ├── offline_prep.py            # B 的离线预处理主流程（Steps B.1-B.6）
│   └── online_responder.py        # B 的在线响应（Steps 3, 6-8）
│       # CompareToCentroids 函数
│       # ColumnWiseMatching 函数
│
├── party_a/
│   ├── local_prep.py              # A 的本地预处理（Steps A.1-A.6）
│   └── online_querier.py          # A 的在线逻辑（Steps 2, 4-5, 9）
│
├── protocol/
│   └── orchestrator.py            # 协议主循环：模拟网络通信，协调 A 与 B
│
├── evaluation/
│   ├── metrics.py                 # Accuracy, Precision, Recall, F1
│   ├── communication_cost.py      # 序列化密文大小统计（使用 TenSEAL 序列化）
│   └── benchmark.py               # 时间、内存消耗测量
│
├── datasets/
│   ├── ncvr_loader.py             # NCVR 数据集加载与 ground truth 构建
│   ├── libcat_loader.py           # 图书馆目录数据集加载
│   └── census_loader.py           # US Census 合成数据集生成
│
├── tests/
│   ├── test_minhash.py
│   ├── test_clustering.py
│   ├── test_ckks_ops.py
│   └── test_end_to_end.py
│
├── artifacts/                     # B 的离线产物（持久化存储）
│   ├── centroids.npy              # C_k，形状 (k, 200)
│   ├── cluster_matrix.npy         # C，形状 (k, max_size, 50)
│   ├── scaler_mean.npy            # StandardScaler mean_，长度 200
│   └── scaler_scale.npy           # StandardScaler scale_，长度 200
│
└── README.md
```

---

## 十三、模块职责划分

### 13.1 `config/params.py`（全局配置，单一真相来源）

存储所有超参数，其他模块从此处 import，禁止硬编码：

```
# MinHash 参数
SHINGLE_SIZE = 3
NUM_PERMUTATIONS_CLUSTER = 200   # EL for clustering / centroid matching
NUM_PERMUTATIONS_MATCH = 50      # EL for column-wise name matching
MAX_HASH = 2**20
HASH_SEED = <固定种子>            # 保证 A/B 使用完全一致的置换函数

# 聚类参数
K_CLUSTERS_FUNC = lambda n: int(n**0.5)  # 默认 k ≈ √n
KMEANS_ITERATIONS = 20

# CKKS 参数
POLY_MODULUS_DEGREE = 8192
COEFF_MOD_BIT_SIZES = [60, 40, 40, 60]
SCALE = 2**40

# 协议参数
SIMILARITY_THRESHOLD = 0.9

# 路径
ARTIFACTS_DIR = "./artifacts"
DATA_DIR = "./data"
```

### 13.2 `minhash/encoder.py`

职责：
- 给定名字（字符串），生成指定长度的 MinHash 签名向量
- 生成 n-gram 集合（含空格）
- 应用哈希函数（SHA-256 mod max_hash）
- 应用置换函数集（通过固定种子生成 a, b 系数）
- 返回 numpy 数组，形状 (num_permutations,)
- 批量处理整个数据集，返回 (N, num_permutations) 矩阵

### 13.3 `preprocessing/normalizer.py`

职责：
- L2 归一化：输入 (N, d) 矩阵，返回每行 L2 归一化后的 (N, d) 矩阵
- StandardScaler 拟合：在 B 侧对 N̂_B_200 拟合，保存 mean_ 和 scale_
- StandardScaler 变换：在 A 侧用 B 共享的 mean_/scale_ 变换 N̂_A_200

### 13.4 `clustering/kmeans_cosine.py`

职责：
- 实现余弦相似度 K-Means（可基于 sklearn 的 K-Means + 预归一化实现余弦距离）
- 对 N̂_Bs_200 执行聚类，返回 cluster_assignments 和 C_k（质心）
- 构建列式矩阵 C（形状 (k, max_size, 50)）：
  - 按 cluster_assignments 对 N̂_B_50 分组
  - 计算 max_size
  - 零填充所有 cluster 至 max_size
  - 组织为三维张量

### 13.5 `ckks/operations.py`

职责：
- 实现 DotProduct_CT_PT（Algorithm 4 第一变体）：
  - 输入：TenSEAL 密文向量 + numpy 明文向量
  - 输出：TenSEAL 密文标量（或向量，取决于实现方式）
- 实现 DotProduct_CT_CT（Algorithm 4 第二变体）：
  - 输入：两个 TenSEAL 密文向量
  - 输出：TenSEAL 密文标量
- 实现 CKKSAddConst（密文减常数 τ）
- 实现 CKKSMultConst（密文乘随机数 r）

### 13.6 `party_b/online_responder.py`

两个核心函数：

**CompareToCentroids(C_k, E_N̂_As_200)**：
- 输入：C_k 质心矩阵（明文），E(N̂_As_200)（密文）
- 对每个质心 c_j 执行 DotProduct_CT_PT
- 序列化加密相似度分数列表
- 返回序列化的 E(sim_scores)

**ColumnWiseMatching(C, E_N̂_A_50, E_S)**：
- 输入：C（列式矩阵，明文），E(N̂_A_50)（密文），E(S)（密文）
- 逐列循环（j = 0 to max_size-1）：
  - 提取 col_j = C[:, j, :]（k × 50 明文矩阵）
  - DotProduct_CT_PT(E(S), col_j) → E(selected_name_j)（50 维密文）
  - DotProduct_CT_CT(E(N̂_A_50), E(selected_name_j)) → E(cos_score_j)（标量密文）
  - E(score_j) = r_j × (E(cos_score_j) - τ)
  - 序列化 E(score_j) 并"发送"（yield 或回调）
- **支持流式发送（每列立即发送）**，A 可边接收边解密

### 13.7 `evaluation/communication_cost.py`

职责：
- 使用 TenSEAL 的 `.serialize()` 方法测量每个密文的字节大小
- 统计每步骤的通信量：
  - Step 2（A→B）：E(N̂_As_200) 大小
  - Step 3（B→A）：E(sim_scores) 大小
  - Step 5（A→B）：E(N̂_A_50) + E(S) 大小
  - Step 6-8（B→A）：每列 E(score_j) 大小 × max_size

---

## 十四、关键数据类型与尺寸一览

| 数据 | 类型 | 形状 / 大小 | 所在方 | 说明 |
|---|---|---|---|---|
| `N_B_200` | numpy float32 | (|N_B|, 200) | B | 原始 MinHash 签名 |
| `N̂_B_200` | numpy float32 | (|N_B|, 200) | B | L2 归一化后 |
| `N̂_Bs_200` | numpy float32 | (|N_B|, 200) | B | 标准化+归一化 |
| `N̂_B_50` | numpy float32 | (|N_B|, 50) | B | L2 归一化 EL=50 |
| `C_k` | numpy float32 | (k, 200) | B（明文缓存）| 质心矩阵 |
| `C` | numpy float32 | (k, max_size, 50) | B（明文缓存）| 列式聚类矩阵 |
| `scaler.mean_` | numpy float32 | (200,) | 共享给 A | 标准化均值 |
| `scaler.scale_` | numpy float32 | (200,) | 共享给 A | 标准化标准差 |
| `N̂_As_200` | numpy float32 | (1, 200) or (m, 200) | A | 标准化+归一化查询 |
| `N̂_A_50` | numpy float32 | (1, 50) or (m, 50) | A | L2 归一化查询 |
| `E(N̂_As_200)` | TenSEAL 密文 | ≈89.1 MB | A→B | 加密的 EL=200 查询 |
| `E(N̂_A_50)` | TenSEAL 密文 | ≈22.3 MB | A→B | 加密的 EL=50 查询 |
| `E(S)` | TenSEAL 密文 | ≈22.3 MB | A→B | 加密的 one-hot 向量（k 维）|
| `E(sim_scores)` | TenSEAL 密文列表 | ≈15.7 MB | B→A | k 个加密质心相似度 |
| `E(score_j)` | TenSEAL 密文 | ≈175 KB/列 | B→A | 每列的加密匹配分数 |

---

## 十五、约束与注意事项

### 15.1 必须严格保持一致的设计决定

1. **置换函数集必须完全一致**：A 和 B 必须使用同一套随机置换参数（同一个 HASH_SEED），否则签名不可比较。

2. **StandardScaler 必须使用 B 的参数**：A 在 Step A.4 中必须使用 B 共享的 mean_ 和 scale_，不能用自己数据 fit 的 scaler。

3. **EL=200 用于质心匹配，EL=50 用于列式匹配**：两者不可互换。C 矩阵存的是 EL=50 的签名，C_k 存的是 EL=200 的质心。

4. **CKKS 噪声控制**：CT-CT 乘法会消耗 HE 的乘法层数（乘法深度）。当前参数 coeff_mod=[60,40,40,60] 支持最多约 2 层乘法（每次乘法消耗一个模数层级）。Step 7 的 DotProduct_CT_CT 含多次乘法，必须确认不超过乘法深度上限。

5. **随机数 r_j 的选取**：每列必须使用不同的随机数 r_j，且 r_j ∈ Z_p*（非零），否则会泄露不同列之间的相似度比值。

6. **填充不影响结果**：C 矩阵中的 dummy 元素是零向量，E(S) × 零向量 = 零向量，DotProduct(E(N̂_A_50), 零向量) = 0，减去 τ 后得负数，不影响 catch 判断。即，dummy 列贡献 score < 0，等同于未匹配。

### 15.2 性能关键点

- **Step B.5（列式矩阵构建）**：max_size 越大，在线阶段需要处理的列数越多，通信量越大。因此 k 的选择至关重要（k 越大，max_size 越小，但 Step 3 计算量越大）。
- **Step 6 可并行**：各列独立，可用多线程/多进程并行执行。
- **Early stopping**：A 一旦收到正分数即可 break，无需等待全部 max_size 列。
- **TenSEAL in-place 操作**：论文指出使用了 in-place 操作节省内存（`iadd`, `imul` 等）。

### 15.3 评估实验设置（对应论文 Section 5）

| 实验维度 | 取值范围 |
|---|---|
| 数据集大小 | 10k, 100k, 1000k |
| 编码长度 EL | 50, 100, 200 |
| 聚类数 k | 0（无聚类/线性搜索）、50、100、150、200、400、500 |
| 查询数 m | 1, 10, 100, 1000 |
| 相似度阈值 τ | 0.5 到 0.95（精度实验）；固定 0.9（主实验）|
| 使用的数据集 | NCVR（2014 vs 2017 快照）、图书馆目录、US Census 合成 |

---

*文档版本：v1.0 | 基于论文完整还原 | 下一步：Python 函数接口定义*
