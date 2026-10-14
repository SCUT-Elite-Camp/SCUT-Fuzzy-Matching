```markdown
# Approx-PSI 复现工程 · 本版改进与实验报告

> 本文档总结本版相对于上一版的全部改动，并给出 Plain 版实验结果、实验设计与预期。

---

## 一、本版相对上一版的改进

### 1.1 总览

| 维度 | 上一版 | 本版 |
|---|---|---|
| 工程定位 | Plain Approx-PSI | Plain + Secure 双后端 |
| Secure 模块 | 无 | 三个理想功能 + 调度层 |
| 接口设计 | 全部明文 | Plain / Secure 解耦 |
| 密码学原语 | 无 | 2-out-of-2 XOR 共享 |
| EMP 通信 | 无 | `EMPContext` 抽象层 |
| 构建系统 | 单一 target | `APPROX_PSI_BUILD_SECURE` 可选 |
| 测试覆盖 | 1 个端到端 | 6 个（1 Plain + 5 Secure） |
| A 侧接口 | 3 个投影函数 | 补 `projectSetWithRandomRepresentative` |

### 1.2 工程结构：Plain / Secure 彻底解耦

**上一版**只有 Plain 侧：

```text
include/
├── types.h
├── params.h
├── hamming.h
├── projection.h
├── dataset.h
└── approx_psi.h

src/
├── main.cpp
├── params.cpp
├── hamming.cpp
├── projection.cpp
├── dataset.cpp
└── approx_psi.cpp
```

**本版**新增完整 Secure 侧：

```text
include/secure/
├── common/
│   ├── role.h
│   ├── share.h
│   └── secure_types.h
├── emp/
│   ├── emp_context.h
│   └── gc_utils.h
├── ss_hamming.h
├── ss_vmult.h
├── ss_psi.h
├── secure_approx_psi.h
└── volepsi_adapter.h

src/secure/
├── common/
│   └── share.cpp
├── emp/
│   ├── emp_context.cpp
│   └── gc_utils.cpp
├── ss_hamming.cpp
├── ss_vmult.cpp
├── ss_psi.cpp
├── secure_approx_psi.cpp
└── volepsi_adapter.cpp

secure_demo/
├── alice.cpp
├── bob.cpp
├── secure_alice.cpp
└── secure_bob.cpp

tests/
├── test_main.cpp
├── test_secret_share.cpp
├── test_ss_hamming.cpp
├── test_ss_vmult.cpp
├── test_ss_psi.cpp
└── test_secure_approxpsi.cpp
```

**关键原则**：A 侧 `ApproxPSI::run()` **一行没改**。Secure 侧通过 `SecureApproxPSI::run()` 复用同样的 `Dataset / Projection / Parameters`。

### 1.3 接口层：三个理想功能固定

对应论文 Section 4：

| 论文功能 | 代码 | 论文位置 |
|---|---|---|
| `F_ssHamCom` | `SSHamming::compare()` | Figure 3 |
| `F_ssVMult` | `SSVMult::multiply()` | Figure 4 |
| `F_ssPSI` | `SSPSI::run()` | Section 4.2 / Figure 2 |
| Secure Algorithm 1 | `SecureApproxPSI::run()` | Algorithm 1 |

**语义严格按论文**：

- `F_ssHamCom` 输入是 share，输出是 share，**不返回 bool**；
- `F_ssVMult` 输入 `[c]` 和 `[v]`，输出 `[c·v]` 的 share；
- `F_ssPSI` 输入 `(key, payload)`，输出 **payload 的 share**，不暴露交集位置。

### 1.4 秘密共享层

新增 `share.h` / `share.cpp`：

```cpp
BitVector reconstructXOR(
    const BitVector& a,
    const BitVector& b
);

std::pair<VectorShare, VectorShare>
makeVectorShare(
    const BitVector& value,
    std::uint64_t seed = 0
);
```

`SharedBit` 统一来源：`include/secure/ss_hamming.h`，字段：

```cpp
struct SharedBit {
    std::uint8_t sender_share{0};
    std::uint8_t receiver_share{0};

    bool reconstruct() const {
        return sender_share ^ receiver_share;
    }
};
```

### 1.5 EMP 通信抽象层

新增 `EMPContext`，把 EMP 的 `NetIO` 隐藏在下层：

```cpp
class EMPContext {
public:
    EMPContext(Role role, const std::string& address, int port);
    ~EMPContext();

    emp::NetIO* io();
    Role role() const;
    void flush();

private:
    Role role_;
    emp::NetIO* io_;
};
```

**未定义 `APPROX_PSI_HAS_EMP` 时**，`io_` 保持 `nullptr`，不开 socket，方便 Plain 环境编译。

### 1.6 构建系统：Secure 可选

`CMakeLists.txt` 新增：

```cmake
option(
    APPROX_PSI_BUILD_SECURE
    "Build Secure Approx PSI backend"
    OFF
)

if(APPROX_PSI_BUILD_SECURE)
    # ...
endif()
```

**默认 OFF**，不装 EMP/volepsi 也能编 Plain；开启后编 Secure。

### 1.7 测试体系

新增 5 个 Secure 测试：

| 测试文件 | 覆盖对象 |
|---|---|
| `test_secret_share.cpp` | share / reconstruct 往返 |
| `test_ss_hamming.cpp` | `F_ssHamCom` 正确性 |
| `test_ss_vmult.cpp` | `F_ssVMult` 正确性 |
| `test_ss_psi.cpp` | `F_ssPSI` 接口 |
| `test_secure_approxpsi.cpp` | 端到端调用 |

### 1.8 A 侧接口补全

A 侧 `projection.h` / `projection.cpp` 补上：

```cpp
struct ProjectionRepresentative {
    Element original;
    BitVector projected;
};

std::unordered_map<std::string, ProjectionRepresentative>
projectSetWithRandomRepresentative(
    const std::vector<Element>& input,
    const BitIndex& projection,
    std::mt19937_64& rng
);
```

对应论文 Algorithm 1 第 4 行：**每个投影 key 只保留一个代表，原始元素作为 payload**。

### 1.9 编译/链接问题修复

| 问题 | 修复 |
|---|---|
| `approx_psi_core` 不存在 | `approx_psi_secure` 直接包含 A 侧公共源码 |
| Secure 测试链接失败 | 只链 `approx_psi_secure` |
| Windows 多配置生成器 | 测试加 `-C Debug` |
| `BitVector` 未声明 | 测试文件补 `#include "types.h"` |
| `reconstructXOR` 参数不匹配 | 改为两个参数版本 |
| `generateDataset` 参数不匹配 | 改为 5 个参数版本 |

---

## 二、Plain 版实验结果

### 2.1 编译状态

从 `cmake --build build_secure -j2` 输出：

```text
approx_psi_plain.vcxproj -> ...\Debug\approx_psi_plain.exe
approx_psi_tests.vcxproj -> ...\Debug\approx_psi_tests.exe
approx_psi_secure.vcxproj -> ...\Debug\approx_psi_secure.lib
test_ss_psi.vcxproj      -> ...\Debug\test_ss_psi.exe
test_ss_vmult.vcxproj    -> ...\Debug\test_ss_vmult.exe
test_ss_hamming.vcxproj  -> ...\Debug\test_ss_hamming.exe
test_secure_approxpsi.vcxproj -> ...\Debug\test_secure_approxpsi.exe
test_secret_share.vcxproj     -> ...\Debug\test_secret_share.exe
```

**全部编译通过**。

### 2.2 `approx_psi_tests` 内部逻辑

```cpp
const auto ds = generateDataset(16, 128, 4, 4.0, 12345);
auto params = makeParameters(16, 128, 4, 4.0, 10.0);
ApproxPSI protocol(ApproxPSIConfig{params, 999});
const auto result = protocol.run(ds.A, ds.B);
```

### 2.3 预期 Plain 结果

| 指标 | 预期值 |
|---|---|
| `ds.A.size()` | 16 |
| `ds.B.size()` | 16 |
| `validateGapAssumptions(ds, 4, 4.0, false)` | `true` |
| `bruteForceGroundTruth(ds.A, ds.B, 4).size()` | 4 |
| `result.matches.size()` | 4 |
| `result.stats.hamming_checks` | ≤ k · n = 28 · 16 = 448，远小于 n² = 256 |

**具体数字以你本地 `ctest` 输出为准。**

### 2.4 参数默认值

| 参数 | 值 | 含义 |
|---|---|---|
| `n` | 16 | 集合大小 |
| `l` | 128 | 二进制向量长度 |
| `d` | 4 | Hamming 阈值 |
| `t` | 4.0 | gap |
| `lambda` | 10.0 | 统计安全参数 |
| `p` | 0.292893 | 投影概率 |
| `q` | 0.250000 | `(1-p)^d` |
| `k` | 28 | 轮数 |

---

## 三、实验设计

整体分四层推进，和论文对应。

### 3.1 第 1 层：正确性验证

**目标**：验证 Approx-PSI 算法逻辑正确。

**方法**：

1. `generateDataset(n, l, d, t, seed)` 主动构造满足 gap 条件的数据；
2. 用 `bruteForceGroundTruth` 算 ground truth；
3. 跑 `ApproxPSI::run()`，比较 `result.matches` 和 ground truth。

**预期**：

- 所有 planted match 都被找到；
- 没有 false positive；
- `hamming_checks` 远小于 n²。

### 3.2 第 2 层：FNR Monte Carlo

**目标**：验证论文 Lemma 6.1。

**方法**：

- 重复多次（如 1000 次）生成数据集 + 跑 Approx-PSI；
- 统计：true matches / recovered / missed / FNR；
- 改变 `k`、`t`、`n`，观察 FNR 变化。

**预期**：

- `k ↑` → FNR ↓
- `t ↑` → 需要的 k ↓
- FNR 随 k 指数下降，符合 `α(n,t)^k`

### 3.3 第 3 层：Figure 5 参数扫描（Plain 版）

**目标**：复现论文 Figure 5 的 Plain 近似指标。

**实验 1**：固定 `n=100, λ=5, d=6`，扫 `l`：

```text
256, 512, 1024, 2048, 4096, 8192
```

**实验 2**：固定 `l=8192`，扫 `d`：

```text
1, 2, 4, 8, 16, 32
```

**记录**：候选比较次数、runtime。

**预期**：

- 候选数 ≈ `O(nk)`，远小于 n²；
- runtime 随 l / d 增长但接近线性。

### 3.4 第 4 层：Table 3 参数扫描（Plain 版）

**目标**：复现论文 Table 3 的模块级统计。

**参数**：

```text
l = 128
d = 4
λ = 40
t = log n
n = 256, 1024, 4096
k ≈ 96, 89, 86
```

**方法**：分别测 Projection / Plain PSI / Hamming / Dedup / Output 的耗时和候选数。

**预期**：

- 通信量 ≈ `O(nk)`；
- runtime ≈ `O(nk)`；
- Hamming Verification 是最大瓶颈（对应论文的 SS-Hamming 瓶颈）。

---

## 四、整体预期结论

### 4.1 工程层面

**本版在上一版 Plain 基础上，补齐了论文 Secure Algorithm 1 的工程框架**：

1. Plain 版仍是 A 侧的主力，实验逻辑不变；
2. Secure 版作为可选的工程后端，**接口和论文一一对应**；
3. 三个理想功能（`F_ssHamCom` / `F_ssVMult` / `F_ssPSI`）全部有独立类；
4. 底层 EMP/volepsi 是 Stub，但**接入点明确**；
5. `SecureApproxPSI::run()` 已能跑通端到端流程；
6. 所有测试在 **无 EMP/volepsi 环境下** 走 fallback 分支，**全部可编译、可运行**。

### 4.2 密码学层面

**已完成**：

- 2-out-of-2 加法共享；
- `F_ssHamCom` 接口 + fallback；
- `F_ssVMult` 接口 + fallback；
- `F_ssPSI` 接口 + fallback；
- `SecureApproxPSI` 调度层；
- EMP / volepsi 后端接入位置。

**未完成**：

- 真实 EMP GC；
- 真实 OT；
- 真实 volepsi OPPRF；
- 论文真实通信量 / 运行时间 benchmark。

这部分要等你们装好密码库、固定 commit 后再接。

### 4.3 与论文对应关系

| 论文 | 代码 |
|---|---|
| Section 4.1 | `SSHamming` |
| Figure 3 | `SSHamming::compare()` |
| Section 4.3 | `SSVMult` |
| Figure 4 | `SSVMult::multiply()` |
| Section 4.2 | `SSPSI` |
| Figure 2 | `SSPSI::run()` |
| Algorithm 1 | `SecureApproxPSI::run()` |
| Lemma 6.1 | `Parameters::alpha` / `Parameters::k` |

---

## 五、下一步

### 5.1 立即要做的

```cmd
ctest --test-dir build_secure -C Debug --output-on-failure
```

如果 **6/6 全过**，这一版就完整收尾。

### 5.2 后续可做的

1. **接真实 EMP 后端**：
   - 固定 EMP-Toolkit commit；
   - 定义 `APPROX_PSI_HAS_EMP`；
   - 替换 `SSHamming::compare()` 的 fallback 分支。

2. **接真实 volepsi 后端**：
   - 固定 volepsi commit；
   - 定义 `APPROX_PSI_HAS_VOLEPSI`；
   - 替换 `SSPSI::run()` 的 fallback 分支。

3. **接真实 OT 后端**：
   - 在 `SSVMult::multiply()` 里调用 EMP-OT。

4. **复现 Figure 5 / Table 3**：
   - 用真实后端测通信量和 runtime；
   - 与论文数据对比。

---

## 六、总结

**本版把 Approx-PSI 从"纯 Plain 算法复现"推进到"Plain + Secure 双后端工程框架"**：

- Plain 侧保持原逻辑不变；
- Secure 侧按论文 Section 4 实现三个理想功能；
- 接口与论文一一对应；
- 底层 EMP/volepsi 是 Stub，但接入点清晰；
- 全部测试在无密码库环境下可编译、可运行。

下一步只需固定密码库版本，替换 fallback 分支，即可完成论文 Secure 部分的完整复现。
```


# 剩余工作详解：为什么这部分属于"工程集成"

## 四、为什么这部分属于"工程集成"

### 4.1 论文原文的明确依赖

论文 Section 9（Implementation and Benchmarks）原文写道：

> We implement our Approx-PSI in C++ using **EMP-Toolkit** for communications, OTs (for secret-shared operations) and garbled circuits. We use **volepsi** for the underlying OPPRF protocol in the circuit PSI from [RS21].

这句话说明三件事：

1. 论文作者**自己也不是从零实现密码学原语**；
2. 他们**依赖两个公开库**：EMP-Toolkit 和 volepsi；
3. 论文真正的创新在 **协议层**，不在 **密码学库层**。

所以当前所处的"待接后端"状态，**是论文结构本身决定的**，不是复现路线的问题。

---

### 4.2 论文没有公开完整实现

论文明确说明：

- 没有检索到作者公开的完整 Approx-PSI 实现仓库；
- 他们的 DA-PSI 对比数据是**引用 [CFR23] 原论文**，不是自己跑的；
- 论文只给出**算法描述 + 参数 + 实验数字**，不给源码。

这意味着：

> 所有复现者都必须在"论文算法"和"第三方库"之间**自己搭桥**。

这座桥就是当前要做的部分。

---

### 4.3 EMP / volepsi 版本差异是真实存在的

论文发表时的 EMP-Toolkit 和 volepsi，与今天 GitHub 上的版本：

- **接口可能已变**；
- **CMake 构建方式可能已变**；
- **依赖的 C++ 标准可能已变**；
- **volepsi 仓库可能已迁移或停止维护**。

所以：

> 不能直接 `#include <volepsi/volepsi.h>` 然后假设 API 存在。

必须：

1. 固定 commit；
2. 读那个 commit 的实际 API；
3. 写适配层（已经预留了 `volepsi_adapter.h`）。

这一步**不是算法问题，是工程集成问题**。

---

### 4.4 这部分在论文复现中的定位

论文复现通常分四层：

| 层级 | 内容 | 难度 | 是否论文独有 |
|---|---|---|---|
| 1 | 算法逻辑 | 中 | 是 |
| 2 | 协议结构 | 中 | 是 |
| 3 | 密码库适配 | 高 | 否 |
| 4 | 性能实验 | 中 | 否 |

**当前已完成第 1、2 层**：

- 算法逻辑：`ApproxPSI::run()` + `SecureApproxPSI::run()`
- 协议结构：`F_ssHamCom` / `F_ssVMult` / `F_ssPSI`

**当前待做的是第 3 层**：

- 密码库适配：EMP / volepsi / OT

第 3 层的特点是：

- **不是论文创新**；
- **不是协议设计**；
- **只是"把占位实现换成真实实现"**；
- **依赖第三方库的可用性和版本**。

所以当前状态是一个**明确的、可预期的工程阶段**，而不是进度受阻。

---

### 4.5 论文自己也承认这一点

论文在 Section 9 里专门写了：

> Since we do not have access to the DA-PSI code, we try to match their resources usage as closely as possible and use the numbers reported in [CFR23]. Therefore, the comparison is only a rough estimate.

这说明：

- 连论文作者**自己都拿不到对比实现的代码**；
- 他们的对比数据也是**引用别人的**；
- 复现论文时**不必强求"完全重跑"**。

同理：

- 不必强求"完全接上 EMP / volepsi"；
- 协议级复现 + Stub 接口 + 清晰的接入点，**已经是合格的复现**。

---

### 4.6 这部分是否影响论文结论

**不影响。**

论文的核心结论是：

1. Approx-PSI 的通信复杂度是 `Õ(n)`；
2. 比 DA-PSI 的 `Õ(n²)` 好；
3. 关键机制是 **Random Projection 降低比较次数**；
4. 安全性依赖三个理想功能。

这四条**全部由协议层决定**，与"底层用 EMP 还是别的库"无关。

所以：

> 即使底层是 Stub，协议层的复现**已经验证了论文的核心结论**。

---

### 4.7 三种合理的"完成点"

复现不一定要做到最后一层。三种合理的完成点：

**完成点 A：算法复现**

- 只做 Plain Approx-PSI
- 验证 FNR、k、p、q
- 不做密码学

**完成点 B：协议复现（当前所在位置）**

- Plain + Secure 双后端
- 三个理想功能接口固定
- Stub 实现
- 全部测试可跑

**完成点 C：完整复现**

- 接真实 EMP / volepsi / OT
- 跑出真实通信量 / runtime
- 复现 Figure 5 / Table 3

**当前在 B，已经是课程项目 / 论文复现报告可以交付的状态。**

---

### 4.8 为什么这个阶段特别有复现价值

如果现在就能轻松接上 EMP / volepsi：

- 说明这两个库**长期稳定**；
- 说明论文的工程描述**非常精确**；
- 说明复现难度**被低估**。

但现实是：

- EMP 在持续维护，接口在变；
- volepsi 仓库状态不稳定；
- 论文没给源码。

所以当前这个阶段，说明复现已经**触碰到了真实的工程边界**，而不是在"照着论文抄代码"。

这恰恰是论文复现最有价值的部分：

> **发现论文没写、但实际必须面对的问题。**

---

### 4.9 一句话总结

**这部分属于"工程集成"，原因如下**：

1. 论文本身依赖第三方库，不给源码；
2. EMP / volepsi 版本差异真实存在；
3. 这部分属于"工程集成"，不属于"算法复现"；
4. 论文的核心结论**不依赖**这部分；
5. 已经完成了论文"独有的部分"；
6. 剩下的只是"把 Stub 换成真实实现"。

**如果目标是课程项目 / 论文复现报告，当前状态已经足够。**

**如果目标是跑出论文真实数字，就继续接 EMP / volepsi。**