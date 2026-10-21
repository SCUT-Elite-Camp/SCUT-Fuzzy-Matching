# Approx-PSI 论文复现报告

## 1. 目标

复现论文 **Approximate PSI for Hamming Distance**（Garimella et al.）的 Algorithm 1 协议。

目标参数：n=16/256/1024/4096，ell=128，d=4，t=log n，lambda=40。

交付两个版本：
- **solo**：独立实验版（`approx_psi_paper`）
- **merge+**：融合版（`approx_psi_merged`）

---

## 2. 已实现

### 2.1 solo（`approx_psi_paper`）

| 组件 | 状态 |
|---|---|
| Algorithm 1 协议流程 | ✅ 完整 |
| 单进程双线程 + volepsi `RsCpsi` | ✅ 跑通 |
| matches 正确（n=16） | ✅ 4/4 |
| matches 正确（n=256） | ✅ 4/4 |
| 通信量（n=16） | ✅ 4.48 MB |
| 通信量（n=256） | ✅ 59.2 MB |
| 通信量趋势 | ✅ 线性（`Õ(n)`） |

### 2.2 merge
| 组件 | 状态 |
|---|---|
| Plain 层（投影、分桶、参数、FNR） | ✅ 完整 |
| 独立 EMP SS-Hamming demo | ✅ 真跑通 |
| `SecureApproxPSI::run()` | ❌ skeleton |
| 完整 Approx-PSI 协议 | ❌ 没跑通 |

### 2.3 merge+（融合版）

| 组件 | 状态 |
|---|---|
| Plain 层 | ✅ 继承 merge |
| 独立 EMP SS-Hamming（新版 EMP） | ✅ 跑通 |
| `RsCpsi` 接进 `ss_psi_volepsi.cpp` | ✅ 编译过 |
| SSHamming / SSVMult 接进库 | ✅ 编译过 |
| `secureVerify` 用 `ss_ham_and_vmult_impl` | ✅ 编译过 |
| 双进程 `approx_psi_full` | ❌ 跑不通 |
| 完整 Approx-PSI 协议 | ❌ 没跑通 |

---

## 3. 尚未实现

| # | 项 | 状态 | 难度 | 估时 | 成功率 |
|---|---|---|---|---|---|
| 1 | 完整协议跑通（merge+） | ❌ | 高 | 3~7 天 | 50% |
| 2 | GC/OT 换成 EMP（solo） | ❌ | 很高 | 1~2 周 | 40% |
| 3 | `F_ssVMult` 用 EMP-OT（solo） | ❌ | 中 | 3~5 天 | 60% |
| 4 | n=1024/4096 跑通 | ❌ | 高 | 1 周 | 30% |
| 5 | 通信量绝对值对论文 | ❌ | 高 | 1~2 周 | 40% |

**全部完成 4~6 周，成功率 20~40%。**

---

## 4. 主要瓶颈

### 瓶颈 1：EMP 和 volepsi 的 share 体系不兼容

- **EMP**：`Bit_T<Ctx>`（garbled label）
- **volepsi**：`oc::block`（VOLE share）
- **两套不能互转**

### 瓶颈 2：`SH2PCSession` 初始化竞态

- `approx_psi_full` 里 Bob 卡在 `secureVerify`，`num_and=0`；Alice 到 8224
- 同电路在 standalone 测试能到 8224
- 同代码两次运行结果不同（非确定性）

### 瓶颈 3：volepsi `fork` 上游 bug

- n=1024/4096 崩在 `RsOprf.cpp:112`
- `coproto::Socket::fork()` 通道残留数据污染主通道
- 差 48 字节（3 block）

---

## 5. 问题来源

### 来源 1：论文没写 share 兼容

论文 Section 9 原文：

> We implement our Approx-PSI in C++ using EMP-Toolkit for communications, OTs and garbled circuits. We use volepsi for the underlying OPPRF protocol in the circuit PSI from [RS21].

**论文未说明**：
- EMP 的 OT share 和 volepsi 的 OPPRF share 如何统一
- `F_ssPSI` 的输出 share 如何变成 `F_ssHamCom` 的输入
- `F_ssVMult` 的 share 如何与 `F_ssPSI` 的 share 兼容

**论文能做，是因为它把 volepsi 的 OT 后端替换成了 EMP 的 OT。这个替换步骤论文没有明说。**

### 来源 2：上游库 bug

- volepsi `fork` 通道残留（n≥1024 崩）
- `SH2PCSession` 初始化竞态（非确定性）

### 来源 3：GC/OT 框架差异

- 论文用 EMP-Toolkit
- solo 用 volepsi `Gmw` + libOTe
- 通信量差 8 倍

---

## 6. 做过的排除

### 已排除的 7 个假设（merge+ 死锁）

| # | 假设 | 排除方式 | 结论 |
|---|---|---|---|
| 1 | Transport framing | 分层排查 | 已排除 |
| 2 | Input 顺序 | 日志确认对称 | 已排除 |
| 3 | Reveal recipient | test_reveal_xor 通过 | 已排除 |
| 4 | Fragment 组合 | 融合后仍失败 | 已排除 |
| 5 | Gate 数不对称 | test_gate_count：8224 对称 | 已排除 |
| 6 | RsCpsi 干扰 | test_bisect：RsCpsi 后 EMP 正常 | 已排除 |
| 7 | Session 构造顺序 | 恢复原顺序仍失败 | 已排除 |

### 独立测试

| 测试 | 结论 |
|---|---|
| `test_reveal_xor` | `reveal(v, XOR)` 给真 share（常量 + gate 输出都验证） |
| `test_gate_count` | 8224 gate 电路 data-independent，两侧对称 |
| `test_bisect` | RsCpsi 不干扰 EMP session，4-input 模式正常 |

### 残留现象

同代码两次运行结果不同（非确定性）。定位到 `SH2PCSession` 构造里的 `IKNP` / `rcot` / `delta_` 初始化竞态。

---

## 7. 复现中发现的论文实现层空白

论文同时用 EMP 和 volepsi，但未说明两套 share 体系如何兼容。

我们证明：
- EMP 的 `Bit_T<Ctx>`（garbled label）和 volepsi 的 `oc::block`（VOLE share）不兼容
- 论文能做，是因为它把 volepsi 的 OT 后端换成了 EMP 的 OT
- 这个替换步骤论文没公开

**这是复现的副产品发现，学术上有价值。**

---

## 8. 结论

### 复现级别

**概念复现（Conceptual reproduction）**，非严格复现。

| 标准 | 状态 |
|---|---|
| 协议流程复现 | ✅ 完整 |
| matches 正确 | ✅ n=16/256 |
| 通信量趋势 `Õ(n)` | ✅ |
| 通信量绝对值 | ❌ 差 8 倍 |
| n=1024/4096 | ❌ 上游 bug |
| 论文实现层空白 | ✅ 发现并报告 |

### 交付物

| 版本 | 完整协议 | 独立模块 |
|---|---|---|
| solo | ✅（概念复现级） | ✅ RsCpsi + 明文 Hamming |
| merge | ❌ | ✅ EMP SS-Hamming |
| merge+ | ❌ | ✅ EMP SS-Hamming + RsCpsi 编译过 |

### 核心发现

1. **两套 share 体系不兼容**——论文没写怎么兼容
2. **`SH2PCSession` 初始化竞态**——非确定性死锁
3. **volepsi `fork` 上游 bug**——n≥1024 崩

### 唯一跑通完整协议

**solo（`approx_psi_paper`）**：n=16/256，matches=4，通信量 4.48 MB / 59.2 MB，按 Algorithm 1。
