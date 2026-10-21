# Approx-PSI 安全过滤性能优化记录

日期：2026-10-18
对照论文：*Approximate PSI with Near-Linear Communication*（Chongchitmate, Lu, Ostrovsky），Table 3
对照基线：`approx_psi_merged` 整轮批处理版本（逐候选 16 位累加器）

## 1. 结论（分阶段实测，两进程 loopback）

单位 MB，单端点 sent+recv（两端对称）。

| n | 阶段 | 我们 | 论文 | 比值 |
|---|---|---|---|---|
| 256 | SS-PSI | 72.2 | 214.57 | 0.34× |
| 256 | Hamming+VM+Open | 352.5 | 251.1 | 1.40× |
| 256 | **合计** | **424.6** | **465.68** | **0.91×** |
| 1024 | SS-PSI | 80.1 | 848.65 | 0.09× |
| 1024 | Hamming+VM+Open | 1220.3 | 930.7 | 1.31× |
| 1024 | **合计** | **1300.5** | **1779.3** | **0.73×** |
| 4096 | SS-PSI | 120.0 | 3279.3 | 0.04× |
| 4096 | Hamming+VM+Open | 4801.6 | 3590.7 | 1.34× |
| 4096 | **合计** | **4921.7** | **6870** | **0.72×** |

相对优化前（总通信，含 PSI）：

| n | 优化前 | 优化后 | 降幅 |
|---|---|---|---|
| 16 | 731.20 MiB | 56.5 MB | 13.6× |
| 256 | 4,309.9 MiB | 424.6 MB | 10.6× |
| 1024 | ≈15.15 GB | 1.30 GB | 11.6× |
| 4096 | 113.55 GiB（历史） | 4.92 GB | 24.8× |
| 4096 | ≈58 GB（本机外推基线） | 4.92 GB | 11.8× |

质量不变：n=16/256/1024/4096 全部 `matched_local_elements=4`；
n=256 的 `local_id/first_match_round` 与优化前逐行一致（2:1, 0:1, 1:3, 3:4），
n=1024 与「比较全部 128 位」的 A/B 逐行一致（2:0, 1:3, 3:8, 0:33）。
轮次不变（每 bin 仍只公开一次 keep），n=4096 为 903,947。

## 2. 差距在哪（现在只剩桶填充）

| n | Hamming 字节/候选 | 论文（按每轮 ≤ n 对） | 比值 |
|---|---|---|---|
| 256 | 10.3 KB | 10.2 KB | 1.01× |
| 1024 | 10.5 KB | 10.2 KB | 1.03× |
| 4096 | 10.6 KB | 10.2 KB | 1.04× |

**每候选的 GC 成本已经和论文持平**；Hamming 段的 1.3–1.4× 差距来自 cuckoo 填充桶：

| n | bins/轮 | 元素数 | 填充比 |
|---|---|---|---|
| 256 | 356 | 256 | 1.39× |
| 1024 | 1305 | 1024 | 1.27× |
| 4096 | 5256 | 4096 | 1.28× |

这些 bin 的 `local_valid` 是私有的，直接过滤会泄漏“哪一侧有元素”，要省掉必须再叠协议。

同时 PSI 比论文便宜 3–27×（payload 只有 8 字节 id vs 论文分享 128 位向量），
所以总量反超论文 9–28%。

## 3. 三处改动

### 3.1 Hamming 电路：16 位累加器 → Wallace 截断树

`include/secure/hamming_threshold.h`：

- 原：`UInt_T<Ctx,16>`，每个 diff 位一次 16 位 mux + 16 位 ripple add，**3,987 AND/候选**（≈128 KB）。
- 中间版：4 位叶子 + 3 位饱和合并树，**605 AND/128 位**。
- 现版：**carry-save（Wallace）压缩到权重 1/2/4，权重 8 的进位直接 OR 成 sticky reject**，
  **357 AND/128 位（2.79 AND/bit）**，比 16 位累加器少 11.2×，比树版少 1.69×。
  d ≤ 6 用 Wallace；d > 6 回退到通用饱和计数器。

### 3.2 跳过公开投影坐标

`secureFilterBatch(..., compare_mask)`：每轮投影下标 `I` 公开，命中的一对元素在 `I` 上必然相等，
贡献恒为 0，因此只比较其余坐标（n=1024: 128→103 位，n=4096: 128→101 位）。
非近似、不新增泄漏；`APPROX_PSI_COMPARE_ALL=1` 可关闭该优化做 A/B。

### 3.3 分阶段字节计量

- `psi_sock.bytesSent()/bytesReceived()` 逐轮累加 → `psi_bytes_*`
- `NetIO::send_counter/recv_counter` 逐轮累加 → `emp_bytes_*`
- 结束打印 `stage_bytes: psi_sent=... psi_recv=... emp_sent=... emp_recv=...`

## 4. 质量验证

| 项目 | 方法 | 结果 |
|---|---|---|
| 电路逻辑 | `bench_circuit ... check`，d=0/2/4/6，距离 0..d+4 | 0 mismatches |
| 端到端命中 | n=16/256/1024/4096 | 全部 4/4 |
| 掩码正确性 | n=1024 掩码 vs 全 128 位 A/B | id/轮次逐行一致 |
| 优化不改变结果 | n=256 优化前/后 | id/轮次逐行一致 |
| 明文路径 | `approx_psi_tests` | All tests passed |

## 5. 实验规模

- 规模：n = 16 / 256 / 1024 / 4096，ell = 128，d = 4，t = 4 / 8 / 10 / 12，λ = 10 / 40。
- k = 96 / 89 / 86，与论文一致；候选数 2,585 / 34,176 / 116,137 / 451,886。
- 双进程端到端（loopback）：4096 约 2.5 min，1024 约 1 min，256 约 20 s。
- 变体：原版基线、tree、wallace、全量比较 A/B；电路微基准 u16/sat/tree/wallace 各 d。
- 未做：重复运行的方差统计、广域网/带宽受限测试、安全路径的 FNR 扫描。

## 6. 复现

```bash
cmake -S . -B build_local \
  -DAPPROX_PSI_BUILD_SECURE=ON -DAPPROX_PSI_HAS_EMP=ON \
  -DAPPROX_PSI_HAS_VOLEPSI=ON -DAPPROX_PSI_BUILD_FULL_DEMO=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$HOME/emp-install"
cmake --build build_local --target approx_psi_full -j8

./run_batch_benchmark.sh build_local 256  128 4 8  40
./run_batch_benchmark.sh build_local 1024 128 4 10 40
./run_batch_benchmark.sh build_local 4096 128 4 12 40

# 电路微基准
cmake -S . -B build_local -DAPPROX_PSI_BUILD_CIRCUIT_BENCH=ON \
  -DCMAKE_PREFIX_PATH="$HOME/emp-install"
cmake --build build_local --target bench_circuit -j8
./build_local/bench_circuit 1 13500 wallace 128 64 4 check & \
./build_local/bench_circuit 2 13500 wallace 128 64 4 check
```

## 7. 口径与注意事项

- 字节数为每轮在本端点用计数器增量累加，两端对称；论文 Table 3 未说明单向/双向，按双向理解。
- 墙钟不可直接比：论文为 8 vCPU 单线程 VM；本机 32 核 WSL loopback。
- 需保留安装版 volePSI 的 `RsCpsiSender::Sharing::mMapping` 修正，否则 0 命中。
- `secureFilterBin`（逐 bin 旧接口）仍用原电路，新电路只服务 `secureFilterBatch`。
- 未覆盖论文的 PAMC 与 edit/Euclidean/angular 距离。
