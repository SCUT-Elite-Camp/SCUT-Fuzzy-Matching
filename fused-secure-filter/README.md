# 融合安全过滤：C++ 研究分支

本目录是远端提交引入的 Approx-PSI + EMP 按轮融合过滤实现，独立保留在 `approx-psi-reproduction/` 原始复现路径之外。**这里不是 Python CKKS 主线，也不是多属性评测目录。**

原来混放在 `10_25/` 下的单姓名 Python CKKS 快照已迁至仓库 `experiments/single_attribute/ckks_fusion/`；其收到的结果在该快照的 `results/received_2026-10-25/`，没有把旧快照覆盖到 `baseline_Integration/`。

| 路径 | 内容 |
|---|---|
| `src/`、`include/`、`secure_demo/`、`tests/` | C++ 明文与融合安全过滤源码、演示和测试 |
| `BATCH_OPTIMIZATION_README.md` | 当前按轮融合、Hamming 电路优化、构建命令与安全/计量边界 |
| `PERFORMANCE_ANALYSIS.md` | 同事原有优化测量记录，包含特定依赖和环境的前提 |
| `benchmark_logs/` | 远端随代码提供的历史运行日志 |
| `docs/legacy_overview.md` | 原骨架时期的 README，保留为历史说明，不作为当前实现状态的唯一依据 |

依赖无关的本地回归：

```bash
cmake -S . -B build_plain -DAPPROX_PSI_HAS_EMP=OFF -DAPPROX_PSI_HAS_VOLEPSI=OFF
cmake --build build_plain
ctest --test-dir build_plain --output-on-failure
```

这些检查只覆盖明文/回退分支，**不证明完整安全性，也不复现真实 EMP/volePSI 的吞吐**。真实后端构建与运行应使用 `BATCH_OPTIMIZATION_README.md` 中的环境配置；当前交付不包含其要求的 volePSI 上游补丁与服务器安装。
