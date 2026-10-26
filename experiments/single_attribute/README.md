# 单姓名实验

本目录只研究姓名这一项属性，与 `multi_attribute_comparison/` 分开。主线仍是仓库根目录的 `baseline_Integration/`。

| 目录 | 内容 | 执行边界 |
|---|---|---|
| `ckks_fusion/` | 远端融合 CKKS 快照；原 `fused-secure-filter/10_25/` | 自带同名模块，必须在本目录单独启动 Python/pytest，不混入主线进程 |
| `cluster_selection/` | Top-K、绝对阈值、BadEvent、簇内超平面过滤的历史增量 | 以独立补丁保存，使用 rebuild.py 重建后再运行，不直接覆盖当前主线 |
| `rldata_comparison/` | RLdata 两个裸姓名 CSV 与组内结果图 | 无身份真值、源码和原始跑分日志，不能作为已验证性能结论 |

`ckks_fusion/results/received_2026-10-25/` 是接收时的只读结果，带 SHA-256 清单；`artifacts/` 是后续可再生输出，继续忽略。单姓名和多属性的数据集、标签与阈值不同，报告不可横向拼成控制变量实验。

快速验证（从仓库根目录进入对应工作目录）：

```powershell
cd experiments/single_attribute/ckks_fusion
python scripts/verify_claims.py --level quick
python -m pytest fusion/tests tests/test_random_hyperplane_projection.py -q
```

以上不验证完整 EMP/volePSI 安全协议，也不重新测量原报告的完整性能。历史实验的重建方式见 `cluster_selection/README.md`。
