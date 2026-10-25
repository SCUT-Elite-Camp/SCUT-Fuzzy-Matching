# 历史选簇与过滤实验增量

四个 ZIP 基于同一套旧代码分别修改，并不是可叠加的升级版本。这里保存**每个变体自己的补丁**，而不是四份重复项目、重复数据，或覆盖当前主线的代码。

`manifest.json` 记录原始 ZIP 指纹、基准 commit、补丁指纹和涉及的文件。`patches/` 内的补丁只包含实验相关 Python/说明文件，不包含旧 GUI、整套数据和损坏文件名的历史文档。原始 ZIP 仍在本机 `deliveries/local/2026-10-25/`，未上传。

| variant | 内容 | 新增测试 |
|---|---|---|
| `topk` | 各排名簇独立 one-hot 验证，查询级 OR | `tests/test_topk_selection.py` |
| `fixed_threshold` | 绝对中心分数阈值、空集 Top-1 回退；含串行 HE 对照 | `tests/test_fixed_threshold.py` |
| `bad_event` | 校准分差阈值的逐查询动态 K；仅明文候选研究 | `tests/test_bad_event_topk.py` |
| `hyperplane` | Top-1 簇内 50 维向量的多轮随机超平面过滤；仅明文 | `tests/test_random_hyperplane_projection.py` |

从仓库根目录重建任一变体到**新的目录**，不修改当前主线：

```powershell
python experiments/single_attribute/cluster_selection/rebuild.py --variant bad_event --output "$env:TEMP/scut-bad-event"
cd "$env:TEMP/scut-bad-event/baseline_Integration"
python -m pytest tests/test_bad_event_topk.py -q
python scripts/evaluate_bad_event_topk.py
```

已有输出目录会被拒绝，不会覆盖。需要完整 Git 历史中的 manifest 基准提交；浅克隆需先取回该提交。四个变体必须分别重建到不同目录，不要将补丁叠加。

随 BadEvent 包交付的原始结果保存在 `results/bad_event_received/`，它们是历史明文证据，不是当前运行生成的结果。详细尺度问题、样本分母差异和安全边界见仓库 `docs/reviews/2026-10-25/` 中的审阅报告。

当前主线已有独立 tiled probes 和另一种聚类特征随机投影；此次归档没有接入动态 K，也没有改变生产阈值、默认参数或隐私输出语义。
