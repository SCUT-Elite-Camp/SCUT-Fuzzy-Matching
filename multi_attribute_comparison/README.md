# 多属性对比实验（multi-attribute comparison）

本目录把「**多属性**模糊匹配（姓名 + 生日 + 地址/电话等）与**单姓名**方案对比」这一轮
实验的全部产物收拢在一起：数据集、脚本、运行日志与对比报告。实验本身跑在
`baseline_Integration/` 的 schema 驱动管线（KMeans + CKKS）上，这里只放"围绕这次实验"
的东西，不改动主工程。

## 目录结构

```
multi_attribute_comparison/
├── README.md              ← 本文件，模块入口
├── report.md              ← 对比报告：单姓名 vs 多属性两方案表 + 口径 + 已知问题
├── table1_multi_attribute.md ← 文档用 Table 1 格式对比表（FEBRL vs Multilingual 双列）
├── datasets/              ← 多属性 A/B/ground-truth 数据（1.1 MB，随仓库提交）
│   ├── README.md          ← 字段、真值格式、NCVR 排除理由
│   ├── febrl4_{A,B,ground_truth}.csv
│   ├── multiling_{A,B,ground_truth}.csv
│   └── manifest.json
├── scripts/               ← 数据准备、编码、评测、通信量实测
│   ├── prepare_multi_attribute_datasets.py
│   ├── encode_multi_attr_to_binary.py
│   ├── evaluate_multi_attribute.py
│   └── measure_multi_attribute_comm.py
├── logs/                  ← 两次正式运行的原始终端日志
│   ├── ckks_febrl.log
│   └── ckks_multiling.log
└── experiments_topk_10k/  ← 10K 数据集上的 Top-k / 投影网格实验（report.md + 逐查询结果 + 日志）
```

## 快速开始（在仓库根目录执行）

```bash
# 1) 整理数据集（FEBRL 原始 CSV 需先下：cd baseline_Integration && python3 scripts/fetch_dataset.py --dataset febrl）
python3 multi_attribute_comparison/scripts/prepare_multi_attribute_datasets.py

# 2) 跑多属性实验（在 baseline_Integration/ 下，job 文件指向 config/examples/）
cd baseline_Integration
python3 scripts/demo_multi_attribute_dataset.py \
  --config config/examples/febrl_multi_attribute.json \
  --limit 100 --db-limit 0 --no-color \
  --output-dir artifacts/demo/febrl_ma_100
cd ..

# 3) 算指标 / 量通信量
python3 multi_attribute_comparison/scripts/evaluate_multi_attribute.py \
  --run baseline_Integration/artifacts/demo/febrl_ma_100/demo_multi_attribute_dataset.json \
  --queries 100 --online-seconds 748.378
python3 multi_attribute_comparison/scripts/measure_multi_attribute_comm.py \
  --config baseline_Integration/config/examples/febrl_multi_attribute.json \
  --db-limit 0 --sample 8 --probes 3 \
  --out baseline_Integration/artifacts/demo/febrl_ma_100/comm_8q.json
```

## 结果速览

| 方案 | 数据集 | P | R | F1 | 时延 | 通信量 |
|---|---|---|---|---|---|---|
| KMeans + CKKS | FEBRL 4a/4b（5000 库 / 100 查询） | 1.0000\* | 0.9400 | 0.9691 | 748.38 s | 5.68 GB |
| KMeans + CKKS | 多语言融合（300 库 / 50 查询） | 1.0000 | 0.9000 | 0.9474 | 121.09 s | 2.05 GB |
| ApproxPSI + EMP GC | 两个多属性数据集 | N/A | N/A | N/A | N/A | N/A |

\* FEBRL 全为正例，precision 恒为 1，不具区分度。
ApproxPSI 为 N/A 的原因（无 `--input`；Hamming 编码 d≤4 仅 25.2% 可达）见 `report.md`
「关键结论 3」。

## 说明

- **数据来源**：FEBRL 4a/4b 与 `fused_multilingual` 都是仓库既有真实数据，`scripts/prepare_*`
  只做字段整理与真值导出，不重新采样、不生成合成数据。
- **指标口径**：协议只回传"是否越阈"一个比特，不返回命中记录 id，所以 P/R/F1 是
  **查询级检测口径**（`TP = 标签为真 且 CATCH`）。详见 `report.md`「实验口径」。
- **更大的产物不在这里**：逐条查询的 JSON/CSV 明细在
  `baseline_Integration/artifacts/demo/`（已被 `.gitignore` 排除），原始 FEBRL 在
  `baseline_Integration/dataset/`（同样被排除）。
