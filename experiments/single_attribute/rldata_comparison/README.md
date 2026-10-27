# RLdata10000 单姓名对比实验

> 本目录比较两条技术路线在 RLdata10000 单姓名任务上的表现：
>
> - **方案一**：KMeans 选簇 + CKKS 同态加密（主线 `baseline_Integration/`）
> - **方案二**：随机投影候选 + 安全汉明验证（C++ `fused-secure-filter/` 的应用层适配）
>
> 原独立目录 `comparison/` 已于 2026-10-27 合并到本目录；这里现在是该对比实验的
> 唯一入口，不再保留平行的 `comparison/` 目录。

## 结论与口径

| 文件 | 内容 |
|---|---|
| `results/scheme_comparison.png` | 两方案全密文管线对比图（时延、通信量等） |
| `results/baseline_he_test.json` | 方案一（真实 TenSEAL CKKS）三次重复的测量结果 |
| `results/secure_e2e.json` | 方案二（volePSI + EMP 安全两方）端到端结果 |
| `results/reference_table1.png` | 主线 Table 1 参考图（零块跳过），不是本目录跑出来的 |

数值来自接收时的记录。**本次整理只做目录合并与路径修正，没有重新跑 CKKS 或
EMP/volePSI 安全后端**，不把原图里的数字当成新一轮独立验证。

原始交付只有两个裸姓名 CSV 和一张结果图，缺少记录 ID、身份真值、源码和逐次日志。
本目录后来补齐了可复现的数据构造与两方案 runner，但历史结果仍需按上述边界解读。

## 目录结构

```text
rldata_comparison/
├── README.md
├── requirements.txt
├── data/
│   ├── database.csv               # 10000 条 B 侧记录（含身份、生日）
│   ├── queries.csv                # 200 条评测查询（100 正 / 100 负）
│   ├── calibration_queries.csv    # 200 条不重叠的标定查询
│   └── manifest.json              # 组成与生成参数
├── scripts/
│   ├── names.py                   # 多语言姓名归一化（两条路线共用）
│   ├── encoding.py                # b-bit MinHash 二值编码与 Hamming
│   ├── prepare_rldata.py          # 从 RLdata10000.rda 生成 data/ 下的数据文件
│   ├── baseline_top1.py           # 方案一：KMeans + CKKS 的 Top-1 runner
│   ├── fused_top1.py              # 方案二：随机投影 + Hamming 的明文实现
│   ├── secure_export.py           # 导出两方输入与明文参考（安全路径）
│   ├── secure_runner.py           # 通过 WSL 跑 EMP/volePSI 端到端管线
│   └── make_figure.py             # 由两份 JSON 结果渲染对比图
└── results/
    ├── baseline_he_test.json
    ├── secure_e2e.json
    ├── secure_e2e_seed20261023.json
    ├── secure/                    # 每 seed 的两方输入、参考预测与日志
    ├── scheme_comparison.png
    └── reference_table1.png
```

## 脚本各自做什么

| 脚本 | 作用 | 依赖 |
|---|---|---|
| `names.py` | Unicode 归一化、德语变音转写、非拉丁文字转 ASCII | `anyascii` |
| `encoding.py` | 把姓名变成 b-bit MinHash 二值向量，并计算打包 Hamming 距离 | `numpy` |
| `prepare_rldata.py` | 解析 `RLdata10000.rda`，在 `data/` 下生成库、评测查询、标定查询和 manifest | `rdata` |
| `baseline_top1.py` | 方案一的 Top-1 扩展：每查询返回一条记录，而不是全候选二值结果 | 主线 CKKS 依赖（`tenseal`、`scikit-learn`） |
| `fused_top1.py` | 方案二的明文模拟：随机投影分桶 + Hamming 阈值 Top-1 | `numpy`、`pandas` |
| `secure_export.py` | 把两方输入编码成二进制，并生成同口径的明文参考 | `numpy`、`pandas` |
| `secure_runner.py` | 启动 C++ `name_secure_pipeline` 两方进程，逐查询核对结果 | WSL + EMP + volePSI |
| `make_figure.py` | 读取 `baseline_he_test.json` 与 `secure_e2e.json`，输出对比图 | `matplotlib` |

## 快速开始

以下命令都在**仓库根目录**执行，使用项目已有的 `.venv`：

```powershell
# 0) 安装本实验依赖（主线的 CKKS 依赖另见 baseline_Integration/requirements.txt）
.venv\Scripts\python.exe -m pip install -r experiments\single_attribute\rldata_comparison\requirements.txt

# 1) 生成可复现数据；需要 baseline_Integration/dataset/rldata10000/RLdata10000.rda
.venv\Scripts\python.exe experiments\single_attribute\rldata_comparison\scripts\prepare_rldata.py

# 2) 方案二：明文模拟跑一遍
.venv\Scripts\python.exe experiments\single_attribute\rldata_comparison\scripts\fused_top1.py `
    --rounds 32 --output experiments\single_attribute\rldata_comparison\results\fused_plaintext.json

# 3) 方案一：真实 CKKS Top-1
.venv\Scripts\python.exe experiments\single_attribute\rldata_comparison\scripts\baseline_top1.py `
    --output experiments\single_attribute\rldata_comparison\results\baseline_he_test.json

# 4) 画对比图（默认读取 results/baseline_he_test.json 与 results/secure_e2e.json）
.venv\Scripts\python.exe experiments\single_attribute\rldata_comparison\scripts\make_figure.py
```

安全路径（方案二端到端）需要先按 `fused-secure-filter/patches/README.md` 构建
`name_secure_pipeline`，然后：

```powershell
.venv\Scripts\python.exe experiments\single_attribute\rldata_comparison\scripts\secure_runner.py --seeds 20261023,20261040,20261057
```

它会先把两方输入写到 `results/secure/`，再通过 WSL 启动 server/client，
逐查询比对明文参考，最后写 `results/secure_e2e*.json`。

## 已知问题

1. **Top-1 helper 依赖主线版本。** `baseline_top1.py` 需要
   `party_a.online_querier.check_tiled_score_top1_debug`，并要求
   `party_b.online_responder.tiled_batch_matching` 支持
   `mask_scope="per_query"`。这两个接口已在 2026-10-27 补回主线；如果换到更早的
   分支，需要同步这两个改动，否则脚本会在导入或调用时失败。
2. **Windows 控制台编码。** 脚本把含非 ASCII 字符的 JSON 直接打印到终端时，GBK
   控制台可能报 `UnicodeEncodeError`；输出文件通常已经写好。可加
   `PYTHONIOENCODING=utf-8` 再运行。
3. **安全路径不是纯 Python。** `secure_runner.py` 依赖 WSL、EMP/volePSI 和已编译的
   C++ 二进制；没有后端时只能跑 `fused_top1.py` 的明文模拟。

## 来源与重复文件

- 原始目录 `comparison/` 的代码、数据和结果已整体移入本目录；生成数据现在平铺在
  `data/` 下，不再保留 `data/rldata10000/` 子目录。
- 原交付的两份裸姓名 CSV（`A_queries_names.csv`、`B_database_names.csv`）已删除；
  `data/database.csv` 与 `data/queries.csv` 是带记录 ID、生日和标签的可复现版本。
- 以下内容与已有文件逐字节相同，或只是原始压缩包，未重复保留，已移到被 Git 忽略的
  `deliveries/local/2026-10-27/comparison_archive/`：
  `rldata10000.rar`、中文命名交付目录及其 ZIP、重复的 `scheme_comparison.png`。
- 原 `result/scheme_comparison.png` 已并入 `results/`，统一使用复数目录名。
