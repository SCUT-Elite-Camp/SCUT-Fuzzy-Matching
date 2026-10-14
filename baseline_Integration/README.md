# 主线：基于同态加密的模糊姓名匹配

> 论文启发：*Privacy-preserving Fuzzy Name Matching for Sharing Financial Intelligence*
> 作为checkpoint1的研究主题，已实现高效的端到端可用的两方隐私保护匹配协议。

## 解决什么问题

反洗钱、金融情报共享等场景中，两个机构想核对"同一个人是否同时出现在双方名单里"，但谁都不愿交出明文名单。难点在于姓名很少完全一致：拼写错误、缩写、文字系统不同，比如汉字到拉丁的转写等场景，都会让精确匹配失效，所以需要的是**模糊匹配**，而且要在**密文上**完成。

## 论文方法

论文的核心思路是：把姓名编码成数值签名向量，双方交换密文，在密文上计算相似度。为了不做全库扫描，数据方离线时对库做聚类，查询方先在密文上选出最可能的簇，再只对簇内候选做精细比对——搜索空间从全库缩小到一个簇。

## 对于论文复现的设计

**论文给的是单语言、单属性、逐条查询的方案。我们围绕真实场景，面对国际化金融企业的真实业务需求，做了四个方向的扩展：**

```mermaid
graph TD
    subgraph 论文方案
        A[姓名 MinHash 签名] --> B[密文上选簇]
        B --> C[簇内密文比对]
    end
    subgraph 我们的扩展
        D[多语言归一化<br/>汉字/阿拉伯/缅甸文等<br/>统一进入协议]
        E[多属性加权<br/>姓名+出生日期+电话+城市<br/>声明式 schema 配置]
        F[批量加密<br/>一个密文同时承载<br/>整批查询]
        G[评测与标定工具链<br/>阈值自动标定 + 可复现演示]
    end
    A --> D
    A --> E
    C --> F
    G --> E
```

- **多语言**（`data_pipeline/`）：不同文字系统的姓名先归一化，再连同拉丁转写变体一起入库，查询用任何一种形式都能命中。例如拉丁转写 `LiaoXueGuang` 能命中原生汉字 `廖學廣`。
- **多属性**（`multi_attribute/`）：不止匹配姓名，还能把出生日期、电话、城市等属性按权重加权打分。属性类型、权重、阈值全部由一份 JSON schema 声明，阈值用带标注数据自动标定，不靠手调。
- **批量加密**（`ckks/tiling.py`）：把整批查询打包进同一个密文并行计算，通信量相比逐条处理降低两个数量级。
- **工具链**（`scripts/`、`evaluation/`）：数据准备、阈值标定、演示、评测全部脚本化，固定种子可复现。

## 协议流程

```mermaid
sequenceDiagram
    participant A as 查询方
    participant B as 数据方
    Note over B: 离线：清洗归一化 → 编码 → 聚类建库
    A->>A: 查询归一化、编码、加密
    A->>B: 第一轮：加密查询
    B-->>A: 加密簇相似度
    A->>A: 解密选出目标簇，加密选择向量
    A->>B: 第二轮：加密选择向量
    B-->>A: 掩码后的加密比对结果
    A->>A: 解密判定：命中 / 未命中
    Note over A,B: 全程双方只见密文；查询方只得到二值结论
```

## 快速开始

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest tests -q
```

依赖使用正式包名 `scikit-learn`，不要安装旧的 `sklearn` 占位包。

demo演示入口：

```powershell
python scripts/demo_sage_cross_script.py        # 多语言跨文字匹配
python scripts/demo_multi_attribute_dataset.py --config config/examples/febrl_multi_attribute.json --limit 50   # 多属性匹配
python scripts/demo_multi_attribute_dataset.py --config config/examples/fused_multilingual_multi_attribute.json --limit 0   # 多语言+多属性融合（全部 50 条）
python scripts/demo_multi_attribute_dataset.py --config config/examples/fused_multilingual_demo10.json --limit 0   # 融合数据集演示子集（10 条查询约几十秒，适合现场演示）
```

`demo_multi_attribute_dataset.py` 不带 `--limit` 时默认只跑前 20 条查询（每条查询都是一次
完整的双轮 HE 运算，全量跑要耐心等）；`--limit 0` 表示跑全部查询。终端报告带 ANSI 颜色，
管道/重定向时自动降级为无色纯文本，也可以加 `--no-color` 强制关闭；逐查询表显示查询样本的
真实姓名（qxx 代号只用于内部关联与落盘产物）。

## 数据

| 数据集 | 位置 | 用途 |
| :--- | :--- | :--- |
| NCVR 10K | `data/ncvr_10k/` | 英文姓名真实子集，主评测基准 |
| SAGE | `data/sage/` | 12 万行、23 国多语言姓名库，含跨文字验证查询 |
| FEBRL | `dataset/febrl/`| 六属性记录链接公开基准 |
| fused_multilingual | `data/fused_multilingual/` | 自建库 多语言姓名+合成属性的小数据集 |

## 目录结构

```text
baseline_Integration/
├── ckks/                  # CKKS 上下文、密钥与二维批处理布局
├── clustering/            # K-Means 聚类
├── config/                # 全局参数与 job 配置示例
├── data_pipeline/         # 数据适配、多语言归一化、评测扰动
├── data/                  # 入库的小型数据集
├── dataset/               # 本地大数据集
├── docs/                  # 设计文档与阶段报告
├── evaluation/            # 指标、通信量、报告
├── minhash/               # MinHash 编码
├── multi_attribute/       # 多属性 schema、编码、协议、标定
├── party_a/               # 查询方：准备、选簇、判定
├── party_b/               # 数据方：建库、密文比对
├── preprocessing/         # 姓名清洗
├── protocol/              # 协议编排
├── scripts/               # 演示、评测、标定、数据生成脚本
├── tests/                 # 回归测试
└── gui_server.py          # demo 级本地可视化
```

## 当前阶段与验证状态

协议端到端可用，回归测试通过。关键验收结果见需求文档： `../specs/matching/matching_fuzzy_match.req.md` ，实测日志： `../specs/matching/evidence/`。

历史性能数据的口径说明：2026-07 的一组数字与后续 30% 扰动评测的运行条件不同，**不能互相比较得出加速结论**；

## 下一步

- 密文计算核下沉到 C++ 层，进一步压缩在线耗时
- 更强的模糊查询集：缩写、姓名顺序变化、昵称
- 与 `../approx-psi-reproduction/` 路径在统一数据集与评测口径下做对比
