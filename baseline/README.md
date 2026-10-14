# baseline（历史留档）

> 本目录是项目早期五人分工阶段的模块原型与接口规范，是主线实现 `baseline_Integration/` 的前身。
> **新工作请前往 `baseline_Integration/`；本目录保留作历史参考，不再迭代。**

## 内容

```text
baseline/
├── ckks/                              # CKKS 模块早期原型
├── minhash&encoding/                  # MinHash 与编码早期原型
├── clustering/                        # 聚类早期原型
├── experimental_evaluation&ablation/  # 早期评测与消融实验
├── pipeline_spec.md                   # 当时的流水线设计说明
├── 项目分工与接口规范.pdf              # 五人分工接口约定
└── canvas.png                         # 系统架构草图
```

## 历史定位

项目按论文 *Privacy-preserving Fuzzy Name Matching for Sharing Financial
Intelligence* 的基线方案起步：MinHash 签名 + CKKS 全同态加密 + K-Means
聚类剪枝。本目录对应"Stage 1 复刻"阶段的模块分工成果；模块随后被整理、
打通并扩展为 `baseline_Integration/` 的正式工程（多语言、多属性、批量
加密等新能力均在后者中实现）。
