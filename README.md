# Fuzzy_Matching

> 复现并优化论文 *Privacy-preserving Fuzzy Name Matching for Sharing Financial Intelligence* 的基线方案，在此基础上探索改进空间，实现更高精度、更低开销的隐私保护模糊匹配系统。

---

## 项目目标

### Stage 1: Baseline 复刻
严格按论文算法还原完整协议流程，确保各项指标与论文报道一致：
- **MinHash + CKKS FHE + K-Means 聚类** 的完整端到端实现
- **精度 (Precision)**：维持完美精度 (~100%)
- **召回 (Recall)**：≥ 96%（基线目标）
- **通信开销**：相比无聚类线性搜索降低 30–300 倍
- **延迟**：1000 条查询在 10k / 100k / 1M 数据量下的端到端耗时

### Stage 2: 优化改进
在基线可用、指标可复现的基础上，探索以下改进维度：

| 方向 | 潜在改进点 |
|------|-----------|
| **编码优化** | 尝试 SimHash、Learned Hash、或融合字符级特征的更紧致签名 |
| **聚类策略** | 层次聚类、密度聚类 (HDBSCAN)、自适应 k 估计，减少边界召回损失 |
| **阈值策略** | 动态阈值、多阈值级联，平衡 Precision / Recall |
| **批处理效率** | 更优的 CKKS slot packing、GPU 加速同态运算 |
| **通信压缩** | 密文压缩、增量传输、early stopping 策略优化 |
| **混合协议** | 引入 PSI (Private Set Intersection) 预处理，缩小候选集 |

---