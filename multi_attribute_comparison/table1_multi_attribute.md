# Table 1. Comprehensive Experimental Evaluation of Multi-Attribute Matching

**Setup, Accuracy, Performance, and Communication Overhead of Schema-Driven Multi-Attribute HE Pipeline**

> 数据来源：本目录 `logs/ckks_febrl.log`、`logs/ckks_multiling.log` 两次实跑；口径与已知问题见 `report.md`。
> 第三列上下两行分别为 **FEBRL** 与 **Multilingual** 两组实验。

| Category / Dimension | Specific Metric / Parameter | Experimental Value / Result |
|---|---|---|
| **I. System Configuration Setup** | | |
| Configuration | Dataset | FEBRL 4a/4b（英文多属性脏数据）<br>Fused Multilingual（中/阿/缅/拉丁多语言脏数据） |
| | Source DB Records / Query Count | 5,000 / 100<br>300 / 50 |
| | Encoded Attributes | 6 属性：name, address, dob, ssn, postcode, state<br>4 属性：name（原生+转写）, dob, city, phone |
| | Cluster Feature Dim. / Match Feature Dim. | 2,156 / 1,936<br>3,272 / 3,122 |
| | Positive / Negative Queries | 100 / 0<br>40 / 10 |
| | Cluster Count *k* / Probes per Query | 70 / 3<br>17 / 3 |
| | Similarity Threshold τ / Decrypt Eps | 0.600 / 1e-4<br>0.485 / 1e-4 |
| | CKKS Polynomial Degree / Scale | 8192 / 2^40（coeff mod [60, 40, 40, 60]）<br>8192 / 2^40（同上） |
| | Batching Scheme | Per-query two-round（未启用 SIMD 批量）<br>Per-query two-round（同上） |
| | Early Stop / Context Reuse | True / False<br>True / False |
| | Zero-Block Skipping | Disabled<br>Disabled |
| **II. Accuracy & Confusion Matrix Evaluation** | | |
| Core Model Metrics | Precision | 1.0000 \*<br>1.0000 |
| | Recall | 0.9400<br>0.9000 |
| | F1-Score | 0.9691<br>0.9474 |
| | Accuracy | N/A（无 TN，无定义）<br>0.9200 |
| Confusion Matrix | True Positive / False Positive | 94 / 0<br>36 / 0 |
| | False Negative / True Negative | 6 / N/A<br>4 / 10 |
| Retrieval Consistency | Cluster Recall（τ-free） | 95 / 100 = 95.00%<br>37 / 40 = 92.50% |
| | Top-1 Cluster Recall | 86 / 100 = 86.00%<br>36 / 40 = 90.00% |
| | Should-Catch Recall（τ-dependent） | 94 / 99 = 94.95%<br>36 / 36 = 100.00% |
| **III. Execution Timing & Resource Overhead** | | |
| Execution Timing | Offline Preprocessing Time | 2.01 s（encode 0.59 + k-means/index 1.42）<br>0.10 s（encode 0.03 + k-means/index 0.07） |
| | Total Online Execution Time | 748.38 s<br>121.09 s |
| | Average Online Time per Query | 7.48 s<br>2.42 s |
| | Round 1 / Round 2 Split | Not measured（日志未分轮计时）<br>Not measured（同上） |
| Resource Overhead | Data Sent per Query | 35.42 MB<br>35.43 MB |
| | Data Received per Query | 22.75 MB<br>6.60 MB |
| | Total Communication Overhead | 5.68 GB<br>2.05 GB |
| | Communication per Query | 58.18 MB<br>42.03 MB |
| | Communication without Public Context | 24.35 MB（total 2.38 GB）<br>8.19 MB（total 0.40 GB） |
| | Scan Cost (columns scanned) | 56.26 / 5,000 = 1.13%<br>22.22 / 300 = 7.41% |
| | Peak Memory | Not measured; no memory claim<br>Not measured; no memory claim |
| **IV. Optimization Location & Work Reduction** | | |
| Modified Kernel | Public-Context Reuse | 预研：`prepare_party_a_multi_query` 每查询重建并序列化 context → 改为会话级复用<br>同一改动（同一函数） |
| Operation Counts | Context Serializations per Run | 100 → 1<br>50 → 1 |
| | Per-Query Communication (after) | 58.18 → 24.35 MB（−58.1%）<br>42.03 → 8.19 MB（−80.5%） |
| | Total Communication (after) | 5.68 → 2.38 GB（2.39×）<br>2.05 → 0.40 GB（5.13×） |
| Observed Gain (Estimated) | Accuracy Impact | None（不改协议与判定逻辑，仅改 context 生命周期）<br>None（同上） |
| | Robustness Side Effect | 消除长跑 `KSwitchKeys data is invalid` 崩溃（200 查询首跑在第 22 条复现）<br>同一改动（同一函数） |

## Notes

1. `*` **FEBRL 的 Precision 恒为 1，不具区分度**：其查询集全为正例（4b 的 5,000 条均为 4a 的 dup），没有负例可被误报；同时 TN=N/A 使 Accuracy 无定义。真正可用的 Precision 来自 Multilingual（10 条负例，实测 0 FP）。
2. **两组数据规模不同**（5,000 库 / 100 查询 vs 300 库 / 50 查询），时延与通信量**不可逐值横比**，只能看量级与趋势（如"每查询时延随库规模近线性、通信量由 context 主导"）。
3. 指标为**查询级 detection 口径**：协议只回传"是否越阈"一个比特，不返回命中记录 id，故 `TP = 标签为真 且 CATCH`。
4. 通信量按 **1 MB = 1024 KB、1 GB = 1024 MB** 计；"without public context" 与第 IV 节优化收益均基于实测的单查询公开 context **33.83 MB**。
5. **第 IV 节全部为预研估算，尚未实跑**：仅由实测的 context 占比推算，不改任何协议步骤；其中 `KSwitchKeys` 崩溃为实测现象（已发生一次），复用 context 能否根除属推断。
6. 早期报告里写的 2.43 GB / 0.48 GB 是十进制换算（÷1000）；此处统一 1024 进制，对应 2.38 GB / 0.40 GB。"约 4 倍"的直觉落在实测区间 2.39×–5.13× 之间。
7. 首列留空表示与上一行同属一组（粘进 Word 后可纵向合并单元格）；第三列的 `<br>` 若在纯文本编辑器里外露，替换成换行即可。
