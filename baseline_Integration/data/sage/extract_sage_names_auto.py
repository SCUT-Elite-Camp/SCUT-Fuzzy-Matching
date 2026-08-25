"""
自动检测列名，从 SAGE parquet 中提取姓名数据
"""

import pyarrow.parquet as pq
import pandas as pd
import os
import gc
from pathlib import Path

# ============================================================
# 配置
# ============================================================
PARQUET_FILE = "sage.parquet"
OUTPUT_DIR = "sage_names"
OUTPUT_TOTAL = "sage_names_all.csv"
BATCH_SIZE = 50000
MAX_ROWS_PER_COUNTRY = 10000
MIN_NAMES_PER_COUNTRY = 100

Path(OUTPUT_DIR).mkdir(exist_ok=True)

if not os.path.exists(PARQUET_FILE):
    print(f"❌ 文件 {PARQUET_FILE} 不存在")
    exit()

print(f"正在读取 {PARQUET_FILE} ...")
parquet = pq.ParquetFile(PARQUET_FILE)
columns = parquet.schema_arrow.names
print(f"可用列: {columns}")

# ============================================================
# 自动检测列名
# ============================================================
# 候选姓名列名（按优先级）
candidate_name_col = None
for candidate in ['candidate_name', 'candidate', 'cand_name', 'name', 'full_name']:
    if candidate in columns:
        candidate_name_col = candidate
        break

# 候选国家列名
country_col = None
for candidate in ['country', 'country_name', 'nation', 'state']:
    if candidate in columns:
        country_col = candidate
        break

if candidate_name_col is None:
    raise RuntimeError(f"未找到姓名列。可用列：{columns}")
if country_col is None:
    print(f"⚠️ 未找到国家列，将使用 'Unknown'")
    country_col = None

print(f"✅ 使用姓名列: {candidate_name_col}")
if country_col:
    print(f"✅ 使用国家列: {country_col}")
else:
    print("⚠️ 未使用国家列")

# ============================================================
# 分批提取
# ============================================================
cols_to_read = [candidate_name_col]
if country_col:
    cols_to_read.append(country_col)

print(f"\n开始提取 {cols_to_read} ...")
all_names = []
country_counts = {}

batch_count = 0
processed = 0

for batch in parquet.iter_batches(batch_size=BATCH_SIZE, columns=cols_to_read):
    df = batch.to_pandas()
    
    # 去掉空值
    df = df.dropna(subset=[candidate_name_col])
    
    # 去重
    df = df.drop_duplicates(subset=cols_to_read)
    
    for _, row in df.iterrows():
        name = str(row[candidate_name_col]).strip()
        if country_col and pd.notna(row[country_col]):
            country = str(row[country_col]).strip()
        else:
            country = "Unknown"
        
        if name and len(name) > 1:
            all_names.append((name, country))
            country_counts[country] = country_counts.get(country, 0) + 1
    
    processed += len(df)
    batch_count += 1
    if batch_count % 10 == 0:
        print(f"已处理 {processed:,} 行，已收集 {len(all_names):,} 个唯一姓名", end="\r")
    
    del df
    gc.collect()

print(f"\n✅ 提取完成！共收集 {len(all_names):,} 个唯一姓名")
if country_col:
    print(f"覆盖 {len(country_counts)} 个国家")

# ============================================================
# 保存结果
# ============================================================
print("\n正在保存...")
total_df = pd.DataFrame(all_names, columns=['candidate_name', 'country'])
total_df.to_csv(OUTPUT_TOTAL, index=False, encoding='utf-8')
print(f"✅ 总文件: {OUTPUT_TOTAL}，{len(total_df):,} 行")

# 按国家分拆
if country_col and len(country_counts) > 1:
    country_data = {}
    for name, country in all_names:
        country_data.setdefault(country, []).append(name)
    
    for country, names in country_data.items():
        if len(names) > MAX_ROWS_PER_COUNTRY:
            import random
            random.seed(42)
            names = random.sample(names, MAX_ROWS_PER_COUNTRY)
        if len(names) < MIN_NAMES_PER_COUNTRY:
            continue
        safe_name = country.replace(' ', '_').replace('/', '_')
        output_file = os.path.join(OUTPUT_DIR, f"sage_{safe_name}.csv")
        pd.DataFrame(names, columns=['candidate_name']).to_csv(output_file, index=False, encoding='utf-8')
        print(f"  ✅ {country}: {len(names):,} → {output_file}")

print("\n✅ 完成！")