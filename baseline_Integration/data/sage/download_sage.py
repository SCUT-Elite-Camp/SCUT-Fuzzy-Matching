"""
下载 SAGE 顶层 parquet 文件并转换为 CSV
数据来源：https://storage.googleapis.com/sage-archive/sage.parquet
"""

import requests
import pandas as pd
import os

# 下载配置
url = "https://storage.googleapis.com/sage-archive/sage.parquet"
local_parquet = "sage.parquet"
output_csv = "sage_data.csv"

# 如果文件已存在，跳过下载
if os.path.exists(local_parquet):
    print(f"✅ {local_parquet} 已存在，跳过下载")
else:
    print(f"正在下载 {url} ... (约 857 MB，请耐心等待)")
    try:
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            print(f"❌ 下载失败，状态码 {response.status_code}")
            exit()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        with open(local_parquet, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size:
                        progress = downloaded / total_size * 100
                        print(f"下载进度: {progress:.1f}%", end="\r")
        print("\n✅ 下载完成")
    except Exception as e:
        print(f"❌ 下载出错: {e}")
        exit()

# 读取 parquet
print("正在读取 parquet 文件...")
try:
    df = pd.read_parquet(local_parquet)
    print(f"✅ 读取成功，共 {len(df)} 行，{len(df.columns)} 列")
    print(f"列名: {df.columns.tolist()}")
except Exception as e:
    print(f"❌ 读取失败: {e}")
    exit()

# 导出为 CSV（分批导出，避免内存不足）
print(f"正在导出为 {output_csv} ...")
try:
    # 如果数据量太大，可以只导出部分列
    # 这里先导出全部列，如果内存不足可以只选需要的列
    df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"✅ 已导出 {output_csv}")
    print(f"文件大小: {os.path.getsize(output_csv) / 1024 / 1024:.2f} MB")
except Exception as e:
    print(f"❌ 导出失败: {e}")