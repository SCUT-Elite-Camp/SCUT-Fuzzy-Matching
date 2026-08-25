import gcsfs

fs = gcsfs.GCSFileSystem(token='anon')

# 列出 parquet 目录
print("sage-archive/parquet/ 下内容：")
try:
    items = fs.ls('sage-archive/parquet/')
    print(items)
except Exception as e:
    print(f"失败：{e}")

# 检查 sage.parquet 是否是文件
print("\n检查 sage-archive/sage.parquet：")
try:
    info = fs.info('sage-archive/sage.parquet')
    print(info)
except Exception as e:
    print(f"失败：{e}")
