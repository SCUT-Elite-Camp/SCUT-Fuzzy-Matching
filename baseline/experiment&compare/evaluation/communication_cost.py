"""
通信开销测量：密文序列化大小
"""

def measure_ct_size(ct) -> int:
    """
    测量 TenSEAL 密文对象的序列化字节大小。

    Args:
        ct: TenSEAL CKKSVector 或密文对象

    Returns:
        int: 序列化后的字节数
    """
    if hasattr(ct, 'serialize'):
        return len(ct.serialize())
    else:
        # 如果不是 TenSEAL 对象（如 mock 数据），回退到 pickled 大小（仅用于调试）
        import pickle
        return len(pickle.dumps(ct))

class CommStats:
    """简单的通信量累计器（用于内部测试）"""
    def __init__(self):
        self.sent_bytes = 0
        self.recv_bytes = 0

    def add_sent(self, obj):
        self.sent_bytes += measure_ct_size(obj)

    def add_recv(self, obj):
        self.recv_bytes += measure_ct_size(obj)

    def total(self):
        return self.sent_bytes + self.recv_bytes

    def __repr__(self):
        return f"Sent: {self.sent_bytes / 1024**2:.2f} MB, Recv: {self.recv_bytes / 1024**2:.2f} MB"