import sys
import os
from pathlib import Path

# 确保能导入 evaluation 包
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from evaluation.error_analyzer import CKKSApproximationError

if __name__ == "__main__":
    analyzer = CKKSApproximationError()
    # 运行 500 次随机向量对的误差分析
    results = analyzer.batch_analysis(num_samples=500, vec_dim=50, tau=0.9)
    
    print("\n=== 批量误差分析结果 ===")
    print(f"CT-PT 点积绝对误差 - 均值: {results['ct_pt_abs_error']['mean']:.6e}")
    print(f"CT-PT 点积绝对误差 - 最大值: {results['ct_pt_abs_error']['max']:.6e}")
    print(f"CT-CT 点积绝对误差 - 均值: {results['ct_ct_abs_error']['mean']:.6e}")
    print(f"CT-CT 点积绝对误差 - 最大值: {results['ct_ct_abs_error']['max']:.6e}")
    print(f"阈值判断失败率: {results['threshold_failure_rate']['mean']:.4f}")
