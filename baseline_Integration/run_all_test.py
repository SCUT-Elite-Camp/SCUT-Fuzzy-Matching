"""运行所有测试文件"""

import subprocess
import sys

test_files = [
    "tests/test_interface_shapes.py",
    "tests/test_choose_k.py",
    "tests/test_security_boundary.py",
    "tests/test_correctness.py",
    "tests/test_evaluation_reporting.py",
    "tests/integration_test.py",
    "tests/test_end_to_end.py",
    "tests/test_ncvr_10k_loader.py",
    "tests/test_benchmark_real.py",
]

for test_file in test_files:
    print(f"\n{'='*60}")
    print(f"Running: {test_file}")
    print('='*60)
    result = subprocess.run([sys.executable, test_file])
    if result.returncode != 0:
        print(f" {test_file} failed")
        break
else:
    print("\n All tests passed")
