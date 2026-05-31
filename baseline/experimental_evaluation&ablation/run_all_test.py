"""运行所有测试文件"""

import subprocess
import sys

test_files = [
    "tests/test_interface_shapes.py",
    "tests/test_security_boundary.py",
    "tests/test_correctness.py",
    "tests/test_mock_flow.py",      # 原 test_experiment_flow.py 重命名后的文件
    "tests/integration_test.py",
]

for test_file in test_files:
    print(f"\n{'='*60}")
    print(f"Running: {test_file}")
    print('='*60)
    result = subprocess.run([sys.executable, test_file])
    if result.returncode != 0:
        print(f"❌ {test_file} failed")
        break
else:
    print("\n🎉 All tests passed!")