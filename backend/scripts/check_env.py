"""检查系统环境：CUDA、GPU 显存、模型文件可用性。"""

import os
import sys
import torch

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import MODEL_PATHS

print("=" * 50)
print("系统环境检查")
print("=" * 50)

# 检查 CUDA
print(f"\nCUDA 是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"GPU 数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        total_mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"  显存总量: {total_mem:.2f} GB")
        print(f"  显存已用: {allocated:.2f} GB")
        print(f"  显存保留: {reserved:.2f} GB")
else:
    print("⚠ 警告: CUDA 不可用，训练将使用 CPU，速度会非常慢！")

# 检查 MPS（Mac）
print(f"\nMPS 是否可用: {torch.backends.mps.is_available()}")

# 检查 PyTorch 版本
print(f"\nPyTorch 版本: {torch.__version__}")

# 检查项目模型目录
print(f"\n学生模型路径: {MODEL_PATHS['student']}")
print(f"学生模型存在: {os.path.exists(MODEL_PATHS['student'])}")
print(f"教师模型路径: {MODEL_PATHS['teacher']}")
print(f"教师模型存在: {os.path.exists(MODEL_PATHS['teacher'])}")
print(f"蒸馏模型路径: {MODEL_PATHS['distilled']}")
print(f"蒸馏模型存在: {os.path.exists(MODEL_PATHS['distilled'])}")

print("\n" + "=" * 50)
