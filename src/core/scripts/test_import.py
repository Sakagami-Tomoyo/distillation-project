"""验证 transformers 和模型加载是否正常工作。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config_shared import MODEL_PATHS, MODEL_LOAD_CONFIG

print(f"Python版本: {sys.version}")
print(f"Python路径: {sys.executable}")

print("\n=== 测试1: 基本导入 ===")
try:
    import transformers
    print(f"✅ transformers导入成功，版本: {transformers.__version__}")
except ImportError as e:
    print(f"❌ 导入transformers失败: {e}")
    sys.exit(1)

print("\n=== 测试2: 导入AutoModelForCausalLM ===")
try:
    from transformers import AutoModelForCausalLM
    print("✅ AutoModelForCausalLM导入成功")
except ImportError as e:
    print(f"❌ 导入AutoModelForCausalLM失败: {e}")

print("\n=== 测试3: transformers版本兼容性 ===")
if transformers.__version__ >= "4.0.0":
    print(f"✅ transformers版本 {transformers.__version__} 支持AutoModelForCausalLM")
else:
    print(f"⚠ transformers版本 {transformers.__version__} 可能不支持")

print("\n=== 测试4: 快速测试模型加载(CPU) ===")
try:
    import torch
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATHS["student"],
        dtype=torch.float16,
        device_map="cpu",
        local_files_only=True,
        trust_remote_code=True,
    )
    print("✅ 模型加载成功")
    del model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
except Exception as e:
    print(f"❌ 模型加载失败: {e}")

print("\n=== 诊断完成 ===")
