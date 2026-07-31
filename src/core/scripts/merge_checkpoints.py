"""将现有 SFT checkpoint 合并为完整模型，存入 merged/sft/。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from config_shared import PROJECT_ROOT, MODEL_PATHS, MODEL_LOAD_CONFIG

checkpoints_dir = os.path.join(PROJECT_ROOT, "outputs", "checkpoints", "sft")
merged_dir = os.path.join(PROJECT_ROOT, "outputs", "merged", "sft")
os.makedirs(merged_dir, exist_ok=True)

student_path = MODEL_PATHS["student"]
tokenizer_path = MODEL_PATHS["student"]

for name in sorted(os.listdir(checkpoints_dir)):
    ckpt_path = os.path.join(checkpoints_dir, name)
    if not os.path.isdir(ckpt_path):
        continue
    if not os.path.exists(os.path.join(ckpt_path, "adapter_config.json")):
        print(f"  跳过（非 LoRA checkpoint）: {name}")
        continue

    out_dir = os.path.join(merged_dir, f"merged-{name}")
    if os.path.exists(out_dir):
        print(f"  跳过（已存在）: merged-{name}")
        continue

    print(f"  合并: {name} ...")
    base = AutoModelForCausalLM.from_pretrained(
        student_path,
        dtype=getattr(torch, MODEL_LOAD_CONFIG["dtype"]),
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True,
    )
    peft = PeftModel.from_pretrained(base, ckpt_path, local_files_only=True)
    merged = peft.merge_and_unload()
    merged.save_pretrained(out_dir)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, trust_remote_code=True)
    tokenizer.save_pretrained(out_dir)
    print(f"  ✅ merged-{name}")

print("完成。")
