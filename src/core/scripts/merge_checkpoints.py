"""将 checkpoints/ 下所有 LoRA checkpoint 合并为完整模型，存入 merged/。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from config_shared import PROJECT_ROOT, MODEL_PATHS, MODEL_LOAD_CONFIG

ckpt_root = os.path.join(PROJECT_ROOT, "outputs", "checkpoints")
merged_root = os.path.join(PROJECT_ROOT, "outputs", "merged")

# 教师 SFT 用教师模型作为基座，其余用学生模型
TEACHER_CATEGORIES = {"teacher-sft"}

student_path = MODEL_PATHS["student"]
teacher_path = MODEL_PATHS["teacher"]
tokenizer_path = MODEL_PATHS["student"]

for category in sorted(os.listdir(ckpt_root)):
    cat_dir = os.path.join(ckpt_root, category)
    if not os.path.isdir(cat_dir):
        continue

    base_path = teacher_path if category in TEACHER_CATEGORIES else student_path
    out_dir = os.path.join(merged_root, category)
    os.makedirs(out_dir, exist_ok=True)

    for name in sorted(os.listdir(cat_dir)):
        ckpt_path = os.path.join(cat_dir, name)
        if not os.path.isdir(ckpt_path):
            continue

        # 跳过已是完整模型的（含 config.json）
        if os.path.exists(os.path.join(ckpt_path, "config.json")):
            print(f"  跳过（已是完整模型）: {category}/{name}")
            continue

        # 必须有 adapter_config.json 才是 LoRA checkpoint
        if not os.path.exists(os.path.join(ckpt_path, "adapter_config.json")):
            print(f"  跳过（非 checkpoint）: {category}/{name}")
            continue

        merged_path = os.path.join(out_dir, f"merged-{name}")
        if os.path.exists(merged_path):
            print(f"  跳过（已存在）: merged-{name}")
            continue

        print(f"  合并: {category}/{name} ...")
        base = AutoModelForCausalLM.from_pretrained(
            base_path,
            dtype=getattr(torch, MODEL_LOAD_CONFIG["dtype"]),
            device_map="auto",
            local_files_only=True,
            trust_remote_code=True,
        )
        peft = PeftModel.from_pretrained(base, ckpt_path, local_files_only=True)
        merged = peft.merge_and_unload()
        merged.save_pretrained(merged_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, trust_remote_code=True)
        tokenizer.save_pretrained(merged_path)
        print(f"  ✅ merged-{name}")

print("完成。")
