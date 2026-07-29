"""验证模型加载、分词器一致性和数据集 token 长度分布。

使用更清晰的导入结构替代了旧的根级 test.py。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from config.settings import MODEL_PATHS, DATA_PATHS, MODEL_LOAD_CONFIG
from data.preprocessing import load_jsonl


def test_model_load(model_name: str):
    """测试通过配置键名加载模型。"""
    path = MODEL_PATHS.get(model_name)
    print(f"测试加载: {path}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=True, trust_remote_code=True,
        )
        print("  ✅ Tokenizer 加载成功")

        import torch
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=getattr(torch, MODEL_LOAD_CONFIG["torch_dtype"]),
            device_map=MODEL_LOAD_CONFIG["device_map"],
            local_files_only=True,
            trust_remote_code=True,
        )
        print("  ✅ 模型加载成功")
        params = sum(p.numel() for p in model.parameters()) / 1e9
        print(f"  参数量: {params:.2f}B")
        return True, tokenizer
    except Exception as e:
        print(f"  ❌ 加载失败: {e}")
        return False, None


def check_tokenizer_consistency(tok1, tok2, name1: str, name2: str) -> bool:
    """检查两个分词器是否兼容用于蒸馏训练。"""
    print(f"\n=== 检查 {name1} 和 {name2} 的Tokenizer一致性 ===")

    checks = [
        ("vocab_size", tok1.vocab_size, tok2.vocab_size),
        ("tokenizer_type", type(tok1).__name__, type(tok2).__name__),
        ("eos_token", repr(tok1.eos_token), repr(tok2.eos_token)),
        ("eos_token_id", tok1.eos_token_id, tok2.eos_token_id),
        ("bos_token", repr(tok1.bos_token), repr(tok2.bos_token)),
        ("bos_token_id", tok1.bos_token_id, tok2.bos_token_id),
        ("pad_token", repr(tok1.pad_token), repr(tok2.pad_token)),
        ("pad_token_id", tok1.pad_token_id, tok2.pad_token_id),
        ("unk_token", repr(tok1.unk_token), repr(tok2.unk_token)),
        ("unk_token_id", tok1.unk_token_id, tok2.unk_token_id),
        ("model_max_length", tok1.model_max_length, tok2.model_max_length),
        ("padding_side", tok1.padding_side, tok2.padding_side),
    ]

    all_match = True
    for name, v1, v2 in checks:
        match = v1 == v2
        status = "✅" if match else "❌"
        print(f"  {status} {name}: {name1}={v1}, {name2}={v2}")
        if not match:
            all_match = False

    if all_match:
        print("✅ 两个Tokenizer完全一致，可以进行蒸馏训练")
    else:
        print("❌ 两个Tokenizer存在不一致，请检查模型版本")
    return all_match


def analyze_token_lengths(data, tokenizer, dataset_name: str):
    """输出 token 长度分布统计信息。"""
    print(f"\n=== {dataset_name} Token 长度分布统计 ===")

    lengths = []
    for item in data:
        if "prompt" in item and "answer" in item:
            text = item["prompt"] + item["answer"]
        elif "instruction" in item and "output" in item:
            text = item["instruction"] + item.get("input", "") + item["output"]
        else:
            text = str(item.get("question", "")) + str(item.get("answer", ""))

        tokens = tokenizer.encode(text, add_special_tokens=False)
        lengths.append(len(tokens))

    lengths = np.array(lengths)

    print(f"样本总数: {len(lengths)}")
    print(f"最小长度: {np.min(lengths)}")
    print(f"最大长度: {np.max(lengths)}")
    print(f"平均长度: {np.mean(lengths):.2f}")
    print(f"中位数: {np.median(lengths)}")
    print(f"标准差: {np.std(lengths):.2f}")

    for p in [50, 75, 90, 95, 99]:
        print(f"  P{p}: {np.percentile(lengths, p)}")

    bins = [0, 128, 256, 512, 1024, 2048, 4096]
    hist, _ = np.histogram(lengths, bins=bins)
    print("\n长度区间分布:")
    for i in range(len(bins) - 1):
        count = hist[i]
        pct = count / len(lengths) * 100
        print(f"  [{bins[i]}, {bins[i+1]}): {count} ({pct:.1f}%)")

    print("\n建议 max_seq_len 设置:")
    for p in [90, 95, 99]:
        print(f"  - 覆盖{p}%样本: {np.percentile(lengths, p):.0f}")

    return lengths


if __name__ == "__main__":
    print("=== 检查模型文件 ===")
    success1, student_tok = test_model_load("student")
    success2, teacher_tok = test_model_load("teacher")

    if student_tok and teacher_tok:
        check_tokenizer_consistency(
            student_tok, teacher_tok,
            "学生模型(Qwen2.5-0.5B)",
            "教师模型(Qwen2.5-3B)",
        )

    tokenizer = student_tok or teacher_tok
    if not tokenizer:
        print("\n❌ Tokenizer 加载失败，无法统计token长度")
        sys.exit(1)

    print(f"\n=== 检查数据集文件 ===")
    for name, path in [("训练集", DATA_PATHS["train"]), ("测试集", DATA_PATHS["test"])]:
        if os.path.exists(path):
            data = load_jsonl(path)
            print(f"✅ {name}加载成功，样本数: {len(data)}")
            analyze_token_lengths(data, tokenizer, name)
        else:
            print(f"❌ {name}文件不存在: {path}")
