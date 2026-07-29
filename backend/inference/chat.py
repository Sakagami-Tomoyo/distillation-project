"""与蒸馏后学生模型的交互式 CLI 对话。

用法：
    python inference/chat.py
"""

import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config.settings import MODEL_PATHS, MODEL_LOAD_CONFIG, GENERATION_CONFIG
from inference.generator import generate_response

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    model_path = MODEL_PATHS["merged"]

    logger.info("Loading distilled model from %s", model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=getattr(torch, MODEL_LOAD_CONFIG["torch_dtype"]),
        device_map=MODEL_LOAD_CONFIG["device_map"],
        local_files_only=MODEL_LOAD_CONFIG["local_files_only"],
        trust_remote_code=MODEL_LOAD_CONFIG["trust_remote_code"],
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("模型已加载。开始对话（输入 'quit' 退出）。")

    history = []
    gen_config = dict(GENERATION_CONFIG)
    gen_config["max_new_tokens"] = 4096

    while True:
        try:
            user_input = input("\n用户: ")
        except (EOFError, KeyboardInterrupt):
            print("\n对话结束")
            break

        if user_input.lower() in ("quit", "exit", "退出"):
            print("对话结束")
            break

        response = generate_response(
            model, tokenizer, user_input,
            history=history,
            generation_config=gen_config,
        )
        print(f"助手: {response}")
        print("-" * 60)

        history.append({"user": user_input, "assistant": response})
        if len(history) > 5:
            history = history[-5:]


if __name__ == "__main__":
    main()
