"""数据预处理工具。

将原始高考 JSON 数据转换为适用于 SFTDataset 的 train/test JSONL 分割文件。
"""

import json
import os
import glob
import logging
from typing import List, Dict, Any, Set

from transformers import AutoTokenizer

from config.settings import (
    DATA_PATHS,
    DATASET_CONFIG,
    MODEL_PATHS,
)

logger = logging.getLogger(__name__)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """加载 JSONL 文件（每行一个 JSON 对象）。"""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def _load_tokenizer():
    """加载用于长度检查的分词器。"""
    return AutoTokenizer.from_pretrained(
        MODEL_PATHS["student"],
        trust_remote_code=True,
        local_files_only=True,
    )


def build_splits(
    train_years: Set[int] = DATASET_CONFIG["train_years"],
    test_years: Set[int] = DATASET_CONFIG["test_years"],
    max_token_length: int = DATASET_CONFIG["max_token_length"],
) -> tuple:
    """处理所有原始数据文件并生成训练/测试分割。

    Returns:
        (训练数据, 测试数据, 过滤掉的训练样本数, 过滤掉的测试样本数)
    """
    original_dir = DATA_PATHS["original"]
    train_dir = DATA_PATHS["train_dir"]
    os.makedirs(train_dir, exist_ok=True)

    tokenizer = _load_tokenizer()

    train_data: List[Dict[str, Any]] = []
    test_data: List[Dict[str, Any]] = []
    filtered_train = 0
    filtered_test = 0

    for folder in ["Objective_Questions", "Subjective_Questions"]:
        folder_path = os.path.join(original_dir, folder)
        if not os.path.exists(folder_path):
            continue
        json_files = glob.glob(os.path.join(folder_path, "*.json"))
        for json_file in json_files:
            logger.info("Processing %s", os.path.basename(json_file))
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            for item in data.get("example", []):
                year = int(item.get("year", 0))
                question = item.get("question", "")
                answer = item.get("answer", "")
                analysis = item.get("analysis", "")

                if isinstance(answer, list):
                    answer = "".join(answer)

                full_answer = f"答案: {answer}\n解析: {analysis}"
                prompt = f"请解答以下高考题目：\n\n{question}。输出要求：首先显示答案是什么，然后再开始推理，例如：答案: D\n解析:......："

                total_text = prompt + full_answer
                token_count = len(tokenizer.encode(total_text, add_special_tokens=False))

                if token_count > max_token_length:
                    if year in train_years:
                        filtered_train += 1
                    elif year in test_years:
                        filtered_test += 1
                    continue

                entry = {
                    "prompt": prompt,
                    "answer": full_answer,
                    "year": year,
                    "source": os.path.basename(json_file),
                }

                if year in train_years:
                    train_data.append(entry)
                elif year in test_years:
                    test_data.append(entry)

    logger.info(
        "Train: %d samples (filtered %d), Test: %d samples (filtered %d)",
        len(train_data), filtered_train, len(test_data), filtered_test,
    )
    return train_data, test_data, filtered_train, filtered_test


def save_splits(train_data: List[Dict[str, Any]], test_data: List[Dict[str, Any]]) -> None:
    """将训练/测试分割写入 JSONL 文件。"""
    train_path = DATA_PATHS["train"]
    test_path = DATA_PATHS["test"]

    with open(train_path, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(test_path, "w", encoding="utf-8") as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info("Train set saved to %s", train_path)
    logger.info("Test set saved to %s", test_path)
