"""用于监督微调 / 知识蒸馏的 SFTDataset。

使用 Qwen 对话模板对问答对进行分词，生成适用于训练的
input_ids、attention_mask 和 labels 张量。
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, List, Any


class SFTDataset(Dataset):
    """监督微调数据集。

    支持多种数据格式：
      - instruction/input/output  (Alpaca 风格)
      - prompt/answer
      - question/answer
    """

    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer: Any,
        max_seq_len: int,
    ) -> None:
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.padding_id = tokenizer.pad_token_id

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        line = self.data[index]

        # -- 从数据字典中提取问题和答案 --------------------------------
        if "instruction" in line:
            query = line["instruction"] + line.get("input", "")
            answer = line["output"] + self.tokenizer.eos_token
        elif "prompt" in line:
            query = line["prompt"]
            answer = line["answer"] + self.tokenizer.eos_token
        else:
            query = line.get("question", "")
            answer = line.get("answer", "") + self.tokenizer.eos_token

        # -- 构建对话格式的提示语 -------------------------------------
        messages: List[Dict[str, str]] = [
            {"role": "user", "content": query},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        answer_ids = self.tokenizer.encode(answer, add_special_tokens=False)

        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids
        attention_mask = [1] * len(input_ids)
        text_len = len(input_ids)

        # -- 填充或截断至 max_seq_len --------------------------------
        if text_len > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
            labels = labels[:self.max_seq_len]
            attention_mask = attention_mask[:self.max_seq_len]
        else:
            pad_len = self.max_seq_len - text_len
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
            labels = labels + [-100] * pad_len
            attention_mask = attention_mask + [0] * pad_len

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels),
        }
