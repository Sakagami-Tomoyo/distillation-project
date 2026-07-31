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

        # -- 提取问题和答案 -----------------------------------------
        if "instruction" in line:
            query = line["instruction"] + line.get("input", "")
            answer = line["output"] + self.tokenizer.eos_token
        elif "prompt" in line:
            query = line["prompt"]
            answer = line["answer"] + self.tokenizer.eos_token
        else:
            query = line.get("question", "")
            answer = line.get("answer", "") + self.tokenizer.eos_token

        # -- 构建对话格式（含 system role，与推理格式一致）--------
        system_prompt = "你是一位经验丰富的高考解题教练。你的核心任务是：不仅给出正确答案，更要展示清晰、分步的推理过程，帮助学生理解解题思路。\n请解答以下题目。请严格按以下格式输出：\n答案: [选择题：你的选项；其他题型：答案]\n解析:\n核心思路：[用一句话概括解题方向，比如“本题考察有氧呼吸的第二阶段”或“本题可以用韦达定理”]\n关键推理步骤：\n[分步列出推导过程，比如“由 A 可得 B”、“将 B 代入 C 得 D”]\n[对于选择题，请对每个选项给出判断]\n最终结论：[确认最终答案]\n\n"
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
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
            "subject": line.get("subject", "未知"),
        }
