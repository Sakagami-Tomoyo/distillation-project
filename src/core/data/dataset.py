"""用于监督微调 / 知识蒸馏的 SFTDataset。

使用 Qwen 对话模板对问答对进行分词，生成适用于训练的
input_ids、attention_mask 和 labels 张量。
"""

import json

import torch
from torch.utils.data import Dataset
from typing import Dict, List, Any


class SFTDataset(Dataset):
    """监督微调数据集。

    api_mode=True 时，按智能业务助手的格式处理数据：output 为工具调用
    JSON（dict），序列化为文本并使用 API_SYSTEM_PROMPT。
    """

    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer: Any,
        max_seq_len: int,
        api_mode: bool = False,
    ) -> None:
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.api_mode = api_mode
        self.padding_id = tokenizer.pad_token_id

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        line = self.data[index]

        if "instruction" in line:
            user_msg = line["instruction"] + line.get("input", "")
            answer_text = line["output"]
            system_prompt = None  # 后面用 SYSTEM_PROMPT
        elif "prompt" in line:
            user_msg = line["prompt"]
            answer_text = line["answer"]
            system_prompt = None
        else:
            user_msg = line.get("question", "")
            answer_text = line.get("answer", "")
            system_prompt = None

        if self.api_mode:
            from config_shared import API_SYSTEM_PROMPT
            if isinstance(answer_text, dict):
                answer_text = json.dumps(answer_text, ensure_ascii=False)
            system_prompt = API_SYSTEM_PROMPT
            # api_mode 的用户消息就是问题本身，任务指令统一放在 system prompt 里，
            # 与推理时的 make_api_prompt 保持一致（避免训练/推理分布不一致）
            user_msg = line.get("input", "")
        else:
            from config_shared import SYSTEM_PROMPT
            if system_prompt is None:
                system_prompt = SYSTEM_PROMPT

        # 构建完整对话（含 assistant reply + <|im_end|>）
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": answer_text},
        ]
        full_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )

        # prompt-only 版本，用于确定 label 边界
        prompt_text = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt},
             {"role": "user", "content": user_msg}],
            tokenize=False, add_generation_prompt=True,
        )

        full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)

        input_ids = full_ids
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        attention_mask = [1] * len(input_ids)
        text_len = len(input_ids)

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
