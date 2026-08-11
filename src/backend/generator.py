"""共享的文本生成工具。

提供一个无状态的 generate_response 函数，供 API 服务器和 CLI 对话工具共同使用。
"""

import logging
from typing import List, Dict, Any, Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from config_shared import SYSTEM_PROMPT, STOP_STRINGS
from backend.config import GENERATION_CONFIG

logger = logging.getLogger(__name__)


def generate_response(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    history: Optional[List[Dict[str, str]]] = None,
    generation_config: Optional[Dict[str, Any]] = None,
) -> str:
    """根据给定的提示语，使用模型生成回复。"""
    device = next(model.parameters()).device

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    gen_kwargs = dict(GENERATION_CONFIG)
    if generation_config:
        gen_kwargs.update(generation_config)
    gen_kwargs.setdefault("pad_token_id", tokenizer.pad_token_id)
    gen_kwargs.setdefault("eos_token_id", tokenizer.eos_token_id)

    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.replace(text, "").strip()
    # 截断 stop strings（防止模型输出下一轮对话标签）
    for stop in STOP_STRINGS:
        idx = response.find(stop)
        if idx != -1:
            response = response[:idx].strip()
            break
    return response


