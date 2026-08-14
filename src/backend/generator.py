"""共享的文本生成工具。

提供无状态的 generate_response 函数，供 API 服务器和 CLI 对话工具共同使用；
并提供智能问答（意图识别 → 工具调用）所需的 prompt 与生成函数。
"""

import logging
from typing import List, Dict, Any, Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from config_shared import SYSTEM_PROMPT, STOP_STRINGS, API_SYSTEM_PROMPT
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


def make_chat_prompt(tokenizer: PreTrainedTokenizer, question: str) -> str:
    """构建推理 prompt：system + 题目，与训练数据 instruction/input 格式一致。"""
    from config_shared import SYSTEM_PROMPT
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


# ---------------------------------------------------------------------------
# 智能问答（意图识别 → 工具调用）
# ---------------------------------------------------------------------------

def make_api_prompt(tokenizer: PreTrainedTokenizer, question: str) -> str:
    """构建智能问答 prompt：system + 用户问题。"""
    messages = [
        {"role": "system", "content": API_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def generate_api_response(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    question: str,
    generation_config: Optional[Dict[str, Any]] = None,
) -> str:
    """智能问答：根据 API_SYSTEM_PROMPT 生成工具调用 JSON 文本。"""
    device = next(model.parameters()).device
    prompt = make_api_prompt(tokenizer, question)

    gen_kwargs = dict(GENERATION_CONFIG)
    gen_kwargs["max_new_tokens"] = 256  # 只需生成简短 JSON
    if generation_config:
        gen_kwargs.update(generation_config)
    gen_kwargs.setdefault("pad_token_id", tokenizer.pad_token_id)
    gen_kwargs.setdefault("eos_token_id", tokenizer.eos_token_id)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    # 只解码「生成部分」，避免把整个 prompt 一起带出来
    response = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()
    return response
