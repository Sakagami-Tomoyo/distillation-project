"""共享的文本生成工具。

提供一个无状态的 generate_response 函数，供 API 服务器和 CLI 对话工具共同使用。
"""

import logging
from typing import List, Dict, Any, Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from config.settings import GENERATION_CONFIG

logger = logging.getLogger(__name__)


def generate_response(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    history: Optional[List[Dict[str, str]]] = None,
    generation_config: Optional[Dict[str, Any]] = None,
) -> str:
    """根据给定的提示语，使用模型生成回复。

    Args:
        model: 已加载的 HuggingFace 模型。
        tokenizer: 已加载的分词器。
        prompt: 用户输入的文本。
        history: 可选的过去对话历史，格式为 {"user": ..., "assistant": ...} 列表。
        generation_config: 可选的生成参数覆盖。

    Returns:
        模型的文本回复（已去除输入提示语）。
    """
    device = next(model.parameters()).device

    # 根据历史记录和当前提示语构建消息列表
    messages: List[Dict[str, str]] = []
    if history:
        for turn in history:
            messages.append({"role": "user", "content": turn["user"]})
            if "assistant" in turn and turn["assistant"]:
                messages.append({"role": "assistant", "content": turn["assistant"]})
    messages.append({"role": "user", "content": prompt})

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    # 合并配置
    gen_kwargs = dict(GENERATION_CONFIG)
    if generation_config:
        gen_kwargs.update(generation_config)
    gen_kwargs.setdefault("pad_token_id", tokenizer.pad_token_id)
    gen_kwargs.setdefault("eos_token_id", tokenizer.eos_token_id)

    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 从回复中去除输入提示语
    response = response.replace(text, "").strip()
    return response


def make_chat_prompt(
    tokenizer: PreTrainedTokenizer,
    question: str,
) -> str:
    """构建单轮对话提示字符串。

    Args:
        tokenizer: 已加载的分词器。
        question: 用户的问题文本。

    Returns:
        格式化后的提示字符串，可直接用于 model.generate()。
    """
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
