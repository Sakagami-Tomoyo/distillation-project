"""后端推理服务配置。"""

import os
import logging
import sys

# 添加项目根目录到 sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_shared import PROJECT_ROOT, MODEL_PATHS, DATA_PATHS, RESULT_PATHS  # noqa: F401, E402

logger = logging.getLogger(__name__)

FRONTEND_DIR = os.path.join(PROJECT_ROOT, "src", "frontend", "dist")

SERVER_CONFIG = {"host": "0.0.0.0", "port": 5000}

GENERATION_CONFIG = {
    "max_new_tokens": 2048, "temperature": 0.7, "top_p": 0.9,
    "repetition_penalty": 1.2, "do_sample": True,
}

# 遇到以下字符串立即停止生成（防止模型输出下一轮对话标签）
STOP_STRINGS = ["\nHuman:", "Human:\n", "\nuser:", "\nUser:", "<|im_start|>"]

SYSTEM_PROMPT = (
    "你是一位经验丰富的高考解题教练。"
    "你的核心任务是：不仅给出正确答案，更要展示清晰、分步的推理过程，帮助学生理解解题思路。"
)

USER_PROMPT_TEMPLATE = (
    "请解答以下题目。请严格按以下格式输出：\n"
    "答案:\n"
    "解析:\n"
    "核心思路：[用一句话概括解题方向]\n"
    "关键推理步骤：\n"
    "[分步列出推导过程]\n"
    "最终结论：[确认最终答案，并说明它为什么正确]\n"
    "\n"
    "题目：{question}"
)
