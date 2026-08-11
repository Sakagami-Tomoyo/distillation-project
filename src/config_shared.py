"""项目级共享配置（路径、HF_TOKEN、环境加载）。"""

import os
import logging

logger = logging.getLogger(__name__)

# src/config_shared.py → src/ → 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# 自动加载 .env
# ---------------------------------------------------------------------------

def _load_dotenv():
    env_path = os.path.join(PROJECT_ROOT, ".env")
    if not os.path.isfile(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if value and key not in os.environ:
                os.environ[key] = value

_load_dotenv()

# 强制离线
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    logger.warning("未找到 HF_TOKEN。")

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------

MODEL_PATHS = {
    "student": os.path.join(PROJECT_ROOT, "models", "Qwen2.5-0.5B-Instruct"),
    "teacher": os.path.join(PROJECT_ROOT, "models", "Qwen2.5-3B-Instruct"),
    "distilled": os.path.join(PROJECT_ROOT, "outputs", "checkpoints", "distillation"),
    "merged": os.path.join(PROJECT_ROOT, "outputs", "merged", "distillation"),
}

DATA_PATHS = {
    "original": os.path.join(PROJECT_ROOT, "data", "Original_Data"),
    "train": os.path.join(PROJECT_ROOT, "data", "train_data", "train.jsonl"),
    "test": os.path.join(PROJECT_ROOT, "data", "train_data", "test.jsonl"),
    "train_dir": os.path.join(PROJECT_ROOT, "data", "train_data"),
}

RESULT_PATHS = {
    "output": os.path.join(PROJECT_ROOT, "outputs", "checkpoints", "distillation"),
    "saves": os.path.join(PROJECT_ROOT, "outputs", "saves"),
    "runs": os.path.join(PROJECT_ROOT, "outputs", "runs"),
}

MODEL_LOAD_CONFIG = {
    "dtype": "float16",
    "device_map": "auto",
    "local_files_only": True,
    "trust_remote_code": True,
}

QUANTIZATION_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": "float16",
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_quant_type": "nf4",
}

# ---------------------------------------------------------------------------
# 共享 Prompt（训练 & 推理统一）
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "你是一位经验丰富的高考解题教练。你的核心任务是：不仅给出正确答案，"
    "更要展示清晰、分步的推理过程，帮助学生理解解题思路。\n"
    "请严格按以下格式输出：\n"
    "答案: [选择题写选项，其他题型写答案]\n"
    "解析:\n"
    "核心思路：[一句话概括解题方向]\n"
    "关键推理步骤：\n"
    "[分步推导，选择题需逐项判断]\n"
    "最终结论：[确认答案]\n"
    "\n"
    "只回答当前这道题，回答完毕后立即停止，不要生成任何用户反馈或后续对话。"
)

# 遇到以下字符串立即停止生成（防止模型脑补下一轮对话）
STOP_STRINGS = [
    "\nHuman:", "\nHuman:\n",
    "\nuser:", "\nUser:",
    "<|im_start|>",
    "\n（",           # 中文括号开头：感谢/追问
    "\n用户",         # 中文"用户"开头：用户追问
    "\n非常",         # "非常感谢""非常详细"
]
