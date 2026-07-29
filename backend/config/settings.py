"""Distallation 项目的集中化配置。

所有路径、训练超参数和生成设置均在此处定义。
敏感值（如 HF_TOKEN）从 .env 文件中自动加载。
"""

import os
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 项目根目录
# ---------------------------------------------------------------------------

# backend/config/settings.py → 向上 3 级到项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# 自动加载 .env 文件
# ---------------------------------------------------------------------------

def _load_dotenv():
    """从项目根目录的 .env 文件加载环境变量（无需额外依赖）。"""
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

# ---------------------------------------------------------------------------
# 环境变量
# ---------------------------------------------------------------------------

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    logger.warning("未找到 HF_TOKEN。请在项目根目录创建 .env 文件并设置 HF_TOKEN。"
                   "可参考 .env.example 模板。")
else:
    logger.info("HF_TOKEN 已从 .env 加载。")

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------

MODEL_PATHS = {
    "student": os.path.join(PROJECT_ROOT, "models", "Qwen2.5-0.5B-Instruct"),
    "teacher": os.path.join(PROJECT_ROOT, "models", "Qwen2.5-3B-Instruct"),
    "distilled": os.path.join(PROJECT_ROOT, "results", "checkpoint-1068"),
    "merged": os.path.join(PROJECT_ROOT, "merged_student_model"),
}

DATA_PATHS = {
    "original": os.path.join(PROJECT_ROOT, "data", "Original_Data"),
    "train": os.path.join(PROJECT_ROOT, "data", "train_data", "train.jsonl"),
    "test": os.path.join(PROJECT_ROOT, "data", "train_data", "test.jsonl"),
    "train_dir": os.path.join(PROJECT_ROOT, "data", "train_data"),
}

RESULT_PATHS = {
    "output": os.path.join(PROJECT_ROOT, "results"),
    "saves": os.path.join(PROJECT_ROOT, "saves"),
    "runs": os.path.join(PROJECT_ROOT, "runs"),
}

FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend", "dist")

# ---------------------------------------------------------------------------
# 服务器配置
# ---------------------------------------------------------------------------

SERVER_CONFIG = {
    "host": "0.0.0.0",
    "port": 5000,
}

# ---------------------------------------------------------------------------
# 生成配置
# ---------------------------------------------------------------------------

GENERATION_CONFIG = {
    "max_new_tokens": 2048,
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.05,
    "do_sample": True,
}

# ---------------------------------------------------------------------------
# 数据集配置
# ---------------------------------------------------------------------------

DATASET_CONFIG = {
    "train_years": set(range(2010, 2021)),   # 2010-2020
    "test_years": {2021, 2022},              # 2021-2022
    "max_token_length": 1536,
    "max_seq_len": 1536,
}

# ---------------------------------------------------------------------------
# 模型加载配置
# ---------------------------------------------------------------------------

MODEL_LOAD_CONFIG = {
    "torch_dtype": "float16",          # 序列化为字符串
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
# LoRA 配置
# ---------------------------------------------------------------------------

LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "lora_dropout": 0.1,
}

# ---------------------------------------------------------------------------
# 训练配置
# ---------------------------------------------------------------------------

TRAINING_ARGS = {
    "num_train_epochs": 5,
    "do_train": True,
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "logging_steps": 1,
    "report_to": "tensorboard",
    "save_strategy": "epoch",
    "save_total_limit": None,
    "learning_rate": 2e-4,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.03,
    "dataloader_num_workers": 2,
    "dataloader_pin_memory": True,
    "fp16": True,
    "optim": "adamw_torch",
    "max_grad_norm": 1.0,
    "remove_unused_columns": False,
    "skip_memory_metrics": True,
}
