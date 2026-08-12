"""训练配置。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config_shared import (  # noqa: F401, E402
    PROJECT_ROOT, MODEL_PATHS, DATA_PATHS, RESULT_PATHS,
    MODEL_LOAD_CONFIG, QUANTIZATION_CONFIG,
)

DATASET_CONFIG = {
    "train_years": set(range(2010, 2021)),
    "test_years": {2021, 2022},
    "max_token_length": 1536,
    "max_seq_len": 1536,
}

LORA_CONFIG = {
    "r": 4,
    "lora_alpha": 32,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "lora_dropout": 0.1,
}

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
