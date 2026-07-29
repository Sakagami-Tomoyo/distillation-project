"""知识蒸馏训练入口点。

用法：
    python training/train.py [--checkpoint PATH] [--temp-max 3.0] [--temp-min 1.0]
"""

import os
import sys
import time
import logging
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DefaultDataCollator,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType

from config.settings import (
    MODEL_PATHS,
    DATA_PATHS,
    RESULT_PATHS,
    QUANTIZATION_CONFIG,
    LORA_CONFIG,
    TRAINING_ARGS,
    DATASET_CONFIG,
    MODEL_LOAD_CONFIG,
    HF_TOKEN,
    PROJECT_ROOT,
)
from data.dataset import SFTDataset
from data.preprocessing import load_jsonl
from training.trainer import KGTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="知识蒸馏训练")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="从检查点目录恢复训练")
    parser.add_argument("--temp-max", type=float, default=3.0,
                        help="温度上限（训练初期）")
    parser.add_argument("--temp-min", type=float, default=1.0,
                        help="温度下限（训练末期）")
    parser.add_argument("--no-lora", action="store_true",
                        help="禁用 LoRA（全量微调）")
    args = parser.parse_args()

    if not HF_TOKEN:
        logger.warning("HF_TOKEN is not set. Set it in your .env file or environment.")

    # ------------------------------------------------------------------
    # 1. 加载学生模型
    # ------------------------------------------------------------------
    logger.info("Loading student model from %s", MODEL_PATHS["student"])
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATHS["student"],
        torch_dtype=getattr(torch, MODEL_LOAD_CONFIG["torch_dtype"]),
        device_map=MODEL_LOAD_CONFIG["device_map"],
        local_files_only=MODEL_LOAD_CONFIG["local_files_only"],
        trust_remote_code=MODEL_LOAD_CONFIG["trust_remote_code"],
    )

    # ------------------------------------------------------------------
    # 2. 加载分词器
    # ------------------------------------------------------------------
    logger.info("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATHS["student"],
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------------
    # 3. 加载教师模型（4 位量化）
    # ------------------------------------------------------------------
    logger.info("Loading teacher model from %s (4-bit)", MODEL_PATHS["teacher"])
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=QUANTIZATION_CONFIG["load_in_4bit"],
        bnb_4bit_compute_dtype=getattr(torch, QUANTIZATION_CONFIG["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=QUANTIZATION_CONFIG["bnb_4bit_use_double_quant"],
        bnb_4bit_quant_type=QUANTIZATION_CONFIG["bnb_4bit_quant_type"],
    )
    teacher_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATHS["teacher"],
        torch_dtype=getattr(torch, MODEL_LOAD_CONFIG["torch_dtype"]),
        quantization_config=bnb_config,
        device_map=MODEL_LOAD_CONFIG["device_map"],
        local_files_only=MODEL_LOAD_CONFIG["local_files_only"],
        trust_remote_code=MODEL_LOAD_CONFIG["trust_remote_code"],
    )
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    # ------------------------------------------------------------------
    # 4. LoRA
    # ------------------------------------------------------------------
    if not args.no_lora:
        logger.info("Applying LoRA: r=%d, alpha=%d", LORA_CONFIG["r"], LORA_CONFIG["lora_alpha"])
        lora_config = LoraConfig(
            r=LORA_CONFIG["r"],
            lora_alpha=LORA_CONFIG["lora_alpha"],
            target_modules=LORA_CONFIG["target_modules"],
            lora_dropout=LORA_CONFIG["lora_dropout"],
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # 5. 训练参数
    # ------------------------------------------------------------------
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    training_args = TrainingArguments(
        output_dir=RESULT_PATHS["output"],
        logging_dir=os.path.join(RESULT_PATHS["runs"], f"run_{timestamp}"),
        **{k: v for k, v in TRAINING_ARGS.items() if k not in ("output_dir", "logging_dir")},
    )

    # ------------------------------------------------------------------
    # 6. 数据集
    # ------------------------------------------------------------------
    logger.info("Loading data")
    train_data = load_jsonl(DATA_PATHS["train"])
    test_data = load_jsonl(DATA_PATHS["test"])
    logger.info("Train: %d samples, Test: %d samples", len(train_data), len(test_data))

    train_dataset = SFTDataset(train_data, tokenizer=tokenizer,
                               max_seq_len=DATASET_CONFIG["max_seq_len"])
    eval_dataset = SFTDataset(test_data, tokenizer=tokenizer,
                              max_seq_len=DATASET_CONFIG["max_seq_len"])

    data_collator = DefaultDataCollator()

    # ------------------------------------------------------------------
    # 7. 训练器
    # ------------------------------------------------------------------
    trainer = KGTrainer(
        model=model,
        teacher_model=teacher_model,
        if_use_entropy=True,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    trainer.temp_max = args.temp_max
    trainer.temp_min = args.temp_min
    trainer.proc = tokenizer
    logger.info("Training starting. temp: %.1f -> %.1f (cosine decay)", trainer.temp_max, trainer.temp_min)
    logger.info("TensorBoard: %s", training_args.logging_dir)
    if args.checkpoint:
        logger.info("Resuming from checkpoint: %s", args.checkpoint)

    trainer.train(resume_from_checkpoint=args.checkpoint)

    # ------------------------------------------------------------------
    # 8. 保存最终模型
    # ------------------------------------------------------------------
    saves_path = RESULT_PATHS["saves"]
    trainer.save_model(saves_path)
    trainer.save_state()

    logger.info("Merging LoRA adapter and saving full model...")
    merged_model = model.merge_and_unload()
    merged_path = MODEL_PATHS["merged"]
    merged_model.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)
    logger.info("Final merged model saved to %s", merged_path)


if __name__ == "__main__":
    # 开始前清理 CUDA 缓存
    torch.cuda.empty_cache()
    logger.info("Initial GPU memory: %.2f GB", torch.cuda.memory_allocated(0) / 1024**3)
    main()
