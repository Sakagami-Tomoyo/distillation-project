"""知识蒸馏训练入口点。

用法：
    cd train && python train.py --sft --base-model ../checkpoints/sft/checkpoint-2855 --epochs 3
"""

import os
import sys
import time
import logging
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, DefaultDataCollator,
    TrainingArguments, BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType

from config_shared import (
    MODEL_PATHS, DATA_PATHS, RESULT_PATHS, QUANTIZATION_CONFIG,
    MODEL_LOAD_CONFIG, PROJECT_ROOT,
)
from core.train.config import LORA_CONFIG, TRAINING_ARGS, DATASET_CONFIG
from core.data.dataset import SFTDataset
from core.data.preprocessing import load_jsonl
from core.train.trainer import KGTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="知识蒸馏 / SFT 训练")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="恢复训练状态（optimizer/scheduler），模型从 --base-model 加载")
    parser.add_argument("--base-model", type=str, default=None,
                        help="基座模型路径。SFT 模式下请指定检查点路径，如 results/checkpoint-1713")
    parser.add_argument("--temp-max", type=float, default=3.0,
                        help="温度上限（训练初期）")
    parser.add_argument("--temp-min", type=float, default=1.0,
                        help="温度下限（训练末期）")
    parser.add_argument("--no-lora", action="store_true",
                        help="禁用 LoRA（全量微调）")
    parser.add_argument("--sft", action="store_true",
                        help="SFT 模式：纯 CE 微调学生模型，不加载教师")
    parser.add_argument("--teacher-sft", action="store_true",
                        help="教师 SFT 模式：微调教师模型（3B），用于后续蒸馏")
    parser.add_argument("--no-feature", action="store_true",
                        help="禁用中间特征蒸馏")
    parser.add_argument("--epochs", type=int, default=None,
                        help="训练轮数（覆盖配置文件中的默认值）")
    parser.add_argument("--subjects", type=str, default=None,
                        help="只训练指定学科，逗号分隔。如：历史,地理,语文")
    args = parser.parse_args()

    if not __import__("config_shared").HF_TOKEN:
        logger.warning("HF_TOKEN is not set.")

    # ------------------------------------------------------------------
    # 1. 加载基座模型 / 教师模型
    # ------------------------------------------------------------------
    if args.teacher_sft:
        # 教师 SFT：微调教师模型本身
        base_model_path = args.base_model or MODEL_PATHS["teacher"]
        teacher_model = None  # 教师 SFT 模式下没有单独的教师
    else:
        base_model_path = args.base_model or MODEL_PATHS["student"]

    # 相对路径 → 基于项目根目录解析
    if not os.path.isabs(base_model_path):
        base_model_path = os.path.join(PROJECT_ROOT, base_model_path)
    base_model_path = os.path.abspath(base_model_path)
    logger.info("基座模型路径（解析后）: %s", base_model_path)

    is_checkpoint = not os.path.exists(os.path.join(base_model_path, "config.json")) \
                    and os.path.exists(os.path.join(base_model_path, "adapter_config.json"))

    if is_checkpoint:
        # LoRA 检查点：先加载基座模型，再合并 LoRA 适配器
        base_for_ckpt = MODEL_PATHS["teacher"] if args.teacher_sft else MODEL_PATHS["student"]
        logger.info("检测到 LoRA 检查点: %s，基座: %s", base_model_path, base_for_ckpt)
        bnb_cfg = None
        load_kwargs = dict(
            dtype=getattr(torch, MODEL_LOAD_CONFIG["dtype"]),
            device_map=MODEL_LOAD_CONFIG["device_map"],
            local_files_only=MODEL_LOAD_CONFIG["local_files_only"],
            trust_remote_code=MODEL_LOAD_CONFIG["trust_remote_code"],
        )
        if args.teacher_sft:
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=QUANTIZATION_CONFIG["load_in_4bit"],
                bnb_4bit_compute_dtype=getattr(torch, QUANTIZATION_CONFIG["bnb_4bit_compute_dtype"]),
                bnb_4bit_use_double_quant=QUANTIZATION_CONFIG["bnb_4bit_use_double_quant"],
                bnb_4bit_quant_type=QUANTIZATION_CONFIG["bnb_4bit_quant_type"],
            )
            load_kwargs["quantization_config"] = bnb_cfg
        model = AutoModelForCausalLM.from_pretrained(base_for_ckpt, **load_kwargs)
        from peft import PeftModel
        logger.info("加载 LoRA 适配器: %s", base_model_path)
        model = PeftModel.from_pretrained(model, base_model_path, local_files_only=True)
        model = model.merge_and_unload()
        model.eval()
        logger.info("LoRA 适配器已合并，基座模型就绪")
    else:
        logger.info("加载基座模型: %s", base_model_path)
        bnb_cfg = None
        load_kwargs = dict(
            dtype=getattr(torch, MODEL_LOAD_CONFIG["dtype"]),
            device_map=MODEL_LOAD_CONFIG["device_map"],
            local_files_only=MODEL_LOAD_CONFIG["local_files_only"],
            trust_remote_code=MODEL_LOAD_CONFIG["trust_remote_code"],
        )
        if args.teacher_sft:
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=QUANTIZATION_CONFIG["load_in_4bit"],
                bnb_4bit_compute_dtype=getattr(torch, QUANTIZATION_CONFIG["bnb_4bit_compute_dtype"]),
                bnb_4bit_use_double_quant=QUANTIZATION_CONFIG["bnb_4bit_use_double_quant"],
                bnb_4bit_quant_type=QUANTIZATION_CONFIG["bnb_4bit_quant_type"],
            )
            load_kwargs["quantization_config"] = bnb_cfg
        model = AutoModelForCausalLM.from_pretrained(base_model_path, **load_kwargs)

    # ------------------------------------------------------------------
    # 2. 加载分词器
    # ------------------------------------------------------------------
    tok_path = base_model_path if os.path.exists(os.path.join(base_model_path, "tokenizer_config.json")) else MODEL_PATHS["student"]
    logger.info("加载分词器: %s", tok_path)
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------------
    # 3. 加载教师模型（SFT / 教师 SFT 模式跳过）
    # ------------------------------------------------------------------
    if not args.teacher_sft and not args.sft:
        logger.info("Loading teacher model from %s (4-bit)", MODEL_PATHS["teacher"])
        teacher_model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATHS["teacher"],
            dtype=getattr(torch, MODEL_LOAD_CONFIG["dtype"]),
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=QUANTIZATION_CONFIG["load_in_4bit"],
                bnb_4bit_compute_dtype=getattr(torch, QUANTIZATION_CONFIG["bnb_4bit_compute_dtype"]),
                bnb_4bit_use_double_quant=QUANTIZATION_CONFIG["bnb_4bit_use_double_quant"],
                bnb_4bit_quant_type=QUANTIZATION_CONFIG["bnb_4bit_quant_type"],
            ),
            device_map=MODEL_LOAD_CONFIG["device_map"],
            local_files_only=MODEL_LOAD_CONFIG["local_files_only"],
            trust_remote_code=MODEL_LOAD_CONFIG["trust_remote_code"],
        )
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
    elif not args.teacher_sft:
        teacher_model = None
        logger.info("SFT 模式：跳过教师模型，使用纯 CE 损失")

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
    mode = "teacher-sft" if args.teacher_sft else ("sft" if args.sft else "distillation")

    # 允许 --epochs 覆盖配置文件中的轮数
    train_kwargs = {k: v for k, v in TRAINING_ARGS.items() if k not in ("output_dir", "logging_dir")}
    if args.epochs is not None:
        train_kwargs["num_train_epochs"] = args.epochs

    checkpoint_dir = os.path.join(PROJECT_ROOT, "outputs", "checkpoints", mode)
    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        logging_dir=os.path.join(RESULT_PATHS["runs"], f"run_{timestamp}"),
        **train_kwargs,
    )

    # ------------------------------------------------------------------
    # 6. 数据集
    # ------------------------------------------------------------------
    logger.info("Loading data")
    train_data = load_jsonl(DATA_PATHS["train"])
    test_data = load_jsonl(DATA_PATHS["test"])

    # 按学科过滤（训练集只保留选定科目，测试集保持全科以监测遗忘）
    if args.subjects:
        target_subjects = set(s.strip() for s in args.subjects.split(","))
        train_data = [d for d in train_data if d.get("subject", "未知") in target_subjects]
        logger.info("训练集过滤 %s → %d 条（测试集保持全科 %d 条）",
                    target_subjects, len(train_data), len(test_data))
    else:
        logger.info("Train: %d samples, Test: %d samples", len(train_data), len(test_data))

    train_dataset = SFTDataset(train_data, tokenizer=tokenizer,
                               max_seq_len=DATASET_CONFIG["max_seq_len"])
    eval_dataset = SFTDataset(test_data, tokenizer=tokenizer,
                              max_seq_len=DATASET_CONFIG["max_seq_len"])

    data_collator = DefaultDataCollator()

    # ------------------------------------------------------------------
    # 7. 训练器
    # ------------------------------------------------------------------
    is_sft = args.sft or args.teacher_sft

    trainer = KGTrainer(
        model=model,
        teacher_model=teacher_model,
        if_use_entropy=not is_sft,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        feature_loss_weight=0.0 if (is_sft or args.no_feature) else 0.05,
        temp_max=1.0 if is_sft else args.temp_max,
        temp_min=1.0 if is_sft else args.temp_min,
        lambda_max=0.0 if is_sft else 0.5,
        lambda_min=0.0 if is_sft else 0.1,
    )
    trainer.proc = tokenizer
    trainer.mode = "teacher-sft" if args.teacher_sft else ("sft" if args.sft else "distillation")
    if args.teacher_sft:
        logger.info("教师 SFT 模式：微调教师模型（3B），纯 CE 训练")
    elif args.sft:
        logger.info("SFT 模式：纯 CE 训练（无教师、无 KL、无特征蒸馏）")
    else:
        logger.info("KD 模式：temp %.1f→%.1f  λ=%.2f→%.2f  feature_loss=%.2f",
                    trainer.temp_max, trainer.temp_min, trainer.lambda_max, trainer.lambda_min,
                    trainer.feature_loss_weight)
    logger.info("TensorBoard: %s", training_args.logging_dir)
    if args.checkpoint:
        # 解析 checkpoint 相对路径
        ckpt_path = args.checkpoint
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(PROJECT_ROOT, ckpt_path)
        ckpt_path = os.path.abspath(ckpt_path)
        logger.info("Resuming from checkpoint: %s", ckpt_path)
    else:
        ckpt_path = None

    prefix_map = {"sft": "sft", "distillation": "distillation", "teacher-sft": "teacher"}
    prefix = prefix_map.get(mode, mode)

    def _save_interrupt():
        """Ctrl+C 时保存当前检查点和合并模型。"""
        step = trainer.state.global_step
        epoch = round(trainer.state.epoch, 2) if trainer.state.epoch else step

        # 保存 checkpoint（LoRA + optimizer/scheduler）
        ckpt_name = f"checkpoint-{prefix}-interrupt-epoch{epoch}"
        ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
        trainer.save_model(ckpt_path)
        trainer.save_state()
        logger.info("中断检查点已保存: %s", ckpt_path)

        # 合并保存完整模型
        merged_name = f"merged-{prefix}-interrupt-epoch{epoch}"
        merged_path = os.path.join(PROJECT_ROOT, "outputs", "merged", mode, merged_name)
        os.makedirs(merged_path, exist_ok=True)
        m = trainer.model.merge_and_unload()
        m.save_pretrained(merged_path)
        tokenizer.save_pretrained(merged_path)
        logger.info("中断合并模型已保存: %s", merged_path)

        logger.info("恢复: --base-model %s --checkpoint %s", merged_path, ckpt_path)

    try:
        trainer.train(resume_from_checkpoint=ckpt_path)
    except KeyboardInterrupt:
        logger.info("\n⚠ Ctrl+C — 正在保存中断检查点...")
        _save_interrupt()
        return

    # ------------------------------------------------------------------
    # 8. 保存最终模型
    # ------------------------------------------------------------------
    final_epoch = int(trainer.state.epoch) if trainer.state.epoch else trainer.args.num_train_epochs

    # 保存最终 checkpoint（含 optimizer/scheduler）
    final_ckpt_dir = os.path.join(checkpoint_dir, f"checkpoint-{prefix}-final")
    trainer.save_model(final_ckpt_dir)
    trainer.save_state()
    logger.info("Final checkpoint saved to %s", final_ckpt_dir)

    # 合并保存完整模型
    logger.info("Merging LoRA adapter and saving full model [%s]...", mode)
    merged_model = model.merge_and_unload()
    merged_epoch_dir = os.path.join(PROJECT_ROOT, "outputs", "merged", mode, f"merged-{prefix}-final")
    os.makedirs(merged_epoch_dir, exist_ok=True)
    merged_model.save_pretrained(merged_epoch_dir)
    tokenizer.save_pretrained(merged_epoch_dir)
    logger.info("Final merged model saved to %s", merged_epoch_dir)


if __name__ == "__main__":
    # 开始前清理 CUDA 缓存
    torch.cuda.empty_cache()
    logger.info("Initial GPU memory: %.2f GB", torch.cuda.memory_allocated(0) / 1024**3)
    main()
