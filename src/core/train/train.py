"""知识蒸馏训练入口点。

用法：
    cd src && python core/train/train.py --sft --epochs 3
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
from core.train.kd_trainer import KDTrainer
from core.train.sft_trainer import SFTTrainer

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
                        help="基座模型路径")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--sft", action="store_true",
                        help="SFT 模式：前向 KL（标准答案‖模型）≡ 交叉熵，不加载教师")
    mode_group.add_argument("--distill", action="store_true",
                        help="蒸馏模式：KL + CE + Feature Loss，需加载教师")
    parser.add_argument("--temp-max", type=float, default=3.0,
                        help="温度上限（训练初期）")
    parser.add_argument("--temp-min", type=float, default=1.0,
                        help="温度下限（训练末期）")
    parser.add_argument("--no-lora", action="store_true",
                        help="禁用 LoRA（全量微调）")
    parser.add_argument("--no-feature", action="store_true",
                        help="禁用中间特征蒸馏")
    parser.add_argument("--epochs", type=int, default=None,
                        help="训练轮数（覆盖配置文件中的默认值）")
    parser.add_argument("--subjects", type=str, default=None,
                        help="只训练指定学科，逗号分隔。如：历史,地理,语文")
    parser.add_argument("--train-data", type=str, default=None,
                        help="自定义训练数据路径（默认 data/train_data/train.jsonl）")
    parser.add_argument("--eval-data", type=str, default=None,
                        help="自定义评估数据路径（默认 data/train_data/test.jsonl）")
    parser.add_argument("--api-mode", action="store_true",
                        help="智能问答（意图识别）模式：训练 data/apidata 数据，输出工具调用 JSON（配合 --sft 使用）")
    args = parser.parse_args()

    if not __import__("config_shared").HF_TOKEN:
        logger.warning("HF_TOKEN is not set.")

    # ------------------------------------------------------------------
    # 1. 加载模型
    # ------------------------------------------------------------------
    base_model_path = args.base_model or MODEL_PATHS["student"]

    if not os.path.isabs(base_model_path):
        base_model_path = os.path.join(PROJECT_ROOT, base_model_path)
    base_model_path = os.path.abspath(base_model_path)

    # 自动检测是否为教师模型（路径包含 "3B"）→ 需要 4-bit 量化
    is_teacher = "3B" in base_model_path or "3b" in base_model_path
    logger.info("基座模型路径（解析后）: %s [%s]", base_model_path,
                "教师, 4-bit" if is_teacher else "学生")

    is_checkpoint = not os.path.exists(os.path.join(base_model_path, "config.json")) \
                    and os.path.exists(os.path.join(base_model_path, "adapter_config.json"))

    def _load_model(path: str, quantize: bool = False):
        """加载模型，可选 4-bit 量化。"""
        load_kwargs = dict(
            dtype=getattr(torch, MODEL_LOAD_CONFIG["dtype"]),
            device_map=MODEL_LOAD_CONFIG["device_map"],
            local_files_only=MODEL_LOAD_CONFIG["local_files_only"],
            trust_remote_code=MODEL_LOAD_CONFIG["trust_remote_code"],
        )
        if quantize:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=QUANTIZATION_CONFIG["load_in_4bit"],
                bnb_4bit_compute_dtype=getattr(torch, QUANTIZATION_CONFIG["bnb_4bit_compute_dtype"]),
                bnb_4bit_use_double_quant=QUANTIZATION_CONFIG["bnb_4bit_use_double_quant"],
                bnb_4bit_quant_type=QUANTIZATION_CONFIG["bnb_4bit_quant_type"],
            )
        return AutoModelForCausalLM.from_pretrained(path, **load_kwargs)

    if is_checkpoint:
        # LoRA 检查点：先加载基座，再合并适配器
        ckpt_base = MODEL_PATHS["teacher"] if is_teacher else MODEL_PATHS["student"]
        logger.info("检测到 LoRA 检查点: %s，基座: %s", base_model_path, ckpt_base)
        model = _load_model(ckpt_base, quantize=is_teacher)
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, base_model_path, local_files_only=True)
        model = model.merge_and_unload()
        model.eval()
        logger.info("LoRA 适配器已合并，基座模型就绪")
    else:
        model = _load_model(base_model_path, quantize=is_teacher)

    # ------------------------------------------------------------------
    # 2. 加载分词器
    # ------------------------------------------------------------------
    tok_path = base_model_path if os.path.exists(os.path.join(base_model_path, "tokenizer_config.json")) else MODEL_PATHS["student"]
    logger.info("加载分词器: %s", tok_path)
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------------
    # 3. 加载教师模型
    # ------------------------------------------------------------------
    if args.sft:
        teacher_model = None
        logger.info("SFT 模式：跳过教师模型，使用纯 CE 损失")
    else:
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
    mode = "sft" if args.sft else "distillation"
    if args.api_mode:
        mode = "wenda"

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
    if args.api_mode:
        apidata_dir = os.path.join(PROJECT_ROOT, "data", "apidata")
        train_path = args.train_data or os.path.join(apidata_dir, "intent_train.jsonl")
        test_path = args.eval_data or os.path.join(apidata_dir, "intent_test.jsonl")
    else:
        train_path = args.train_data or DATA_PATHS["train"]
        test_path = args.eval_data or DATA_PATHS["test"]
    logger.info("Loading data: train=%s, test=%s", train_path, test_path)
    train_data = load_jsonl(train_path)
    test_data = load_jsonl(test_path)

    # 按学科过滤（训练集只保留选定科目，测试集保持全科以监测遗忘）
    if args.subjects:
        target_subjects = set(s.strip() for s in args.subjects.split(","))
        train_data = [d for d in train_data if d.get("subject", "未知") in target_subjects]
        logger.info("训练集过滤 %s → %d 条（测试集保持全科 %d 条）",
                    target_subjects, len(train_data), len(test_data))
    else:
        logger.info("Train: %d samples, Test: %d samples", len(train_data), len(test_data))

    train_dataset = SFTDataset(train_data, tokenizer=tokenizer,
                               max_seq_len=DATASET_CONFIG["max_seq_len"],
                               api_mode=args.api_mode)
    eval_dataset = SFTDataset(test_data, tokenizer=tokenizer,
                              max_seq_len=DATASET_CONFIG["max_seq_len"],
                              api_mode=args.api_mode)

    data_collator = DefaultDataCollator()

    # ------------------------------------------------------------------
    # 7. 训练器
    # ------------------------------------------------------------------
    if args.sft:
        # SFTTrainer：前向 KL（标准答案‖模型）≡ 交叉熵，无教师、无特征蒸馏、无 λ/T 调度
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        model_type = "教师(3B)" if is_teacher else "学生(0.5B)"
        logger.info("SFT 模式 [%s]：前向 KL（标准答案‖模型）≡ 交叉熵，无教师、无特征蒸馏", model_type)
    else:
        # KDTrainer：skewed-FKL + CE + 特征蒸馏 + 动态 λ/T（必须有教师）
        trainer = KDTrainer(
            model=model,
            teacher_model=teacher_model,
            if_use_entropy=True,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            feature_loss_weight=0.0 if args.no_feature else 0.05,
            temp_max=args.temp_max,
            temp_min=args.temp_min,
            lambda_max=0.5,
            lambda_min=0.1,
        )
        logger.info("KD 模式：temp %.1f→%.1f  λ=%.2f→%.2f  feature_loss=%.2f",
                    trainer.temp_max, trainer.temp_min, trainer.lambda_max, trainer.lambda_min,
                    trainer.feature_loss_weight)
    trainer.proc = tokenizer
    logger.info("TensorBoard: %s", training_args.logging_dir)
    if args.checkpoint:
        ckpt_path = args.checkpoint
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(PROJECT_ROOT, ckpt_path)
        ckpt_path = os.path.abspath(ckpt_path)
        logger.info("Resuming from checkpoint: %s", ckpt_path)
    else:
        ckpt_path = None

    prefix_map = {"sft": "sft", "distillation": "distillation", "wenda": "wenda"}
    prefix = prefix_map.get(mode, mode)

    # 设置手动检查点触发器
    trainer._save_trigger_file = os.path.join(PROJECT_ROOT, "outputs", "SAVE_NOW")
    trainer._ckpt_dir = checkpoint_dir

    def _save_interrupt():
        """Ctrl+C 时保存当前检查点和合并模型。"""
        step = trainer.state.global_step
        epoch = round(trainer.state.epoch, 2) if trainer.state.epoch else step
        ckpt_name = f"checkpoint-{prefix}-interrupt-epoch{epoch}"
        ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
        trainer.save_model(ckpt_path)
        trainer.save_state()
        logger.info("中断检查点已保存: %s", ckpt_path)
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
        logger.info("\nCtrl+C — 正在保存中断检查点...")
        _save_interrupt()
        return

    # ------------------------------------------------------------------
    # 8. 保存最终模型
    # ------------------------------------------------------------------
    final_epoch = int(trainer.state.epoch) if trainer.state.epoch else trainer.args.num_train_epochs

    final_ckpt_dir = os.path.join(checkpoint_dir, f"checkpoint-{prefix}-final")
    trainer.save_model(final_ckpt_dir)
    trainer.save_state()
    logger.info("Final checkpoint saved to %s", final_ckpt_dir)

    logger.info("Merging LoRA adapter and saving full model [%s]...", mode)
    merged_model = model.merge_and_unload()
    merged_epoch_dir = os.path.join(PROJECT_ROOT, "outputs", "merged", mode, f"merged-{prefix}-final")
    os.makedirs(merged_epoch_dir, exist_ok=True)
    merged_model.save_pretrained(merged_epoch_dir)
    tokenizer.save_pretrained(merged_epoch_dir)
    logger.info("Final merged model saved to %s", merged_epoch_dir)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    logger.info("Initial GPU memory: %.2f GB", torch.cuda.memory_allocated(0) / 1024**3)
    main()
