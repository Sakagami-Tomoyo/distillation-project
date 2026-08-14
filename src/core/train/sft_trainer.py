"""
SFTTrainer：监督微调训练器。

SFT 损失 ：以「标准答案」为真实分布、「模型输出」为预测分布，
标准答案是一段 one-hot 文本，故该前向 KL 在数学上退化为交叉熵，
损失实现直接用 HF 的 outputs.loss。

SFT 基座默认 0.5B
训练只需提供训练集与验证集；评估为通用的验证集 token 准确率（不分学科）。
检查点保存等训练基础设施在这里共享，KDTrainer（见 kd_trainer.py）继承本类，
只补充蒸馏特有的损失与调度。
"""

import os
import logging
from typing import Optional, Dict, Any

import torch
from transformers import Trainer, TrainingArguments
from peft import PeftModel

logger = logging.getLogger(__name__)


class SFTTrainer(Trainer):
    """监督微调训练器（前向 KL ≡ 交叉熵，无教师）。

    通用 SFT：训练集 / 验证集由训练入口指定，评估不分学科、不区分数据用途。
    """

    def __init__(
        self,
        model=None,
        args: TrainingArguments = None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        model_init=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
        preprocess_logits_for_metrics=None,
    ) -> None:
        super().__init__(
            model=model, args=args, data_collator=data_collator,
            train_dataset=train_dataset, eval_dataset=eval_dataset,
            model_init=model_init, compute_metrics=compute_metrics,
            callbacks=callbacks, optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        # 监控指标（日志用）
        self._ce_loss: Optional[float] = None
        self._accuracy: Optional[float] = None

        # KDTrainer 的生成对比日志需要的分词器（由 train.py 赋值）
        self.proc = None

    # ------------------------------------------------------------------
    # 损失：交叉熵
    # ------------------------------------------------------------------

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """SFT 损失：前向 KL（标准答案分布 ‖ 模型分布）≡ 交叉熵 = outputs.loss。

        必须覆写：HF Trainer 每步都会调用 compute_loss 取损失，同时在这里
        填充 _ce_loss / _accuracy 两个监控标量供日志展示。
        """
        outputs = model(**inputs)

        # 交叉熵：标准答案为 one-hot 真实分布，模型为预测分布
        ce_loss = outputs.loss
        self._ce_loss = ce_loss.detach().cpu().item()

        # 准确率（监控指标，不参与损失）
        labels = inputs["labels"]
        logits_shifted = outputs.logits[:, :-1, :]
        valid_mask = labels[:, 1:] != -100
        accuracy = (logits_shifted.argmax(-1) == labels[:, 1:]) & valid_mask
        accuracy = accuracy.sum().float() / valid_mask.sum().float()
        self._accuracy = accuracy.detach().cpu().item()

        return (ce_loss, outputs) if return_outputs else ce_loss

    # ------------------------------------------------------------------
    # 评估
    # ------------------------------------------------------------------

    def _eval_generic(self, max_samples: int = 50, log_prefix: str = "Eval") -> Dict[str, float]:
        """
        SFT 训练集 / 验证集由训练入口自行指定，这里只做整体前向评估。
        """
        if self.eval_dataset is None:
            return {}
        device = next(self.model.parameters()).device
        total_correct = 0
        total_valid = 0
        for i in range(len(self.eval_dataset)):
            if i >= max_samples:
                break
            sample = self.eval_dataset[i]
            input_ids = sample["input_ids"].unsqueeze(0).to(device)
            attention_mask = sample["attention_mask"].unsqueeze(0).to(device)
            labels = sample["labels"].unsqueeze(0).to(device)

            with torch.no_grad():
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits

            # 同训练时的 token 准确率计算
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            preds = shift_logits.argmax(dim=-1)
            valid = shift_labels != -100
            total_correct += ((preds == shift_labels) & valid).sum().item()
            total_valid += valid.sum().item()

        acc = total_correct / total_valid if total_valid > 0 else 0.0
        logger.info("[%s] 验证集 token 准确率: %.1f%% (%d/%d)",
                    log_prefix, acc * 100, total_correct, total_valid)
        return {"eval_accuracy": round(acc, 4)}

    # ------------------------------------------------------------------
    # 检查点
    # ------------------------------------------------------------------

    def on_epoch_end(self) -> None:
        super().on_epoch_end()
        from config_shared import PROJECT_ROOT

        current_epoch = int(self.state.epoch) if self.state.epoch else 0
        # 输出目录由训练入口按模式决定（sft / wenda / distillation），这里直接复用
        mode_dir = os.path.basename(self.args.output_dir)
        logger.info("Epoch %d/%d [%s] — saving model...", current_epoch, self.args.num_train_epochs, mode_dir)

        # 通用评估：验证集 token 准确率
        self._eval_generic(max_samples=50, log_prefix=f"Epoch{current_epoch}")

        # 保存 checkpoint（LoRA + optimizer/scheduler）
        ckpt_dir = os.path.join(PROJECT_ROOT, "outputs", "checkpoints", mode_dir, f"checkpoint-{mode_dir}-v3-epoch_{current_epoch}")
        self.save_model(ckpt_dir)
        self.save_state()
        logger.info("Checkpoint saved to %s", ckpt_dir)

        # 合并保存完整模型
        saves_path = os.path.join(PROJECT_ROOT, "outputs", "saves")
        self.save_model(saves_path)

        merged = self.model.merge_and_unload()
        merged_dir = os.path.join(PROJECT_ROOT, "outputs", "merged", mode_dir, f"merged-{mode_dir}-v3-epoch_{current_epoch}")
        merged.save_pretrained(merged_dir)
        logger.info("Merged model saved to %s", merged_dir)

        self.model = PeftModel.from_pretrained(self.model, saves_path, local_files_only=True)
        self.model = self.model.to(self.model.device)

    # ------------------------------------------------------------------
    # 日志
    # ------------------------------------------------------------------

    def log(self, logs: Dict[str, Any], start_time: Optional[float] = None) -> None:
        if self._ce_loss is not None:
            logs["ce_loss"] = self._ce_loss
        if self._accuracy is not None:
            logs["accuracy"] = self._accuracy

        # 每 2 步轻量通用评估（验证集前 max_samples 条）
        step = self.state.global_step
        if step > 0 and step % 2 == 0:
            try:
                eval_res = self._eval_generic(max_samples=20, log_prefix=f"Step{step}")
                for k, v in eval_res.items():
                    logs[k] = v
            except Exception as exc:
                logger.warning("通用评估失败: %s", exc)

        super().log(logs, start_time)
