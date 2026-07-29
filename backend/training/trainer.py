"""KGTrainer：用于知识蒸馏的自定义 HuggingFace Trainer。

在基础 Trainer 上扩展了以下功能：
  - 学生与教师 logits 之间的偏向前向 KL 散度损失 (skewed F-KL)
  - 可选的交叉熵损失混合（熵模式）
  - 余弦衰减温度调度：temp = temp_min + (temp_max - temp_min) * (1 + cos(π·p)) / 2
  - 每轮保存检查点并合并 LoRA
  - 基于步数的定期检查点
"""

import os
import json
import time
import math
import logging
from typing import Optional, Dict, Any, List

import torch
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments
from transformers.modeling_outputs import CausalLMOutputWithPast
from peft import PeftModel

from losses.kl_divergence import compute_skewed_fkl

logger = logging.getLogger(__name__)


class KGTrainer(Trainer):
    """使用余弦衰减温度调度的知识蒸馏训练器。

    在学生与教师 logit 分布之间使用偏向前向 KL 散度。
    温度沿余弦曲线从 temp_max（初期软目标）平滑下降到
    temp_min（后期硬目标），混合交叉熵损失。
    """

    def __init__(
        self,
        model=None,
        teacher_model=None,
        if_use_entropy: bool = False,
        temp_max: float = 3.0,
        temp_min: float = 1.0,
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
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        self.teacher_model = teacher_model
        self.if_use_entropy = if_use_entropy
        self._kl_loss: Optional[float] = None
        self._ce_loss: Optional[float] = None
        self._accuracy: Optional[float] = None
        self.temp: float = temp_max
        self.temp_max: float = temp_max
        self.temp_min: float = temp_min

        # 生成日志
        self.proc = None  # 由 train.py 注入
        self._gen_log_path: Optional[str] = None

    # ------------------------------------------------------------------
    # 检查点
    # ------------------------------------------------------------------

    def on_epoch_end(self) -> None:
        """每轮结束时保存合并后的模型并保留 LoRA 继续训练。"""
        super().on_epoch_end()
        current_epoch = int(self.state.epoch) if self.state.epoch else 0
        total_epochs = self.args.num_train_epochs

        logger.info(
            "Epoch %d/%d complete (%.0f%%). Saving distilled model...",
            current_epoch, total_epochs,
            current_epoch / total_epochs * 100 if total_epochs else 0,
        )

        from config.settings import PROJECT_ROOT

        # 先保存当前 LoRA 适配器，再合并（确保首次 epoch 也不会丢失）
        saves_path = os.path.join(PROJECT_ROOT, "saves")
        self.save_model(saves_path)

        merged_model = self.model.merge_and_unload()
        epoch_model_path = os.path.join(
            PROJECT_ROOT, f"merged_student_model_epoch_{current_epoch}",
        )
        merged_model.save_pretrained(epoch_model_path)
        logger.info("Epoch model saved to %s", epoch_model_path)

        # 重新包装 LoRA 以继续训练
        self.model = PeftModel.from_pretrained(self.model, saves_path)
        self.model = self.model.to(self.model.device)

    # ------------------------------------------------------------------
    # 温度调度
    # ------------------------------------------------------------------

    def _update_temperature(self) -> None:
        """余弦衰减：temp = temp_min + (temp_max - temp_min) * (1 + cos(π * progress)) / 2

        训练早期温度高（~temp_max），软目标利于探索；
        训练后期温度低（~temp_min），硬目标利于精炼。
        """
        progress = self.state.epoch / self.args.num_train_epochs
        progress = min(progress, 1.0)
        self.temp = self.temp_min + (self.temp_max - self.temp_min) * 0.5 * (1.0 + math.cos(math.pi * progress))

    # ------------------------------------------------------------------
    # 生成日志：定期对比学生/教师/标准答案
    # ------------------------------------------------------------------

    def _log_generations(self) -> None:
        """取一条 eval 样本，让学生和教师分别生成回答，与标准答案一起写入日志。"""
        if self.proc is None or self.eval_dataset is None:
            return
        if self._gen_log_path is None:
            from config.settings import PROJECT_ROOT
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            os.makedirs(os.path.join(PROJECT_ROOT, "results"), exist_ok=True)
            self._gen_log_path = os.path.join(
                PROJECT_ROOT, "results", f"training_generations_{timestamp}.jsonl",
            )

        # 取一条 eval 样本
        sample = self.eval_dataset[min(len(self.eval_dataset) - 1,
                                       self.state.global_step % len(self.eval_dataset))]
        input_ids = sample["input_ids"].unsqueeze(0)
        attention_mask = sample["attention_mask"].unsqueeze(0)
        labels = [t for t in sample["labels"].tolist() if t != -100]

        # 解码题目（prompt 部分，labels=-100 的位置）
        prompt_ids = [t for t, l in zip(sample["input_ids"].tolist(), sample["labels"].tolist())
                      if l == -100]
        question = self.proc.decode(prompt_ids, skip_special_tokens=True)

        # 解码标准答案
        ground_truth = self.proc.decode(labels, skip_special_tokens=True)

        # 让教师生成
        device = next(self.teacher_model.parameters()).device
        teacher_inputs = {"input_ids": input_ids.to(device), "attention_mask": attention_mask.to(device)}
        with torch.no_grad():
            teacher_out = self.teacher_model.generate(
                **teacher_inputs, max_new_tokens=1536, temperature=0.7,
                do_sample=True, pad_token_id=self.proc.pad_token_id,
                eos_token_id=self.proc.eos_token_id,
            )
        teacher_answer = self.proc.decode(teacher_out[0], skip_special_tokens=True)
        teacher_answer = teacher_answer.replace(question, "").strip()

        # 让学生生成
        device = next(self.model.parameters()).device
        student_inputs = {"input_ids": input_ids.to(device), "attention_mask": attention_mask.to(device)}
        with torch.no_grad():
            student_out = self.model.generate(
                **student_inputs, max_new_tokens=1536, temperature=0.7,
                do_sample=True, pad_token_id=self.proc.pad_token_id,
                eos_token_id=self.proc.eos_token_id,
            )
        student_answer = self.proc.decode(student_out[0], skip_special_tokens=True)
        student_answer = student_answer.replace(question, "").strip()

        # 写入日志
        entry = {
            "step": self.state.global_step,
            "epoch": round(self.state.epoch, 2),
            "question": question,
            "ground_truth": ground_truth,
            "student": student_answer,
            "teacher": teacher_answer,
        }
        with open(self._gen_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        logger.info(
            "[GenLog] Step %d: 已记录。GT 长度=%d, 学生长度=%d, 教师长度=%d —> %s",
            self.state.global_step, len(ground_truth),
            len(student_answer), len(teacher_answer), self._gen_log_path,
        )

    # ------------------------------------------------------------------
    # 损失计算
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        model,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ):
        """计算知识蒸馏损失（KL + 可选的交叉熵）。"""
        outputs: CausalLMOutputWithPast = model(**inputs)

        device = outputs.logits.device
        teacher_inputs = {k: v.to(self.teacher_model.device) for k, v in inputs.items()}

        with torch.no_grad():
            teacher_outputs = self.teacher_model(**teacher_inputs)

        ce_loss = outputs.loss
        logits = outputs.logits
        teacher_logits = teacher_outputs.logits.to(device)

        # 处理词表大小不匹配
        if logits.shape[-1] != teacher_logits.shape[-1]:
            teacher_logits = teacher_logits[:, :, :logits.shape[-1]]

        # 为下一个 token 预测做偏移
        labels = inputs["labels"]
        logits_shifted = logits[:, :-1, :]
        teacher_logits_shifted = teacher_logits[:, :-1, :]
        labels_shifted = labels[:, 1:]

        kl_loss = compute_skewed_fkl(
            logits_shifted, teacher_logits_shifted, labels_shifted,
            padding_id=-100, temp=self.temp, skew_lambda=0.1,
        ).mean()

        # 准确率监控
        preds = logits_shifted.argmax(dim=-1)
        valid_mask = labels_shifted != -100
        correct = (preds == labels_shifted) & valid_mask
        accuracy = correct.sum().float() / valid_mask.sum().float()

        self._kl_loss = kl_loss.detach().cpu().item()
        self._ce_loss = ce_loss.detach().cpu().item()
        self._accuracy = accuracy.detach().cpu().item()

        if self.if_use_entropy:
            loss_total = 0.2 * kl_loss + 0.8 * ce_loss
        else:
            loss_total = kl_loss

        return (loss_total, outputs) if return_outputs else loss_total

    # ------------------------------------------------------------------
    # 日志记录
    # ------------------------------------------------------------------

    def log(self, logs: Dict[str, Any], start_time: Optional[float] = None) -> None:
        """注入自定义指标、更新温度，并定期记录模型生成对比。"""
        if self._kl_loss is not None:
            logs["kl_loss"] = self._kl_loss
        if self._ce_loss is not None:
            logs["ce_loss"] = self._ce_loss
        if self._accuracy is not None:
            logs["accuracy"] = self._accuracy

        self._update_temperature()
        logs["temp"] = round(self.temp, 2)

        # 每步记录学生/教师/标准答案对比
        try:
            self._log_generations()
        except Exception as exc:
            logger.warning("生成日志记录失败: %s", exc)

        super().log(logs, start_time)
