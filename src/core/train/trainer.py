"""KGTrainer：用于知识蒸馏的自定义 HuggingFace Trainer。

功能：
  - skewed F-KL 散度损失 + 可选交叉熵损失混合
  - λ 余弦衰减：早期重 KL（向教师学），后期重 CE（向标准答案学）
  - 温度余弦衰减：temp_max → temp_min
  - 中间特征蒸馏：匹配师生中间层 hidden states
  - 每轮保存检查点并合并 LoRA
"""

import os
import json
import time
import math
import logging
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments
from transformers.modeling_outputs import CausalLMOutputWithPast
from peft import PeftModel

from core.train.losses import compute_skewed_fkl

logger = logging.getLogger(__name__)

# 师生层匹配：学生 24 层，教师 36 层，等比对应 6 个位置
LAYER_MATCHES = [
    (4, 6), (8, 12), (12, 18), (16, 24), (20, 30), (23, 35),
]

from config_shared import STOP_STRINGS


def _trim_stop_strings(text: str) -> str:
    for s in STOP_STRINGS:
        idx = text.find(s)
        if idx != -1:
            return text[:idx].strip()
    return text


class KGTrainer(Trainer):
    """知识蒸馏训练器：余弦衰减温度 + 余弦衰减 λ + 中间特征蒸馏。"""

    def __init__(
        self,
        model=None,
        teacher_model=None,
        if_use_entropy: bool = False,
        temp_max: float = 3.0,
        temp_min: float = 1.0,
        lambda_max: float = 0.5,
        lambda_min: float = 0.1,
        feature_loss_weight: float = 0.05,
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
        self.teacher_model = teacher_model
        self.if_use_entropy = if_use_entropy
        self._kl_loss: Optional[float] = None
        self._ce_loss: Optional[float] = None
        self._accuracy: Optional[float] = None
        self._lambda: float = lambda_max
        self.temp: float = temp_max
        self.temp_max = temp_max
        self.temp_min = temp_min
        self.lambda_max = lambda_max
        self.lambda_min = lambda_min
        self.feature_loss_weight = feature_loss_weight
        self._feature_loss: Optional[float] = None

        # 生成日志
        self.proc = None
        self._gen_log_path: Optional[str] = None
        self._gen_step_counter: int = 0

        # 中间特征投影层（学生 896 → 教师 2048）
        self._proj = None

    # ------------------------------------------------------------------
    # 检查点
    # ------------------------------------------------------------------

    def _eval_per_subject(self, max_per_subject: int = 20, log_prefix: str = "Eval"):
        """按学科评估 token 级准确率（与训练指标一致），仅做前向传播，不生成。"""
        if self.eval_dataset is None:
            return {}
        from collections import defaultdict
        import torch

        device = next(self.model.parameters()).device
        subj_correct = defaultdict(int)
        subj_total = defaultdict(int)

        for i in range(len(self.eval_dataset)):
            sample = self.eval_dataset[i]
            subj = sample.get("subject", "未知")
            if subj_total[subj] >= max_per_subject:
                continue

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

            correct = (preds == shift_labels) & valid
            subj_correct[subj] += correct.sum().item()
            subj_total[subj] += valid.sum().item()

        result = {}
        for subj in sorted(subj_total.keys()):
            total = subj_total[subj]
            correct = subj_correct[subj]
            acc = correct / total if total > 0 else 0
            result[subj] = round(acc, 4)

        parts = "  ".join(f"{s}:{v:.1%}" for s, v in result.items())
        overall_correct = sum(subj_correct.values())
        overall_total = sum(subj_total.values())
        overall_acc = overall_correct / overall_total if overall_total > 0 else 0
        logger.info("[%s] 总体:%.1f%%  %s", log_prefix, overall_acc * 100, parts)
        return result

    def on_epoch_end(self) -> None:
        super().on_epoch_end()
        from config_shared import PROJECT_ROOT

        current_epoch = int(self.state.epoch) if self.state.epoch else 0
        mode = getattr(self, "mode", "sft" if self.teacher_model is None else "distillation")
        prefix_map = {"sft": "sft", "distillation": "distillation", "teacher-sft": "teacher"}
        prefix = prefix_map.get(mode, "sft")
        logger.info("Epoch %d/%d [%s] — saving model...", current_epoch, self.args.num_train_epochs, mode)

        # 按学科评估（每科 20 条）
        self._eval_per_subject(max_per_subject=30, log_prefix=f"Epoch{current_epoch}")

        # 保存 checkpoint（LoRA + optimizer/scheduler）
        ckpt_dir = os.path.join(PROJECT_ROOT, "outputs", "checkpoints", mode, f"checkpoint-{prefix}-v3-epoch_{current_epoch}")
        self.save_model(ckpt_dir)
        self.save_state()
        logger.info("Checkpoint saved to %s", ckpt_dir)

        # 合并保存完整模型
        saves_path = os.path.join(PROJECT_ROOT, "outputs", "saves")
        self.save_model(saves_path)

        merged = self.model.merge_and_unload()
        merged_dir = os.path.join(PROJECT_ROOT, "outputs", "merged", mode, f"merged-{prefix}-v3-epoch_{current_epoch}")
        merged.save_pretrained(merged_dir)
        logger.info("Merged model saved to %s", merged_dir)

        self.model = PeftModel.from_pretrained(self.model, saves_path, local_files_only=True)
        self.model = self.model.to(self.model.device)

    # ------------------------------------------------------------------
    # 调度
    # ------------------------------------------------------------------

    def _update_schedules(self) -> None:
        """余弦衰减温度 和 λ（SFT 模式下固定 temp=1, λ=0）。"""
        if self.teacher_model is None:
            self.temp = 1.0
            self._lambda = 0.0
            return

        progress = min(self.state.epoch / self.args.num_train_epochs, 1.0)
        coef = 0.5 * (1.0 + math.cos(math.pi * progress))

        self.temp = self.temp_min + (self.temp_max - self.temp_min) * coef
        self._lambda = self.lambda_min + (self.lambda_max - self.lambda_min) * coef

    # ------------------------------------------------------------------
    # 生成日志（每 50 步一次，避免拖慢训练）
    # ------------------------------------------------------------------

    def _log_generations(self) -> None:
        if self.proc is None or self.eval_dataset is None:
            return
        if self.teacher_model is None:
            return
        if self._gen_log_path is None:
            from config_shared import PROJECT_ROOT
            ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            os.makedirs(os.path.join(PROJECT_ROOT, "outputs"), exist_ok=True)
            self._gen_log_path = os.path.join(PROJECT_ROOT, "outputs", f"training_generations_{ts}.jsonl")

        sample = self.eval_dataset[min(len(self.eval_dataset) - 1, self.state.global_step % len(self.eval_dataset))]
        input_ids = sample["input_ids"].unsqueeze(0)
        attention_mask = sample["attention_mask"].unsqueeze(0)
        labels = [t for t in sample["labels"].tolist() if t != -100]

        prompt_ids = [t for t, l in zip(sample["input_ids"].tolist(), sample["labels"].tolist()) if l == -100]
        question = self.proc.decode(prompt_ids, skip_special_tokens=True)
        ground_truth = self.proc.decode(labels, skip_special_tokens=True)

        gen_kw = dict(max_new_tokens=512, temperature=0.7, do_sample=True,
                      pad_token_id=self.proc.pad_token_id, eos_token_id=self.proc.eos_token_id)

        device = next(self.teacher_model.parameters()).device
        with torch.no_grad():
            t_out = self.teacher_model.generate(input_ids.to(device), attention_mask=attention_mask.to(device), **gen_kw)
        t_ans = self.proc.decode(t_out[0], skip_special_tokens=True).replace(question, "").strip()
        t_ans = _trim_stop_strings(t_ans)

        device = next(self.model.parameters()).device
        with torch.no_grad():
            s_out = self.model.generate(input_ids.to(device), attention_mask=attention_mask.to(device), **gen_kw)
        s_ans = self.proc.decode(s_out[0], skip_special_tokens=True).replace(question, "").strip()
        s_ans = _trim_stop_strings(s_ans)

        with open(self._gen_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"step": self.state.global_step, "epoch": round(self.state.epoch, 2),
                                "question": question, "ground_truth": ground_truth,
                                "student": s_ans, "teacher": t_ans}, ensure_ascii=False) + "\n")

        logger.info("[GenLog] Step %d — GT:%d  student:%d  teacher:%d",
                     self.state.global_step, len(ground_truth), len(s_ans), len(t_ans))

    # ------------------------------------------------------------------
    # 损失计算
    # ------------------------------------------------------------------

    def _get_feature_loss(self, student_outputs, teacher_outputs) -> torch.Tensor:
        """中间特征蒸馏损失：匹配师生对应层的 hidden states（余弦相似度）。"""
        if self.feature_loss_weight <= 0 or self.teacher_model is None:
            return torch.tensor(0.0, device=student_outputs.logits.device)

        s_hidden = student_outputs.get("hidden_states")
        t_hidden = teacher_outputs.get("hidden_states")
        if s_hidden is None or t_hidden is None:
            return torch.tensor(0.0, device=student_outputs.logits.device)

        # 延迟创建投影层
        if self._proj is None:
            s_dim = s_hidden[0].shape[-1]
            t_dim = t_hidden[0].shape[-1]
            self._proj = nn.Linear(s_dim, t_dim, bias=False).to(s_hidden[0].device)

        losses = []
        for s_idx, t_idx in LAYER_MATCHES:
            if s_idx >= len(s_hidden) or t_idx >= len(t_hidden):
                continue
            s_h = self._proj(s_hidden[s_idx])
            t_h = t_hidden[t_idx].detach()
            # 余弦相似度损失：1 - cos(s, t)
            cos = F.cosine_similarity(s_h, t_h, dim=-1)
            losses.append((1.0 - cos).mean())

        return sum(losses) / len(losses) if losses else torch.tensor(0.0, device=s_hidden[0].device)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """知识蒸馏损失：KL + CE + 中间特征蒸馏。"""
        # 学生前向（打开 hidden states）
        outputs: CausalLMOutputWithPast = model(**inputs, output_hidden_states=True)

        if self.teacher_model is not None:
            device = outputs.logits.device
            teacher_inputs = {k: v.to(self.teacher_model.device) for k, v in inputs.items()}
            with torch.no_grad():
                teacher_outputs = self.teacher_model(**teacher_inputs, output_hidden_states=True)

            ce_loss = outputs.loss
            logits = outputs.logits
            teacher_logits = teacher_outputs.logits.to(device)

            if logits.shape[-1] != teacher_logits.shape[-1]:
                teacher_logits = teacher_logits[:, :, :logits.shape[-1]]

            labels = inputs["labels"]
            kl_loss = compute_skewed_fkl(
                logits[:, :-1, :], teacher_logits[:, :-1, :], labels[:, 1:],
                padding_id=-100, temp=self.temp, skew_lambda=0.1,
            ).mean()

            # 中间特征蒸馏
            feat_loss = self._get_feature_loss(outputs, teacher_outputs)
            self._feature_loss = feat_loss.detach().cpu().item()
        else:
            # SFT 模式：纯 CE
            ce_loss = outputs.loss
            kl_loss = torch.tensor(0.0, device=outputs.logits.device)
            self._feature_loss = 0.0

        # 准确率
        labels = inputs["labels"]
        logits_shifted = outputs.logits[:, :-1, :]
        valid_mask = labels[:, 1:] != -100
        accuracy = (logits_shifted.argmax(-1) == labels[:, 1:]) & valid_mask
        accuracy = accuracy.sum().float() / valid_mask.sum().float()

        self._kl_loss = kl_loss.detach().cpu().item() if isinstance(kl_loss, torch.Tensor) else 0.0
        self._ce_loss = ce_loss.detach().cpu().item()
        self._accuracy = accuracy.detach().cpu().item()

        if self.teacher_model is not None and self.if_use_entropy:
            loss_total = self._lambda * kl_loss + (1.0 - self._lambda) * ce_loss
        elif self.teacher_model is not None:
            loss_total = kl_loss
        else:
            loss_total = ce_loss

        if self._feature_loss and self._feature_loss > 0:
            loss_total = loss_total + self.feature_loss_weight * torch.tensor(self._feature_loss, device=loss_total.device)

        return (loss_total, outputs) if return_outputs else loss_total

    # ------------------------------------------------------------------
    # 日志
    # ------------------------------------------------------------------

    def log(self, logs: Dict[str, Any], start_time: Optional[float] = None) -> None:
        if self._ce_loss is not None:
            logs["ce_loss"] = self._ce_loss
        if self._accuracy is not None:
            logs["accuracy"] = self._accuracy

        # 蒸馏模式下才记录 KL、温度、λ、特征损失
        if self.teacher_model is not None:
            if self._kl_loss is not None:
                logs["kl_loss"] = self._kl_loss
            if self._feature_loss is not None:
                logs["feature_loss"] = self._feature_loss
            self._update_schedules()
            logs["temp"] = round(self.temp, 2)
            logs["lambda"] = round(self._lambda, 3)

        # 每 2 步轻量学科评估（每科 20 条）
        step = self.state.global_step
        if step > 0 and step % 2 == 0:
            try:
                per_subj = self._eval_per_subject(max_per_subject=20, log_prefix=f"Step{step}")
                for subj, loss in per_subj.items():
                    logs[f"loss_{subj}"] = loss
            except Exception as exc:
                logger.warning("学科评估失败: %s", exc)

        # 每步生成对比日志（蒸馏模式）
        if self.teacher_model is not None:
            try:
                self._log_generations()
            except Exception as exc:
                logger.warning("生成日志失败: %s", exc)

        super().log(logs, start_time)
