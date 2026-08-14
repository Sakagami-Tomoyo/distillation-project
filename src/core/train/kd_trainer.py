"""KDTrainer：知识蒸馏训练器（继承 SFTTrainer 的共享逻辑）。

蒸馏特有：skewed-FKL + CE + 中间特征蒸馏 + 动态 λ/T 调度 + 生成对比日志。
日志、通用评估、检查点保存等共享逻辑在 core.train.sft_trainer.SFTTrainer 里，
这里只补充蒸馏差异。
"""

import os
import json
import time
import math
import logging
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import TrainingArguments
from transformers.modeling_outputs import CausalLMOutputWithPast

from core.train.losses import compute_skewed_fkl
from core.train.sft_trainer import SFTTrainer

logger = logging.getLogger(__name__)

from config_shared import STOP_STRINGS

# 师生层匹配：学生 24 层，教师 36 层，等比对应 6 个位置
LAYER_MATCHES = [
    (4, 6), (8, 12), (12, 18), (16, 24), (20, 30), (23, 35),
]


def _trim_stop_strings(text: str) -> str:
    for s in STOP_STRINGS:
        idx = text.find(s)
        if idx != -1:
            return text[:idx].strip()
    return text


class KDTrainer(SFTTrainer):
    """知识蒸馏训练器：余弦衰减温度 + 余弦衰减 λ + 中间特征蒸馏。

    必须有教师模型；SFT（无教师）直接用 SFTTrainer。
    """

    def __init__(
        self,
        model=None,
        teacher_model=None,
        if_use_entropy: bool = True,
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
        # 蒸馏特有字段
        self.teacher_model = teacher_model
        self.if_use_entropy = if_use_entropy
        self._kl_loss: Optional[float] = None
        self._lambda: float = lambda_max
        self.temp: float = temp_max
        self.temp_max = temp_max
        self.temp_min = temp_min
        self.lambda_max = lambda_max
        self.lambda_min = lambda_min
        self.feature_loss_weight = feature_loss_weight
        self._feature_loss = 0.0

        # 中间特征投影层（学生 896 → 教师 2048）
        self._proj = None

        # 生成对比日志
        self._gen_log_path: Optional[str] = None
        self._gen_step_counter: int = 0

    # ------------------------------------------------------------------
    # 调度
    # ------------------------------------------------------------------

    def _update_schedules(self) -> None:
        """余弦衰减温度 和 λ"""
        progress = min(self.state.epoch / self.args.num_train_epochs, 1.0)
        coef = 0.5 * (1.0 + math.cos(math.pi * progress))

        self.temp = self.temp_min + (self.temp_max - self.temp_min) * coef
        self._lambda = self.lambda_min + (self.lambda_max - self.lambda_min) * coef

    # ------------------------------------------------------------------
    # 特征蒸馏
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

    # ------------------------------------------------------------------
    # 损失
    # ------------------------------------------------------------------

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """知识蒸馏损失：KL + CE + 中间特征蒸馏（必须有教师模型）。"""
        if self.teacher_model is None:
            raise ValueError(
                "KDTrainer 是蒸馏训练器，必须提供 teacher_model（请用 --distill 模式加载教师）；"
                "SFT（无教师）请用 SFTTrainer。"
            )

        # 学生前向（打开 hidden states）
        outputs: CausalLMOutputWithPast = model(**inputs, output_hidden_states=True)

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

        # 准确率
        logits_shifted = outputs.logits[:, :-1, :]
        valid_mask = labels[:, 1:] != -100
        accuracy = (logits_shifted.argmax(-1) == labels[:, 1:]) & valid_mask
        accuracy = accuracy.sum().float() / valid_mask.sum().float()

        self._kl_loss = kl_loss.detach().cpu().item() if isinstance(kl_loss, torch.Tensor) else 0.0
        self._ce_loss = ce_loss.detach().cpu().item()
        self._accuracy = accuracy.detach().cpu().item()

        if self.if_use_entropy:
            loss_total = self._lambda * kl_loss + (1.0 - self._lambda) * ce_loss
        else:
            loss_total = kl_loss

        if self._feature_loss and self._feature_loss > 0:
            loss_total = loss_total + self.feature_loss_weight * torch.tensor(self._feature_loss, device=loss_total.device)

        return (loss_total, outputs) if return_outputs else loss_total

    # ------------------------------------------------------------------
    # 生成对比日志（每 50 步一次，避免拖慢训练）
    # ------------------------------------------------------------------

    def _log_generations(self) -> None:
        if self.proc is None or self.eval_dataset is None:
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
    # 日志
    # ------------------------------------------------------------------

    def log(self, logs: Dict[str, Any], start_time: Optional[float] = None) -> None:
        # 蒸馏专属指标：KL + 特征损失 + 温度 + λ
        if self._kl_loss is not None:
            logs["kl_loss"] = self._kl_loss
        if self._feature_loss is not None:
            logs["feature_loss"] = self._feature_loss
        self._update_schedules()
        logs["temp"] = round(self.temp, 2)
        logs["lambda"] = round(self._lambda, 3)

        try:
            self._log_generations()
        except Exception as exc:
            logger.warning("生成日志失败: %s", exc)

        super().log(logs, start_time)
