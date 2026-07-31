"""共享的模型和分词器加载模块。

提供单一、一致的入口点，用于加载学生模型、教师模型、蒸馏（检查点）模型和分词器。
"""

import os
import logging
from typing import Optional, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from peft import PeftModel

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from config_shared import MODEL_PATHS, MODEL_LOAD_CONFIG, QUANTIZATION_CONFIG

logger = logging.getLogger(__name__)


class ModelLoader:
    """蒸馏流程中所有模型的集中加载器。"""

    def __init__(self, device: Optional[torch.device] = None):
        """初始化加载器。

        Args:
            device: 目标设备。如未提供则自动检测。
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self._tokenizer: Optional[PreTrainedTokenizer] = None

    # ------------------------------------------------------------------
    # 分词器
    # ------------------------------------------------------------------

    def load_tokenizer(self) -> PreTrainedTokenizer:
        """从学生模型路径加载分词器。首次调用后缓存。"""
        if self._tokenizer is not None:
            return self._tokenizer

        logger.info("Loading tokenizer from %s", MODEL_PATHS["student"])
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATHS["student"],
            trust_remote_code=MODEL_LOAD_CONFIG["trust_remote_code"],
            local_files_only=MODEL_LOAD_CONFIG["local_files_only"],
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self._tokenizer = tokenizer
        return tokenizer

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        """便捷属性：返回已缓存的分词器（首次访问时加载）。"""
        return self.load_tokenizer()

    # ------------------------------------------------------------------
    # 学生模型（0.5B，全精度）
    # ------------------------------------------------------------------

    def load_student(self) -> PreTrainedModel:
        """加载原始学生模型（Qwen2.5-0.5B-Instruct）。"""
        logger.info("Loading student model from %s", MODEL_PATHS["student"])
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATHS["student"],
            dtype=getattr(torch, MODEL_LOAD_CONFIG["dtype"]),
            device_map=MODEL_LOAD_CONFIG["device_map"],
            local_files_only=MODEL_LOAD_CONFIG["local_files_only"],
            trust_remote_code=MODEL_LOAD_CONFIG["trust_remote_code"],
        )
        model.eval()
        return model

    # ------------------------------------------------------------------
    # 教师模型（3B，4 位量化）
    # ------------------------------------------------------------------

    def load_teacher(self) -> PreTrainedModel:
        """使用 4 位量化加载教师模型（Qwen2.5-3B-Instruct）。"""
        logger.info("Loading teacher model from %s (4-bit quantized)", MODEL_PATHS["teacher"])
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=QUANTIZATION_CONFIG["load_in_4bit"],
            bnb_4bit_compute_dtype=getattr(torch, QUANTIZATION_CONFIG["bnb_4bit_compute_dtype"]),
            bnb_4bit_use_double_quant=QUANTIZATION_CONFIG["bnb_4bit_use_double_quant"],
            bnb_4bit_quant_type=QUANTIZATION_CONFIG["bnb_4bit_quant_type"],
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATHS["teacher"],
            dtype=getattr(torch, MODEL_LOAD_CONFIG["dtype"]),
            quantization_config=bnb_config,
            device_map=MODEL_LOAD_CONFIG["device_map"],
            local_files_only=MODEL_LOAD_CONFIG["local_files_only"],
            trust_remote_code=MODEL_LOAD_CONFIG["trust_remote_code"],
        )
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        return model

    # ------------------------------------------------------------------
    # 蒸馏 / 检查点模型
    # ------------------------------------------------------------------

    def load_distilled(
        self,
        checkpoint_path: Optional[str] = None,
        merge: bool = True,
    ) -> Optional[PreTrainedModel]:
        """加载蒸馏检查点（LoRA 适配器与学生模型合并）。

        Args:
            checkpoint_path: 检查点目录路径。默认使用配置中的路径。
            merge: 若为 True，合并 LoRA 适配器并返回合并后的模型。

        Returns:
            蒸馏后的模型，若检查点不存在则返回 None。
        """
        if checkpoint_path is None:
            checkpoint_path = MODEL_PATHS["distilled"]

        if not os.path.exists(checkpoint_path):
            logger.warning("Distilled checkpoint not found at %s", checkpoint_path)
            return None

        logger.info("Loading distilled model from %s", checkpoint_path)

        # 首先加载基础学生模型
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATHS["student"],
            dtype=getattr(torch, MODEL_LOAD_CONFIG["dtype"]),
            device_map=MODEL_LOAD_CONFIG["device_map"],
            local_files_only=MODEL_LOAD_CONFIG["local_files_only"],
            trust_remote_code=MODEL_LOAD_CONFIG["trust_remote_code"],
        )

        # 加载 LoRA 适配器（不传 device_map，base_model 已在正确设备上）
        peft_model = PeftModel.from_pretrained(
            base_model,
            checkpoint_path,
            local_files_only=MODEL_LOAD_CONFIG["local_files_only"],
        )

        if merge:
            model = peft_model.merge_and_unload()
            model.eval()
            return model

        peft_model.eval()
        return peft_model

    def load_merged(self, merged_path: Optional[str] = None) -> PreTrainedModel:
        """加载已合并的模型（例如训练后通过 merge_and_unload 保存的模型）。"""
        if merged_path is None:
            merged_path = MODEL_PATHS["merged"]

        logger.info("Loading merged model from %s", merged_path)
        model = AutoModelForCausalLM.from_pretrained(
            merged_path,
            dtype=getattr(torch, MODEL_LOAD_CONFIG["dtype"]),
            device_map=MODEL_LOAD_CONFIG["device_map"],
            local_files_only=MODEL_LOAD_CONFIG["local_files_only"],
            trust_remote_code=MODEL_LOAD_CONFIG["trust_remote_code"],
        )
        model.eval()
        return model

    # ------------------------------------------------------------------
    # 便捷方法：一次性加载全部
    # ------------------------------------------------------------------

    def load_all(
        self,
        include_distilled: bool = True,
    ) -> Tuple[PreTrainedModel, PreTrainedModel, Optional[PreTrainedModel], PreTrainedTokenizer]:
        """一次性加载全部三个模型和分词器。

        Returns:
            (学生模型, 教师模型, 蒸馏模型或None, 分词器)
        """
        tokenizer = self.load_tokenizer()
        student = self.load_student()
        teacher = self.load_teacher()
        distilled = self.load_distilled() if include_distilled else None
        return student, teacher, distilled, tokenizer
