"""API 服务器的模型生命周期管理。

将模型加载、缓存和推理包装在单个类中。
支持普通生成和流式生成两种模式。
支持从多个检查点中同时加载并对比蒸馏模型。
"""

import os
import logging
import asyncio
import threading
from typing import Dict, Optional, List, AsyncIterator

import torch
from transformers import TextIteratorStreamer

from loader import ModelLoader
from backend.generator import make_chat_prompt
from backend.config import GENERATION_CONFIG
from config_shared import STOP_STRINGS
from config_shared import PROJECT_ROOT

logger = logging.getLogger(__name__)


class ModelManager:
    """管理学生、教师和蒸馏模型的生命周期。

    模型在首次访问时延迟加载。蒸馏模型支持同时从多个检查点加载，
    方便对比不同训练阶段的输出效果。
    """

    def __init__(self) -> None:
        self._loader = ModelLoader()
        self._student = None
        self._teacher = None
        self._checkpoints: Dict[str, object] = {}  # name → model
        self._tokenizer = None
        self._loaded = False
        self._wenda = None
        self._wenda_checked = False

    # ------------------------------------------------------------------
    # 加载
    # ------------------------------------------------------------------

    def load_all(self) -> None:
        """预先加载学生、教师和分词器。检查点按需加载。"""
        if self._loaded:
            return

        logger.info("加载基础模型...")
        self._tokenizer = self._loader.load_tokenizer()
        self._student = self._loader.load_student()
        self._teacher = self._loader.load_teacher()
        self._loaded = True
        logger.info("基础模型加载完成（检查点按需加载）。")

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def student(self):
        if self._student is None:
            self._student = self._loader.load_student()
        return self._student

    @property
    def teacher(self):
        if self._teacher is None:
            self._teacher = self._loader.load_teacher()
        return self._teacher

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = self._loader.load_tokenizer()
        return self._tokenizer

    @property
    def wenda(self):
        """智能问答（wenda）微调模型；未训练或不存在时返回 None。"""
        if not self._wenda_checked:
            self._wenda_checked = True
            self._wenda = self._load_wenda_model()
        return self._wenda

    def _load_wenda_model(self):
        """加载最新的 wenda 合并模型（outputs/merged/wenda/ 下 mtime 最新的）。"""
        wenda_root = os.path.join(PROJECT_ROOT, "outputs", "merged", "wenda")
        if not os.path.isdir(wenda_root):
            return None
        candidates = [
            os.path.join(wenda_root, name)
            for name in os.listdir(wenda_root)
            if os.path.isdir(os.path.join(wenda_root, name))
            and os.path.exists(os.path.join(wenda_root, name, "config.json"))
        ]
        if not candidates:
            return None
        candidates.sort(key=os.path.getmtime, reverse=True)
        logger.info("加载 wenda 模型: %s", candidates[0])
        model = self._loader.load_merged(candidates[0])
        model.eval()
        return model

    # ------------------------------------------------------------------
    # 合并模型管理（从 merged/ 直接加载，无需 LoRA 合并，更快）
    # ------------------------------------------------------------------

    def list_checkpoints(self) -> List[Dict[str, str]]:
        """扫描 merged/ 目录，返回所有可用的合并模型。"""
        merged_root = os.path.join(PROJECT_ROOT, "outputs", "merged")
        if not os.path.exists(merged_root):
            return []

        checkpoints = []
        for category in os.listdir(merged_root):
            cat_dir = os.path.join(merged_root, category)
            if not os.path.isdir(cat_dir):
                continue
            for name in os.listdir(cat_dir):
                full = os.path.join(cat_dir, name)
                if not os.path.isdir(full):
                    continue
                if os.path.exists(os.path.join(full, "config.json")):
                    checkpoints.append({"name": f"{category}/{name}", "path": full})

        checkpoints.sort(key=lambda c: os.path.getmtime(c["path"]), reverse=True)
        return checkpoints

    def get_or_load_checkpoint(self, checkpoint_name: str):
        """获取或按需加载指定的合并模型（线程安全，带缓存）。

        checkpoint_name 格式: "sft/merged-sft-epoch_3" 或 "distillation/merged-distill-epoch_0"
        """
        if checkpoint_name in self._checkpoints:
            return self._checkpoints[checkpoint_name]

        if "/" not in checkpoint_name:
            for cat in ["sft", "distillation"]:
                candidate = os.path.join(PROJECT_ROOT, "outputs", "merged", cat, checkpoint_name)
                if os.path.exists(candidate):
                    model_path = candidate
                    break
            else:
                raise ValueError(f"合并模型不存在: {checkpoint_name}")
        else:
            model_path = os.path.join(PROJECT_ROOT, "outputs", "merged", checkpoint_name)
        if not os.path.exists(model_path):
            raise ValueError(f"合并模型不存在: {checkpoint_name}")

        if not hasattr(self, '_cp_lock'):
            self._cp_lock = threading.Lock()
        with self._cp_lock:
            if checkpoint_name in self._checkpoints:
                return self._checkpoints[checkpoint_name]

            logger.info("加载合并模型: %s", checkpoint_name)
            model = self._loader.load_merged(model_path)
            model.eval()
            self._checkpoints[checkpoint_name] = model
            logger.info("合并模型 %s 加载完成", checkpoint_name)
        return model

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------

    def model_status(self) -> Dict:
        """返回当前可用的模型列表和检查点信息。"""
        return {
            "student": self._student is not None,
            "teacher": self._teacher is not None,
            "checkpoints": self.list_checkpoints(),
        }

    # ------------------------------------------------------------------
    # 模型查找
    # ------------------------------------------------------------------

    def _get_model(self, model_name: str):
        """根据名称获取模型。

        支持：
          - "student" / "teacher" — 标准模型
          - "checkpoint:xxx" — 指定检查点
          - "distilled" — 最新检查点（兼容旧逻辑）
        """
        if model_name == "student":
            return self.student
        elif model_name == "teacher":
            return self.teacher
        elif model_name == "distilled":
            # 兼容旧逻辑：使用最新检查点
            cps = self.list_checkpoints()
            if not cps:
                raise ValueError("没有可用的检查点")
            return self.get_or_load_checkpoint(cps[0]["name"])
        elif model_name.startswith("checkpoint:"):
            cp_name = model_name.split(":", 1)[1]
            return self.get_or_load_checkpoint(cp_name)
        else:
            raise ValueError(f"未知模型: {model_name}")

    def get_display_name(self, model_name: str) -> str:
        """返回模型的显示名称。"""
        display_map = {
            "student": "学生模型",
            "teacher": "教师模型",
            "distilled": "蒸馏模型",
        }
        if model_name.startswith("checkpoint:"):
            return model_name.split(":", 1)[1]
        return display_map.get(model_name, model_name)

    # ------------------------------------------------------------------
    # 普通推理
    # ------------------------------------------------------------------

    def generate(self, model_name: str, question: str) -> str:
        """使用指定模型生成回复。"""
        from generator import generate_response

        model = self._get_model(model_name)
        prompt = make_chat_prompt(self.tokenizer, question)
        return generate_response(model, self.tokenizer, prompt)

    def generate_multi(self, question: str, model_names: List[str]) -> Dict[str, str]:
        """使用多个模型生成回复。"""
        results: Dict[str, str] = {}
        for name in model_names:
            try:
                results[name] = self.generate(name, question)
            except Exception as exc:
                logger.error("生成失败 %s: %s", name, exc)
                results[name] = f"生成失败: {exc}"
        return results

    # ------------------------------------------------------------------
    # 流式推理
    # ------------------------------------------------------------------

    async def generate_stream(
        self,
        question: str,
        model_names: List[str],
        temperature: float = 0.7,
        seed: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """顺序流式生成：一个模型一个模型地输出。

        避免 CPU 环境下多模型并行争抢资源，速度反而更快。

        SSE 事件格式：
            {"model": "student", "token": "这是"}
            {"model": "student", "done": true}
            {"all_done": true}
        """
        tokenizer = self.tokenizer
        prompt = make_chat_prompt(tokenizer, question)
        inputs = tokenizer(prompt, return_tensors="pt")
        loop = asyncio.get_running_loop()

        gen_config = dict(GENERATION_CONFIG)
        gen_config["temperature"] = temperature
        if seed is not None:
            gen_config["seed"] = seed
        gen_config["pad_token_id"] = tokenizer.pad_token_id
        gen_config["eos_token_id"] = tokenizer.eos_token_id

        for model_name in model_names:
            # 预加载模型
            try:
                model = self._get_model(model_name)
            except Exception as exc:
                logger.error("加载模型失败 %s: %s", model_name, exc)
                yield f"data: {_json_dumps({'model': model_name, 'error': str(exc), 'done': True})}\n\n"
                continue

            # 在本线程中用 streamer + queue 实现流式输出
            q: asyncio.Queue = asyncio.Queue()

            def _run():
                try:
                    streamer = TextIteratorStreamer(
                        tokenizer, skip_prompt=True, skip_special_tokens=True,
                    )
                    gen_kwargs = {**gen_config, "streamer": streamer}
                    model_inputs = {k: v.to(model.device) for k, v in inputs.items()}

                    t = threading.Thread(target=lambda: model.generate(**model_inputs, **gen_kwargs), daemon=True)
                    t.start()

                    accumulated = ""
                    for token in streamer:
                        cleaned = _clean_stream_token(token)
                        if cleaned:
                            accumulated += cleaned
                            loop.call_soon_threadsafe(
                                lambda tk=cleaned: asyncio.ensure_future(
                                    q.put({"model": model_name, "token": tk})
                                )
                            )
                            # 检测 stop strings，遇到立即停止
                            if any(s in accumulated for s in STOP_STRINGS):
                                break

                    loop.call_soon_threadsafe(
                        lambda: asyncio.ensure_future(
                            q.put({"model": model_name, "done": True})
                        )
                    )
                except Exception as exc:
                    logger.error("流式生成失败 %s: %s", model_name, exc)
                    loop.call_soon_threadsafe(
                        lambda e=str(exc): asyncio.ensure_future(
                            q.put({"model": model_name, "error": e, "done": True})
                        )
                    )

            threading.Thread(target=_run, daemon=True).start()

            # 等待当前模型完成
            while True:
                item = await q.get()
                is_done = item.get("done", False)
                yield f"data: {_json_dumps(item)}\n\n"
                if is_done:
                    break

        yield f"data: {_json_dumps({'all_done': True})}\n\n"


def _clean_stream_token(token: str) -> str:
    """清理流式 token。模型的 system prompt 设为"你是一位高考解题助手"，
    不会输出 Qwen 默认提示词，此过滤仅作兜底。"""
    if token in ("<|im_start|>", "<|im_end|>"):
        return ""
    for skip_marker in ["system", "user", "assistant", "human",
                        "You are Qwen", "Alibaba Cloud"]:
        if token.strip().startswith(skip_marker):
            return ""
    return token


def _json_dumps(obj: dict) -> str:
    """紧凑的 JSON 序列化。"""
    import json
    return json.dumps(obj, ensure_ascii=False)
