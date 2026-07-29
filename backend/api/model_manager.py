"""API 服务器的模型生命周期管理。

将模型加载、缓存和推理包装在单个类中。
支持普通生成和流式生成两种模式。
支持从多个检查点中选择蒸馏模型。
"""

import os
import logging
import asyncio
import threading
from typing import Dict, Optional, List, AsyncIterator

import torch
from transformers import TextIteratorStreamer

from models.loader import ModelLoader
from inference.generator import make_chat_prompt
from config.settings import GENERATION_CONFIG, PROJECT_ROOT

logger = logging.getLogger(__name__)


class ModelManager:
    """管理学生、教师和蒸馏模型的生命周期。

    模型在首次访问时延迟加载。在启动时调用 load_all() 来预先加载所有模型，
    或让各个端点自行触发加载。
    蒸馏模型支持从多个检查点中动态选择。
    """

    def __init__(self) -> None:
        self._loader = ModelLoader()
        self._student = None
        self._teacher = None
        self._distilled = None
        self._current_checkpoint: Optional[str] = None
        self._tokenizer = None
        self._loaded = False

    # ------------------------------------------------------------------
    # 加载
    # ------------------------------------------------------------------

    def load_all(self) -> None:
        """预先加载学生、教师和分词器。蒸馏模型按需加载。"""
        if self._loaded:
            return

        logger.info("加载所有模型...")
        self._tokenizer = self._loader.load_tokenizer()
        self._student = self._loader.load_student()
        self._teacher = self._loader.load_teacher()
        self._loaded = True
        logger.info("基础模型加载完成（蒸馏模型按需加载）。")

    # ------------------------------------------------------------------
    # 属性（延迟访问）
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
    def distilled(self):
        return self._distilled

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = self._loader.load_tokenizer()
        return self._tokenizer

    # ------------------------------------------------------------------
    # 检查点管理
    # ------------------------------------------------------------------

    def list_checkpoints(self) -> List[Dict[str, str]]:
        """扫描 results/ 目录，返回所有可用检查点。"""
        results_dir = os.path.join(PROJECT_ROOT, "results")
        if not os.path.exists(results_dir):
            return []

        checkpoints = []
        for name in os.listdir(results_dir):
            full = os.path.join(results_dir, name)
            if not os.path.isdir(full):
                continue
            adapter_file = os.path.join(full, "adapter_config.json")
            if os.path.exists(adapter_file):
                checkpoints.append({
                    "name": name,
                    "path": full,
                })

        # 按修改时间倒序（最新的在前）
        checkpoints.sort(key=lambda c: os.path.getmtime(c["path"]), reverse=True)
        return checkpoints

    def load_checkpoint(self, checkpoint_name: str) -> None:
        """加载指定的检查点作为蒸馏模型。"""
        if self._current_checkpoint == checkpoint_name and self._distilled is not None:
            return

        checkpoint_path = os.path.join(PROJECT_ROOT, "results", checkpoint_name)
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"检查点不存在: {checkpoint_name}")

        logger.info("加载蒸馏模型检查点: %s", checkpoint_name)
        # 释放旧模型
        if self._distilled is not None:
            del self._distilled
            self._distilled = None
            torch.cuda.empty_cache()
        self._distilled = self._loader.load_distilled(checkpoint_path, merge=True)
        self._current_checkpoint = checkpoint_name

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------

    def model_status(self) -> Dict:
        """返回当前可用的模型列表和检查点信息。"""
        return {
            "student": self._student is not None,
            "teacher": self._teacher is not None,
            "distilled": self._distilled is not None or len(self.list_checkpoints()) > 0,
            "current_checkpoint": self._current_checkpoint,
        }

    # ------------------------------------------------------------------
    # 普通推理
    # ------------------------------------------------------------------

    def generate(
        self,
        model_name: str,
        question: str,
        checkpoint: Optional[str] = None,
    ) -> str:
        """使用指定模型生成回复。"""
        from inference.generator import generate_response

        # 如果请求蒸馏模型，自动加载检查点
        if model_name == "distilled":
            if not checkpoint:
                cps = self.list_checkpoints()
                if cps:
                    checkpoint = cps[0]["name"]
            if checkpoint:
                self.load_checkpoint(checkpoint)

        model_map = {
            "student": self.student,
            "teacher": self.teacher,
            "distilled": self.distilled,
        }
        model = model_map.get(model_name)
        if model is None:
            raise ValueError(f"模型 '{model_name}' 不可用")

        prompt = make_chat_prompt(self.tokenizer, question)
        return generate_response(model, self.tokenizer, prompt)

    def generate_multi(
        self,
        question: str,
        model_names: List[str],
        checkpoint: Optional[str] = None,
    ) -> Dict[str, str]:
        """使用多个模型生成回复。"""
        results: Dict[str, str] = {}
        for name in model_names:
            try:
                results[name] = self.generate(name, question, checkpoint=checkpoint)
            except Exception as exc:
                logger.error("生成失败 %s: %s", name, exc)
                results[name] = f"生成失败: {exc}"
        return results

    # ------------------------------------------------------------------
    # 流式推理
    # ------------------------------------------------------------------

    def _get_model(self, model_name: str, checkpoint: Optional[str] = None):
        """根据名称获取模型。"""
        if model_name == "distilled":
            # 如果没有指定检查点，自动选第一个可用的
            if not checkpoint:
                cps = self.list_checkpoints()
                if cps:
                    checkpoint = cps[0]["name"]
            if checkpoint:
                self.load_checkpoint(checkpoint)

        model_map = {
            "student": self.student,
            "teacher": self.teacher,
            "distilled": self.distilled,
        }
        model = model_map.get(model_name)
        if model is None:
            raise ValueError(f"模型 '{model_name}' 不可用")
        return model

    async def generate_stream(
        self,
        question: str,
        model_names: List[str],
        temperature: float = 0.7,
        seed: Optional[int] = None,
        checkpoint: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """流式生成：实时产出 token 的 SSE 事件。

        每个模型在独立线程中生成，token 通过队列汇集，
        以 JSON 格式输出 SSE 事件。

        参数：
            question: 用户输入的题目
            model_names: 要使用的模型列表
            temperature: 采样温度（0.1-2.0），越高越随机。默认 0.7
            seed: 随机种子。不传则每次结果不同；传固定值可复现

        事件格式：
            {"model": "student", "token": "这是"}
            {"model": "student", "done": true}
            {"model": "teacher", "token": "答案"}
            ...
            {"all_done": true}
        """
        tokenizer = self.tokenizer
        prompt = make_chat_prompt(tokenizer, question)
        inputs = tokenizer(prompt, return_tensors="pt")

        queue: asyncio.Queue = asyncio.Queue()
        threads = []
        loop = asyncio.get_running_loop()

        gen_config = dict(GENERATION_CONFIG)
        gen_config["temperature"] = temperature
        if seed is not None:
            gen_config["seed"] = seed
        gen_config["pad_token_id"] = tokenizer.pad_token_id
        gen_config["eos_token_id"] = tokenizer.eos_token_id

        def _run_model(model_name: str):
            """在独立线程中运行单个模型的生成。"""
            try:
                model = self._get_model(model_name, checkpoint=checkpoint)
                streamer = TextIteratorStreamer(
                    tokenizer,
                    skip_prompt=True,
                    skip_special_tokens=True,
                )
                gen_kwargs = {**gen_config, "streamer": streamer}
                # 将 inputs 移到模型所在设备
                model_inputs = {k: v.to(model.device) for k, v in inputs.items()}

                def _generate():
                    with torch.no_grad():
                        model.generate(**model_inputs, **gen_kwargs)

                gen_thread = threading.Thread(target=_generate, daemon=True)
                gen_thread.start()

                # 从 streamer 读取 token 并放入队列
                for token in streamer:
                    cleaned = _clean_stream_token(token)
                    if cleaned:
                        loop.call_soon_threadsafe(
                            lambda t=cleaned, m=model_name: asyncio.ensure_future(
                                queue.put({"model": m, "token": t})
                            )
                        )

                # 该模型生成完毕
                loop.call_soon_threadsafe(
                    lambda m=model_name: asyncio.ensure_future(
                        queue.put({"model": m, "done": True})
                    )
                )
            except Exception as exc:
                logger.error("流式生成失败 %s: %s", model_name, exc)
                loop.call_soon_threadsafe(
                    lambda m=model_name, e=str(exc): asyncio.ensure_future(
                        queue.put({"model": m, "error": e, "done": True})
                    )
                )

        # 启动所有模型的生成线程
        for name in model_names:
            t = threading.Thread(target=_run_model, args=(name,), daemon=True)
            t.start()
            threads.append(t)

        # 统计已完成的模型
        done_count = 0
        total = len(model_names)

        # 从队列中读取并产出 SSE 事件
        while done_count < total:
            item = await queue.get()
            if item.get("done"):
                done_count += 1
            yield f"data: {_json_dumps(item)}\n\n"

        yield f"data: {_json_dumps({'all_done': True})}\n\n"

        # 等待所有线程结束
        for t in threads:
            t.join(timeout=10)


def _clean_stream_token(token: str) -> str:
    """清理流式 token，去掉 system/user/assistant 标签和无关内容。"""
    # 跳过包含角色标签的 token
    if token in ("system", "user", "assistant"):
        return ""
    # 跳过 system 前缀内容
    for skip_marker in [
        "You are Qwen",
        "You are a helpful",
        "Alibaba Cloud",
    ]:
        if skip_marker in token:
            return ""
    return token


def _json_dumps(obj: dict) -> str:
    """紧凑的 JSON 序列化（不依赖 json 模块的 import 开销）。"""
    import json
    return json.dumps(obj, ensure_ascii=False)
