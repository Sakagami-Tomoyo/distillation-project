"""高考题模型对比的 FastAPI 应用。

由 main.py 启动。服务器加载全部三个模型（学生、教师、蒸馏后），
并托管 React 前端，用于并排对比各模型的回答。
"""

import os
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse

from config import SERVER_CONFIG, FRONTEND_DIR
from config_shared import HF_TOKEN  # noqa: F401
from model_manager import ModelManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 应用和模型管理器
# ---------------------------------------------------------------------------

app = FastAPI(
    title="高考题模型对比",
    description="对比教师、学生、蒸馏后模型的回答",
)

model_manager = ModelManager()

# ---------------------------------------------------------------------------
# 静态文件（前端）
# ---------------------------------------------------------------------------

if os.path.exists(FRONTEND_DIR):
    assets_dir = os.path.join(FRONTEND_DIR, "assets")
    if os.path.exists(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")
    logger.info("Frontend directory: %s", FRONTEND_DIR)
else:
    logger.info("前端尚未构建。请运行: cd frontend && npm run build")

# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------


@app.get("/api/health")
async def health_check():
    """健康检查端点。返回模型可用性。"""
    return {
        "status": "ok",
        "models": model_manager.model_status(),
    }


@app.get("/api/models")
async def get_models():
    """返回当前已加载的模型列表。"""
    return model_manager.model_status()


@app.get("/api/checkpoints")
async def get_checkpoints():
    """返回 results/ 中所有可用的检查点。"""
    return {"checkpoints": model_manager.list_checkpoints()}


@app.post("/api/generate")
async def api_generate(request: Request):
    """使用选定的模型为给定题目生成回答。

    请求体：
        {"question": "...", "models": ["student", "teacher", "distilled"], "checkpoint": "checkpoint-1068"}
    """
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="请求体必须是有效的JSON")

    question = data.get("question", "")
    models = data.get("models", [])
    checkpoint = data.get("checkpoint")

    if not question.strip():
        raise HTTPException(status_code=400, detail="请输入题目")
    if not models:
        raise HTTPException(status_code=400, detail="请至少选择一个模型")

    valid_names = {"student", "teacher", "distilled"}
    for name in models:
        if name not in valid_names:
            raise HTTPException(status_code=400, detail=f"无效的模型名称: {name}")

    results = model_manager.generate_multi(question, models, checkpoint=checkpoint)

    if not results:
        raise HTTPException(status_code=500, detail="没有可用的模型，请先加载模型")

    return {"results": results}


@app.post("/api/generate/stream")
async def api_generate_stream(request: Request):
    """流式生成：实时返回每个模型生成的 token。

    请求体：
        {"question": "...", "models": ["student", "teacher", "distilled"], "checkpoint": "checkpoint-1068"}

    返回 SSE 流，事件格式：
        data: {"model": "student", "token": "这是"}
        data: {"model": "student", "done": true}
        data: {"all_done": true}
    """
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="请求体必须是有效的JSON")

    question = data.get("question", "")
    models = data.get("models", [])
    checkpoint = data.get("checkpoint")

    if not question.strip():
        raise HTTPException(status_code=400, detail="请输入题目")
    if not models:
        raise HTTPException(status_code=400, detail="请至少选择一个模型")

    valid_names = {"student", "teacher", "distilled"}
    for name in models:
        if name in valid_names or name.startswith("checkpoint:"):
            continue
        raise HTTPException(status_code=400, detail=f"无效的模型名称: {name}")

    temperature = data.get("temperature", 0.7)
    seed = data.get("seed")

    return StreamingResponse(
        model_manager.generate_stream(question, models, temperature=temperature, seed=seed),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/")
async def root():
    """托管 React 前端（如果尚未构建则显示回退消息）。"""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "请先构建前端: cd frontend && npm run build"}

# ---------------------------------------------------------------------------
# 启动
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    """服务器启动时加载模型。"""
    logger.info("CUDA available: %s", __import__("torch").cuda.is_available())
    logger.info("Loading models...")
    model_manager.load_all()
    logger.info("服务器就绪。")
