"""后端推理服务配置。"""

import os
import logging
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_shared import (  # noqa: F401, E402
    PROJECT_ROOT, MODEL_PATHS, DATA_PATHS, RESULT_PATHS,
    SYSTEM_PROMPT, STOP_STRINGS,
)

logger = logging.getLogger(__name__)

FRONTEND_DIR = os.path.join(PROJECT_ROOT, "src", "frontend", "dist")

SERVER_CONFIG = {"host": "0.0.0.0", "port": 5000}

GENERATION_CONFIG = {
    "max_new_tokens": 2048, "temperature": 0.7, "top_p": 0.9,
    "repetition_penalty": 1.2, "do_sample": True,
}
