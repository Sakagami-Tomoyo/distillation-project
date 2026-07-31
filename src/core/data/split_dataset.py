"""将原始高考数据分割为训练/测试 JSONL 文件。

使用 src.data 中的预处理模块。
"""

import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from preprocessing import build_splits, save_splits
from core.train.config import DATASET_CONFIG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("开始构建训练/测试分割（训练: %s, 测试: %s）",
                DATASET_CONFIG["train_years"], DATASET_CONFIG["test_years"])

    train_data, test_data, filtered_train, filtered_test = build_splits()

    logger.info("训练集: %d 样本（过滤 %d）", len(train_data), filtered_train)
    logger.info("测试集: %d 样本（过滤 %d）", len(test_data), filtered_test)

    save_splits(train_data, test_data)
    logger.info("完成。")
