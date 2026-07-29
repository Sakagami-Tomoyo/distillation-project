"""交互式模型对比工具。

用户可以选择对比学生模型、教师模型、蒸馏模型中的任意一个或多个。
启动后先选择模型，再进入问答交互循环。

用法:
    python scripts/compare_models.py [--checkpoint PATH]
"""

import os
import sys
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.loader import ModelLoader
from inference.generator import generate_response, make_chat_prompt
from config.settings import MODEL_PATHS, GENERATION_CONFIG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---- 模型配置 ----
MODEL_INFO = {
    "1": {
        "key": "student",
        "name": "学生模型 (Qwen2.5-0.5B-Instruct)",
        "icon": "📚",
    },
    "2": {
        "key": "teacher",
        "name": "教师模型 (Qwen2.5-3B-Instruct)",
        "icon": "👨‍🏫",
    },
    "3": {
        "key": "distilled",
        "name": "蒸馏模型",
        "icon": "✨",
    },
}


def _find_latest_checkpoint():
    """在 results 目录中查找最新的 checkpoint。"""
    results_parent = os.path.dirname(MODEL_PATHS["distilled"])
    if not os.path.exists(results_parent):
        return None

    checkpoints = []
    for f in os.listdir(results_parent):
        full = os.path.join(results_parent, f)
        if os.path.isdir(full) and f.startswith("checkpoint"):
            if os.path.exists(os.path.join(full, "adapter_config.json")):
                checkpoints.append(f)

    if not checkpoints:
        return None

    checkpoints.sort(
        key=lambda x: os.path.getmtime(os.path.join(results_parent, x)),
        reverse=True,
    )
    return os.path.join(results_parent, checkpoints[0])


def _select_models() -> list[str]:
    """交互式选择要对比的模型。返回选中的模型 key 列表。"""
    print("\n" + "=" * 60)
    print("          📝 模型对比工具")
    print("=" * 60)
    print("\n可选模型：")
    for num, info in MODEL_INFO.items():
        print(f"  [{num}] {info['icon']} {info['name']}")
    print(f"  [0] 🚀 全部对比")
    print(f"  [q] ❌ 退出")
    print()

    while True:
        choice = input("请选择要对比的模型 (如: 1,2 表示选前两个): ").strip()

        if choice.lower() in ("q", "quit", "exit", "退出"):
            return []

        if choice == "0":
            return [info["key"] for info in MODEL_INFO.values()]

        # 解析用户输入：支持 "1,2" 或 "1 2" 或 "12"
        selected = []
        # 替换中文逗号、空格
        cleaned = choice.replace("，", ",").replace(" ", ",")
        parts = cleaned.split(",")

        valid = True
        for part in parts:
            part = part.strip()
            if part in MODEL_INFO:
                selected.append(MODEL_INFO[part]["key"])
            elif part:
                # 如果是连续数字如 "12"，拆开处理
                valid = False
                for ch in part:
                    if ch in MODEL_INFO:
                        selected.append(MODEL_INFO[ch]["key"])
                    else:
                        valid = False
                        break

        # 去重
        selected = list(dict.fromkeys(selected))

        if not selected:
            print("⚠ 未识别到有效的模型编号，请重新输入（如: 1,2,3 或 0 选全部）")
            continue

        names = [MODEL_INFO[k]["name"] for k, v in MODEL_INFO.items()
                 if v["key"] in selected for k in [k]][:len(selected)]
        # 用另一种方式获取名字
        name_map = {v["key"]: v["name"] for v in MODEL_INFO.values()}
        selected_names = [name_map[k] for k in selected]

        print(f"\n已选择: {', '.join(selected_names)}")
        confirm = input("确认选择？[Y/n]: ").strip().lower()
        if confirm in ("", "y", "yes", "是"):
            return selected
        print("请重新选择。")


def main(checkpoint_path=None):
    """主入口：选择模型 → 加载模型 → 交互问答。"""
    loader = ModelLoader()
    tokenizer = loader.load_tokenizer()

    # ---- 第一步：交互式选择模型 ----
    selected_keys = _select_models()
    if not selected_keys:
        print("已退出。")
        return

    # ---- 第二步：按需加载模型 ----
    print("\n" + "-" * 60)
    print("正在加载模型...")
    print("-" * 60)

    loaded_models: dict[str, object] = {}  # key → model

    for key in selected_keys:
        if key == "student":
            logger.info("加载学生模型...")
            loaded_models["student"] = loader.load_student()
            print("  ✅ 学生模型加载完成")
        elif key == "teacher":
            logger.info("加载教师模型...")
            loaded_models["teacher"] = loader.load_teacher()
            print("  ✅ 教师模型加载完成")
        elif key == "distilled":
            logger.info("加载蒸馏模型...")
            if checkpoint_path is None:
                checkpoint_path = _find_latest_checkpoint()
                if checkpoint_path:
                    logger.info("使用最新 checkpoint: %s", checkpoint_path)
            if checkpoint_path is None:
                print("  ⚠ 未找到有效的蒸馏模型 checkpoint，跳过")
                continue
            distilled = loader.load_distilled(checkpoint_path)
            if distilled is not None:
                loaded_models["distilled"] = distilled
                print("  ✅ 蒸馏模型加载完成")
            else:
                print("  ⚠ 蒸馏模型加载失败，跳过")

    if not loaded_models:
        print("❌ 没有可用的模型，退出。")
        return

    print("\n" + "=" * 60)
    print("模型加载完成！输入问题开始对比 (输入 quit 退出)")
    print("=" * 60)

    gen_config = dict(GENERATION_CONFIG)
    gen_config["max_new_tokens"] = 4096

    # ---- 第三步：问答交互循环 ----
    while True:
        try:
            user_input = input("\n用户: ")
        except (EOFError, KeyboardInterrupt):
            print("\n对话结束。")
            break

        if user_input.lower() in ("quit", "exit", "退出"):
            print("对话结束。")
            break

        if not user_input.strip():
            continue

        prompt = make_chat_prompt(tokenizer, user_input)

        print("\n" + "=" * 70)
        print(f"问题: {user_input}")

        # 按固定顺序输出
        for display_order_key in ["student", "teacher", "distilled"]:
            if display_order_key not in loaded_models:
                continue

            model = loaded_models[display_order_key]
            for num, info in MODEL_INFO.items():
                if info["key"] == display_order_key:
                    print(f"\n{info['icon']} {info['name']}:")
                    break
            print("-" * 70)

            try:
                response = generate_response(
                    model, tokenizer, prompt, generation_config=gen_config,
                )
                print(response)
            except Exception as exc:
                logger.error("生成失败: %s", exc)
                print(f"❌ 生成失败: {exc}")

        print("\n" + "=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="交互式模型对比工具")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="蒸馏模型 checkpoint 路径",
    )
    args = parser.parse_args()
    main(args.checkpoint)
