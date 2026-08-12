"""ROP-D 黑盒蒸馏流水线（独立于白盒蒸馏，互不干扰）。

Phase 1: 学生模型对每道题用 4 个不同温度各生成一个回答
Phase 2: DeepSeek-V4 生成 4 个标准答案变体 + 逐一批改 + 打分
Phase 3: 将批改结果转为 SFT 修正数据 → 存到 data/Example/

输出格式（Alpaca 风格，可直接用于 train.py --train-data）：
    {"instruction": "...", "input": "学生的错误回答", "output": "教师的完美答案"}
"""

import os, sys, json, time, logging, argparse, re
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config_shared import PROJECT_ROOT, DATA_PATHS, MODEL_PATHS, MODEL_LOAD_CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ====== 配置 ======
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"  # 可改为 deepseek-reasoner 或 deepseek-chat

STUDENT_GEN_COUNT = 4
TEACHER_GEN_COUNT = 4
STUDENT_TEMPS = [0.6, 0.8, 1.0, 1.2]
MAX_QUESTIONS = None

# 输出目录（独立于白盒蒸馏的 outputs/）
ROP_D_DIR = os.path.join(PROJECT_ROOT, "data", "Example")
ROP_D_CACHE = os.path.join(ROP_D_DIR, "cache")

# ====== 教师 Prompt（生成 4 个标准答案变体）======
TEACHER_VARIANT_PROMPT = """你是一位高考解题专家。请对下面这道题生成 4 个不同的标准答案变体。

要求：
1. 每个变体都必须推导正确、最终答案一致
2. 变体之间在推导角度、详略程度、表述风格上有明显差异（例如：一个详细版、一个简洁版、一个代数法、一个几何法）
3. 每个变体以 ---ANSWER_1---、---ANSWER_2--- 等标记开头
4. 严格按以下格式输出：
   答案: [选项]
   解析:
   核心思路: [一句话概括解题方向]
   关键推理步骤: [分步推导]
   最终结论: [确认答案并说明原因]

题目：{question}"""

# ====== 教师 Prompt（批改学生回答 + 打分）======
GRADING_PROMPT = """你是无情的阅卷机器。

下面是:
- 一道高考题
- 4 个标准答案（由另一位专家生成）
- 4 个学生回答（由一个 AI 模型生成）

你的任务：
逐一检查 4 个学生回答，对照标准答案，给出分数和扣分理由。

评分规则（严格执行）：
- 最终选项正确 AND 推导过程完整正确 → 1 分
- 推导过程正确但最终选项选错 → 0 分（致命错误）
- 推导方向正确但中途卡住/未完成 → 0.5 分
- 推导过程有严重逻辑错误 → 0 分
- 只有选项没有推导 → 0 分（不算回答）

题目：
{question}

标准答案：
{teacher_text}

学生回答：
{student_text}

请严格按以下格式输出（不要输出任何其他内容）：

---SCORES---
学生1: X/1 —— [扣分理由，一句话，必须具体指出哪里错了]
学生2: X/1 —— [扣分理由，一句话，必须具体指出哪里错了]
学生3: X/1 —— [扣分理由，一句话，必须具体指出哪里错了]
学生4: X/1 —— [扣分理由，一句话，必须具体指出哪里错了]
---TOTAL---
加权总分: X.X/4
---FEEDBACK---
[总体评语：学生的主要问题类型是什么（如选项混淆、推导跳跃、概念错误），应该如何针对性改进]"""

# ====== 修正数据 Prompt（用于 SFT instruction）======
CORRECTION_INSTRUCTION = """你是一位高考解题教练。下面有一道题，以及一个学生的回答。阅卷老师指出了这个回答的问题。请根据反馈，写出一个完全正确的解答。

题目：{question}

阅卷反馈：{reason}
{feedback}

请按照格式输出：
答案: [选项]
解析:
核心思路: [一句话]
关键推理步骤: [分步推导]
最终结论: [确认答案]"""


# ====== DeepSeek API ======

def call_deepseek(messages: List[Dict], temperature: float = 0.7, label: str = "") -> str:
    import requests
    logger.info("  🌐 调用 DeepSeek API %s ...", label)
    resp = requests.post(
        f"{DEEPSEEK_BASE_URL}/v1/chat/completions",
        headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                  "Content-Type": "application/json"},
        json={"model": DEEPSEEK_MODEL, "messages": messages,
              "temperature": temperature, "max_tokens": 4096},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ====== Phase 1: 学生生成 ======

def student_generate_answers(model, tokenizer, question: str, count: int = 4) -> List[str]:
    from backend.generator import generate_response

    answers = []
    for i in range(count):
        temp = STUDENT_TEMPS[i % len(STUDENT_TEMPS)]
        gen_cfg = {"temperature": temp, "max_new_tokens": 2048, "do_sample": True}
        ans = generate_response(model, tokenizer, question, generation_config=gen_cfg)
        # 只保留 assistant 回答，去掉 system/user 前缀（省 API 费）
        idx = ans.rfind("assistant\n")
        if idx != -1:
            ans = ans[idx + len("assistant\n"):].strip()
        answers.append(ans)
        logger.info("  学生 %d/%d (T=%.1f) —— %d 字", i+1, count, temp, len(ans))
    return answers


# ====== Phase 2: 教师生成 + 批改 ======

def teacher_generate_variants(question: str, count: int = 4) -> List[str]:
    prompt = TEACHER_VARIANT_PROMPT.format(question=question)
    resp = call_deepseek([{"role": "user", "content": prompt}], temperature=0.8,
                         label="生成标准答案变体")

    variants = []
    for i in range(1, count + 1):
        marker = f"---ANSWER_{i}---"
        parts = resp.split(marker)
        if len(parts) > 1:
            body = parts[1]
            for j in range(1, count + 1):
                nm = f"---ANSWER_{j}---"
                if nm in body and nm != marker:
                    body = body.split(nm)[0]
                    break
            variants.append(body.strip())
        else:
            variants.append("")
    if not any(variants):
        variants = [resp]
    return variants


def _strip_preamble(text: str) -> str:
    """去掉 system/user 前缀，只保留 assistant 部分。降费 40%。"""
    idx = text.rfind("assistant\n")
    return text[idx + len("assistant\n"):].strip() if idx != -1 else text.strip()


def teacher_grade(question: str, student_answers: List[str],
                  teacher_answers: List[str]) -> Dict:
    student_text = "\n\n".join(
        f"=== 学生回答 {i+1} ===\n{_strip_preamble(a)}"
        for i, a in enumerate(student_answers))
    teacher_text = "\n\n".join(
        f"=== 标准答案 {i+1} ===\n{a}" for i, a in enumerate(teacher_answers))

    prompt = GRADING_PROMPT.format(
        question=question, teacher_text=teacher_text, student_text=student_text)
    resp = call_deepseek([{"role": "user", "content": prompt}], temperature=0.3,
                         label="批改打分")
    return _parse_grading(resp)


def _parse_grading(text: str) -> Dict:
    result = {"raw": text, "scores": [], "total": "", "feedback": ""}

    scores_section = text
    if "---SCORES---" in text:
        scores_section = text.split("---SCORES---")[1]
        if "---TOTAL---" in scores_section:
            scores_section = scores_section.split("---TOTAL---")[0]

    for line in scores_section.strip().split("\n"):
        m = re.search(r"学生(\d).*?([\d.]+)/([\d.]+)", line)
        if m:
            result["scores"].append({
                "student": int(m.group(1)),
                "score": float(m.group(2)),
                "max": float(m.group(3)),
                "reason": line.split("——")[-1].strip() if "——" in line else line.strip(),
            })

    total_m = re.search(r"加权总分.*?([\d.]+)/([\d.]+)", text)
    if total_m:
        result["total"] = f"{total_m.group(1)}/{total_m.group(2)}"

    if "---FEEDBACK---" in text:
        result["feedback"] = text.split("---FEEDBACK---")[1].strip()

    return result


# ====== Phase 3: 构建 SFT 修正数据 ======

def build_correction_data(question: str, student_answers: List[str],
                          teacher_answers: List[str], grading: Dict) -> List[Dict]:
    records = []
    scores = grading.get("scores", [])
    feedback = grading.get("feedback", "")

    for s in scores:
        idx = s["student"] - 1
        if idx < 0 or idx >= len(student_answers):
            continue
        if s["score"] >= 1.0:
            continue

        best_teacher = teacher_answers[0] if teacher_answers else ""
        reason = s.get("reason", "")

        instruction = CORRECTION_INSTRUCTION.format(
            question=question, reason=reason, feedback=feedback)

        records.append({
            "instruction": instruction,
            "input": student_answers[idx],
            "output": best_teacher,
            "meta": {"student_score": s["score"], "reason": reason},
        })

    return records


# ====== 主流程 ======

def main():
    parser = argparse.ArgumentParser(description="ROP-D 黑盒蒸馏（独立于白盒蒸馏）")
    parser.add_argument("--base-model", type=str, default=None,
                        help="学生模型路径（默认用 student）")
    parser.add_argument("--max", type=int, default=MAX_QUESTIONS,
                        help="最多处理题数（默认全部）")
    parser.add_argument("--skip-student", action="store_true",
                        help="跳过学生生成（读缓存）")
    parser.add_argument("--skip-teacher", action="store_true",
                        help="跳过教师批改（读缓存）")
    parser.add_argument("--only-teacher", action="store_true",
                        help="只运行教师生成变体+批改（需已有学生缓存）")
    parser.add_argument("--test", action="store_true",
                        help="使用测试集（2021-2022年题）而非训练集")
    parser.add_argument("--output", type=str, default=None,
                        help="输出文件名（默认 data/Example/ropd_data_{timestamp}.jsonl）")
    args = parser.parse_args()

    if not DEEPSEEK_API_KEY:
        logger.error("请设置 DEEPSEEK_API_KEY 环境变量")
        return

    os.makedirs(ROP_D_DIR, exist_ok=True)
    os.makedirs(ROP_D_CACHE, exist_ok=True)

    # ====== 加载学生模型 ======
    student_model = None
    tokenizer = None
    if args.only_teacher:
        args.skip_student = True
        args.skip_teacher = False
    if not args.skip_student:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_path = args.base_model or MODEL_PATHS["student"]
        if not os.path.isabs(model_path):
            model_path = os.path.join(PROJECT_ROOT, model_path)
        model_path = os.path.abspath(model_path)
        logger.info("加载学生模型: %s", model_path)
        student_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=getattr(torch, MODEL_LOAD_CONFIG["dtype"]),
            device_map=MODEL_LOAD_CONFIG["device_map"],
            local_files_only=MODEL_LOAD_CONFIG["local_files_only"],
            trust_remote_code=MODEL_LOAD_CONFIG["trust_remote_code"],
        )
        student_model.eval()
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    # ====== 加载数据 ======
    from core.data.preprocessing import load_jsonl
    data_path = DATA_PATHS["test"] if args.test else DATA_PATHS["train"]
    label = "测试集" if args.test else "训练集"
    all_data = load_jsonl(data_path)
    if args.max:
        all_data = all_data[:args.max]
    logger.info("使用 %s: %s (%d 道题)", label, data_path, len(all_data))

    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    out_path = args.output or os.path.join(ROP_D_DIR, f"ropd_data_{ts}.jsonl")

    student_cache = os.path.join(ROP_D_CACHE, f"student_{ts}.json")

    # ====== Phase 1 ======
    student_all = {}
    if not args.skip_student:
        logger.info("=== Phase 1: 学生生成（4 个温度 × %d 题）===", len(all_data))
        for idx, item in enumerate(all_data):
            q = item.get("input") or item.get("prompt") or item.get("question", "")
            logger.info("[%d/%d] %s", idx+1, len(all_data), q[:80])
            student_all[q] = student_generate_answers(
                student_model, tokenizer, q, STUDENT_GEN_COUNT)
            if (idx + 1) % 10 == 0:
                with open(student_cache, "w", encoding="utf-8") as f:
                    json.dump(student_all, f, ensure_ascii=False, indent=2)
                logger.info("缓存已保存 (%d 条)", len(student_all))
        with open(student_cache, "w", encoding="utf-8") as f:
            json.dump(student_all, f, ensure_ascii=False, indent=2)
    else:
        caches = sorted([f for f in os.listdir(ROP_D_CACHE) if f.startswith("student_")],
                        reverse=True)
        if caches:
            with open(os.path.join(ROP_D_CACHE, caches[0]), encoding="utf-8") as f:
                student_all = json.load(f)
            logger.info("从缓存加载学生回答: %s (%d 条)", caches[0], len(student_all))

    # ====== Phase 2 & 3 ======
    all_records = []
    if not args.skip_teacher:
        logger.info("=== Phase 2 & 3: DeepSeek-V4 批改 + 生成修正数据 ===")
        for idx, item in enumerate(all_data):
            q = item.get("input") or item.get("prompt") or item.get("question", "")
            student_answers = student_all.get(q, [])
            if not student_answers:
                continue

            logger.info("[%d/%d] %s", idx+1, len(all_data), q[:80])

            try:
                teacher_answers = teacher_generate_variants(q, TEACHER_GEN_COUNT)
                grading = teacher_grade(q, student_answers, teacher_answers)
                records = build_correction_data(
                    q, student_answers, teacher_answers, grading)
                all_records.extend(records)

                total_score = sum(s["score"] for s in grading.get("scores", []))
                logger.info("  总分 %s → %d 条修正数据",
                            grading.get("total", f"{total_score}/4"), len(records))
            except Exception as exc:
                logger.error("  处理失败: %s", exc)

            if (idx + 1) % 5 == 0 and all_records:
                _save_records(out_path, all_records)
                logger.info("已写入 %d 条", len(all_records))

        _save_records(out_path, all_records)

    logger.info("=== 完成 ===")
    logger.info("修正数据: %d 条 → %s", len(all_records), out_path)
    logger.info("")
    logger.info("下一步（黑盒微调，极低学习率）:")
    logger.info("  python core/train/train.py --sft --base-model <模型路径> \\")
    logger.info("      --train-data %s --epochs 1 --lr 5e-6", out_path)


def _save_records(path, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
