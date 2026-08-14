"""智能问答的意图识别。

优先用（微调后的）模型生成工具调用 JSON，失败时用规则正则兜底，
保证在模型尚未针对意图微调时也能正确查询。
"""

import json
import logging
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_CN_NUM = {
    "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6,
    "七": 7, "八": 8, "九": 9, "十": 10, "十一": 11, "十二": 12,
}

# 合法工具名白名单：校验模型输出，防止幻觉出不存在的工具（如 query_project_address）
KNOWN_TOOLS = {
    "query_project_status",
    "query_device_info",
    "query_maintenance_record",
    "query_project_review",
}


def parse_tool_json(text: str) -> Optional[Dict[str, Any]]:
    """从模型输出中提取第一个合法的工具调用 JSON。"""
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", text.strip())
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except (ValueError, TypeError):
        return None
    if isinstance(obj, dict) and "tool" in obj and "parameters" in obj:
        return {"tool": obj["tool"], "parameters": obj.get("parameters", {})}
    return None


def _extract_month(q: str) -> Optional[str]:
    m = re.search(r"(\d{1,2})\s*月", q)
    if m:
        return m.group(1)
    m = re.search(r"([一二两三四五六七八九十]{1,3})\s*月", q)
    if m:
        return str(_CN_NUM.get(m.group(1), m.group(1)))
    return None


# 字段级意图：把宽泛的工具查询收窄到具体字段（列名 → 触发关键词）。
# 顺序即优先级：更具体的关键词排在前面。
_TOOL_FIELDS = {
    "query_project_status": [
        ("reject_reason", ["驳回原因", "拒绝原因", "为什么驳回", "因为什么", "原因"]),
        ("address", ["地址", "在哪", "位置", "哪里", "何处", "坐落", "什么地方"]),
        ("cost", ["预算", "金额", "多少钱", "费用", "造价", "成本", "价格"]),
        ("submitted_at", ["提交时间", "提交日期", "什么时候交"]),
        ("reviewed_at", ["审核时间", "审核日期", "什么时候审"]),
        ("contact_phone", ["电话", "手机", "联系方式", "号码"]),
        ("contact_name", ["联系人", "负责人", "谁负责", "姓名"]),
        ("project_name", ["名称", "叫什么", "项目名", "标题"]),
        ("category", ["类别", "类型", "哪类", "什么类"]),
        ("status", ["状态", "进度", "审核", "批没批", "通过", "驳回"]),
    ],
    "query_device_info": [
        ("serial_no", ["序列号", "SN"]),
        ("manufacturer", ["厂家", "制造商", "品牌", "生产商"]),
        ("model", ["型号"]),
        ("last_online_at", ["最近在线", "上次在线"]),
        ("online_status", ["运行状态", "在线", "离线", "是否在线"]),
        ("review_status", ["审核状态"]),
        ("access_status", ["接入状态"]),
        ("contact_phone", ["电话", "手机", "联系方式", "号码"]),
        ("contact_name", ["联系人", "负责人", "谁负责"]),
        ("robot_name", ["名称", "叫什么", "名字"]),
        ("category", ["类别", "类型"]),
    ],
}


def _extract_field(question: str, tool: str) -> Optional[str]:
    """从问句中提取字段级意图，例如「地址」「预算」「运行状态」。"""
    fields = _TOOL_FIELDS.get(tool)
    if not fields:
        return None
    for field, keywords in fields:
        if any(kw in question for kw in keywords):
            return field
    return None


def fallback_intent(question: str) -> Dict[str, Any]:
    """规则兜底：正则提取意图与参数。"""
    dev = re.search(r"(苏E-[A-Za-z]-\d{5})", question)
    year = re.search(r"(20\d{2})\s*年?", question)
    month = _extract_month(question)
    proj = re.search(r"(?:项目|编号|申请)[\s号编号：:]*(\d+)", question)
    if not proj:
        proj = re.search(r"(\d+)\s*号?\s*项目", question)
    project_id = proj.group(1) if proj else None

    # 维保
    if any(k in question for k in ("维保", "保养", "巡检", "维修")):
        params: Dict[str, str] = {}
        if dev:
            params["device_no"] = dev.group(1)
        if year:
            params["year"] = year.group(1)
        if month:
            params["month"] = month
        return {"tool": "query_maintenance_record", "parameters": params}

    # 审核历程
    if any(k in question for k in ("审核历程", "审核步骤", "审核历史", "审核动作", "怎么批", "被谁审核", "审核到")):
        return {"tool": "query_project_review",
                "parameters": {"project_id": project_id} if project_id else {}}

    # 设备
    if dev:
        return {"tool": "query_device_info", "parameters": {"device_no": dev.group(1)}}

    # 项目状态
    if project_id or any(k in question for k in ("项目", "审核状态", "预算", "地址", "状态", "批")):
        return {"tool": "query_project_status",
                "parameters": {"project_id": project_id} if project_id else {}}

    return {"tool": None, "parameters": {}}


def detect_intent(question: str, model=None, tokenizer=None) -> Dict[str, Any]:
    """意图识别：模型优先，正则兜底；再叠加字段级意图收窄。

    返回的 dict 除 tool/parameters 外，额外带 model_raw：模型原始输出文本
    （模型未启用或失败时为 None）。若问句提到具体字段（如「地址」「预算」），
    会在 parameters 里补一个 field 键，用于只返回该字段而非整条记录。
    """
    model_raw = None
    tool_call = None
    if model is not None and tokenizer is not None:
        try:
            from backend.generator import generate_api_response

            raw = generate_api_response(model, tokenizer, question)
            model_raw = raw
            candidate = parse_tool_json(raw)
            if candidate and candidate.get("tool") in KNOWN_TOOLS:
                tool_call = candidate
            else:
                logger.info("模型工具名不合法或未产出 JSON，改用规则兜底。模型输出: %r", raw)
        except Exception as exc:  # noqa: BLE001
            logger.warning("模型意图识别异常，改用规则兜底: %s", exc)
    if tool_call is None:
        tool_call = fallback_intent(question)
    tool_call["model_raw"] = model_raw

    # 字段级意图：只问「地址」就只回地址，不回整条项目/设备记录
    params = tool_call.get("parameters") or {}
    if isinstance(params, dict) and tool_call.get("tool") and "field" not in params:
        field = _extract_field(question, tool_call["tool"])
        if field:
            params["field"] = field
            tool_call["parameters"] = params

    return tool_call
