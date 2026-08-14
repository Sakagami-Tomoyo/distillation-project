"""智能业务助手：数据库查询接口（MySQL）。

连接 MySQL 数据库（库名 kzj），每个工具函数对应一个查询，返回 JSON 可序列化的结果。
连接信息可通过环境变量覆盖（DB_HOST / DB_PORT / DB_USER / DB_PASSWORD / DB_NAME）。

工具清单：
  - query_project_status     查项目审核状态 / 预算 / 地址
  - query_device_info        查设备档案 / 运行状态
  - query_maintenance_record 查设备维保记录
  - query_project_review     查项目审核历程
"""

import os
import logging
import decimal
from typing import Any, Dict

import pymysql

logger = logging.getLogger(__name__)

DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "192.168.130.8"),
    "port": int(os.environ.get("DB_PORT", "3306")),
    "user": os.environ.get("DB_USER", "root"),
    "password": os.environ.get("DB_PASSWORD", "root"),
    "database": os.environ.get("DB_NAME", "kzj"),
    "charset": "utf8mb4",
    "cursorclass": pymysql.cursors.DictCursor,
    "connect_timeout": 5,
    # 禁用 TLS：pymysql 默认 PREFERRED 模式会尝试加载 Windows 证书库，
    # 在部分 Python/OpenSSL 环境下因证书库损坏抛 [ASN1: NOT_ENOUGH_DATA]。
    "ssl_disabled": True,
}


def _connect() -> pymysql.Connection:
    """获取数据库连接。"""
    return pymysql.connect(**DB_CONFIG)


def _to_jsonable(value: Any) -> Any:
    """递归把 Decimal 等 JSON 不可序列化的类型转成可序列化形式。

    MySQL 的 DECIMAL 列会被 pymysql 返回为 decimal.Decimal，直接交给
    FastAPI 序列化会抛 TypeError，这里统一转成字符串保留精度。
    """
    if isinstance(value, decimal.Decimal):
        return str(value)
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def query_project_status(project_id: str, field: str = None) -> Dict[str, Any]:
    """查询项目审核状态、预算、地址等信息；指定 field 时只返回该字段。"""
    if not project_id:
        return {"found": False, "message": "缺少项目编号"}
    sql = (
        "SELECT project_id, project_name, category, status, cost, address, "
        "submitted_at, reviewed_at, reject_reason, contact_name, contact_phone "
        "FROM applications WHERE project_id = %s"
    )
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, [project_id])
            row = cur.fetchone()
    if row is None:
        return {"found": False, "project_id": project_id, "message": "未找到该项目"}
    if field and field in row:
        return {"found": True, "project_id": row["project_id"], field: row[field]}
    return {"found": True, **row}


def query_device_info(device_no: str, field: str = None) -> Dict[str, Any]:
    """查询设备型号、厂家、运行状态等档案信息；指定 field 时只返回该字段。"""
    if not device_no:
        return {"found": False, "message": "缺少设备编号"}
    sql = (
        "SELECT device_no, category, robot_name, model, serial_no, manufacturer, "
        "review_status, access_status, online_status, contact_name, contact_phone, "
        "machinery_info_id, cert_date, reviewed_at, accessed_at, last_online_at "
        "FROM devices WHERE device_no = %s"
    )
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, [device_no])
            row = cur.fetchone()
    if row is None:
        return {"found": False, "device_no": device_no, "message": "未找到该设备"}
    if field and field in row:
        return {"found": True, "device_no": row["device_no"], field: row[field]}
    return {"found": True, **row}


def query_maintenance_record(device_no: str, year: str = None, month: str = None) -> Dict[str, Any]:
    """查询设备在指定年份 / 月份的维保记录。"""
    if not device_no:
        return {"found": False, "message": "缺少设备编号"}

    sql = (
        "SELECT m.id, m.maintenance_date, m.maintenance_type, m.attachment_url, "
        "m.created_by, m.created_at "
        "FROM device_maintenances m "
        "JOIN project_devices pd ON m.project_device_id = pd.id "
        "JOIN devices d ON pd.device_id = d.id "
        "WHERE d.device_no = %s AND m.is_deleted = 0"
    )
    params: list = [device_no]
    if year:
        sql += " AND YEAR(m.maintenance_date) = %s"
        params.append(year)
    if month:
        sql += " AND MONTH(m.maintenance_date) = %s"
        params.append(month)
    sql += " ORDER BY m.maintenance_date DESC"

    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
    return {
        "found": bool(rows),
        "device_no": device_no,
        "year": year,
        "month": month,
        "records": rows,
    }


def query_project_review(project_id: str) -> Dict[str, Any]:
    """查询项目审核历程（提交 / 驳回 / 撤回 / 通过 等动作及操作人）。"""
    if not project_id:
        return {"found": False, "message": "缺少项目编号"}
    sql = (
        "SELECT r.action, r.comment, r.created_at, a.name AS operator_name "
        "FROM application_reviews r "
        "LEFT JOIN accounts a ON r.operator_id = a.id "
        "WHERE r.application_id = (SELECT id FROM applications WHERE project_id = %s) "
        "ORDER BY r.id"
    )
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, [project_id])
            rows = cur.fetchall()
    return {"found": bool(rows), "project_id": project_id, "reviews": rows}


TOOL_FUNCS = {
    "query_project_status": query_project_status,
    "query_device_info": query_device_info,
    "query_maintenance_record": query_maintenance_record,
    "query_project_review": query_project_review,
}


def execute_tool(tool: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """根据 tool 名称与 parameters 调用对应查询函数。"""
    func = TOOL_FUNCS.get(tool)
    if func is None:
        return {"error": f"未知工具: {tool}"}
    try:
        return _to_jsonable(func(**parameters))
    except Exception as exc:  # noqa: BLE001
        logger.exception("查询失败 tool=%s params=%s", tool, parameters)
        return {"error": f"查询失败: {exc}"}
