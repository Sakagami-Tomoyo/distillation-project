# 智能业务助手 — 数据库表结构与模拟数据

> 本文档描述企业智能业务助手所需的数据库表结构，以及用于生成 SFT 微调数据的真实模拟记录。

---

## 一、表结构

### 1. accounts — 系统用户账户表

系统用户基础表，包含账号与基本信息。

| 字段名 | 数据类型 | 约束 | 中文说明 / 业务含义 |
| --- | --- | --- | --- |
| id | bigint(20) | 主键, 自增 | 主键 |
| username | varchar(100) | 唯一, 非空 | 登录用户名（唯一索引） |
| name | varchar(300) | 可空 | 真实姓名 / 昵称 |
| hashed_password | varchar(255) | 非空 | 加密后的密码 |
| role | varchar(20) | 非空 | 角色权限（admin / user / reviewer） |
| department | varchar(100) | 可空 | 所属部门 |
| is_active | tinyint(1) | 非空 | 账号是否启用（1:启用, 0:停用） |
| is_safe | tinyint(1) | 非空 | 安全状态标记 |
| last_login_at | datetime | 可空 | 最后一次登录时间 |
| created_at | datetime | 非空 | 创建时间（默认当前时间） |
| created_by | bigint(20) | 可空 | 创建人 ID |
| updated_at | datetime | 非空 | 更新时间（自动更新） |
| updated_by | bigint(20) | 可空 | 更新人 ID |
| entity_id | bigint(20) | 索引 | 关联实体 / 业务线 ID |

### 2. applications — 业务申请单主表

核心业务表，记录设备 / 项目的正式申请。

| 字段名 | 数据类型 | 约束 | 中文说明 / 业务含义 |
| --- | --- | --- | --- |
| id | bigint(20) | 主键, 自增 | 主键 |
| project_id | bigint(20) | 唯一, 非空 | 项目编号（唯一索引） |
| category | varchar(20) | 非空 | 申请类别 |
| status | varchar(20) | 非空 | 当前审核 / 流程状态 |
| is_joint | tinyint(1) | 非空 | 是否联合申请 |
| joint_statement_path | varchar(500) | 可空 | 联合声明文件路径 |
| address | varchar(300) | 可空 | 申请项目 / 设备所在地址 |
| cost | decimal(15,2) | 可空 | 申请预算金额 |
| area | decimal(10,4) | 可空 | 占地面积 / 数据 |
| start_date | date | 可空 | 项目开始日期 |
| end_date | date | 可空 | 项目结束日期 |
| plan_file_path | varchar(500) | 可空 | 项目计划文件路径 |
| submitted_at | datetime | 可空 | 提交申请的时间 |
| reviewed_at | datetime | 可空 | 审核完成的时间 |
| reject_reason | varchar(2000) | 可空 | 驳回 / 拒绝理由 |
| created_at | datetime | 非空 | 创建时间 |
| created_by | bigint(20) | 可空 | 申请人 ID |
| updated_at | datetime | 非空 | 更新时间 |
| updated_by | bigint(20) | 可空 | 更新人 ID |
| contact_name | varchar(50) | 可空 | 业务联系人姓名 |
| contact_phone | varchar(50) | 可空 | 业务联系电话 |
| ent_snapshot_ids | varchar(500) | 可空 | 企业快照 ID 串 |
| ent_snapshot_names | varchar(2000) | 可空 | 企业快照名称串 |
| supervision_no | varchar(100) | 可空 | 监管号 |
| project_name | varchar(200) | 可空 | 项目名称 |

### 3. application_reviews — 申请单审核记录表

记录每一次的审批 / 审核动作。

| 字段名 | 数据类型 | 约束 | 中文说明 / 业务含义 |
| --- | --- | --- | --- |
| id | bigint(20) | 主键, 自增 | 主键 |
| application_id | bigint(20) | 索引, 非空 | 关联的申请单 ID |
| action | varchar(20) | 非空 | 审核操作（submit / withdraw / rejected / approved） |
| operator_id | bigint(20) | 索引, 可空 | 审核人 / 操作员 ID |
| comment | text | 可空 | 审核意见 / 评语 |
| created_at | datetime | 非空 | 审核时间 |
| created_by | bigint(20) | 可空 | 创建人 ID |

### 4. devices — 设备 / 机器主表

设备档案核心表，字段丰富，用于记录设备状态和属性。

| 字段名 | 数据类型 | 约束 | 中文说明 / 业务含义 |
| --- | --- | --- | --- |
| id | bigint(20) | 主键, 自增 | 主键 |
| provider_id | bigint(20) | 索引, 非空 | 服务商 / 供应商 ID |
| device_no | varchar(50) | 唯一, 可空 | 设备编号（设备唯一标识） |
| category | varchar(20) | 可空 | 设备大类 |
| robot_type | varchar(20) | 可空 | 机器人 / 细分类型 |
| robot_name | varchar(100) | 可空 | 设备显示名称 |
| model | varchar(100) | 非空 | 设备型号 |
| serial_no | varchar(100) | 非空 | 设备序列号（硬件串号） |
| manufacturer | varchar(200) | 非空 | 制造商名称 |
| contact_name | varchar(50) | 非空 | 设备联系人 |
| contact_phone | varchar(11) | 非空 | 联系电话 |
| cert_date | date | 可空 | 设备证书 / 认证日期 |
| source_type | varchar(20) | 可空 | 设备来源类型 |
| panoramic_photo_url | varchar(200) | 可空 | 全景 / 外观照片地址 |
| nameplate_photo_url | varchar(200) | 可空 | 铭牌照片地址 |
| certificate_url | varchar(200) | 可空 | 证书文件地址 |
| property_proof_url | varchar(200) | 可空 | 资产证明文件地址 |
| tech_params_url | varchar(200) | 可空 | 技术参数文件地址 |
| notify_status | varchar(50) | 可空 | 通知状态 |
| install_notify_no | varchar(50) | 可空 | 安装通知单号 |
| install_location | varchar(200) | 可空 | 设备安装位置 |
| usage_status | varchar(20) | 可空 | 使用状态 |
| usage_reg_no | varchar(50) | 可空 | 使用登记编号 |
| usage_reg_date | date | 可空 | 使用登记日期 |
| review_status | varchar(20) | 非空 | 审核状态 |
| reviewed_at | datetime | 可空 | 审核时间 |
| access_status | varchar(20) | 非空 | 准入 / 接入状态 |
| accessed_at | datetime | 可空 | 准入审批时间 |
| machinery_info_id | varchar(50) | 可空 | 关联机械信息 ID |
| device_status | varchar(50) | 可空 | 设备在线运行状态 |
| created_at | datetime | 非空 | 创建时间 |
| created_by | bigint(20) | 可空 | 创建人 |
| updated_at | datetime | 非空 | 更新时间 |
| updated_by | bigint(20) | 可空 | 更新人 |

### 5. device_maintenances — 设备维保记录表

记录设备的定期巡检、保养和维修历史。

| 字段名 | 数据类型 | 约束 | 中文说明 / 业务含义 |
| --- | --- | --- | --- |
| id | bigint(20) | 主键, 自增 | 主键 |
| device_id | bigint(20) | 索引, 非空 | 关联的设备 ID |
| machinery_info_id | varchar(50) | 非空 | 关联机械信息 ID |
| record_id | int(11) | 非空 | 维保记录编号 |
| year | int(11) | 可空 | 维保年份 |
| month | int(11) | 可空 | 维保月份 |
| maintenance_person | varchar(50) | 可空 | 维保人 / 责任人 |
| created_at | datetime | 非空 | 维保记录登记时间 |
| created_by | bigint(20) | 可空 | 登记人 |

---

## 二、实际数据（模拟）

### 1. accounts — 人员账户表

| username | name | role | department |
| --- | --- | --- | --- |
| 13900001111 | 测试监管 | supervisor | 测试部门 |
| 91320583MA21L1 | 苏州傲之途智慧科技 | provider | (NULL) |
| 17996751873 | 王丢 | navigator | (NULL) |
| 17379403893 | 彭于晏 | supervisor | 综合科 |
| 18070547717 | 王仁骏 | navigator | (NULL) |

### 2. applications — 项目主表

| project_id | category | status | cost | address |
| --- | --- | --- | --- | --- |
| 8 | 房屋建筑 | pending | 4101.66 | 江苏省苏州市昆山市... |
| 11 | 房屋建筑 | approved | 2458.00 | 昆山市陆家镇顺星路... |
| 19 | 市政基础设施 | rejected | 332.00 | 昆山市华成路8号 |
| 18 | 装饰装修 | approved | 72001.24 | 陆家镇香花路东侧、... |
| 34 | 房屋建筑 | not_started | 510.72 | 昆山旅游度假区淀山湖... |

### 3. application_reviews — 项目审核动作表

| application_id | action | operator_id | comment |
| --- | --- | --- | --- |
| 11 | submit | 25 | (NULL) |
| 11 | withdraw | 25 | (NULL) |
| 11 | rejected | 3 | 重新填写 |
| 11 | approved | 3 | (NULL) |
| 19 | submit | 26 | (NULL) |

### 4. devices — 设备主表

> 注：部分字段在原始数据中为 NULL，这里如实记录。

| device_no | category | robot_name | model | manufacturer |
| --- | --- | --- | --- | --- |
| 苏E-M-00001 | robot | 地面工序 | 211 | 呜呜呜 |
| 苏E-B-00002 | building_machir | (NULL) | xx | xx |
| 苏E-T-00001 | tower_crane | (NULL) | 测试 | 测试 |
| 苏E-M-00002 | robot | 地面工序 | R-TEST-007 | 苏州测试装备有限 |
| 苏E-M-00003 | robot | 地面工序 | B-TEST-007 | 苏州测试装备有限 |

### 5. device_maintenances — 设备维保记录表

> 注：此表关键信息是 device_id 与维保时间、维保人。

| device_id | year | month | maintenance_person |
| --- | --- | --- | --- |
| 11 | 2024 | 11 | 俄方 |
| 11 | 2024 | 8 | 上午 |
| 11 | 2024 | 2 | 222 |
| 11 | 2024 | 2 | 9996 |
| 11 | 2024 | 1 | 111 |
