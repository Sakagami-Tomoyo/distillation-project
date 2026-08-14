import { useState, useRef, useEffect } from 'react'
import './index.css'

const FIELD_LABELS = {
  project_id: '项目编号', project_name: '项目名称', category: '类别', status: '状态',
  cost: '预算金额', address: '地址', start_date: '开始日期', end_date: '结束日期',
  contact_name: '联系人', contact_phone: '联系电话', reject_reason: '驳回原因',
  submitted_at: '提交时间', reviewed_at: '审核时间',
  device_no: '设备编号', robot_type: '机器人类型', robot_name: '设备名称', model: '型号',
  serial_no: '序列号', manufacturer: '制造商', install_location: '安装位置',
  usage_status: '使用状态', device_status: '运行状态', review_status: '审核状态',
  access_status: '接入状态', online_status: '运行状态', last_online_at: '最近在线时间',
  cert_date: '认证日期', accessed_at: '准入时间',
  year: '年份', month: '月份', maintenance_person: '维保人', maintenance_pers: '维保人',
  record_id: '记录编号', machinery_info_id: '机械信息ID',
  maintenance_date: '维保日期', maintenance_type: '维保类型', attachment_url: '附件',
  action: '动作', operator_id: '操作人ID', operator_name: '操作人',
  comment: '意见', created_at: '时间',
}

const ACTION_ZH = { submit: '提交', withdraw: '撤回', rejected: '驳回', approved: '通过' }
const STATUS_ZH = { pending: '待审核', approved: '已通过', rejected: '已驳回', not_started: '未开始' }

const label = (k) => FIELD_LABELS[k] || k
const fmt = (k, v) => {
  if (v === null || v === undefined || v === '') return '—'
  if (k === 'action') return ACTION_ZH[v] || v
  if (k === 'status') return STATUS_ZH[v] || v
  return String(v)
}

function ResultView({ data }) {
  if (!data) return <span>无返回结果</span>
  if (data.error) return <div className="chat-error">❌ {data.error}</div>

  const { intent, result } = data
  const tool = intent?.tool
  const params = intent?.parameters || {}

  // 列表型结果（维保记录 / 审核历程）
  const listKey = tool === 'query_maintenance_record' ? 'records'
    : tool === 'query_project_review' ? 'reviews' : null
  const ok = result && !result.error && result.found !== false
  const list = ok && listKey ? result[listKey] : null
  const scalarKeys = ok ? Object.keys(result).filter((k) =>
    !['found', 'message', 'records', 'reviews', 'id'].includes(k)) : []

  return (
    <div className="chat-result">
      <div className="chat-tool-bar">
        <span className="chat-tool-tag">🔧 {tool}</span>
        <code className="chat-params">{JSON.stringify(params)}</code>
      </div>

      {!result && <div className="chat-empty">查询无结果</div>}
      {result && result.error && <div className="chat-error">❌ {result.error}</div>}
      {result && result.found === false && (
        <div className="chat-empty">未找到相关记录{result.message ? `（${result.message}）` : ''}</div>
      )}

      {scalarKeys.length > 0 && (
        <table className="result-table"><tbody>
          {scalarKeys.map((k) => (
            <tr key={k}><td className="k">{label(k)}</td><td>{fmt(k, result[k])}</td></tr>
          ))}
        </tbody></table>
      )}

      {list && list.length > 0 && (
        <table className="result-table">
          <thead>
            <tr>{Object.keys(list[0]).filter((k) => k !== 'id' && k !== 'device_id').map((k) => (
              <th key={k}>{label(k)}</th>
            ))}</tr>
          </thead>
          <tbody>
            {list.map((r, i) => (
              <tr key={i}>{Object.keys(r).filter((k) => k !== 'id' && k !== 'device_id').map((k) => (
                <td key={k}>{fmt(k, r[k])}</td>
              ))}</tr>
            ))}
          </tbody>
        </table>
      )}

      <details className="chat-raw">
        <summary>查看原始 JSON</summary>
        <div className="raw-block">
          <div className="raw-label">模型返回的 JSON（intent）</div>
          <pre className="msg-pre">{JSON.stringify(intent, null, 2)}</pre>
        </div>
        <div className="raw-block">
          <div className="raw-label">查询结果（result）</div>
          <pre className="msg-pre">{JSON.stringify(result, null, 2)}</pre>
        </div>
      </details>
    </div>
  )
}

function ChatApp() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const listRef = useRef(null)

  useEffect(() => {
    if (listRef.current) listRef.current.scrollTop = listRef.current.scrollHeight
  }, [messages, loading])

  const send = async () => {
    const q = input.trim()
    if (!q || loading) return
    setInput('')
    setLoading(true)
    setMessages((prev) => [...prev, { role: 'user', text: q }])

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q }),
      })
      if (!res.ok) {
        let detail = `HTTP ${res.status}`
        try { const j = await res.json(); if (j.detail) detail = j.detail } catch { /* ignore */ }
        throw new Error(`后端请求失败（${detail}）`)
      }
      const data = await res.json()
      setMessages((prev) => [...prev, { role: 'assistant', data }])
    } catch (err) {
      const msg = /Failed to fetch|NetworkError|加载失败|Unexpected end of JSON/i.test(String(err))
        ? '无法连接后端服务。请先启动后端：cd src && uvicorn backend.main:app --host 0.0.0.0 --port 5000'
        : String(err)
      setMessages((prev) => [...prev, { role: 'assistant', data: { error: msg } }])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="chat-container">
      <header className="chat-header">
        <a className="back-link" href="#/">← 返回高考问答</a>
        <h1>🤖 智能业务助手</h1>
        <p>项目 / 设备 / 维保 / 审核历程 查询</p>
      </header>

      <div className="chat-messages" ref={listRef}>
        {messages.length === 0 && (
          <div className="chat-hint">
            <p>可以这样问我：</p>
            <ul>
              <li>「8 号项目现在什么状态？」</li>
              <li>「查一下设备 苏E-M-00001 的档案」</li>
              <li>「苏E-M-00001 2024 年 11 月的维保记录」</li>
              <li>「项目 11 的审核历程」</li>
            </ul>
          </div>
        )}

        {messages.map((m, i) => (
          <div key={i} className={`chat-msg ${m.role}`}>
            <div className="chat-role">{m.role === 'user' ? '我' : '助手'}</div>
            <div className="chat-bubble">
              {m.role === 'user' ? m.text : <ResultView data={m.data} />}
            </div>
          </div>
        ))}

        {loading && (
          <div className="chat-msg assistant">
            <div className="chat-role">助手</div>
            <div className="chat-bubble"><span className="waiting">正在查询...</span></div>
          </div>
        )}
      </div>

      <div className="chat-input-row">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && send()}
          placeholder="请输入业务问题..."
          disabled={loading}
        />
        <button onClick={send} disabled={loading}>发送</button>
      </div>
    </div>
  )
}

export default ChatApp
