import { useState, useEffect, useRef } from 'react'
import { marked } from 'marked'
import katex from 'katex'
import './index.css'

const BASE_MODELS = {
  student: { name: '学生模型', icon: '📚', color: 'student' },
  teacher: { name: '教师模型', icon: '👨‍🏫', color: 'teacher' },
}

marked.setOptions({ breaks: true, gfm: true })

function renderContent(text) {
  const mathBlocks = []
  const MATH_PLACEHOLDER = '%%MATH_%d%%'
  let html = text

  // 修复双反斜杠：\\frac → \frac（模型可能输出 JSON 转义风格的 LaTeX）
  html = html.replace(/\\\\/g, '\\')

  // 预处理：统一 LaTeX 分隔符
  html = html.replace(/\\\(([\s\S]+?)\\\)/g, (_, f) => `$${f.trim()}$`)
  html = html.replace(/\\\[([\s\S]+?)\\\]/g, (_, f) => `$$\n${f.trim()}\n$$`)

  // 0. [ ... ] 显示公式
  html = html.replace(/\[([^\[\]]+)\](?!\()/g, (_, formula) => {
    const idx = mathBlocks.length
    try {
      mathBlocks.push(katex.renderToString(formula.trim(), { throwOnError: false, displayMode: true }))
    } catch { mathBlocks.push(`<div class="math-error">[${formula}]</div>`) }
    return MATH_PLACEHOLDER.replace('%d', idx)
  })

  // 1. $$...$$
  html = html.replace(/\$\$([\s\S]+?)\$\$/g, (_, formula) => {
    const idx = mathBlocks.length
    try {
      mathBlocks.push(katex.renderToString(formula.trim(), { throwOnError: false, displayMode: true }))
    } catch { mathBlocks.push(`<div class="math-error">$$${formula}$$</div>`) }
    return MATH_PLACEHOLDER.replace('%d', idx)
  })

  // 2. $...$
  html = html.replace(/(?<!\$)\$(?!\$)([^$]+?)\$(?!\$)/g, (_, formula) => {
    const idx = mathBlocks.length
    try {
      mathBlocks.push(katex.renderToString(formula.trim(), { throwOnError: false, displayMode: false }))
    } catch { mathBlocks.push(`<span class="math-error">$${formula}$</span>`) }
    return MATH_PLACEHOLDER.replace('%d', idx)
  })

  // 3. 裸 LaTeX
  const LATEX_CMDS = [
    'sqrt', 'frac', 'vec', 'overrightarrow', 'bar', 'hat', 'tilde', 'dot', 'ddot',
    'text', 'textbf', 'textit', 'mathrm', 'mathbb', 'mathcal', 'mathfrak',
    'binom', 'underbrace', 'overline', 'underline',
    'Rightarrow', 'Leftarrow', 'Leftrightarrow', 'leftarrow', 'rightarrow',
    'implies', 'iff', 'to', 'mapsto', 'uparrow', 'downarrow',
    'neq', 'times', 'div', 'pm', 'mp', 'cdot', 'cdots', 'vdots', 'ddots',
    'leq', 'geq', 'approx', 'equiv', 'sim', 'propto', 'll', 'gg',
    'forall', 'exists', 'in', 'notin', 'subset', 'supset', 'subseteq', 'supseteq',
    'cup', 'cap', 'emptyset', 'varnothing',
    'parallel', 'perp', 'angle', 'triangle', 'square', 'circ', 'diamond',
    'infty', 'partial', 'nabla', 'aleph',
    'sin', 'cos', 'tan', 'cot', 'sec', 'csc',
    'arcsin', 'arccos', 'arctan',
    'log', 'ln', 'lg', 'det', 'dim', 'gcd', 'hom', 'ker', 'lim', 'sup', 'inf',
    'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta',
    'iota', 'kappa', 'lambda', 'mu', 'nu', 'xi', 'pi', 'rho', 'sigma', 'tau',
    'upsilon', 'phi', 'chi', 'psi', 'omega',
    'Gamma', 'Delta', 'Theta', 'Lambda', 'Xi', 'Pi', 'Sigma', 'Upsilon', 'Phi', 'Psi', 'Omega',
    'sum', 'prod', 'int', 'iint', 'iiint', 'oint',
  ].join('|')
  html = html.replace(new RegExp(`\\\\(?:${LATEX_CMDS})(?:\\s*\\{[^}]*\\})*`, 'g'), (match) => {
    const idx = mathBlocks.length
    try {
      mathBlocks.push(katex.renderToString(match, { throwOnError: false, displayMode: false }))
    } catch { mathBlocks.push(`<span class="math-error">${match}</span>`) }
    return MATH_PLACEHOLDER.replace('%d', idx)
  })

  // 4. Markdown 渲染
  html = marked(html)

  // 5. 替换回 KaTeX
  html = html.replace(/%%MATH_(\d+)%%/g, (_, idx) => mathBlocks[parseInt(idx)] || '')
  return html
}

function App() {
  const [question, setQuestion] = useState('')
  const [selectedModels, setSelectedModels] = useState([])
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState({})
  const [streaming, setStreaming] = useState({})
  const [submittedQuestion, setSubmittedQuestion] = useState('')
  const [error, setError] = useState('')
  const [streamDone, setStreamDone] = useState({})
  const [temperature, setTemperature] = useState(0.8)
  const [availableModels, setAvailableModels] = useState({ student: true, teacher: true })
  const [checkpoints, setCheckpoints] = useState([])
  const [rawVisible, setRawVisible] = useState({})
  const abortRef = useRef(null)

  // 加载模型状态和检查点列表
  useEffect(() => {
    // 并行获取模型状态和检查点列表
    Promise.all([
      fetch('/api/models').then(r => r.json()),
      fetch('/api/checkpoints').then(r => r.json()).catch(() => ({ checkpoints: [] })),
    ]).then(([models, cpData]) => {
      setAvailableModels({ student: models.student, teacher: models.teacher })
      const cps = cpData.checkpoints || models.checkpoints || []
      setCheckpoints(cps)
      // 默认选中第一个检查点
      if (cps.length > 0 && selectedModels.length === 0) {
        setSelectedModels(['checkpoint:' + cps[0].name])
      }
    }).catch(err => console.error('获取模型状态失败:', err))
  }, [])

  const toggleModel = (key) => {
    setSelectedModels(prev => {
      if (prev.includes(key)) {
        if (prev.length === 1) return prev
        return prev.filter(m => m !== key)
      }
      return [...prev, key]
    })
  }

  // 显示名称
  const getDisplayName = (key) => {
    if (BASE_MODELS[key]) return BASE_MODELS[key].name
    if (key.startsWith('checkpoint:')) return key.split(':')[1]
    return key
  }
  const getIcon = (key) => {
    if (BASE_MODELS[key]) return BASE_MODELS[key].icon
    return '🔄'
  }
  const getColor = (key) => {
    if (BASE_MODELS[key]) return BASE_MODELS[key].color
    // 为不同检查点分配不同颜色
    const colors = ['checkpoint-a', 'checkpoint-b', 'checkpoint-c', 'checkpoint-d', 'checkpoint-e']
    const idx = [...selectedModels].indexOf(key)
    return colors[idx % colors.length]
  }

  // ---------- 流式请求 ----------
  const fetchStream = async (q, models, temp, signal) => {
    const response = await fetch('/api/generate/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q, models, temperature: temp }),
      signal,
    })
    if (!response.ok) throw new Error('STREAM_NOT_AVAILABLE')

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue
        try {
          const data = JSON.parse(line.slice(6))
          if (data.all_done) {
            setStreaming(prev => { setResults(prev); return prev })
            setLoading(false)
            return
          }
          if (data.done) { setStreamDone(prev => ({ ...prev, [data.model]: true })); continue }
          if (data.token) {
            setStreaming(prev => ({ ...prev, [data.model]: (prev[data.model] || '') + data.token }))
          }
        } catch { /* ignore */ }
      }
    }
    setLoading(false)
  }

  const handleSubmit = async () => {
    if (!question.trim()) { setError('请输入题目'); return }
    if (selectedModels.length === 0) { setError('请至少选择一个模型'); return }
    if (loading) { abortRef.current?.abort(); setLoading(false); return }

    setLoading(true)
    setError('')
    setResults({})
    setStreaming({})
    setStreamDone({})
    setSubmittedQuestion(question.trim())

    const initial = {}
    selectedModels.forEach(m => { initial[m] = '' })
    setStreaming(initial)

    const controller = new AbortController()
    abortRef.current = controller

    try {
      await fetchStream(question.trim(), [...selectedModels], temperature, controller.signal)
    } catch (err) {
      if (err.name === 'AbortError') { setLoading(false); return }
      setError(err.message === 'STREAM_NOT_AVAILABLE' ? '后端不支持流式，请重启服务' : err.message)
      setLoading(false)
    }
  }

  const isStreaming = (key) => loading && !streamDone[key] && selectedModels.includes(key)
  const displayContent = (key) => {
    if (loading || streamDone[key]) return streaming[key] || ''
    return results[key] || ''
  }
  const hasAnyContent = () => {
    return Object.values(streaming).some(v => v && v.length > 0)
      || Object.values(results).some(v => v && v.length > 0)
  }

  // 所有可选模型（基础 + 检查点）
  const allModelKeys = [
    ...Object.keys(BASE_MODELS),
    ...checkpoints.map(cp => 'checkpoint:' + cp.name),
  ]

  return (
    <div className="app-container">
      <header className="header">
        <h1>📝 高考题模型对比</h1>
        <p>对比学生、教师、各检查点模型对高考题的回答</p>
      </header>

      <main className="main-content">
        <section className="input-section">
          <label htmlFor="question-input">题目</label>
          <textarea
            id="question-input"
            placeholder="请输入高考题目..."
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
          />
        </section>

        {/* 基础模型 */}
        <section className="model-section">
          <label>基础模型</label>
          <div className="model-checkboxes">
            {Object.entries(BASE_MODELS).map(([key, config]) => (
              <label
                key={key}
                className={`model-checkbox ${config.color} ${!availableModels[key] ? 'disabled' : ''}`}
              >
                <input
                  type="checkbox"
                  checked={selectedModels.includes(key)}
                  onChange={() => toggleModel(key)}
                  disabled={!availableModels[key]}
                />
                <span>{config.icon}</span>
                <span>{config.name}</span>
              </label>
            ))}
          </div>
        </section>

        {/* 检查点模型 */}
        {checkpoints.length > 0 && (
          <section className="model-section">
            <label>蒸馏检查点（可多选对比）</label>
            <div className="model-checkboxes">
              {checkpoints.map((cp, i) => {
                const key = 'checkpoint:' + cp.name
                const colorCls = ['checkpoint-a', 'checkpoint-b', 'checkpoint-c', 'checkpoint-d', 'checkpoint-e'][i % 5]
                return (
                  <label key={key} className={`model-checkbox ${colorCls}`}>
                    <input
                      type="checkbox"
                      checked={selectedModels.includes(key)}
                      onChange={() => toggleModel(key)}
                    />
                    <span>🔄</span>
                    <span>{cp.name}</span>
                  </label>
                )
              })}
            </div>
          </section>
        )}

        {/* 温度 */}
        <section className="temp-section">
          <label>
            🌡️ 随机性（温度）: <span className="temp-value">{temperature.toFixed(1)}</span>
          </label>
          <div className="temp-control">
            <span>精确</span>
            <input type="range" min="0.1" max="2.0" step="0.1"
              value={temperature} onChange={(e) => setTemperature(parseFloat(e.target.value))} />
            <span>随机</span>
          </div>
        </section>

        <button className="submit-btn" onClick={handleSubmit}>
          {loading ? '⏹ 停止生成' : '生成回答'}
        </button>

        {error && <div className="error-message">❌ {error}</div>}

        {submittedQuestion && hasAnyContent() && (
          <section className="results-section">
            <div className="question-card">
              <div className="question-header">📋 题目</div>
              <div className="question-content">{submittedQuestion}</div>
            </div>

            <h2 className="results-title">回答结果</h2>

            {selectedModels.map(key => {
              const content = displayContent(key)
              const active = isStreaming(key)
              const displayName = getDisplayName(key)
              const icon = getIcon(key)
              const colorCls = getColor(key)

              return (
                <div key={key} className={`result-card ${active ? 'streaming' : ''}`}>
                  <div className={`result-header ${colorCls}`}>
                    <span className="icon">{icon}</span>
                    <span>{displayName}</span>
                    {active && <span className="streaming-dot"></span>}
                    {streamDone[key] && <span className="done-mark">✅</span>}
                    {content && (
                      <button
                        className="raw-toggle-btn"
                        onClick={() => setRawVisible(prev => ({ ...prev, [key]: !prev[key] }))}
                        title={rawVisible[key] ? '查看渲染' : '查看原文'}
                      >
                        {rawVisible[key] ? '📄' : '🔍'}
                      </button>
                    )}
                  </div>
                  <div className="result-content">
                    {content ? (
                      rawVisible[key] ? (
                        <pre className="raw-content">{content}</pre>
                      ) : (
                        <div className="answer" dangerouslySetInnerHTML={{ __html: renderContent(content) }} />
                      )
                    ) : (
                      <span className="waiting">等待生成...</span>
                    )}
                    {active && <span className="cursor-blink">|</span>}
                  </div>
                </div>
              )
            })}
          </section>
        )}
      </main>
    </div>
  )
}

export default App
