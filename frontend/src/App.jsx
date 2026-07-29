import { useState, useEffect, useRef } from 'react'
import { marked } from 'marked'
import katex from 'katex'
import './index.css'

const MODEL_CONFIG = {
  student: { name: '学生模型', icon: '📚', color: 'student' },
  teacher: { name: '教师模型', icon: '👨‍🏫', color: 'teacher' },
  distilled: { name: '蒸馏模型', icon: '✨', color: 'distilled' },
}

marked.setOptions({ breaks: true, gfm: true })

function renderContent(text) {
  // ---- 先保护 LaTeX，避免被 marked 转义反斜杠 ----
  const mathBlocks = []
  const MATH_PLACEHOLDER = '%%MATH_%d%%'
  let html = text

  // -- 预处理：统一 LaTeX 分隔符 --
  // \(...\) → $...$  （LaTeX 行内公式）
  html = html.replace(/\\\(([\s\S]+?)\\\)/g, (_, f) => `$${f.trim()}$`)
  // \[...\] → $$...$$（LaTeX 显示公式）
  html = html.replace(/\\\[([\s\S]+?)\\\]/g, (_, f) => `$$\n${f.trim()}\n$$`)

  // 0. 保护 [ ... ] 显示公式（模型常用 LaTeX 风格 [ ... ] 替代 $$...$$）
  //    排除 Markdown 链接 [text](url)，KaTeX 渲染失败则保留原文
  html = html.replace(/\[([^\[\]]+)\](?!\()/g, (_, formula) => {
    const idx = mathBlocks.length
    try {
      const rendered = katex.renderToString(formula.trim(), { throwOnError: false, displayMode: true })
      mathBlocks.push(rendered)
    } catch {
      mathBlocks.push(`<div class="math-error">[${formula}]</div>`)
    }
    return MATH_PLACEHOLDER.replace('%d', idx)
  })

  // 1. 保护 $$...$$ 块级公式
  html = html.replace(/\$\$([\s\S]+?)\$\$/g, (_, formula) => {
    const idx = mathBlocks.length
    try {
      const rendered = katex.renderToString(formula.trim(), { throwOnError: false, displayMode: true })
      mathBlocks.push(rendered)
    } catch {
      mathBlocks.push(`<div class="math-error">$$${formula}$$</div>`)
    }
    return MATH_PLACEHOLDER.replace('%d', idx)
  })

  // 2. 保护 $...$ 行内公式（不匹配 $$）
  html = html.replace(/(?<!\$)\$(?!\$)([^$]+?)\$(?!\$)/g, (_, formula) => {
    const idx = mathBlocks.length
    try {
      const rendered = katex.renderToString(formula.trim(), { throwOnError: false, displayMode: false })
      mathBlocks.push(rendered)
    } catch {
      mathBlocks.push(`<span class="math-error">$${formula}$</span>`)
    }
    return MATH_PLACEHOLDER.replace('%d', idx)
  })

  // 3. 处理裸 LaTeX（出现在 $ 外面的命令）
  //    支持 \sqrt{3}, \frac{a}{b} 等带参数命令，也支持 \Rightarrow, \neq 等符号
  const LATEX_CMDS = [
    // 带参数
    'sqrt', 'frac', 'vec', 'overrightarrow', 'bar', 'hat', 'tilde', 'dot', 'ddot',
    'text', 'textbf', 'textit', 'mathrm', 'mathbb', 'mathcal', 'mathfrak',
    'binom', 'underbrace', 'overline', 'underline',
    // 符号
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
    // 函数 / 运算符
    'sum', 'prod', 'int', 'iint', 'iiint', 'oint',
    // 括号缩放
    'left', 'right', 'big', 'Big', 'bigg', 'Bigg',
  ].join('|')

  // 匹配 \cmd（可选 {arg}... + 可选的尾随分隔符如 \left( → \left(）
  html = html.replace(new RegExp(`\\\\(?:${LATEX_CMDS})(?:\\s*\\{[^}]*\\})*[()\\[\\]|.]?`, 'g'), (match) => {
    const idx = mathBlocks.length
    try {
      const rendered = katex.renderToString(match, { throwOnError: false, displayMode: false })
      mathBlocks.push(rendered)
    } catch {
      mathBlocks.push(`<span class="math-error">${match}</span>`)
    }
    return MATH_PLACEHOLDER.replace('%d', idx)
  })

  // 3. Markdown 渲染
  html = marked(html)

  // 4. 把占位符换回渲染好的 KaTeX
  html = html.replace(/%%MATH_(\d+)%%/g, (_, idx) => {
    return mathBlocks[parseInt(idx)] || ''
  })

  return html
}

function App() {
  const [question, setQuestion] = useState('')
  const [selectedModels, setSelectedModels] = useState(['distilled'])
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState({})
  const [streaming, setStreaming] = useState({})
  const [submittedQuestion, setSubmittedQuestion] = useState('')
  const [error, setError] = useState('')
  const [availableModels, setAvailableModels] = useState({
    student: true,
    teacher: true,
    distilled: true,
  })
  const [streamDone, setStreamDone] = useState({})
  const [temperature, setTemperature] = useState(0.8)
  const [checkpoints, setCheckpoints] = useState([])
  const [selectedCheckpoint, setSelectedCheckpoint] = useState('')
  const abortRef = useRef(null)

  useEffect(() => {
    fetch('/api/models')
      .then(res => res.json())
      .then(data => {
        setAvailableModels(data)
        const available = Object.keys(data).filter(key => data[key])
        if (available.length > 0 && !selectedModels.some(m => available.includes(m))) {
          setSelectedModels([available[0]])
        }
      })
      .catch(err => console.error('获取模型状态失败:', err))

    fetch('/api/checkpoints')
      .then(res => res.json())
      .then(data => {
        const cps = data.checkpoints || []
        setCheckpoints(cps)
        if (cps.length > 0 && !selectedCheckpoint) {
          setSelectedCheckpoint(cps[0].name)
        }
      })
      .catch(err => console.error('获取检查点失败:', err))
  }, [])

  const toggleModel = (modelKey) => {
    if (!availableModels[modelKey]) return
    setSelectedModels(prev => {
      if (prev.includes(modelKey)) {
        if (prev.length === 1) return prev
        return prev.filter(m => m !== modelKey)
      }
      return [...prev, modelKey]
    })
  }

  // ---------- 普通请求（兜底） ----------
  const fetchNormal = async (q, models, temp, cp, signal) => {
    const response = await fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q, models, temperature: temp, checkpoint: cp }),
      signal,
    })
    const data = await response.json()
    if (!response.ok) throw new Error(data.detail || '请求失败')
    setResults(data.results || {})
    setLoading(false)
  }

  // ---------- 流式请求 ----------
  const fetchStream = async (q, models, temp, cp, signal) => {
    const response = await fetch('/api/generate/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q, models, temperature: temp, checkpoint: cp }),
      signal,
    })

    if (!response.ok) {
      // 流式端点不可用，回退到普通端点
      throw new Error('STREAM_NOT_AVAILABLE')
    }

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
          if (data.done) {
            setStreamDone(prev => ({ ...prev, [data.model]: true }))
            continue
          }
          if (data.token) {
            setStreaming(prev => ({
              ...prev,
              [data.model]: (prev[data.model] || '') + data.token,
            }))
          }
        } catch { /* 忽略解析错误 */ }
      }
    }
    setLoading(false)
  }

  const handleSubmit = async () => {
    if (!question.trim()) { setError('请输入题目'); return }
    if (selectedModels.length === 0) { setError('请至少选择一个模型'); return }

    // 如果正在生成中，点击按钮取消
    if (loading) {
      if (abortRef.current) abortRef.current.abort()
      setLoading(false)
      return
    }

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

    const q = question.trim()
    const models = [...selectedModels]

    const temp = temperature
    const cp = selectedCheckpoint

    try {
      // 先尝试流式请求
      await fetchStream(q, models, temp, cp, controller.signal)
    } catch (err) {
      if (err.name === 'AbortError') {
        setLoading(false)
        return
      }
      if (err.message === 'STREAM_NOT_AVAILABLE') {
        // 流式不可用，回退到普通请求
        try {
          setStreaming({})
          await fetchNormal(q, models, temp, cp, controller.signal)
        } catch (err2) {
          if (err2.name !== 'AbortError') {
            setError(err2.message)
          }
          setLoading(false)
        }
      } else {
        setError(err.message)
        setLoading(false)
      }
    }
  }

  const isStreaming = (modelKey) => {
    return loading && !streamDone[modelKey] && selectedModels.includes(modelKey)
  }

  const displayContent = (modelKey) => {
    if (loading || streamDone[modelKey]) {
      return streaming[modelKey] || ''
    }
    return results[modelKey] || ''
  }

  const hasAnyContent = () => {
    return Object.values(streaming).some(v => v && v.length > 0)
      || Object.values(results).some(v => v && v.length > 0)
  }

  return (
    <div className="app-container">
      <header className="header">
        <h1>📝 高考题模型对比</h1>
        <p>对比学生、教师、蒸馏模型对高考题的回答</p>
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

        <section className="model-section">
          <label>选择模型</label>
          <div className="model-checkboxes">
            {Object.entries(MODEL_CONFIG).map(([key, config]) => (
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
                <span className="badge">{key}</span>
              </label>
            ))}
          </div>
        </section>

        {selectedModels.includes('distilled') && checkpoints.length > 0 && (
          <section className="checkpoint-section">
            <label htmlFor="checkpoint-select">🔬 蒸馏模型检查点</label>
            <select
              id="checkpoint-select"
              value={selectedCheckpoint}
              onChange={(e) => setSelectedCheckpoint(e.target.value)}
            >
              {checkpoints.map(cp => (
                <option key={cp.name} value={cp.name}>{cp.name}</option>
              ))}
            </select>
          </section>
        )}

        <section className="temp-section">
          <label>
            🌡️ 随机性（温度）: <span className="temp-value">{temperature.toFixed(1)}</span>
          </label>
          <div className="temp-control">
            <span>精确</span>
            <input
              type="range"
              min="0.1"
              max="2.0"
              step="0.1"
              value={temperature}
              onChange={(e) => setTemperature(parseFloat(e.target.value))}
            />
            <span>随机</span>
          </div>
          <div className="temp-hint">温度越高，答案越多样。选择题推荐 0.8~1.2</div>
        </section>

        <button className="submit-btn" onClick={handleSubmit}>
          {loading ? '⏹ 停止生成' : '生成回答'}
        </button>

        {error && <div className="error-message">❌ {error}</div>}

        {submittedQuestion && hasAnyContent() && (
          <section className="results-section">
            <div className="question-card">
              <div className="question-header">📋 题目</div>
              <div className="question-content" dangerouslySetInnerHTML={{ __html: renderContent(submittedQuestion) }} />
            </div>

            <h2 className="results-title">回答结果</h2>

            {selectedModels.map(modelKey => {
              const config = MODEL_CONFIG[modelKey]
              const content = displayContent(modelKey)
              const active = isStreaming(modelKey)

              return (
                <div key={modelKey} className={`result-card ${active ? 'streaming' : ''}`}>
                  <div className={`result-header ${config.color}`}>
                    <span className="icon">{config.icon}</span>
                    <span>{config.name}</span>
                    {active && <span className="streaming-dot"></span>}
                    {streamDone[modelKey] && <span className="done-mark">✅</span>}
                  </div>
                  <div className="result-content">
                    {content ? (
                      <div
                        className="answer"
                        dangerouslySetInnerHTML={{ __html: renderContent(content) }}
                      />
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
