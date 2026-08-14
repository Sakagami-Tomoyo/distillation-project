import { StrictMode, useState, useEffect } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App.jsx'
import ChatApp from './ChatApp.jsx'

// 轻量 hash 路由：默认高考问答，`#/chat` 进入智能业务助手
function Router() {
  const [route, setRoute] = useState(window.location.hash)

  useEffect(() => {
    const onChange = () => setRoute(window.location.hash)
    window.addEventListener('hashchange', onChange)
    return () => window.removeEventListener('hashchange', onChange)
  }, [])

  return route.startsWith('#/chat') ? <ChatApp /> : <App />
}

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <Router />
  </StrictMode>,
)
