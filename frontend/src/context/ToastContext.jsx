import { createContext, useContext, useState, useCallback, useRef } from 'react'
import ToastContainer from '../components/ToastContainer'

// no-op default so useToast() is safe outside a provider (unit tests render pages
// in isolation without wrapping them in <ToastProvider>).
const noop = () => {}
const ToastContext = createContext({ success: noop, error: noop, info: noop })

let nextId = 0

export function ToastProvider({ children }) {
  const [toasts, setToasts] = useState([])
  const timers = useRef({})

  const dismiss = useCallback((id) => {
    setToasts((list) => list.filter((t) => t.id !== id))
    if (timers.current[id]) {
      clearTimeout(timers.current[id])
      delete timers.current[id]
    }
  }, [])

  const push = useCallback(
    (type, message, duration = 3500) => {
      const id = ++nextId
      setToasts((list) => [...list, { id, type, message }])
      timers.current[id] = setTimeout(() => dismiss(id), duration)
      return id
    },
    [dismiss]
  )

  const api = useRef({
    success: (m, d) => push('success', m, d),
    error: (m, d) => push('error', m, d),
    info: (m, d) => push('info', m, d),
  })

  return (
    <ToastContext.Provider value={api.current}>
      {children}
      <ToastContainer toasts={toasts} onDismiss={dismiss} />
    </ToastContext.Provider>
  )
}

export function useToast() {
  return useContext(ToastContext)
}
