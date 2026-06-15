import { createContext, useContext, useState, useCallback, useRef } from 'react'
import ToastContainer from '../components/ToastContainer'

const noop = () => {}
const ToastContext = createContext({ success: noop, error: noop, info: noop, infoWithUndo: noop })

let nextId = 0
const EXIT_MS = 300

export function ToastProvider({ children }) {
  const [toasts, setToasts]  = useState([])
  const timers    = useRef({})
  const intervals = useRef({})

  const remove = useCallback((id) => {
    setToasts((list) => list.filter((t) => t.id !== id))
    clearTimeout(timers.current[id]);     delete timers.current[id]
    clearInterval(intervals.current[id]); delete intervals.current[id]
  }, [])

  // Triggers slide-out animation, then removes after EXIT_MS.
  const dismiss = useCallback((id) => {
    setToasts((list) => list.map((t) => t.id === id ? { ...t, leaving: true } : t))
    clearTimeout(timers.current[id]);     delete timers.current[id]
    clearInterval(intervals.current[id]); delete intervals.current[id]
    setTimeout(() => remove(id), EXIT_MS)
  }, [remove])

  const push = useCallback((type, message, duration = 3500, options = {}) => {
    const id = ++nextId
    const hasUndo     = typeof options.onUndo === 'function'
    const initSeconds = hasUndo ? Math.round(duration / 1000) : undefined

    setToasts((list) => [...list, {
      id, type, message, leaving: false,
      onUndo:      hasUndo ? options.onUndo : null,
      secondsLeft: initSeconds,
    }])

    if (hasUndo) {
      intervals.current[id] = setInterval(() => {
        setToasts((list) =>
          list.map((t) =>
            t.id === id
              ? { ...t, secondsLeft: Math.max(0, (t.secondsLeft ?? 1) - 1) }
              : t
          )
        )
      }, 1000)
    }

    timers.current[id] = setTimeout(() => dismiss(id), duration)
    return id
  }, [dismiss])

  const api = useRef({
    success:      (m, d)         => push('success', m, d),
    error:        (m, d)         => push('error',   m, d),
    info:         (m, d)         => push('info',    m, d),
    infoWithUndo: (m, onUndo, d = 10000) => push('info', m, d, { onUndo }),
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
