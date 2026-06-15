// Frontend-only prediction history, persisted in localStorage (per-browser).
// Survives reloads; never sent to the backend. Capped to the most recent MAX entries.
const KEY = 'ntcyber.history'
const MAX = 200

const listeners = new Set()

function read() {
  if (typeof window === 'undefined') return []
  try {
    const raw = window.localStorage.getItem(KEY)
    const parsed = raw ? JSON.parse(raw) : []
    return Array.isArray(parsed) ? parsed : []
  } catch {
    return []
  }
}

function write(list) {
  try {
    window.localStorage.setItem(KEY, JSON.stringify(list))
  } catch {
    /* quota / unavailable — ignore */
  }
  listeners.forEach((cb) => cb(list))
}

export function getHistory() {
  return read()
}

// entry: { kind: 'spam'|'malware', model, label, confidence, summary }
export function recordPrediction(entry) {
  const list = read()
  const record = {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    ts: new Date().toISOString(),
    ...entry,
  }
  const next = [record, ...list].slice(0, MAX)
  write(next)
  return record
}

export function deleteHistoryItems(ids) {
  const idSet = new Set(ids)
  write(read().filter((item) => !idSet.has(item.id)))
}

export function clearHistory() {
  write([])
}

export function restoreHistory(items) {
  write(Array.isArray(items) ? items : [])
}

// Subscribe to changes; returns an unsubscribe function.
export function subscribe(cb) {
  listeners.add(cb)
  return () => listeners.delete(cb)
}
