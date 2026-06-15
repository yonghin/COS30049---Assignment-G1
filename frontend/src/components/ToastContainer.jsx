import styles from './ToastContainer.module.css'

const ICONS = { success: '✓', error: '✕', info: 'ℹ' }

function ToastContainer({ toasts = [], onDismiss }) {
  if (toasts.length === 0) return null
  return (
    <div className={styles.container} role="status" aria-live="polite">
      {toasts.map((t) => (
        <div
          key={t.id}
          className={`${styles.toast} ${styles[t.type] ?? ''} ${t.leaving ? styles.leaving : ''}`}
        >
          <span className={styles.icon} aria-hidden="true">{ICONS[t.type] ?? 'ℹ'}</span>
          <span className={styles.message}>{t.message}</span>
          {t.onUndo && (
            <button
              className={styles.undoBtn}
              onClick={() => { t.onUndo(); onDismiss(t.id) }}
            >
              Undo{t.secondsLeft != null ? ` (${t.secondsLeft}s)` : ''}
            </button>
          )}
          <button className={styles.close} onClick={() => onDismiss(t.id)} aria-label="Dismiss">✕</button>
        </div>
      ))}
    </div>
  )
}

export default ToastContainer
