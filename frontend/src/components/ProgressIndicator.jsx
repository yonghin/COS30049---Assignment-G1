import styles from './ProgressIndicator.module.css'

function ProgressIndicator({ visible = false, label }) {
  if (!visible) return null
  return (
    <div className={styles.wrap} role="status" aria-live="polite">
      <div className={styles.spinner} />
      {label && <div className={styles.label}>{label}</div>}
    </div>
  )
}

export default ProgressIndicator
