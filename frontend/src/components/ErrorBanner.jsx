import styles from './ErrorBanner.module.css'

function ErrorBanner({ message, onDismiss }) {
  if (message === null || message === undefined) return null
  return (
    <div className={styles.banner}>
      <span>{message}</span>
      <button className={styles.dismiss} onClick={onDismiss} aria-label="Dismiss error">
        ✕
      </button>
    </div>
  )
}

export default ErrorBanner
