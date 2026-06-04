import { Link } from 'react-router-dom'
import styles from './Footer.module.css'

const APP_VERSION = 'v2.0'

function Footer() {
  return (
    <footer className={styles.footer}>
      <div className={styles.inner}>
        <div className={styles.brandCol}>
          <span className={styles.brand}>NTCyber AI</span>
          <span className={styles.tagline}>Protect. Detect. Analyze.</span>
        </div>
        <nav className={styles.links} aria-label="Footer">
          <Link to="/dashboard" className={styles.link}>Dashboard</Link>
          <Link to="/spam" className={styles.link}>Spam Detector</Link>
          <Link to="/malware" className={styles.link}>Malware Detector</Link>
          <Link to="/analytics" className={styles.link}>Model Analytics</Link>
          <Link to="/history" className={styles.link}>History</Link>
        </nav>
        <div className={styles.meta}>
          <span>COS30049 · Group 1</span>
          <span className={styles.version}>{APP_VERSION}</span>
        </div>
      </div>
    </footer>
  )
}

export default Footer
