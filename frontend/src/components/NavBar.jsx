import { Link, useLocation } from 'react-router-dom'
import { useTheme } from '../context/ThemeContext'
import styles from './NavBar.module.css'

const LINKS = [
  { to: '/dashboard', label: 'Dashboard' },
  { to: '/spam', label: 'Spam Detector' },
  { to: '/malware', label: 'Malware Detector' },
  { to: '/analytics', label: 'Model Analytics' },
  { to: '/history', label: 'History' },
]

function NavBar() {
  const { pathname } = useLocation()
  const { theme, toggleTheme } = useTheme()
  return (
    <nav className={styles.navbar}>
      <div className={styles.left}>
        <span className={styles.logo}>NTCyber AI</span>
        <div className={styles.links}>
          {LINKS.map((l) => (
            <Link
              key={l.to}
              to={l.to}
              className={pathname === l.to ? `${styles.link} ${styles.active}` : styles.link}
            >
              {l.label}
            </Link>
          ))}
        </div>
      </div>
      <div className={styles.right}>
        <button
          className={styles.themeToggle}
          onClick={toggleTheme}
          aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}
          title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}
        >
          {theme === 'dark' ? '☀️' : '🌙'}
        </button>
        <span className={styles.subtitle}>NTCyber AI Platform</span>
      </div>
    </nav>
  )
}

export default NavBar
