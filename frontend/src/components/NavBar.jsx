import { Link, useLocation } from 'react-router-dom'
import styles from './NavBar.module.css'

const LINKS = [
  { to: '/dashboard', label: 'Dashboard' },
  { to: '/spam', label: 'Spam Detector' },
  { to: '/malware', label: 'Malware Detector' },
  { to: '/analytics', label: 'Model Analytics' },
]

function NavBar() {
  const { pathname } = useLocation()
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
      <span className={styles.subtitle}>NTCyber AI Platform</span>
    </nav>
  )
}

export default NavBar
