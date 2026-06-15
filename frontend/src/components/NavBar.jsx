import { useState, useEffect, useRef } from 'react'
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
  const [menuOpen, setMenuOpen] = useState(false)
  const menuRef = useRef(null)

  // Close menu when clicking outside
  useEffect(() => {
    if (!menuOpen) return
    const handleClick = (e) => {
      if (menuRef.current && !menuRef.current.contains(e.target)) {
        setMenuOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [menuOpen])

  // Close menu on route change
  useEffect(() => { setMenuOpen(false) }, [pathname])

  return (
    <nav className={styles.navbar} ref={menuRef}>
      <div className={styles.left}>
        <Link to="/dashboard" className={styles.logo}>NTCyber AI</Link>
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

        <button
          className={`${styles.burger} ${menuOpen ? styles.burgerOpen : ''}`}
          onClick={() => setMenuOpen((o) => !o)}
          aria-label="Toggle navigation menu"
          aria-expanded={menuOpen}
        >
          <span />
          <span />
          <span />
        </button>
      </div>

      {/* Mobile dropdown menu */}
      {menuOpen && (
        <div className={styles.mobileMenu}>
          {LINKS.map((l) => (
            <Link
              key={l.to}
              to={l.to}
              className={pathname === l.to ? `${styles.mobileLink} ${styles.mobileLinkActive}` : styles.mobileLink}
              onClick={() => setMenuOpen(false)}
            >
              {l.label}
            </Link>
          ))}
        </div>
      )}
    </nav>
  )
}

export default NavBar
