import { Link } from 'react-router-dom'
import styles from './PageHeader.module.css'

// Consistent page title + subtitle + breadcrumb trail across pages.
// crumbs: optional [{ label, to }]; the current page (title) is appended automatically.
function PageHeader({ title, subtitle, crumbs = [{ label: 'Home', to: '/dashboard' }] }) {
  return (
    <header className={styles.header}>
      <nav className={styles.crumbs} aria-label="Breadcrumb">
        {crumbs.map((c) => (
          <span key={c.to} className={styles.crumbItem}>
            <Link to={c.to} className={styles.crumbLink}>{c.label}</Link>
            <span className={styles.sep} aria-hidden="true">/</span>
          </span>
        ))}
        <span className={styles.crumbCurrent}>{title}</span>
      </nav>
      <h1 className={styles.title}>{title}</h1>
      {subtitle && <p className={styles.subtitle}>{subtitle}</p>}
    </header>
  )
}

export default PageHeader
