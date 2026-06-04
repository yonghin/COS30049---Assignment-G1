import NavBar from './NavBar'
import Footer from './Footer'
import styles from './Layout.module.css'

// App shell: fixed NavBar + page content + Footer pinned to the bottom.
function Layout({ children }) {
  return (
    <div className={styles.shell}>
      <NavBar />
      <main className={styles.main}>{children}</main>
      <Footer />
    </div>
  )
}

export default Layout
