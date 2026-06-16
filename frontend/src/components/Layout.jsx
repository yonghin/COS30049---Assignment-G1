import NavBar from './NavBar'
import Footer from './Footer'
import Box from '@mui/material/Box'

// App shell: NavBar + page content + Footer pinned to the bottom.
function Layout({ children }) {
  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
      <NavBar />
      <Box component="main" sx={{ flex: '1 0 auto' }}>
        {children}
      </Box>
      <Footer />
    </Box>
  )
}

export default Layout