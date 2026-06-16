import { useState } from 'react'
import { Link as RouterLink, useLocation } from 'react-router-dom'
import { useTheme } from '../context/ThemeContext'
import AppBar from '@mui/material/AppBar'
import Toolbar from '@mui/material/Toolbar'
import Box from '@mui/material/Box'
import Button from '@mui/material/Button'
import IconButton from '@mui/material/IconButton'
import Drawer from '@mui/material/Drawer'
import List from '@mui/material/List'
import ListItemButton from '@mui/material/ListItemButton'
import ListItemText from '@mui/material/ListItemText'
import MenuIcon from '@mui/icons-material/Menu'
import LightModeIcon from '@mui/icons-material/LightMode'
import DarkModeIcon from '@mui/icons-material/DarkMode'

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

  return (
    <AppBar
      position="sticky"
      elevation={0}
      sx={{ bgcolor: 'background.paper', borderBottom: 1, borderColor: 'divider' }}
    >
      <Toolbar sx={{ minHeight: 60, justifyContent: 'space-between' }}>
        {/* Left: logo + desktop links */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <Box
            component={RouterLink}
            to="/dashboard"
            sx={{
              display: 'flex',
              alignItems: 'center',
              gap: 1,
              textDecoration: 'none',
              color: 'text.primary',
              fontWeight: 700,
              fontSize: 18,
              whiteSpace: 'nowrap',
            }}
          >
            <Box
              component="img"
              src="/NTCyber_AI_Logo.png"
              alt=""
              sx={{ width: 32, height: 32, objectFit: 'contain' }}
            />
            NTCyber AI
          </Box>

          <Box sx={{ display: { xs: 'none', sm: 'flex' }, gap: 0.5 }}>
            {LINKS.map((l) => {
              const active = pathname === l.to
              return (
                <Button
                  key={l.to}
                  component={RouterLink}
                  to={l.to}
                  sx={{
                    color: active ? 'primary.main' : 'text.secondary',
                    borderRadius: 0,
                    borderBottom: 2,
                    borderColor: active ? 'primary.main' : 'transparent',
                    textTransform: 'none',
                    fontSize: 14,
                    '&:hover': { color: 'text.primary', bgcolor: 'action.hover' },
                  }}
                >
                  {l.label}
                </Button>
              )
            })}
          </Box>
        </Box>

        {/* Right: theme toggle + mobile burger */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <IconButton
            onClick={toggleTheme}
            sx={{ color: 'text.primary' }}
            aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}
          >
            {theme === 'dark' ? <LightModeIcon /> : <DarkModeIcon />}
          </IconButton>

          <IconButton
            onClick={() => setMenuOpen(true)}
            sx={{ display: { xs: 'inline-flex', sm: 'none' }, color: 'text.primary' }}
            aria-label="Open navigation menu"
          >
            <MenuIcon />
          </IconButton>
        </Box>
      </Toolbar>

      {/* Mobile drawer */}
      <Drawer anchor="right" open={menuOpen} onClose={() => setMenuOpen(false)}>
        <Box sx={{ width: 240 }} role="presentation">
          <List>
            {LINKS.map((l) => (
              <ListItemButton
                key={l.to}
                component={RouterLink}
                to={l.to}
                selected={pathname === l.to}
                onClick={() => setMenuOpen(false)}
              >
                <ListItemText primary={l.label} />
              </ListItemButton>
            ))}
          </List>
        </Box>
      </Drawer>
    </AppBar>
  )
}

export default NavBar