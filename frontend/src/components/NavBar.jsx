import { useState } from 'react'
import { Link as RouterLink, useLocation } from 'react-router-dom'
import { useTheme } from '../context/ThemeContext'
import AppBar from '@mui/material/AppBar'
import Toolbar from '@mui/material/Toolbar'
import Box from '@mui/material/Box'
import Button from '@mui/material/Button'
import IconButton from '@mui/material/IconButton'
import Collapse from '@mui/material/Collapse'
import MenuIcon from '@mui/icons-material/Menu'
import CloseIcon from '@mui/icons-material/Close'
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
                    '&:hover': {
                      color: 'primary.main',
                      bgcolor: 'action.hover',
                      borderColor: 'primary.main',
                    },
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
            onClick={() => setMenuOpen((o) => !o)}
            sx={{ display: { xs: 'inline-flex', sm: 'none' }, color: 'text.primary' }}
            aria-label={menuOpen ? 'Close navigation menu' : 'Open navigation menu'}
            aria-expanded={menuOpen}
          >
            {menuOpen ? <CloseIcon /> : <MenuIcon />}
          </IconButton>
        </Box>
      </Toolbar>

      {/* Mobile dropdown menu, expands from below the navbar */}
      <Collapse in={menuOpen} timeout="auto" unmountOnExit>
        <Box
          sx={{
            display: { xs: 'flex', sm: 'none' },
            flexDirection: 'column',
            bgcolor: 'background.paper',
            borderTop: 1,
            borderColor: 'divider',
            py: 1,
          }}
        >
          {LINKS.map((l) => {
            const active = pathname === l.to
            return (
              <Box
                key={l.to}
                component={RouterLink}
                to={l.to}
                onClick={() => setMenuOpen(false)}
                sx={{
                  display: 'block',
                  px: 3,
                  py: 1.75,
                  fontSize: 15,
                  textDecoration: 'none',
                  color: active ? 'primary.main' : 'text.secondary',
                  borderLeft: '3px solid',
                  borderColor: active ? 'primary.main' : 'transparent',
                  bgcolor: active ? 'action.hover' : 'transparent',
                  '&:hover': {
                    color: 'primary.main',
                    bgcolor: (theme) => theme.palette.mode === 'dark' ? 'rgba(0,212,255,0.08)' : 'rgba(0,102,204,0.08)',
                    borderColor: 'primary.main',
                  },
                }}
              >
                {l.label}
              </Box>
            )
          })}
        </Box>
      </Collapse>
    </AppBar>
  )
}

export default NavBar