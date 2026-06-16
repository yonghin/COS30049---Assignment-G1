import { createContext, useContext, useLayoutEffect, useState, useCallback, useMemo } from 'react'
import { ThemeProvider as MuiThemeProvider, createTheme } from '@mui/material/styles'
import CssBaseline from '@mui/material/CssBaseline'

const STORAGE_KEY = 'ntcyber.theme'

// Default value so useTheme() works outside a provider (e.g. in unit tests).
const ThemeContext = createContext({ theme: 'dark', toggleTheme: () => {}, setTheme: () => {} })

function readInitialTheme() {
  if (typeof window === 'undefined') return 'dark'
  try {
    const saved = window.localStorage.getItem(STORAGE_KEY)
    if (saved === 'light' || saved === 'dark') return saved
  } catch {
    /* localStorage unavailable */
  }
  return 'dark'
}

export function ThemeProvider({ children }) {
  const [theme, setThemeState] = useState(readInitialTheme)

  // Keep <html data-theme> in sync so the D3 charts and index.css CSS variables still work.
  useLayoutEffect(() => {
    document.documentElement.dataset.theme = theme
    try {
      window.localStorage.setItem(STORAGE_KEY, theme)
    } catch {
      /* ignore persistence failures */
    }
  }, [theme])

  const setTheme = useCallback((next) => setThemeState(next === 'light' ? 'light' : 'dark'), [])
  const toggleTheme = useCallback(() => setThemeState((t) => (t === 'dark' ? 'light' : 'dark')), [])

  // Build an MUI theme that mirrors the light/dark state.
  const muiTheme = useMemo(() => {
    const isDark = theme === 'dark'
    return createTheme({
      palette: {
        mode: theme,
        primary:   { main: isDark ? '#00d4ff' : '#0066cc' },
        secondary: { main: isDark ? '#6c63ff' : '#5b54e6' },
        error:     { main: isDark ? '#ff4d4d' : '#c42b2b' },
        success:   { main: isDark ? '#00cc88' : '#0a8c5e' },
        warning:   { main: isDark ? '#ffb347' : '#b8680e' },
        background: {
          default: isDark ? '#0f1117' : '#d8e8f8',
          paper:   isDark ? '#1a1d2e' : '#eaf3ff',
        },
        text: {
          primary:   isDark ? '#e8eaf0' : '#0c1a30',
          secondary: isDark ? '#8892a4' : '#3e5575',
        },
        divider: isDark ? '#2a2d3e' : '#aec8e4',
      },
      typography: {
        fontFamily: "Inter, 'Segoe UI', system-ui, sans-serif",
      },
      shape: {
        borderRadius: 8,
      },
      components: {
        MuiButton: {
          defaultProps: { disableElevation: true },
          styleOverrides: {
            root: { textTransform: 'none', borderRadius: 8, fontWeight: 600 },
          },
        },
        MuiPaper: {
          styleOverrides: {
            rounded: { borderRadius: 12 },
          },
        },
      },
    })
  }, [theme])

  return (
    <ThemeContext.Provider value={{ theme, toggleTheme, setTheme }}>
      <MuiThemeProvider theme={muiTheme}>
        <CssBaseline />
        {children}
      </MuiThemeProvider>
    </ThemeContext.Provider>
  )
}

export function useTheme() {
  return useContext(ThemeContext)
}