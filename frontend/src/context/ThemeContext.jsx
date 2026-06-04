import { createContext, useContext, useEffect, useState, useCallback } from 'react'

const STORAGE_KEY = 'ntcyber.theme'

// Default value so useTheme() works outside a provider (e.g. in unit tests that
// render a chart/page in isolation) — it just falls back to the dark theme.
const ThemeContext = createContext({ theme: 'dark', toggleTheme: () => {}, setTheme: () => {} })

function readInitialTheme() {
  if (typeof window === 'undefined') return 'dark'
  try {
    const saved = window.localStorage.getItem(STORAGE_KEY)
    if (saved === 'light' || saved === 'dark') return saved
  } catch {
    /* localStorage unavailable — fall through */
  }
  return 'dark' // dark is the default per product decision
}

export function ThemeProvider({ children }) {
  const [theme, setThemeState] = useState(readInitialTheme)

  // Reflect the theme onto <html data-theme> so the CSS variables in index.css apply.
  useEffect(() => {
    document.documentElement.dataset.theme = theme
    try {
      window.localStorage.setItem(STORAGE_KEY, theme)
    } catch {
      /* ignore persistence failures */
    }
  }, [theme])

  const setTheme = useCallback((next) => setThemeState(next === 'light' ? 'light' : 'dark'), [])
  const toggleTheme = useCallback(() => setThemeState((t) => (t === 'dark' ? 'light' : 'dark')), [])

  return (
    <ThemeContext.Provider value={{ theme, toggleTheme, setTheme }}>
      {children}
    </ThemeContext.Provider>
  )
}

export function useTheme() {
  return useContext(ThemeContext)
}
