// Reads a CSS custom property off <html>, falling back to a dark-theme default.
function cssVar(name, fallback) {
  if (typeof window === 'undefined' || !window.getComputedStyle) return fallback
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim()
  return v || fallback
}

// Returns current theme tokens. Call inside a D3 useEffect (with `theme` in deps)
// so charts automatically recolor on light/dark toggle.
export function getThemeColors() {
  return {
    bg:     cssVar('--bg-card',       '#1a1d2e'),
    text:   cssVar('--text-primary',  '#e8eaf0'),
    muted:  cssVar('--text-muted',    '#8892a4'),
    border: cssVar('--border',        '#2a2d3e'),
  }
}

// Brand / semantic colors — theme-invariant.
export const COLORS = {
  accent:  '#00d4ff',
  purple:  '#6c63ff',
  success: '#00cc88',
  danger:  '#ff4d4d',
  warning: '#ffb347',
  muted:   '#8892a4',
}
