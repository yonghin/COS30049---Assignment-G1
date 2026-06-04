// Reads a CSS custom property off <html>, falling back to a dark-theme default
// (jsdom returns '' for custom props, so charts still get sane colors in tests).
function cssVar(name, fallback) {
  if (typeof window === 'undefined' || !window.getComputedStyle) return fallback
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim()
  return v || fallback
}

// Builds a Plotly layout that matches the active theme. Charts call this inside
// their effect (with `theme` in the deps) so they recolor on light/dark toggle.
export function getChartLayout() {
  const bg = cssVar('--bg-card', '#1a1d2e')
  const text = cssVar('--text-primary', '#e8eaf0')
  const muted = cssVar('--text-muted', '#8892a4')
  const border = cssVar('--border', '#2a2d3e')
  return {
    paper_bgcolor: bg,
    plot_bgcolor: bg,
    font: { color: text, family: 'Inter, system-ui, sans-serif', size: 12 },
    xaxis: {
      gridcolor: border, zerolinecolor: border,
      tickfont: { color: muted }, titlefont: { color: muted },
    },
    yaxis: {
      gridcolor: border, zerolinecolor: border,
      tickfont: { color: muted }, titlefont: { color: muted },
    },
    legend: { bgcolor: 'transparent', font: { color: text }, orientation: 'h', y: -0.2 },
    margin: { l: 60, r: 30, t: 50, b: 60 },
    height: 400,
  }
}

// Brand/semantic colors that don't change with the theme.
export const COLORS = {
  accent: '#00d4ff',
  purple: '#6c63ff',
  success: '#00cc88',
  danger: '#ff4d4d',
  warning: '#ffb347',
  muted: '#8892a4',
}

// Back-compat export (static dark layout) for any caller not yet migrated.
export const DARK_LAYOUT = getChartLayout()

export const CHART_CONFIG = {
  responsive: true,
  displayModeBar: true,
  toImageButtonOptions: { format: 'png', scale: 2 },
  modeBarButtonsToRemove: ['sendDataToCloud'],
}
