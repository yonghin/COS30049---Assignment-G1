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

// Download icon: arrow-down + underline (24×24 Material Design path)
const downloadIcon = {
  width: 24,
  height: 24,
  path: 'M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z',
}

// CHART_CONFIG is consumed by every chart component via Plotly.newPlot(div, traces, layout, config).
// - toImage (camera) is removed and replaced with a custom download button placed last.
export function getChartConfig(plotlyInstance) {
  return {
    responsive: true,
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['sendDataToCloud', 'select2d', 'lasso2d', 'toImage'],
    modeBarButtonsToAdd: [
      {
        name: 'Download plot as PNG',
        title: 'Download plot as PNG',
        icon: downloadIcon,
        click: (gd) => plotlyInstance.downloadImage(gd, { format: 'png', scale: 2 }),
      },
    ],
  }
}

// Keep the old static export so any caller that hasn't migrated still compiles.
export const CHART_CONFIG = {
  responsive: true,
  displayModeBar: true,
  displaylogo: false,
  modeBarButtonsToRemove: ['sendDataToCloud', 'select2d', 'lasso2d'],
}
