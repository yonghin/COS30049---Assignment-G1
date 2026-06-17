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

// Shared tooltip box — clean card look (no coloured border), soft shadow.
// Usage: const tip = createTooltip(d3, container)
export function createTooltip(d3, container) {
  const isLight = typeof document !== 'undefined' &&
    document.documentElement.getAttribute('data-theme') === 'light'

  return d3.select(container)
    .append('div')
    .attr('class', 'chart-tooltip')
    .style('position', 'absolute')
    .style('visibility', 'hidden')
    .style('pointer-events', 'none')
    .style('z-index', '20')
    .style('padding', '10px 13px')
    .style('border-radius', '10px')
    .style('font-size', '12.5px')
    .style('line-height', '1.6')
    .style('font-family', 'inherit')
    .style('white-space', 'nowrap')
    .style('color', isLight ? '#1f2733' : '#e8eaf0')
    .style('background', isLight ? '#ffffff' : '#161a28')
    .style('border', `1px solid ${isLight ? 'rgba(0,0,0,0.08)' : 'rgba(255,255,255,0.08)'}`)
    .style('box-shadow', isLight
      ? '0 8px 24px rgba(0,0,0,0.14)'
      : '0 8px 24px rgba(0,0,0,0.5)')
}

// Builds one tooltip line: small colour dot + label + bold value.
//   tipTitle(name)              -> bold heading line
//   tipRow('Accuracy', '97.5%', '#00d4ff')  -> dot + "Accuracy" + bold value
export function tipTitle(name, color) {
  const c = color ? `color:${color};` : ''
  return `<div style="font-weight:700;font-size:13.5px;margin-bottom:5px;${c}">${name}</div>`
}

export function tipRow(label, value, color) {
  const dot = color
    ? `<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${color};margin-right:7px;vertical-align:middle;"></span>`
    : ''
  return `<div style="margin:2px 0;">${dot}<span style="opacity:0.75;">${label}</span>` +
         `<span style="font-weight:700;margin-left:6px;">${value}</span></div>`
}