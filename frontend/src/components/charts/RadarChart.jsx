import { useRef, useEffect } from 'react'
import * as d3 from 'd3'
import { COLORS, getThemeColors } from './chartTheme'
import { useTheme } from '../../context/ThemeContext'

const PALETTE = [COLORS.accent, COLORS.purple, COLORS.success, COLORS.warning, COLORS.danger]

// Multi-model radar chart comparing metrics on one figure.
//   series:  [{ name, values: [0..1, ...], color? }]
//   metrics: ['Accuracy', 'Precision', ...]
function RadarChart({ series = [], metrics = [], title = 'Model Comparison', rangeMin = 0 }) {
  const containerRef = useRef(null)
  const { theme } = useTheme()

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const draw = () => {
      d3.select(container).selectAll('*').remove()

      const { bg, text, muted, border } = getThemeColors()
      const W  = container.clientWidth || 500
      const H  = 400
      const cx = W / 2
      const cy = H / 2 + 10   // nudge down slightly for title
      const radius = Math.min(W, H) * 0.32

      const svg = d3.select(container)
        .append('svg')
        .attr('width', '100%').attr('height', H)
        .attr('viewBox', `0 0 ${W} ${H}`)
        .style('background', bg)

      const N = metrics.length
      if (!N) return

      // Angle for axis i: start at 12-o'clock (-π/2), distribute clockwise
      const angle = i => -Math.PI / 2 + (2 * Math.PI / N) * i

      // Radial value → distance from centre
      const rScale = d3.scaleLinear().domain([rangeMin, 1]).range([0, radius])

      // Polygon helper (array of [x,y] → SVG points string)
      const toPolygon = pts => pts.map(p => p.join(',')).join(' ')

      // ── Grid rings ──────────────────────────────────────────────────────────
      const levels = 4
      for (let lv = 1; lv <= levels; lv++) {
        const r   = (radius / levels) * lv
        const val = rangeMin + (1 - rangeMin) * lv / levels
        const pts = Array.from({ length: N }, (_, i) => [
          cx + r * Math.cos(angle(i)),
          cy + r * Math.sin(angle(i)),
        ])
        svg.append('polygon')
          .attr('points', toPolygon(pts))
          .attr('fill', 'none')
          .attr('stroke', border)
          .attr('stroke-width', 0.8)

        // Level label (show at the rightmost spoke)
        svg.append('text')
          .attr('x', cx + r * Math.cos(angle(0)) + 4)
          .attr('y', cy + r * Math.sin(angle(0)))
          .attr('dominant-baseline', 'middle')
          .style('font-size', '9px').style('fill', muted)
          .text(val.toFixed(2))
      }

      // ── Spokes ──────────────────────────────────────────────────────────────
      Array.from({ length: N }, (_, i) => {
        svg.append('line')
          .attr('x1', cx).attr('y1', cy)
          .attr('x2', cx + radius * Math.cos(angle(i)))
          .attr('y2', cy + radius * Math.sin(angle(i)))
          .attr('stroke', border).attr('stroke-width', 0.8)

        // Axis label
        const labelR = radius * 1.2
        const ax     = angle(i)
        svg.append('text')
          .attr('x', cx + labelR * Math.cos(ax))
          .attr('y', cy + labelR * Math.sin(ax))
          .attr('text-anchor', 'middle').attr('dominant-baseline', 'middle')
          .style('font-size', '11px').style('fill', text)
          .text(metrics[i])
      })

      // ── Series polygons ─────────────────────────────────────────────────────
      // Tooltip
      const tip = d3.select(container)
        .append('div')
        .style('position', 'absolute').style('visibility', 'hidden')
        .style('background', bg).style('color', text)
        .style('padding', '7px 11px').style('border-radius', '6px')
        .style('font-size', '13px').style('pointer-events', 'none')
        .style('border', `1px solid ${muted}`).style('z-index', '20')
        .style('white-space', 'nowrap')

      series.forEach((s, si) => {
        const col  = s.color ?? PALETTE[si % PALETTE.length]
        const vals = s.values ?? []

        const pts = Array.from({ length: N }, (_, i) => {
          const v = Math.max(rangeMin, Math.min(1, vals[i] ?? rangeMin))
          const r = rScale(v)
          return [cx + r * Math.cos(angle(i)), cy + r * Math.sin(angle(i))]
        })

        svg.append('polygon')
          .attr('points', toPolygon(pts))
          .attr('fill', col).attr('fill-opacity', 0.15)
          .attr('stroke', col).attr('stroke-width', 1.8)
          .style('cursor', 'pointer')
          .on('mouseover', function (event) {
            d3.select(this).attr('fill-opacity', 0.35)
            const lines = metrics.map((m, i) => `${m}: ${(vals[i] ?? 0).toFixed(4)}`).join('<br>')
            tip.style('visibility', 'visible').html(`<strong>${s.name}</strong><br>${lines}`)
          })
          .on('mousemove', function (event) {
            const r = container.getBoundingClientRect()
            tip.style('top',  `${event.clientY - r.top  - 10}px`)
               .style('left', `${event.clientX - r.left + 12}px`)
          })
          .on('mouseout', function () {
            d3.select(this).attr('fill-opacity', 0.15)
            tip.style('visibility', 'hidden')
          })

        // Vertex dots
        pts.forEach(([px, py], i) => {
          svg.append('circle').attr('cx', px).attr('cy', py).attr('r', 4)
            .attr('fill', col).attr('stroke', bg).attr('stroke-width', 1)
        })
      })

      // ── Title ────────────────────────────────────────────────────────────────
      svg.append('text').attr('x', W / 2).attr('y', 20).attr('text-anchor', 'middle')
        .style('font-size', '14px').style('fill', text).text(title)

      // ── Legend ────────────────────────────────────────────────────────────────
      const legendY  = H - 14
      const itemW    = Math.min(100, (W - 40) / Math.max(series.length, 1))
      const startX   = (W - itemW * series.length) / 2

      series.forEach((s, i) => {
        const col = s.color ?? PALETTE[i % PALETTE.length]
        const lx  = startX + i * itemW
        svg.append('circle').attr('cx', lx + 6).attr('cy', legendY - 4).attr('r', 5).attr('fill', col)
        svg.append('text').attr('x', lx + 15).attr('y', legendY)
          .style('font-size', '11px').style('fill', text).text(s.name)
      })
    }

    draw()
    window.addEventListener('resize', draw)
    return () => window.removeEventListener('resize', draw)
  }, [series, metrics, title, rangeMin, theme])

  return <div ref={containerRef} style={{ width: '100%', minHeight: '400px', position: 'relative' }} />
}

export default RadarChart
