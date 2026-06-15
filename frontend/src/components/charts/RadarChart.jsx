import { useRef, useEffect } from 'react'
import * as d3 from 'd3'
import { COLORS, getThemeColors } from './chartTheme'
import { useTheme } from '../../context/ThemeContext'
import { addToolbar } from './chartToolbar'

const PALETTE = [COLORS.accent, COLORS.purple, COLORS.success, COLORS.warning, COLORS.danger]

function RadarChart({ series = [], metrics = [], title = 'Model Comparison', rangeMin = 0 }) {
  const containerRef  = useRef(null)
  const hiddenRef     = useRef(new Set())      // persists legend toggle state
  const zoomTransform = useRef(d3.zoomIdentity)
  const { theme } = useTheme()

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const draw = () => {
      d3.select(container).selectAll('*').remove()

      const { bg, text, muted, border } = getThemeColors()
      const W   = container.clientWidth || 500
      const N   = metrics.length
      if (!N || !series.length) return

      // Legend layout — multi-row so long names aren't truncated
      const ITEM_W   = 135
      const PER_ROW  = Math.max(1, Math.min(series.length, Math.floor(W / ITEM_W)))
      const LEG_ROWS = Math.ceil(series.length / PER_ROW)
      const LEG_H    = LEG_ROWS * 22 + 8

      const H       = 400 + Math.max(0, (LEG_ROWS - 1) * 22)
      const legTopY = H - LEG_H + 6

      const cx     = W / 2
      const cy     = (H - LEG_H) / 2 + 30
      const radius = Math.min(W * 0.44 - 30, (H - LEG_H - 84) / 2)

      const svg = d3.select(container)
        .append('svg')
        .attr('width', '100%').attr('height', H)
        .attr('viewBox', `0 0 ${W} ${H}`)
        .style('background', bg)

      // Clip keeps radar body inside the space between title and legend
      svg.append('defs').append('clipPath').attr('id', 'rad-clip')
        .append('rect').attr('y', 30).attr('width', W).attr('height', H - 30 - LEG_H)

      // contentG holds the entire radar body (rings, spokes, labels, polygons, dots).
      // Only this group gets the zoom transform — title and legend stay fixed.
      const contentG = svg.append('g').attr('clip-path', 'url(#rad-clip)')

      const angle  = i => -Math.PI / 2 + (2 * Math.PI / N) * i
      const rScale = d3.scaleLinear().domain([rangeMin, 1]).range([0, radius])
      const toPoly = pts => pts.map(p => p.join(',')).join(' ')

      // ── Grid rings ────────────────────────────────────────────────────────
      const levels = 4
      for (let lv = 1; lv <= levels; lv++) {
        const r   = (radius / levels) * lv
        const val = rangeMin + (1 - rangeMin) * lv / levels
        const pts = Array.from({ length: N }, (_, i) => [
          cx + r * Math.cos(angle(i)),
          cy + r * Math.sin(angle(i)),
        ])
        contentG.append('polygon')
          .attr('points', toPoly(pts))
          .attr('fill', 'none').attr('stroke', border).attr('stroke-width', 0.8)

        contentG.append('text')
          .attr('x', cx + r * Math.cos(angle(0)) + 4)
          .attr('y', cy + r * Math.sin(angle(0)))
          .attr('dominant-baseline', 'middle')
          .style('font-size', '9px').style('fill', muted)
          .text(val.toFixed(2))
      }

      // ── Spokes + axis labels ─────────────────────────────────────────────
      Array.from({ length: N }, (_, i) => {
        contentG.append('line')
          .attr('x1', cx).attr('y1', cy)
          .attr('x2', cx + radius * Math.cos(angle(i)))
          .attr('y2', cy + radius * Math.sin(angle(i)))
          .attr('stroke', border).attr('stroke-width', 0.8)

        const labelR = radius * 1.2
        contentG.append('text')
          .attr('x', cx + labelR * Math.cos(angle(i)))
          .attr('y', cy + labelR * Math.sin(angle(i)))
          .attr('text-anchor', 'middle').attr('dominant-baseline', 'middle')
          .style('font-size', '11px').style('fill', text)
          .text(metrics[i])
      })

      // ── Tooltip ───────────────────────────────────────────────────────────
      const tip = d3.select(container)
        .append('div')
        .style('position', 'absolute').style('visibility', 'hidden')
        .style('background', bg).style('color', text)
        .style('padding', '7px 11px').style('border-radius', '6px')
        .style('font-size', '13px').style('pointer-events', 'none')
        .style('border', `1px solid ${muted}`).style('z-index', '20')
        .style('white-space', 'nowrap')

      // ── Series polygons + dots ────────────────────────────────────────────
      series.forEach((s, si) => {
        const col      = s.color ?? PALETTE[si % PALETTE.length]
        const vals     = s.values ?? []
        const isHidden = hiddenRef.current.has(si)

        const pts = Array.from({ length: N }, (_, i) => {
          const v = Math.max(rangeMin, Math.min(1, vals[i] ?? rangeMin))
          return [cx + rScale(v) * Math.cos(angle(i)), cy + rScale(v) * Math.sin(angle(i))]
        })

        contentG.append('polygon')
          .attr('id', `radar-poly-${si}`)
          .attr('points', toPoly(pts))
          .attr('fill', col).attr('fill-opacity', 0.15)
          .attr('stroke', col).attr('stroke-width', 1.8)
          .attr('visibility', isHidden ? 'hidden' : 'visible')
          .style('cursor', 'pointer')
          .on('mouseover', function (event) {
            if (hiddenRef.current.has(si)) return
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

        pts.forEach(([px, py]) => {
          contentG.append('circle')
            .attr('class', `radar-dot-${si}`)
            .attr('cx', px).attr('cy', py).attr('r', 4)
            .attr('fill', col).attr('stroke', bg).attr('stroke-width', 1)
            .attr('visibility', isHidden ? 'hidden' : 'visible')
        })
      })

      // ── Title — fixed in svg (not inside contentG) ────────────────────────
      svg.append('text').attr('x', W / 2).attr('y', 20).attr('text-anchor', 'middle')
        .style('font-size', '14px').style('fill', text).text(title)

      // ── Clickable legend — fixed in svg ───────────────────────────────────
      series.forEach((s, i) => {
        const col        = s.color ?? PALETTE[i % PALETTE.length]
        const row        = Math.floor(i / PER_ROW)
        const colIdx     = i % PER_ROW
        const itemsInRow = Math.min(PER_ROW, series.length - row * PER_ROW)
        const rowStartX  = (W - itemsInRow * ITEM_W) / 2
        const lx         = rowStartX + colIdx * ITEM_W
        const ly         = legTopY + row * 22

        const legendG = svg.append('g')
          .style('cursor', 'pointer')
          .attr('opacity', hiddenRef.current.has(i) ? 0.4 : 1)

        legendG.append('circle')
          .attr('cx', lx + 6).attr('cy', ly - 5).attr('r', 5).attr('fill', col)

        legendG.append('text')
          .attr('x', lx + 15).attr('y', ly)
          .style('font-size', '11px').style('fill', text)
          .text(s.name)

        // svg.select searches all descendants including contentG children
        legendG.on('click', function () {
          const hidden = hiddenRef.current
          if (hidden.has(i)) hidden.delete(i)
          else hidden.add(i)
          const nowHidden = hidden.has(i)
          svg.select(`#radar-poly-${i}`).attr('visibility', nowHidden ? 'hidden' : 'visible')
          svg.selectAll(`.radar-dot-${i}`).attr('visibility', nowHidden ? 'hidden' : 'visible')
          d3.select(this).attr('opacity', nowHidden ? 0.4 : 1)
        })
      })

      // Content-only zoom: radar body zooms, title and legend stay fixed.
      // translateExtent is anchored to the radar content rect (center ± radius*1.3 covers
      // polygon + axis labels) so D3 always keeps the viewport overlapping the content.
      const zoomBehavior = d3.zoom()
        .filter(e => e.type !== 'wheel')
        .scaleExtent([1, 5])
        .extent([[0, 30], [W, H - LEG_H]])
        .translateExtent([
          [cx - radius * 1.3, cy - radius * 1.3],
          [cx + radius * 1.3, cy + radius * 1.3],
        ])
        .on('zoom', event => {
          zoomTransform.current = event.transform
          contentG.attr('transform', event.transform.toString())
        })

      svg.call(zoomBehavior).call(zoomBehavior.transform, zoomTransform.current)

      addToolbar(container, {
        svgSel: svg, zoomBehavior,
        onReset: () => { zoomTransform.current = d3.zoomIdentity },
        title,
      })
    }

    draw()
    window.addEventListener('resize', draw)
    return () => window.removeEventListener('resize', draw)
  }, [series, metrics, title, rangeMin, theme])

  return <div ref={containerRef} style={{ width: '100%', minHeight: '400px', position: 'relative' }} />
}

export default RadarChart
