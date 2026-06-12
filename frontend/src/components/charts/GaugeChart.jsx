import { useRef, useEffect } from 'react'
import * as d3 from 'd3'
import { COLORS, getThemeColors } from './chartTheme'
import { useTheme } from '../../context/ThemeContext'
import { addToolbar } from './chartToolbar'

// Semicircle gauge: flat side at bottom, 0 at 9-o'clock, 100 at 3-o'clock.
function GaugeChart({ spamProb = null, label }) {
  const containerRef  = useRef(null)
  const zoomTransform = useRef(d3.zoomIdentity)
  const { theme } = useTheme()

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const draw = () => {
      d3.select(container).selectAll('*').remove()

      const { bg, text, muted, border } = getThemeColors()
      const W  = container.clientWidth || 400
      const H  = 300
      const cx = W / 2
      const cy = H * 0.78
      const radius = Math.min(cx * 0.75, cy - 30)

      const svg = d3.select(container)
        .append('svg')
        .attr('width', '100%').attr('height', H)
        .attr('viewBox', `0 0 ${W} ${H}`)
        .style('background', bg)

      // Clip keeps gauge inside the SVG during content-only zoom
      svg.append('defs').append('clipPath').attr('id', 'gauge-clip')
        .append('rect').attr('width', W).attr('height', H)

      // contentG is the only thing that gets the zoom transform.
      // Since the gauge has no separate fixed title/axes, the whole body zooms together.
      const contentG = svg.append('g').attr('clip-path', 'url(#gauge-clip)')

      const val      = spamProb ?? 0
      const barColor = val >= 0.5 ? COLORS.danger : COLORS.success
      const valuePct = Math.round(val * 100)

      // Angle scale: 0 → -π/2 (9-o'clock), 100 → π/2 (3-o'clock)
      const startAngle = -Math.PI / 2
      const endAngle   =  Math.PI / 2
      const aScale = d3.scaleLinear().domain([0, 100]).range([startAngle, endAngle])

      const mkArc = (r0, r1, a0, a1) =>
        d3.arc().innerRadius(r0).outerRadius(r1).startAngle(a0).endAngle(a1)()

      const g = contentG.append('g').attr('transform', `translate(${cx},${cy})`)

      // Track
      g.append('path').attr('d', mkArc(radius * 0.6, radius, startAngle, endAngle))
        .attr('fill', border)

      // Green zone (0-50)
      g.append('path').attr('d', mkArc(radius * 0.6, radius, startAngle, aScale(50)))
        .attr('fill', 'rgba(0,204,136,0.18)')

      // Red zone (50-100)
      g.append('path').attr('d', mkArc(radius * 0.6, radius, aScale(50), endAngle))
        .attr('fill', 'rgba(255,77,77,0.18)')

      // Value arc
      g.append('path').attr('d', mkArc(radius * 0.62, radius * 0.88, startAngle, aScale(valuePct)))
        .attr('fill', barColor)

      // Threshold line
      const a  = aScale(valuePct)
      const sx = Math.sin(a), cy2 = Math.cos(a)
      g.append('line')
        .attr('x1', radius * 0.55 * sx).attr('y1', -radius * 0.55 * cy2)
        .attr('x2', radius       * sx).attr('y2', -radius       * cy2)
        .attr('stroke', barColor).attr('stroke-width', 3)

      // Tick marks + labels (0, 25, 50, 75, 100)
      ;[0, 25, 50, 75, 100].forEach(tick => {
        const ta = aScale(tick)
        const ts = Math.sin(ta), tc = Math.cos(ta)
        g.append('line')
          .attr('x1', radius * 1.02 * ts).attr('y1', -radius * 1.02 * tc)
          .attr('x2', radius * 1.12 * ts).attr('y2', -radius * 1.12 * tc)
          .attr('stroke', muted).attr('stroke-width', 1.5)
        g.append('text')
          .attr('x', radius * 1.25 * ts).attr('y', -radius * 1.25 * tc)
          .attr('text-anchor', 'middle').attr('dominant-baseline', 'middle')
          .style('font-size', '10px').style('fill', muted).text(tick)
      })

      // Centre value
      g.append('text').attr('text-anchor', 'middle').attr('y', -radius * 0.12)
        .style('font-size', '36px').style('font-weight', '700').style('fill', barColor)
        .text(`${valuePct}%`)

      // Label below the arc opening
      g.append('text').attr('text-anchor', 'middle').attr('y', 28)
        .style('font-size', '13px').style('fill', muted)
        .text(label ?? 'Spam Probability')

      // Content-only zoom — the whole gauge body zooms together
      const zoomBehavior = d3.zoom()
        .filter(e => e.type !== 'wheel')
        .scaleExtent([0.5, 5])
        .on('zoom', event => {
          zoomTransform.current = event.transform
          contentG.attr('transform', event.transform.toString())
        })

      svg.call(zoomBehavior).call(zoomBehavior.transform, zoomTransform.current)

      addToolbar(container, {
        svgSel: svg, zoomBehavior,
        onReset: () => { zoomTransform.current = d3.zoomIdentity },
        title: label ?? 'Gauge',
      })
    }

    draw()
    window.addEventListener('resize', draw)
    return () => window.removeEventListener('resize', draw)
  }, [spamProb, label, theme])

  return <div ref={containerRef} style={{ width: '100%', minHeight: '300px', position: 'relative' }} />
}

export default GaugeChart
