import { useRef, useEffect } from 'react'
import * as d3 from 'd3'
import { COLORS, getThemeColors } from './chartTheme'
import { useTheme } from '../../context/ThemeContext'
import { addToolbar } from './chartToolbar'

const DEFAULT_COLORS = [COLORS.success, COLORS.danger, COLORS.warning, COLORS.accent, COLORS.purple]

function DonutChart({ labels = [], values = [], colors, title = 'Class Distribution' }) {
  const containerRef = useRef(null)
  const hiddenRef    = useRef(new Set())
  const { theme } = useTheme()

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const draw = () => {
      d3.select(container).selectAll('*').remove()

      const { bg, text, muted } = getThemeColors()
      const hidden      = hiddenRef.current
      const sliceColors = colors ?? DEFAULT_COLORS.slice(0, labels.length)
      const W = container.clientWidth || 400
      const H = 400
      const margin = { top: 44, right: 20, bottom: 70, left: 20 }
      const radius = Math.min(W - margin.left - margin.right, H - margin.top - margin.bottom) / 2

      const svg = d3.select(container)
        .append('svg')
        .attr('width', '100%')
        .attr('height', H)
        .attr('viewBox', `0 0 ${W} ${H}`)
        .style('background', bg)

      const cx = W / 2
      const cy = margin.top + radius

      const tip = d3.select(container)
        .append('div')
        .style('position', 'absolute')
        .style('visibility', 'hidden')
        .style('background', bg)
        .style('color', text)
        .style('padding', '7px 11px')
        .style('border-radius', '6px')
        .style('font-size', '13px')
        .style('pointer-events', 'none')
        .style('border', `1px solid ${muted}`)
        .style('z-index', '20')
        .style('white-space', 'nowrap')

      if (!labels.length || !values.length) {
        svg.append('text')
          .attr('x', cx).attr('y', cy)
          .attr('text-anchor', 'middle').attr('dominant-baseline', 'middle')
          .style('fill', muted).style('font-size', '13px')
          .text('No data')
        return
      }

      // Compute total ignoring hidden slices
      const visibleTotal = values.reduce((s, v, i) => s + (hidden.has(i) ? 0 : v), 0)

      const arc      = d3.arc().innerRadius(radius * 0.5).outerRadius(radius)
      const arcHover = d3.arc().innerRadius(radius * 0.5).outerRadius(radius * 1.06)
      const pie      = d3.pie().sort(null).value(d => d)

      const g = svg.append('g').attr('transform', `translate(${cx}, ${cy})`)

      g.selectAll('path')
        .data(pie(values))
        .join('path')
        .attr('id', (_, i) => `donut-slice-${i}`)
        .attr('d', arc)
        .attr('fill', (_, i) => sliceColors[i])
        .attr('stroke', bg)
        .attr('stroke-width', 2)
        .attr('opacity', (_, i) => hidden.has(i) ? 0 : 1)
        .style('cursor', 'pointer')
        .on('mouseover', function (event, d) {
          if (hidden.has(d.index)) return
          d3.select(this).transition().duration(120).attr('d', arcHover)
          const pct = visibleTotal > 0 ? ((d.data / visibleTotal) * 100).toFixed(1) : '0.0'
          tip.style('visibility', 'visible')
            .html(`<strong>${labels[d.index]}</strong>: ${d.data} (${pct}%)`)
        })
        .on('mousemove', function (event) {
          const r = container.getBoundingClientRect()
          tip.style('top',  `${event.clientY - r.top  - 10}px`)
             .style('left', `${event.clientX - r.left + 12}px`)
        })
        .on('mouseout', function (_, d) {
          if (hidden.has(d.index)) return
          d3.select(this).transition().duration(120).attr('d', arc)
          tip.style('visibility', 'hidden')
        })

      // Center total (visible only)
      const totalText = g.append('text').attr('text-anchor', 'middle').attr('dy', '-0.25em')
        .style('font-size', '22px').style('font-weight', '700').style('fill', text)
        .text(visibleTotal)
      g.append('text').attr('text-anchor', 'middle').attr('dy', '1.2em')
        .style('font-size', '11px').style('fill', muted).text('total')

      // Title
      svg.append('text')
        .attr('x', cx).attr('y', 20)
        .attr('text-anchor', 'middle')
        .style('font-size', '14px').style('fill', text)
        .text(title)

      // Clickable legend
      const legendY = margin.top + radius * 2 + 22
      const itemW   = Math.min(110, (W - 40) / labels.length)
      const startX  = (W - itemW * labels.length) / 2

      const legend = svg.append('g').attr('transform', `translate(${startX}, ${legendY})`)
      labels.forEach((lbl, i) => {
        const item = legend.append('g')
          .attr('transform', `translate(${i * itemW}, 0)`)
          .style('cursor', 'pointer')
          .attr('opacity', hidden.has(i) ? 0.4 : 1)

        item.append('circle').attr('cx', 7).attr('cy', 7).attr('r', 6).attr('fill', sliceColors[i])
        item.append('text').attr('x', 17).attr('y', 11)
          .style('font-size', '12px').style('fill', text).text(lbl)

        item.on('click', function () {
          if (hidden.has(i)) hidden.delete(i)
          else hidden.add(i)
          const isHidden = hidden.has(i)

          // Toggle slice visibility
          g.select(`#donut-slice-${i}`)
            .attr('opacity', isHidden ? 0 : 1)

          // Update center total to reflect visible slices only
          const newTotal = values.reduce((s, v, j) => s + (hidden.has(j) ? 0 : v), 0)
          totalText.text(newTotal)

          d3.select(this).attr('opacity', isHidden ? 0.4 : 1)
        })
      })

      // Save-only toolbar (no zoom/pan)
      addToolbar(container, { title, hasPanZoom: false })
    }

    draw()
    window.addEventListener('resize', draw)
    return () => window.removeEventListener('resize', draw)
  }, [labels, values, colors, title, theme])

  return <div ref={containerRef} className="chart-container" style={{ width: '100%', minHeight: '400px', position: 'relative' }} />
}

export default DonutChart
