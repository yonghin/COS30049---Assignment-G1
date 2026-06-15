import { useRef, useEffect } from 'react'
import * as d3 from 'd3'
import { COLORS, getThemeColors } from './chartTheme'
import { useTheme } from '../../context/ThemeContext'
import { addToolbar } from './chartToolbar'

let _histCounter = 0

function Histogram({
  values = [],
  title  = 'Confidence Distribution',
  xLabel = 'Confidence',
  color  = COLORS.accent,
  nbins  = 20,
}) {
  const containerRef  = useRef(null)
  const zoomTransform = useRef(d3.zoomIdentity)
  const chartId       = useRef(`hist${++_histCounter}`)
  const { theme } = useTheme()

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const draw = () => {
      d3.select(container).selectAll('*').remove()

      const { bg, text, muted, border } = getThemeColors()
      const cid = chartId.current
      const W  = container.clientWidth || 500
      const H  = 400
      const m  = { top: 44, right: 30, bottom: 60, left: 60 }
      const iW = W - m.left - m.right
      const iH = H - m.top  - m.bottom

      const svg = d3.select(container)
        .append('svg')
        .attr('width', '100%').attr('height', H)
        .attr('viewBox', `0 0 ${W} ${H}`)
        .style('background', bg)

      if (!values.length) {
        svg.append('text')
          .attr('x', W / 2).attr('y', H / 2)
          .attr('text-anchor', 'middle').attr('dominant-baseline', 'middle')
          .style('fill', muted).style('font-size', '13px').text('No data')
        return
      }

      const defs = svg.append('defs')

      // Clip bars to chart area
      defs.append('clipPath').attr('id', `hist-clip-${cid}`)
        .append('rect').attr('width', iW).attr('height', iH)

      // Clip x-axis labels (in xAxisG's local coordinate system: axis line is y=0)
      defs.append('clipPath').attr('id', `hist-xclip-${cid}`)
        .append('rect').attr('x', 0).attr('y', -2).attr('width', iW).attr('height', m.bottom + 4)

      const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`)

      const tip = d3.select(container)
        .append('div')
        .style('position', 'absolute').style('visibility', 'hidden')
        .style('background', bg).style('color', text)
        .style('padding', '7px 11px').style('border-radius', '6px')
        .style('font-size', '13px').style('pointer-events', 'none')
        .style('border', `1px solid ${muted}`).style('z-index', '20')
        .style('white-space', 'nowrap')

      const axisStyle = ax => {
        ax.select('.domain').attr('stroke', border)
        ax.selectAll('.tick line').attr('stroke', border)
        ax.selectAll('.tick text').style('fill', muted)
      }

      const xOrig = d3.scaleLinear().domain(d3.extent(values)).range([0, iW]).nice()
      const bins  = d3.bin().domain(xOrig.domain()).thresholds(nbins)(values)
      const yOrig = d3.scaleLinear().domain([0, d3.max(bins, d => d.length)]).range([iH, 0]).nice()

      // Fixed Y grid + left axis
      g.append('g')
        .call(d3.axisLeft(yOrig).ticks(5).tickSize(-iW).tickFormat(''))
        .call(ax => { ax.select('.domain').remove(); ax.selectAll('.tick line').attr('stroke', border).attr('stroke-dasharray', '3,3') })
      g.append('g').call(d3.axisLeft(yOrig).ticks(5)).call(axisStyle)

      // X axis — clips labels so they don't slide past the chart edges during pan
      const xAxisG = g.append('g')
        .attr('transform', `translate(0,${iH})`)
        .attr('clip-path', `url(#hist-xclip-${cid})`)
        .call(d3.axisBottom(xOrig).ticks(6)).call(axisStyle)

      // Clipped content area for bars
      const contentG = g.append('g').attr('clip-path', `url(#hist-clip-${cid})`)

      const bars = contentG.selectAll('rect').data(bins).join('rect')
        .attr('x', d => xOrig(d.x0) + 1)
        .attr('y', d => yOrig(d.length))
        .attr('width',  d => Math.max(0, xOrig(d.x1) - xOrig(d.x0) - 2))
        .attr('height', d => iH - yOrig(d.length))
        .attr('fill', color).attr('opacity', 0.85)
        .style('cursor', 'pointer')
        .on('mouseover', function (event, d) {
          d3.select(this).attr('opacity', 1)
          tip.style('visibility', 'visible')
            .html(`Range: ${d.x0.toFixed(2)} – ${d.x1.toFixed(2)}<br>Count: <strong>${d.length}</strong>`)
        })
        .on('mousemove', function (event) {
          const r = container.getBoundingClientRect()
          tip.style('top',  `${event.clientY - r.top  - 10}px`)
             .style('left', `${event.clientX - r.left + 12}px`)
        })
        .on('mouseout', function () { d3.select(this).attr('opacity', 0.85); tip.style('visibility', 'hidden') })

      // Title + axis labels
      svg.append('text').attr('x', W / 2).attr('y', 22).attr('text-anchor', 'middle')
        .style('font-size', '14px').style('fill', text).text(title)
      svg.append('text').attr('x', m.left + iW / 2).attr('y', H - 8)
        .attr('text-anchor', 'middle').style('font-size', '12px').style('fill', muted).text(xLabel)
      svg.append('text')
        .attr('transform', `translate(14,${m.top + iH / 2}) rotate(-90)`)
        .attr('text-anchor', 'middle').style('font-size', '12px').style('fill', muted).text('Count')

      // Zoom: X-axis only — bars and x tick labels rescale together, Y stays fixed
      const zoomBehavior = d3.zoom()
        .filter(e => e.type !== 'wheel')
        .scaleExtent([0.5, 10])
        .on('zoom', event => {
          zoomTransform.current = event.transform
          const newX = event.transform.rescaleX(xOrig)

          xAxisG.call(d3.axisBottom(newX).ticks(6)).call(axisStyle)

          bars
            .attr('x', d => newX(d.x0) + 1)
            .attr('width', d => Math.max(0, newX(d.x1) - newX(d.x0) - 2))
            // y and height stay the same (Y fixed)
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
  }, [values, title, xLabel, color, nbins, theme])

  return <div ref={containerRef} style={{ width: '100%', minHeight: '400px', position: 'relative' }} />
}

export default Histogram
