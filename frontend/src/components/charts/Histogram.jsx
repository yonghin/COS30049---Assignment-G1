import { useRef, useEffect } from 'react'
import * as d3 from 'd3'
import { COLORS, getThemeColors, createTooltip, tipTitle, tipRow } from './chartTheme'
import { useTheme } from '../../context/ThemeContext'
import { addToolbar } from './chartToolbar'

let _histCounter = 0

function Histogram({
  values = [],
  title  = 'Confidence Distribution',
  xLabel = 'Confidence',
  color  = COLORS.accent,
  nbins  = 20,
  domainMax = null,   // when set (e.g. 1), the x-axis is fixed to [0, domainMax]
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
        .style('background', 'transparent')

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
        .append('rect').attr('x', -20).attr('y', -2).attr('width', iW + 24).attr('height', m.bottom + 4)

      const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`)

      const tip = createTooltip(d3, container)

      const axisStyle = ax => {
        ax.select('.domain').attr('stroke', border)
        ax.selectAll('.tick line').attr('stroke', border)
        ax.selectAll('.tick text').style('fill', muted)
      }

      // Domain always starts at 0; negative values are never shown.
      // When domainMax is given (e.g. 1 for probabilities), the x-axis is fixed to that range.
      const xMax = domainMax != null ? domainMax : (d3.max(values) || 1)
      const xOrig = d3.scaleLinear().domain([0, xMax]).range([0, iW])

      // Build evenly spaced bin thresholds so every bar has the exact same width,
      // instead of letting d3.bin() pick "nice" but unequal boundaries.
      const thresholds = d3.range(nbins + 1).map((i) => (xMax * i) / nbins)
      const bins  = d3.bin().domain([0, xMax]).thresholds(thresholds)(values)
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
            .html(
              tipTitle(`${d.x0.toFixed(2)} to ${d.x1.toFixed(2)}`) +
              tipRow('Count', d.length, color)
            )
        })
        .on('mousemove', function (event) {
          const r = container.getBoundingClientRect()
          const tipW = tip.node().offsetWidth || 160
          const relX = event.clientX - r.left
          const left = relX + 12 + tipW > r.width ? relX - tipW - 12 : relX + 12
          tip.style('top',  `${event.clientY - r.top - 10}px`)
             .style('left', `${Math.max(4, left)}px`)
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
        .scaleExtent([1, 10])
        // translateExtent x0=0 enforces t.x >= 0 so data x=0 never drifts left of the y-axis
        .translateExtent([[0, 0], [Infinity, Infinity]])
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
  }, [values, title, xLabel, color, nbins, domainMax, theme])

  return <div ref={containerRef} className="chart-container" style={{ width: '100%', minHeight: '400px', position: 'relative' }} />
}

export default Histogram
