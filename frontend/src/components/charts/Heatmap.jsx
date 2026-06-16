import { useRef, useEffect } from 'react'
import * as d3 from 'd3'
import { getThemeColors } from './chartTheme'
import { useTheme } from '../../context/ThemeContext'
import { addToolbar } from './chartToolbar'

// Confusion matrix heatmap. matrix = [[TN, FP], [FN, TP]], labels e.g. ['Ham','Spam'].
function Heatmap({ matrix = [], labels = [], title = 'Confusion Matrix' }) {
  const containerRef  = useRef(null)
  const zoomTransform = useRef(d3.zoomIdentity)
  const { theme } = useTheme()

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const draw = () => {
      d3.select(container).selectAll('*').remove()

      const { bg, text, muted, border } = getThemeColors()
      const W  = container.clientWidth || 500
      const H  = 360
      const m  = { top: 60, right: 30, bottom: 60, left: 80 }
      const iW = W - m.left - m.right
      const iH = H - m.top  - m.bottom

      const svg = d3.select(container)
        .append('svg')
        .attr('width', '100%').attr('height', H)
        .attr('viewBox', `0 0 ${W} ${H}`)
        .style('background', bg)

      if (!matrix.length || !labels.length) {
        svg.append('text').attr('x', W / 2).attr('y', H / 2)
          .attr('text-anchor', 'middle').attr('dominant-baseline', 'middle')
          .style('fill', muted).style('font-size', '13px').text('No data')
        return
      }

      // Clip path keeps cells inside chart area during content-only zoom
      svg.append('defs').append('clipPath').attr('id', 'hm-clip')
        .append('rect').attr('width', iW).attr('height', iH)

      const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`)

      // Axes: predicted on x, actual on y (reversed so TN is top-left)
      const xDomain = labels
      const yDomain = [...labels].reverse()

      const x = d3.scaleBand().domain(xDomain).range([0, iW]).padding(0.05)
      const y = d3.scaleBand().domain(yDomain).range([0, iH]).padding(0.05)

      const allVals = matrix.flat()
      const maxVal  = d3.max(allVals) || 1

      const colorScale = d3.scaleLinear()
        .domain([0, maxVal * 0.35, maxVal * 0.65, maxVal])
        .range([bg, '#4a3aaa', '#0077b6', '#00cc88'])

      // Tooltip
      const tip = d3.select(container)
        .append('div')
        .style('position', 'absolute').style('visibility', 'hidden')
        .style('background', bg).style('color', text)
        .style('padding', '7px 11px').style('border-radius', '6px')
        .style('font-size', '13px').style('pointer-events', 'none')
        .style('border', `1px solid ${muted}`).style('z-index', '20')
        .style('white-space', 'nowrap')

      const cells = []
      matrix.forEach((row, ri) => {
        row.forEach((val, ci) => {
          cells.push({ actual: labels[ri], predicted: labels[ci], value: val })
        })
      })

      // Axes — saved references so tick labels reposition when scales update on zoom
      const axisStyle = ax => {
        ax.select('.domain').attr('stroke', border)
        ax.selectAll('.tick line').attr('stroke', border)
        ax.selectAll('.tick text').style('fill', muted)
      }
      const xAxisG = g.append('g').attr('transform', `translate(0,${iH})`).call(d3.axisBottom(x)).call(axisStyle)
      const yAxisG = g.append('g').call(d3.axisLeft(y)).call(axisStyle)

      // contentG: clipped to chart area
      const contentG = g.append('g').attr('clip-path', 'url(#hm-clip)')

      const rectSel = contentG.selectAll('rect').data(cells).join('rect')
        .attr('x', d => x(d.predicted))
        .attr('y', d => y(d.actual))
        .attr('width',  x.bandwidth())
        .attr('height', y.bandwidth())
        .attr('fill', d => colorScale(d.value))
        .attr('rx', 4)
        .style('cursor', 'pointer')
        .on('mouseover', function (event, d) {
          d3.select(this).attr('opacity', 0.75)
          tip.style('visibility', 'visible')
            .html(`Actual <strong>${d.actual}</strong> / Predicted <strong>${d.predicted}</strong><br>Count: <strong>${d.value}</strong>`)
        })
        .on('mousemove', function (event) {
          const r = container.getBoundingClientRect()
          tip.style('top',  `${event.clientY - r.top  - 10}px`)
             .style('left', `${event.clientX - r.left + 12}px`)
        })
        .on('mouseout', function () { d3.select(this).attr('opacity', 1); tip.style('visibility', 'hidden') })

      const cellTextColor = theme === 'light' ? '#111827' : '#e8eaf0'
      const textSel = contentG.selectAll('.cell-label').data(cells).join('text')
        .attr('class', 'cell-label')
        .attr('x', d => (x(d.predicted) ?? 0) + x.bandwidth() / 2)
        .attr('y', d => (y(d.actual)    ?? 0) + y.bandwidth() / 2)
        .attr('text-anchor', 'middle').attr('dominant-baseline', 'middle')
        .style('font-size', '16px').style('font-weight', '700').style('fill', cellTextColor)
        .text(d => d.value)

      // Axis labels + title — fixed in svg
      svg.append('text').attr('x', m.left + iW / 2).attr('y', H - 8)
        .attr('text-anchor', 'middle').style('font-size', '12px').style('fill', muted).text('Predicted')
      svg.append('text')
        .attr('transform', `translate(14,${m.top + iH / 2}) rotate(-90)`)
        .attr('text-anchor', 'middle').style('font-size', '12px').style('fill', muted).text('Actual')
      svg.append('text').attr('x', W / 2).attr('y', 22).attr('text-anchor', 'middle')
        .style('font-size', '14px').style('fill', text).text(title)

      // Zoom: update band scales + reposition cells + rescale axes; chart stays within data bounds
      const zoomBehavior = d3.zoom()
        .filter(e => e.type !== 'wheel')
        .scaleExtent([0.5, 10])
        .extent([[0, 0], [iW, iH]])
        .translateExtent([[0, 0], [iW, iH]])
        .on('zoom', event => {
          zoomTransform.current = event.transform
          const t = event.transform
          const newX = x.copy().range([t.applyX(0), t.applyX(iW)])
          const newY = y.copy().range([t.applyY(0), t.applyY(iH)])

          xAxisG.call(d3.axisBottom(newX)).call(axisStyle)
          yAxisG.call(d3.axisLeft(newY)).call(axisStyle)

          rectSel
            .attr('x', d => newX(d.predicted))
            .attr('y', d => newY(d.actual))
            .attr('width', newX.bandwidth())
            .attr('height', newY.bandwidth())

          textSel
            .attr('x', d => (newX(d.predicted) ?? 0) + newX.bandwidth() / 2)
            .attr('y', d => (newY(d.actual)    ?? 0) + newY.bandwidth() / 2)
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
  }, [matrix, labels, title, theme])

  return <div ref={containerRef} className="chart-container" style={{ width: '100%', minHeight: '360px', position: 'relative' }} />
}

export default Heatmap
