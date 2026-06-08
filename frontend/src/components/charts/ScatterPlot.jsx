import { useRef, useEffect } from 'react'
import * as d3 from 'd3'
import { COLORS, getThemeColors } from './chartTheme'
import { useTheme } from '../../context/ThemeContext'

// PCA 2-D scatter.
//   pcaData: [[x, y], ...], labels: [], clusters: [], anomalies: [bool], rowIds: []
function ScatterPlot({ pcaData = [], labels = [], clusters = [], anomalies = [], rowIds = [], title = 'PCA Projection' }) {
  const containerRef = useRef(null)
  const { theme } = useTheme()

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const draw = () => {
      d3.select(container).selectAll('*').remove()

      const { bg, text, muted, border } = getThemeColors()
      const W  = container.clientWidth || 600
      const H  = 400
      const m  = { top: 44, right: 30, bottom: 60, left: 60 }
      const iW = W - m.left - m.right
      const iH = H - m.top  - m.bottom

      const svg = d3.select(container)
        .append('svg')
        .attr('width', '100%').attr('height', H)
        .attr('viewBox', `0 0 ${W} ${H}`)
        .style('background', bg)

      if (!pcaData.length) {
        svg.append('text').attr('x', W / 2).attr('y', H / 2)
          .attr('text-anchor', 'middle').attr('dominant-baseline', 'middle')
          .style('fill', muted).style('font-size', '13px').text('No data')
        return
      }

      const points = pcaData.map((coord, i) => ({
        x: coord[0], y: coord[1],
        label:     labels[i],
        cluster:   clusters[i],
        isAnomaly: anomalies[i],
        rowId:     rowIds[i] ?? i + 1,
      }))

      const benignNormal  = points.filter(p => p.label === 'BENIGN'  && !p.isAnomaly)
      const malwareNormal = points.filter(p => p.label === 'MALWARE' && !p.isAnomaly)
      const anomalyPoints = points.filter(p => p.isAnomaly)

      const xs = pcaData.map(d => d[0])
      const ys = pcaData.map(d => d[1])

      const x = d3.scaleLinear().domain(d3.extent(xs)).range([0, iW]).nice()
      const y = d3.scaleLinear().domain(d3.extent(ys)).range([iH, 0]).nice()

      const axisStyle = ax => {
        ax.select('.domain').attr('stroke', border)
        ax.selectAll('.tick line').attr('stroke', border)
        ax.selectAll('.tick text').style('fill', muted)
      }

      // Grid
      const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`)
      g.append('g').call(d3.axisLeft(y).ticks(5).tickSize(-iW).tickFormat(''))
        .call(ax => { ax.select('.domain').remove(); ax.selectAll('.tick line').attr('stroke', border).attr('stroke-dasharray', '3,3') })

      g.append('g').attr('transform', `translate(0,${iH})`).call(d3.axisBottom(x).ticks(6)).call(axisStyle)
      g.append('g').call(d3.axisLeft(y).ticks(5)).call(axisStyle)

      // Tooltip
      const tip = d3.select(container)
        .append('div')
        .style('position', 'absolute').style('visibility', 'hidden')
        .style('background', bg).style('color', text)
        .style('padding', '7px 11px').style('border-radius', '6px')
        .style('font-size', '13px').style('pointer-events', 'none')
        .style('border', `1px solid ${muted}`).style('z-index', '20')
        .style('white-space', 'nowrap')

      const mkTip = function (event, d) {
        d3.select(this).attr('r', d.isAnomaly ? 8 : 7)
        const r = container.getBoundingClientRect()
        tip.style('visibility', 'visible')
          .html(`Row ${d.rowId} | ${d.label} | Cluster ${d.cluster}`)
        tip.style('top',  `${event.clientY - r.top  - 10}px`)
           .style('left', `${event.clientX - r.left + 12}px`)
      }
      const hideTip = function (_, d) {
        d3.select(this).attr('r', d.isAnomaly ? 6 : 5)
        tip.style('visibility', 'hidden')
      }

      const dotProps = (pts, col, r) =>
        g.selectAll(null).data(pts).join('circle')
          .attr('cx', d => x(d.x)).attr('cy', d => y(d.y)).attr('r', r)
          .attr('fill', col).attr('opacity', 0.8)
          .attr('stroke', bg).attr('stroke-width', 1)
          .style('cursor', 'crosshair')
          .on('mouseover', mkTip)
          .on('mousemove', function (event) {
            const r2 = container.getBoundingClientRect()
            tip.style('top',  `${event.clientY - r2.top  - 10}px`)
               .style('left', `${event.clientX - r2.left + 12}px`)
          })
          .on('mouseout', hideTip)

      dotProps(benignNormal,  COLORS.success, 5)
      dotProps(malwareNormal, COLORS.danger,  5)

      // Anomaly ✕ markers
      g.selectAll('.anomaly').data(anomalyPoints).join('text')
        .attr('class', 'anomaly')
        .attr('x', d => x(d.x)).attr('y', d => y(d.y))
        .attr('text-anchor', 'middle').attr('dominant-baseline', 'middle')
        .style('font-size', '14px').style('font-weight', '700').style('fill', COLORS.warning)
        .style('cursor', 'crosshair')
        .text('✕')
        .on('mouseover', function (event, d) {
          const r = container.getBoundingClientRect()
          tip.style('visibility', 'visible')
            .html(`Row ${d.rowId} | ANOMALY | Cluster ${d.cluster}`)
          tip.style('top',  `${event.clientY - r.top  - 10}px`)
             .style('left', `${event.clientX - r.left + 12}px`)
        })
        .on('mousemove', function (event) {
          const r = container.getBoundingClientRect()
          tip.style('top',  `${event.clientY - r.top  - 10}px`)
             .style('left', `${event.clientX - r.left + 12}px`)
        })
        .on('mouseout', () => tip.style('visibility', 'hidden'))

      // Title
      svg.append('text').attr('x', W / 2).attr('y', 22).attr('text-anchor', 'middle')
        .style('font-size', '14px').style('fill', text).text(title)

      // Axis labels
      svg.append('text').attr('x', m.left + iW / 2).attr('y', H - 8)
        .attr('text-anchor', 'middle').style('font-size', '12px').style('fill', muted).text('PC1')
      svg.append('text')
        .attr('transform', `translate(14,${m.top + iH / 2}) rotate(-90)`)
        .attr('text-anchor', 'middle').style('font-size', '12px').style('fill', muted).text('PC2')

      // Legend
      const legend = [
        { c: COLORS.success, label: 'Benign',  shape: 'circle' },
        { c: COLORS.danger,  label: 'Malware', shape: 'circle' },
        { c: COLORS.warning, label: 'Anomaly', shape: 'x' },
      ]
      const legendY = H - 14
      const itemW   = 80
      const startX  = m.left + iW / 2 - (legend.length * itemW) / 2

      legend.forEach(({ c, label, shape }, i) => {
        const lx = startX + i * itemW
        if (shape === 'circle') {
          svg.append('circle').attr('cx', lx + 5).attr('cy', legendY - 5).attr('r', 5).attr('fill', c)
        } else {
          svg.append('text').attr('x', lx + 5).attr('y', legendY - 1)
            .attr('text-anchor', 'middle').style('font-size', '12px').style('font-weight', '700').style('fill', c).text('✕')
        }
        svg.append('text').attr('x', lx + 14).attr('y', legendY)
          .style('font-size', '11px').style('fill', text).text(label)
      })
    }

    draw()
    window.addEventListener('resize', draw)
    return () => window.removeEventListener('resize', draw)
  }, [pcaData, labels, clusters, anomalies, rowIds, title, theme])

  return <div ref={containerRef} style={{ width: '100%', minHeight: '400px', position: 'relative' }} />
}

export default ScatterPlot
