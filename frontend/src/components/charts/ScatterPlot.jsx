import { useRef, useEffect } from 'react'
import * as d3 from 'd3'
import { COLORS, getThemeColors } from './chartTheme'
import { useTheme } from '../../context/ThemeContext'
import { addToolbar } from './chartToolbar'

// PCA 2-D scatter.
//   pcaData: [[x, y], ...], labels: [], clusters: [], anomalies: [bool], rowIds: []
function ScatterPlot({ pcaData = [], labels = [], clusters = [], anomalies = [], rowIds = [], title = 'PCA Projection' }) {
  const containerRef  = useRef(null)
  const zoomTransform = useRef(d3.zoomIdentity)   // persist zoom across redraws
  const { theme } = useTheme()

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const draw = () => {
      d3.select(container).selectAll('*').remove()

      const { bg, text, muted, border } = getThemeColors()
      const W  = container.clientWidth || 600
      // Extra bottom margin so legend and PC1 axis label have clear separation
      const H  = 420
      const m  = { top: 44, right: 30, bottom: 78, left: 60 }
      const iW = W - m.left - m.right
      const iH = H - m.top  - m.bottom

      const svg = d3.select(container)
        .append('svg')
        .attr('width', '100%').attr('height', H)
        .attr('viewBox', `0 0 ${W} ${H}`)
        .style('background', bg)
        .style('cursor', 'grab')

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

      // Clip path — extend 6 px on every side so edge dots aren't half-clipped
      const clipId = 'sc-clip'
      svg.append('defs').append('clipPath').attr('id', clipId)
        .append('rect').attr('x', -6).attr('y', -6).attr('width', iW + 12).attr('height', iH + 12)

      const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`)

      // Grid
      g.append('g').call(d3.axisLeft(y).ticks(5).tickSize(-iW).tickFormat(''))
        .call(ax => { ax.select('.domain').remove(); ax.selectAll('.tick line').attr('stroke', border).attr('stroke-dasharray', '3,3') })

      const xAxisG = g.append('g').attr('transform', `translate(0,${iH})`).call(d3.axisBottom(x).ticks(6)).call(axisStyle)
      const yAxisG = g.append('g').call(d3.axisLeft(y).ticks(5)).call(axisStyle)

      // Tooltip
      const tip = d3.select(container)
        .append('div')
        .style('position', 'absolute').style('visibility', 'hidden')
        .style('background', bg).style('color', text)
        .style('padding', '7px 11px').style('border-radius', '6px')
        .style('font-size', '13px').style('pointer-events', 'none')
        .style('border', `1px solid ${muted}`).style('z-index', '20')
        .style('white-space', 'nowrap')

      // Clipped group — dots stay inside axes during zoom/pan
      const chartArea = g.append('g').attr('clip-path', `url(#${clipId})`)

      const mkTip = function (event, d) {
        d3.select(this).attr('r', 7)
        const r = container.getBoundingClientRect()
        tip.style('visibility', 'visible')
          .html(`Row ${d.rowId} | ${d.label} | Cluster ${d.cluster}`)
        tip.style('top',  `${event.clientY - r.top  - 10}px`)
           .style('left', `${event.clientX - r.left + 12}px`)
      }
      const hideTip = function (_, d) {
        d3.select(this).attr('r', 5)
        tip.style('visibility', 'hidden')
      }

      const addDots = (pts, col) =>
        chartArea.selectAll(null).data(pts).join('circle')
          .attr('cx', d => x(d.x)).attr('cy', d => y(d.y)).attr('r', 5)
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

      addDots(benignNormal,  COLORS.success)
      addDots(malwareNormal, COLORS.danger)

      // Anomaly ✕ markers
      chartArea.selectAll('.anomaly').data(anomalyPoints).join('text')
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

      // Axis labels — PC1 at the very bottom, legend clearly above it
      svg.append('text').attr('x', m.left + iW / 2).attr('y', H - 8)
        .attr('text-anchor', 'middle').style('font-size', '12px').style('fill', muted).text('PC1')
      svg.append('text')
        .attr('transform', `translate(14,${m.top + iH / 2}) rotate(-90)`)
        .attr('text-anchor', 'middle').style('font-size', '12px').style('fill', muted).text('PC2')

      // Legend — 22px gap above PC1 axis label
      const legend = [
        { c: COLORS.success, label: 'Benign',  shape: 'circle' },
        { c: COLORS.danger,  label: 'Malware', shape: 'circle' },
        { c: COLORS.warning, label: 'Anomaly', shape: 'x' },
      ]
      const legendY = H - 30   // was H-14; now 22px clear of PC1 label at H-8
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

      // ── Zoom & pan (buttons only — scroll wheel disabled) ────────────────────
      const zoomBehavior = d3.zoom()
        .filter(e => e.type !== 'wheel')
        .scaleExtent([0.5, 30])
        .extent([[m.left, m.top], [m.left + iW, m.top + iH]])
        .on('zoom', event => {
          zoomTransform.current = event.transform
          const newX = event.transform.rescaleX(x)
          const newY = event.transform.rescaleY(y)
          xAxisG.call(d3.axisBottom(newX).ticks(6)).call(axisStyle)
          yAxisG.call(d3.axisLeft(newY).ticks(5)).call(axisStyle)
          chartArea.selectAll('circle').attr('cx', d => newX(d.x)).attr('cy', d => newY(d.y))
          chartArea.selectAll('.anomaly').attr('x', d => newX(d.x)).attr('y', d => newY(d.y))
        })

      svg.call(zoomBehavior).call(zoomBehavior.transform, zoomTransform.current)

      addToolbar(container, {
        svgSel: svg,
        zoomBehavior,
        onReset: () => { zoomTransform.current = d3.zoomIdentity },
        title,
      })
    }

    draw()
    window.addEventListener('resize', draw)
    return () => window.removeEventListener('resize', draw)
  }, [pcaData, labels, clusters, anomalies, rowIds, title, theme])

  return <div ref={containerRef} style={{ width: '100%', minHeight: '420px', position: 'relative' }} />
}

export default ScatterPlot
