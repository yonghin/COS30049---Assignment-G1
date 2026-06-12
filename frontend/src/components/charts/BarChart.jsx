import { useRef, useEffect } from 'react'
import * as d3 from 'd3'
import { COLORS, getThemeColors } from './chartTheme'
import { useTheme } from '../../context/ThemeContext'
import { addToolbar } from './chartToolbar'

const H_PALETTE = [COLORS.accent, COLORS.purple, COLORS.success, COLORS.warning, COLORS.danger]

function BarChart({ models, accuracy, f1, auc, title = 'Model Performance', horizontal = false, categories, values }) {
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
      const H  = 400
      const m  = horizontal
        ? { top: 44, right: 30, bottom: 50, left: 160 }
        : { top: 44, right: 30, bottom: 60, left: 60 }
      const iW = W - m.left - m.right
      const iH = H - m.top  - m.bottom

      const svg = d3.select(container)
        .append('svg')
        .attr('width', '100%').attr('height', H)
        .attr('viewBox', `0 0 ${W} ${H}`)
        .style('background', bg)

      // Clip path keeps bars inside chart area during content-only zoom
      svg.append('defs').append('clipPath').attr('id', 'bar-clip')
        .append('rect').attr('width', iW).attr('height', iH)

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

      // contentG is the only thing that gets the zoom transform —
      // axes, tick labels, title, and legend all stay fixed.
      let contentG

      if (horizontal) {
        const cats = categories ?? []
        const vals = values ?? []

        const y = d3.scaleBand().domain(cats).range([0, iH]).padding(0.25)
        const x = d3.scaleLinear().domain([0, d3.max(vals) || 1]).range([0, iW]).nice()

        // Grid + axes — stay fixed in g
        g.append('g')
          .call(d3.axisBottom(x).ticks(5).tickSize(iH).tickFormat(''))
          .call(ax => { ax.select('.domain').remove(); ax.selectAll('.tick line').attr('stroke', border).attr('stroke-dasharray', '3,3') })
        g.append('g').attr('transform', `translate(0,${iH})`).call(d3.axisBottom(x).ticks(5)).call(axisStyle)
        g.append('g').call(d3.axisLeft(y)).call(axisStyle)

        // Bars in contentG (zooms independently of axes)
        contentG = g.append('g').attr('clip-path', 'url(#bar-clip)')
        contentG.selectAll('rect').data(vals).join('rect')
          .attr('y', (_, i) => y(cats[i]))
          .attr('x', 0)
          .attr('height', y.bandwidth())
          .attr('width', d => x(d))
          .attr('fill', (_, i) => H_PALETTE[i % H_PALETTE.length])
          .attr('rx', 3)
          .style('cursor', 'pointer')
          .on('mouseover', function (event, d) {
            d3.select(this).attr('opacity', 0.8)
            tip.style('visibility', 'visible').text(`${d.toFixed(4)}`)
          })
          .on('mousemove', function (event) {
            const r = container.getBoundingClientRect()
            tip.style('top',  `${event.clientY - r.top  - 10}px`)
               .style('left', `${event.clientX - r.left + 12}px`)
          })
          .on('mouseout', function () { d3.select(this).attr('opacity', 1); tip.style('visibility', 'hidden') })

      } else {
        const mdls = models ?? []
        const acc  = accuracy ?? []
        const f1s  = f1 ?? []
        const aucs = auc ?? []
        const metrics      = ['Accuracy', 'F1 Score', 'AUC']
        const metricColors = [COLORS.accent, COLORS.purple, COLORS.success]

        const xGroup = d3.scaleBand().domain(mdls).range([0, iW]).padding(0.3)
        const xBar   = d3.scaleBand().domain(metrics).range([0, xGroup.bandwidth()]).padding(0.05)
        const y      = d3.scaleLinear().domain([0, 1.05]).range([iH, 0])

        // Grid + axes — stay fixed
        g.append('g')
          .call(d3.axisLeft(y).ticks(5).tickSize(-iW).tickFormat(''))
          .call(ax => { ax.select('.domain').remove(); ax.selectAll('.tick line').attr('stroke', border).attr('stroke-dasharray', '3,3') })
        g.append('g').attr('transform', `translate(0,${iH})`).call(d3.axisBottom(xGroup)).call(axisStyle)
        g.append('g').call(d3.axisLeft(y).ticks(5)).call(axisStyle)

        // Bars in contentG
        contentG = g.append('g').attr('clip-path', 'url(#bar-clip)')

        const seriesData = [
          { key: 'Accuracy', vals: acc },
          { key: 'F1 Score', vals: f1s },
          { key: 'AUC',      vals: aucs },
        ]
        seriesData.forEach(({ key, vals }, si) => {
          contentG.selectAll(`.bar-${si}`)
            .data(vals).join('rect')
            .attr('class', `bar-${si}`)
            .attr('x', (_, i) => (xGroup(mdls[i]) ?? 0) + (xBar(key) ?? 0))
            .attr('y', d => y(d))
            .attr('width', xBar.bandwidth())
            .attr('height', d => iH - y(d))
            .attr('fill', metricColors[si])
            .attr('rx', 2)
            .style('cursor', 'pointer')
            .on('mouseover', function (event, d) {
              d3.select(this).attr('opacity', 0.75)
              tip.style('visibility', 'visible').html(`${key}: <strong>${d.toFixed(4)}</strong>`)
            })
            .on('mousemove', function (event) {
              const r = container.getBoundingClientRect()
              tip.style('top',  `${event.clientY - r.top  - 10}px`)
                 .style('left', `${event.clientX - r.left + 12}px`)
            })
            .on('mouseout', function () { d3.select(this).attr('opacity', 1); tip.style('visibility', 'hidden') })
        })

        // Legend — fixed in svg (outside g)
        const legendY      = H - 16
        const legendStartX = m.left + iW / 2 - (metrics.length * 90) / 2
        metrics.forEach((key, i) => {
          const lx = legendStartX + i * 90
          svg.append('rect').attr('x', lx).attr('y', legendY - 10).attr('width', 12).attr('height', 12)
            .attr('fill', metricColors[i]).attr('rx', 2)
          svg.append('text').attr('x', lx + 16).attr('y', legendY)
            .style('font-size', '11px').style('fill', text).text(key)
        })
      }

      // Title — fixed in svg
      svg.append('text').attr('x', W / 2).attr('y', 22).attr('text-anchor', 'middle')
        .style('font-size', '14px').style('fill', text).text(title)

      // Content-only zoom: only bars move; axes, tick labels, title, legend stay fixed
      const zoomBehavior = d3.zoom()
        .filter(e => e.type !== 'wheel')
        .scaleExtent([0.5, 10])
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
  }, [models, accuracy, f1, auc, title, horizontal, categories, values, theme])

  return <div ref={containerRef} style={{ width: '100%', minHeight: '400px', position: 'relative' }} />
}

export default BarChart
