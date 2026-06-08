import { useRef, useEffect } from 'react'
import * as d3 from 'd3'
import { COLORS, getThemeColors } from './chartTheme'
import { useTheme } from '../../context/ThemeContext'

// Dual-purpose line chart:
//  - Time series mode: pass `spamSeries` / `malwareSeries` ([{timestamp, count}])
//  - ROC mode:         pass `fpr` / `tpr` / `auc`
function LineChart({ spamSeries, malwareSeries, fpr, tpr, auc, title, color }) {
  const containerRef  = useRef(null)
  const zoomTransform = useRef(d3.zoomIdentity)   // persist zoom across data refreshes
  const { theme } = useTheme()

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const isRoc = Array.isArray(fpr) && Array.isArray(tpr)

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

      const axisStyle = ax => {
        ax.select('.domain').attr('stroke', border)
        ax.selectAll('.tick line').attr('stroke', border)
        ax.selectAll('.tick text').style('fill', muted)
      }

      // Tooltip
      const tip = d3.select(container)
        .append('div')
        .style('position', 'absolute').style('visibility', 'hidden')
        .style('background', bg).style('color', text)
        .style('padding', '7px 11px').style('border-radius', '6px')
        .style('font-size', '13px').style('pointer-events', 'none')
        .style('border', `1px solid ${muted}`).style('z-index', '20')
        .style('white-space', 'nowrap')

      const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`)

      // Clip path so lines don't overflow axes during zoom
      const clipId = 'lc-clip'
      svg.append('defs').append('clipPath').attr('id', clipId)
        .append('rect').attr('width', iW).attr('height', iH)

      const chartArea = g.append('g').attr('clip-path', `url(#${clipId})`)

      // Title
      svg.append('text').attr('x', W / 2).attr('y', 22).attr('text-anchor', 'middle')
        .style('font-size', '14px').style('fill', text)
        .text(isRoc
          ? (title ?? `ROC Curve (AUC = ${auc != null ? auc.toFixed(4) : 'N/A'})`)
          : (title ?? 'Live Predictions'))

      if (isRoc) {
        // ── ROC curve ─────────────────────────────────────────────────────────
        const x = d3.scaleLinear().domain([0, 1]).range([0, iW])
        const y = d3.scaleLinear().domain([0, 1.02]).range([iH, 0])

        // Grid
        g.append('g').call(d3.axisLeft(y).ticks(5).tickSize(-iW).tickFormat(''))
          .call(ax => { ax.select('.domain').remove(); ax.selectAll('.tick line').attr('stroke', border).attr('stroke-dasharray', '3,3') })

        g.append('g').attr('transform', `translate(0,${iH})`).call(d3.axisBottom(x).ticks(6)).call(axisStyle)
        g.append('g').call(d3.axisLeft(y).ticks(5)).call(axisStyle)

        // Diagonal reference
        chartArea.append('line')
          .attr('x1', x(0)).attr('y1', y(0)).attr('x2', x(1)).attr('y2', y(1))
          .attr('stroke', COLORS.muted).attr('stroke-width', 1).attr('stroke-dasharray', '5,5')

        // ROC line
        const lineGen = d3.line().x((_, i) => x(fpr[i])).y((_, i) => y(tpr[i]))
        chartArea.append('path')
          .datum(tpr)
          .attr('d', lineGen)
          .attr('fill', 'none')
          .attr('stroke', color ?? COLORS.accent)
          .attr('stroke-width', 2.5)

        // Axis labels
        svg.append('text').attr('x', m.left + iW / 2).attr('y', H - 8)
          .attr('text-anchor', 'middle').style('font-size', '12px').style('fill', muted).text('False Positive Rate')
        svg.append('text')
          .attr('transform', `translate(14,${m.top + iH / 2}) rotate(-90)`)
          .attr('text-anchor', 'middle').style('font-size', '12px').style('fill', muted).text('True Positive Rate')

        // Legend
        ;[{ c: color ?? COLORS.accent, label: 'ROC' }, { c: COLORS.muted, label: 'Random', dash: '5,5' }].forEach(({ c, label, dash }, i) => {
          const lx = m.left + iW / 2 - 70 + i * 90
          const ly = H - 16
          svg.append('line').attr('x1', lx).attr('y1', ly - 5).attr('x2', lx + 18).attr('y2', ly - 5)
            .attr('stroke', c).attr('stroke-width', 2).attr('stroke-dasharray', dash ?? '')
          svg.append('text').attr('x', lx + 22).attr('y', ly)
            .style('font-size', '11px').style('fill', text).text(label)
        })

      } else {
        // ── Time series ───────────────────────────────────────────────────────
        const toMYT = s => s ? new Date(new Date(s).getTime() + 8 * 60 * 60 * 1000) : new Date(0)

        const spamData    = (spamSeries    ?? []).map(p => ({ date: toMYT(p.timestamp), count: p.count }))
        const malwareData = (malwareSeries ?? []).map(p => ({ date: toMYT(p.timestamp), count: p.count }))
        const allData     = [...spamData, ...malwareData]

        if (!allData.length) {
          g.append('text').attr('x', iW / 2).attr('y', iH / 2)
            .attr('text-anchor', 'middle').attr('dominant-baseline', 'middle')
            .style('fill', muted).style('font-size', '13px').text('No data yet')
          return
        }

        const xExtent = d3.extent(allData, d => d.date)
        const yMax    = d3.max(allData, d => d.count) || 1

        const x = d3.scaleTime().domain(xExtent).range([0, iW])
        const y = d3.scaleLinear().domain([0, yMax]).range([iH, 0]).nice()

        // Grid
        g.append('g').call(d3.axisLeft(y).ticks(5).tickSize(-iW).tickFormat(''))
          .call(ax => { ax.select('.domain').remove(); ax.selectAll('.tick line').attr('stroke', border).attr('stroke-dasharray', '3,3') })

        const xAxisG = g.append('g').attr('transform', `translate(0,${iH})`)
          .call(d3.axisBottom(x).ticks(6).tickFormat(d3.timeFormat('%m-%d %H:%M'))).call(axisStyle)
          .call(ax => ax.selectAll('.tick text').attr('transform', 'rotate(-25)').style('text-anchor', 'end'))

        g.append('g').call(d3.axisLeft(y).ticks(5)).call(axisStyle)

        const mkLine = xScale => d3.line().x(d => xScale(d.date)).y(d => y(d.count))

        const spamPath    = chartArea.append('path').datum(spamData)
          .attr('fill', 'none').attr('stroke', COLORS.accent).attr('stroke-width', 2)
          .attr('d', mkLine(x))

        const malwarePath = chartArea.append('path').datum(malwareData)
          .attr('fill', 'none').attr('stroke', COLORS.danger).attr('stroke-width', 2)
          .attr('d', mkLine(x))

        // Dots
        const mkDots = (data, cls, col, xScale) =>
          chartArea.selectAll(`.${cls}`).data(data).join('circle')
            .attr('class', cls)
            .attr('cx', d => xScale(d.date)).attr('cy', d => y(d.count)).attr('r', 4)
            .attr('fill', col).attr('stroke', bg).attr('stroke-width', 1.5)
            .style('cursor', 'crosshair')
            .on('mouseover', function (event, d) {
              d3.select(this).attr('r', 6)
              tip.style('visibility', 'visible')
                .html(`${d3.timeFormat('%m-%d %H:%M')(d.date)}<br>Count: <strong>${d.count}</strong>`)
            })
            .on('mousemove', function (event) {
              const r = container.getBoundingClientRect()
              tip.style('top',  `${event.clientY - r.top  - 10}px`)
                 .style('left', `${event.clientX - r.left + 12}px`)
            })
            .on('mouseout', function () { d3.select(this).attr('r', 4); tip.style('visibility', 'hidden') })

        mkDots(spamData,    'spam-dot',    COLORS.accent,  x)
        mkDots(malwareData, 'malware-dot', COLORS.danger,  x)

        // Axis labels
        svg.append('text').attr('x', m.left + iW / 2).attr('y', H - 4)
          .attr('text-anchor', 'middle').style('font-size', '12px').style('fill', muted).text('Time (MYT)')
        svg.append('text')
          .attr('transform', `translate(14,${m.top + iH / 2}) rotate(-90)`)
          .attr('text-anchor', 'middle').style('font-size', '12px').style('fill', muted).text('Count')

        // Legend
        ;[{ c: COLORS.accent, label: 'Spam' }, { c: COLORS.danger, label: 'Malware' }].forEach(({ c, label }, i) => {
          const lx = m.left + iW / 2 - 60 + i * 90
          const ly = H - 4
          svg.append('circle').attr('cx', lx + 5).attr('cy', ly - 7).attr('r', 5).attr('fill', c)
          svg.append('text').attr('x', lx + 14).attr('y', ly - 3)
            .style('font-size', '11px').style('fill', text).text(label)
        })

        // Zoom behaviour (preserve state across live refreshes)
        const zoomBehavior = d3.zoom()
          .scaleExtent([1, 50])
          .translateExtent([[0, 0], [iW, iH]])
          .extent([[0, 0], [iW, iH]])
          .on('zoom', event => {
            zoomTransform.current = event.transform
            const newX = event.transform.rescaleX(x)
            xAxisG.call(d3.axisBottom(newX).ticks(6).tickFormat(d3.timeFormat('%m-%d %H:%M'))).call(axisStyle)
              .call(ax => ax.selectAll('.tick text').attr('transform', 'rotate(-25)').style('text-anchor', 'end'))
            spamPath.attr('d',    mkLine(newX)(spamData))
            malwarePath.attr('d', mkLine(newX)(malwareData))
            chartArea.selectAll('.spam-dot').attr('cx',    d => newX(d.date))
            chartArea.selectAll('.malware-dot').attr('cx', d => newX(d.date))
          })

        // Overlay rect to capture zoom events
        svg.append('rect')
          .attr('transform', `translate(${m.left},${m.top})`)
          .attr('width', iW).attr('height', iH)
          .attr('fill', 'none').attr('pointer-events', 'all')
          .call(zoomBehavior)
          // Restore previous zoom if data refreshed while user was zoomed
          .call(zoomBehavior.transform, zoomTransform.current)
      }
    }

    draw()
    window.addEventListener('resize', draw)
    return () => window.removeEventListener('resize', draw)
  }, [spamSeries, malwareSeries, fpr, tpr, auc, title, color, theme])

  return <div ref={containerRef} style={{ width: '100%', minHeight: '400px', position: 'relative' }} />
}

export default LineChart
