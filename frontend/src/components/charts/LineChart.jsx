import { useRef, useEffect } from 'react'
import * as d3 from 'd3'
import { COLORS, getThemeColors } from './chartTheme'
import { useTheme } from '../../context/ThemeContext'
import { addToolbar } from './chartToolbar'

// Dual-purpose line chart:
//  - Time series mode: pass `spamSeries` / `malwareSeries` ([{timestamp, count}])
//  - ROC mode:         pass `fpr` / `tpr` / `auc`
function LineChart({ spamSeries, malwareSeries, fpr, tpr, auc, title, color }) {
  const containerRef   = useRef(null)
  const zoomTransform  = useRef(d3.zoomIdentity)   // persist zoom across data refreshes
  const hiddenLinesRef = useRef(new Set())          // persist legend toggle state
  const { theme } = useTheme()

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const isRoc  = Array.isArray(fpr) && Array.isArray(tpr)
    const hidden = hiddenLinesRef.current

    const draw = () => {
      d3.select(container).selectAll('*').remove()

      const { bg, text, muted, border } = getThemeColors()
      const W = container.clientWidth || 600

      // Time series needs extra bottom margin to separate rotated tick labels,
      // x-axis label, and the clickable legend — so use a taller canvas.
      const H  = isRoc ? 400 : 400
      const m  = isRoc
        ? { top: 44, right: 30, bottom: 60, left: 60 }
        : { top: 44, right: 30, bottom: 68, left: 60 }
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

      // Tooltip div
      const tip = d3.select(container)
        .append('div')
        .style('position', 'absolute').style('visibility', 'hidden')
        .style('background', bg).style('color', text)
        .style('padding', '7px 11px').style('border-radius', '6px')
        .style('font-size', '13px').style('pointer-events', 'none')
        .style('border', `1px solid ${muted}`).style('z-index', '20')
        .style('white-space', 'nowrap')

      const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`)

      // Clip path prevents lines from overflowing axes during zoom
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

        // Gridlines
        g.append('g').call(d3.axisLeft(y).ticks(5).tickSize(-iW).tickFormat(''))
          .call(ax => {
            ax.select('.domain').remove()
            ax.selectAll('.tick line').attr('stroke', border).attr('stroke-dasharray', '3,3')
          })

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

        // Axis labels — placed well above the legend
        svg.append('text').attr('x', m.left + iW / 2).attr('y', H - 28)
          .attr('text-anchor', 'middle').style('font-size', '12px').style('fill', muted)
          .text('False Positive Rate')
        svg.append('text')
          .attr('transform', `translate(14,${m.top + iH / 2}) rotate(-90)`)
          .attr('text-anchor', 'middle').style('font-size', '12px').style('fill', muted)
          .text('True Positive Rate')

        // Legend — at the very bottom, 20+ px below axis label
        const rocItems   = [
          { c: color ?? COLORS.accent, label: 'ROC' },
          { c: COLORS.muted,           label: 'Random', dash: '5,5' },
        ]
        const legTotalW  = rocItems.length * 90
        const legStartX  = m.left + iW / 2 - legTotalW / 2
        rocItems.forEach(({ c, label, dash }, i) => {
          const lx = legStartX + i * 90
          const ly = H - 8
          svg.append('line')
            .attr('x1', lx).attr('y1', ly - 5).attr('x2', lx + 18).attr('y2', ly - 5)
            .attr('stroke', c).attr('stroke-width', 2).attr('stroke-dasharray', dash ?? '')
          svg.append('text').attr('x', lx + 22).attr('y', ly)
            .style('font-size', '11px').style('fill', text).text(label)
        })

        // Content-only zoom: ROC line + diagonal zoom; axes, labels, legend stay fixed.
        // rocZoomState is read by the tooltip rect to un-apply the zoom when bisecting.
        let rocZoomState = zoomTransform.current
        const rocZoom = d3.zoom()
          .filter(e => e.type !== 'wheel')
          .scaleExtent([0.5, 10])
          .on('zoom', event => {
            zoomTransform.current = event.transform
            rocZoomState = event.transform
            chartArea.attr('transform', event.transform.toString())
          })

        svg.call(rocZoom).call(rocZoom.transform, zoomTransform.current)

        // Tooltip overlay in g (fixed position) — converts cursor coords using current zoom state
        g.append('rect')
          .attr('width', iW).attr('height', iH)
          .attr('fill', 'none').attr('pointer-events', 'all')
          .style('cursor', 'crosshair')
          .on('mousemove', function (event) {
            const [mx] = d3.pointer(event)
            // Un-apply the zoom transform to get the content-coordinate x
            const contentX = (mx - rocZoomState.x) / rocZoomState.k
            const x0  = x.invert(contentX)
            const idx = Math.max(0, Math.min(fpr.length - 1, d3.bisectLeft(fpr, x0)))
            tip.style('visibility', 'visible')
              .html(`FPR: <strong>${fpr[idx].toFixed(4)}</strong><br>TPR: <strong>${tpr[idx].toFixed(4)}</strong>`)
            const r = container.getBoundingClientRect()
            tip.style('top',  `${event.clientY - r.top  - 10}px`)
               .style('left', `${event.clientX - r.left + 12}px`)
          })
          .on('mouseleave', () => tip.style('visibility', 'hidden'))

        addToolbar(container, {
          svgSel: svg, zoomBehavior: rocZoom,
          onReset: () => { zoomTransform.current = d3.zoomIdentity },
          title: title ?? `ROC Curve (AUC = ${auc != null ? auc.toFixed(4) : 'N/A'})`,
        })

      } else {
        // ── Time series ───────────────────────────────────────────────────────
        // Convert UTC timestamp to a Date that D3's local-time formatters display as MYT (UTC+8)
        const toMYT = s => {
          if (!s) return new Date(0)
          const d = new Date(s)
          // Get the MYT wall-clock string then parse as a "fake local" Date
          // so d3.timeFormat shows the correct MYT time regardless of browser timezone
          return new Date(d.toLocaleString('en-US', { timeZone: 'Asia/Kuala_Lumpur' }))
        }

        const spamData    = (spamSeries    ?? []).map(p => ({ date: toMYT(p.timestamp), count: p.count }))
        const malwareData = (malwareSeries ?? []).map(p => ({ date: toMYT(p.timestamp), count: p.count }))
        const allData     = [...spamData, ...malwareData]

        if (!allData.length) {
          g.append('text').attr('x', iW / 2).attr('y', iH / 2)
            .attr('text-anchor', 'middle').attr('dominant-baseline', 'middle')
            .style('fill', muted).style('font-size', '13px').text('No data yet')
          return
        }

        const xExtent  = d3.extent(allData, d => d.date)
        const yMax     = d3.max(allData, d => d.count) || 1

        const x = d3.scaleTime().domain(xExtent).range([0, iW])
        const y = d3.scaleLinear().domain([0, yMax]).range([iH, 0]).nice()

        // Pre-compute timestamp arrays for bisector (avoids re-mapping on every mousemove)
        const spamTimes    = spamData.map(d => d.date.getTime())
        const malwareTimes = malwareData.map(d => d.date.getTime())

        // Track the current (possibly zoomed) x scale for tooltip bisector
        let currentX = x

        // Gridlines
        g.append('g').call(d3.axisLeft(y).ticks(5).tickSize(-iW).tickFormat(''))
          .call(ax => {
            ax.select('.domain').remove()
            ax.selectAll('.tick line').attr('stroke', border).attr('stroke-dasharray', '3,3')
          })

        const xAxisG = g.append('g').attr('transform', `translate(0,${iH})`)
          .call(d3.axisBottom(x).ticks(6).tickFormat(d3.timeFormat('%H:%M')))
          .call(axisStyle)

        g.append('g').call(d3.axisLeft(y).ticks(5)).call(axisStyle)

        const mkLine = xScale => d3.line().x(d => xScale(d.date)).y(d => y(d.count))

        // Lines
        const spamPath = chartArea.append('path').datum(spamData)
          .attr('id', 'lc-spam-line')
          .attr('fill', 'none').attr('stroke', COLORS.accent).attr('stroke-width', 2)
          .attr('d', mkLine(x))
          .attr('visibility', hidden.has('spam') ? 'hidden' : 'visible')

        const malwarePath = chartArea.append('path').datum(malwareData)
          .attr('id', 'lc-malware-line')
          .attr('fill', 'none').attr('stroke', COLORS.danger).attr('stroke-width', 2)
          .attr('d', mkLine(x))
          .attr('visibility', hidden.has('malware') ? 'hidden' : 'visible')

        // Data-point dots with hover tooltip
        const mkDots = (data, cls, col, xScale, key) =>
          chartArea.selectAll(`.${cls}`).data(data).join('circle')
            .attr('class', cls)
            .attr('cx', d => xScale(d.date)).attr('cy', d => y(d.count)).attr('r', 4)
            .attr('fill', col).attr('stroke', bg).attr('stroke-width', 1.5)
            .attr('visibility', hidden.has(key) ? 'hidden' : 'visible')
            .style('cursor', 'crosshair')
            .on('mouseover', function (event, d) {
              d3.select(this).attr('r', 6)
              tip.style('visibility', 'visible')
                .html(`${d3.timeFormat('%m-%d %H:%M')(d.date)} (MYT)<br>Count: <strong>${d.count}</strong>`)
            })
            .on('mousemove', function (event) {
              const r = container.getBoundingClientRect()
              tip.style('top',  `${event.clientY - r.top  - 10}px`)
                 .style('left', `${event.clientX - r.left + 12}px`)
            })
            .on('mouseout', function () {
              d3.select(this).attr('r', 4)
              tip.style('visibility', 'hidden')
            })

        mkDots(spamData,    'spam-dot',    COLORS.accent, x, 'spam')
        mkDots(malwareData, 'malware-dot', COLORS.danger, x, 'malware')

        // Axis labels — x label placed above the clickable legend
        svg.append('text').attr('x', m.left + iW / 2).attr('y', H - 32)
          .attr('text-anchor', 'middle').style('font-size', '12px').style('fill', muted)
          .text('Time (MYT)')
        svg.append('text')
          .attr('transform', `translate(14,${m.top + iH / 2}) rotate(-90)`)
          .attr('text-anchor', 'middle').style('font-size', '12px').style('fill', muted)
          .text('Count')

        // Clickable legend — clearly below the axis label
        const tsItems   = [
          { c: COLORS.accent, label: 'Spam',    key: 'spam' },
          { c: COLORS.danger, label: 'Malware', key: 'malware' },
        ]
        const legTotalW = tsItems.length * 90
        const legStartX = m.left + iW / 2 - legTotalW / 2
        tsItems.forEach(({ c, label, key }, i) => {
          const lx = legStartX + i * 90
          const ly = H - 16

          const legendG = svg.append('g')
            .style('cursor', 'pointer')
            .attr('opacity', hidden.has(key) ? 0.4 : 1)

          legendG.append('circle').attr('cx', lx + 5).attr('cy', ly - 7).attr('r', 5).attr('fill', c)
          legendG.append('text').attr('x', lx + 14).attr('y', ly - 3)
            .style('font-size', '11px').style('fill', text).text(label)

          legendG.on('click', function () {
            if (hidden.has(key)) hidden.delete(key)
            else hidden.add(key)
            const isHidden = hidden.has(key)
            svg.select(`#lc-${key}-line`).attr('visibility', isHidden ? 'hidden' : 'visible')
            chartArea.selectAll(`.${key}-dot`).attr('visibility', isHidden ? 'hidden' : 'visible')
            d3.select(this).attr('opacity', isHidden ? 0.4 : 1)
          })
        })

        // Zoom behaviour — scroll wheel disabled; use toolbar buttons
        // scaleExtent minimum < 1 so zoom-out is allowed
        const zoomBehavior = d3.zoom()
          .filter(e => e.type !== 'wheel')
          .scaleExtent([0.2, 50])
          .extent([[0, 0], [iW, iH]])
          .on('zoom', event => {
            zoomTransform.current = event.transform
            const newX = event.transform.rescaleX(x)
            currentX   = newX   // keep tooltip bisector in sync
            xAxisG.call(d3.axisBottom(newX).ticks(6).tickFormat(d3.timeFormat('%H:%M'))).call(axisStyle)
            spamPath.attr('d',    mkLine(newX)(spamData))
            malwarePath.attr('d', mkLine(newX)(malwareData))
            chartArea.selectAll('.spam-dot').attr('cx',    d => newX(d.date))
            chartArea.selectAll('.malware-dot').attr('cx', d => newX(d.date))
          })

        // Overlay rect: zoom + hover-tooltip events
        const overlay = svg.append('rect')
          .attr('transform', `translate(${m.left},${m.top})`)
          .attr('width', iW).attr('height', iH)
          .attr('fill', 'none').attr('pointer-events', 'all')
          .call(zoomBehavior)
          .call(zoomBehavior.transform, zoomTransform.current)
          .on('mousemove.tip', function (event) {
            const [mx]  = d3.pointer(event)
            const xVal  = currentX.invert(mx)
            const t     = xVal.getTime()
            let html    = ''

            if (spamData.length && !hidden.has('spam')) {
              const bi = Math.max(0, Math.min(spamData.length - 1, d3.bisectLeft(spamTimes, t)))
              html += `${d3.timeFormat('%m-%d %H:%M')(spamData[bi].date)} (MYT)<br>Spam: <strong>${spamData[bi].count}</strong><br>`
            }
            if (malwareData.length && !hidden.has('malware')) {
              const bi = Math.max(0, Math.min(malwareData.length - 1, d3.bisectLeft(malwareTimes, t)))
              html += `Malware: <strong>${malwareData[bi].count}</strong>`
            }
            if (html) {
              tip.style('visibility', 'visible').html(html)
              const r = container.getBoundingClientRect()
              tip.style('top',  `${event.clientY - r.top  - 10}px`)
                 .style('left', `${event.clientX - r.left + 12}px`)
            }
          })
          .on('mouseleave.tip', () => tip.style('visibility', 'hidden'))

        addToolbar(container, {
          svgSel: overlay, zoomBehavior,
          onReset: () => { zoomTransform.current = d3.zoomIdentity },
          title: title ?? 'Live Predictions',
        })
      }
    }

    draw()
    window.addEventListener('resize', draw)
    return () => window.removeEventListener('resize', draw)
  }, [spamSeries, malwareSeries, fpr, tpr, auc, title, color, theme])

  return <div ref={containerRef} style={{ width: '100%', minHeight: '400px', position: 'relative' }} />
}

export default LineChart
