import { useRef, useEffect } from 'react'
import * as d3 from 'd3'
import { COLORS, getThemeColors } from './chartTheme'
import { useTheme } from '../../context/ThemeContext'
import { addToolbar } from './chartToolbar'

const H_PALETTE = [COLORS.accent, COLORS.purple, COLORS.success, COLORS.warning, COLORS.danger]

let _chartCounter = 0

function BarChart({ models, accuracy, f1, auc, title = 'Model Performance', horizontal = false, categories, values }) {
  const containerRef  = useRef(null)
  const zoomTransform = useRef(d3.zoomIdentity)
  const hiddenRef     = useRef(new Set())
  const chartId       = useRef(`bc${++_chartCounter}`)
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

      const defs = svg.append('defs')
      const cid  = chartId.current

      // Clip for bars — in g's coordinate system: (0,0) to (iW, iH)
      defs.append('clipPath').attr('id', `bar-clip-${cid}`)
        .append('rect').attr('width', iW).attr('height', iH)

      // Clip for x-axis labels — x=-20 lets the "0" tick label (centered at pixel 0) render fully
      defs.append('clipPath').attr('id', `xaxis-label-clip-${cid}`)
        .append('rect').attr('x', -20).attr('y', -2).attr('width', iW + 20).attr('height', m.bottom + 4)

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

      if (horizontal) {
        // ── Horizontal bar chart (feature importance / category scores) ──────
        const cats  = categories ?? []
        const vals  = values ?? []

        const y     = d3.scaleBand().domain(cats).range([0, iH]).padding(0.25)
        const xOrig = d3.scaleLinear().domain([0, d3.max(vals) || 1]).range([0, iW]).nice()

        // Grid + clipped bars
        const contentG = g.append('g').attr('clip-path', `url(#bar-clip-${cid})`)
        const gridG    = contentG.append('g')

        const drawGrid = (xScale) => {
          gridG.selectAll('*').remove()
          xScale.ticks(5).forEach(tick => {
            gridG.append('line')
              .attr('x1', xScale(tick)).attr('y1', 0)
              .attr('x2', xScale(tick)).attr('y2', iH)
              .attr('stroke', border).attr('stroke-dasharray', '3,3').attr('stroke-width', 0.8)
          })
        }
        drawGrid(xOrig)

        const xAxisG = g.append('g').attr('transform', `translate(0,${iH})`)
          .attr('clip-path', `url(#xaxis-label-clip-${cid})`)
          .call(d3.axisBottom(xOrig).ticks(5)).call(axisStyle)

        g.append('g').call(d3.axisLeft(y)).call(axisStyle)

        const bars = contentG.selectAll('rect.hbar').data(vals).join('rect')
          .attr('class', 'hbar')
          .attr('y', (_, i) => y(cats[i]))
          .attr('x', 0)
          .attr('height', y.bandwidth())
          .attr('width', d => xOrig(d))
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

        // Zoom: rescaleX with translateExtent — drag right to pan, blocked from going left past x=0
        const zoomBehavior = d3.zoom()
          .filter(e => e.type !== 'wheel')
          .scaleExtent([1, 10])
          .translateExtent([[0, 0], [Infinity, Infinity]])
          .on('zoom', event => {
            zoomTransform.current = event.transform
            const newX = event.transform.rescaleX(xOrig)
            xAxisG.call(d3.axisBottom(newX).ticks(5)).call(axisStyle)
            drawGrid(newX)
            bars.attr('width', d => Math.max(0, newX(d)))
          })

        svg.call(zoomBehavior).call(zoomBehavior.transform, zoomTransform.current)

        addToolbar(container, {
          svgSel: svg, zoomBehavior,
          onReset: () => { zoomTransform.current = d3.zoomIdentity },
          title,
        })

      } else {
        // ── Vertical grouped bar chart ────────────────────────────────────────
        const hidden       = hiddenRef.current
        const mdls         = models ?? []
        const acc          = accuracy ?? []
        const f1s          = f1 ?? []
        const aucs         = auc ?? []
        const metrics      = ['Accuracy', 'F1 Score', 'AUC']
        const metricColors = [COLORS.accent, COLORS.purple, COLORS.success]

        const seriesData = [
          { key: 'Accuracy', vals: acc  },
          { key: 'F1 Score', vals: f1s  },
          { key: 'AUC',      vals: aucs },
        ]

        const xGroupOrig = d3.scaleBand().domain(mdls).range([0, iW]).padding(0.3)
        const yOrig      = d3.scaleLinear().domain([0, 1.05]).range([iH, 0])
        const mkXBar     = xG => d3.scaleBand().domain(metrics).range([0, xG.bandwidth()]).padding(0.05)

        // Clipped content area for bars + grid
        const contentG = g.append('g').attr('clip-path', `url(#bar-clip-${cid})`)
        const gridG    = contentG.append('g')

        const drawGrid = (yScale) => {
          gridG.selectAll('*').remove()
          yScale.ticks(5).forEach(tick => {
            gridG.append('line')
              .attr('x1', 0).attr('y1', yScale(tick))
              .attr('x2', iW).attr('y2', yScale(tick))
              .attr('stroke', border).attr('stroke-dasharray', '3,3').attr('stroke-width', 0.8)
          })
        }
        drawGrid(yOrig)

        // Axes — outside contentG (not clipped) but x-axis labels clipped to chart width
        const xAxisG = g.append('g')
          .attr('transform', `translate(0,${iH})`)
          .attr('clip-path', `url(#xaxis-label-clip-${cid})`)
          .call(d3.axisBottom(xGroupOrig)).call(axisStyle)

        const yAxisG = g.append('g')
          .call(d3.axisLeft(yOrig).ticks(5)).call(axisStyle)

        // Draw bars
        const xBarOrig = mkXBar(xGroupOrig)
        seriesData.forEach(({ key, vals }, si) => {
          contentG.selectAll(`.bar-${si}`)
            .data(vals).join('rect')
            .attr('class', `bar-${si}`)
            .attr('x', (_, i) => (xGroupOrig(mdls[i]) ?? 0) + (xBarOrig(key) ?? 0))
            .attr('y', d => yOrig(d))
            .attr('width', xBarOrig.bandwidth())
            .attr('height', d => Math.max(0, iH - yOrig(d)))
            .attr('fill', metricColors[si])
            .attr('rx', 2)
            .attr('visibility', hidden.has(si) ? 'hidden' : 'visible')
            .style('cursor', 'pointer')
            .on('mouseover', function (event, d) {
              if (hidden.has(si)) return
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

        // Zoom: X-axis only — bars spread wider, Y stays fixed so baseline never goes off-screen.
        // translateExtent + extent are both set to the inner chart rect so D3 keeps bars
        // within bounds and panning works symmetrically in both directions.
        const zoomBehavior = d3.zoom()
          .filter(e => e.type !== 'wheel')
          .scaleExtent([1, 10])
          .extent([[m.left, m.top], [m.left + iW, m.top + iH]])
          .translateExtent([[m.left, m.top], [m.left + iW, m.top + iH]])
          .on('zoom', event => {
            zoomTransform.current = event.transform
            const t = event.transform

            // applyX works in SVG space; subtract m.left to convert to g-space range endpoints
            const x0 = t.applyX(m.left) - m.left
            const x1 = t.applyX(m.left + iW) - m.left
            const newXGroup = xGroupOrig.copy().range([x0, x1])
            const newXBar   = mkXBar(newXGroup)

            xAxisG.call(d3.axisBottom(newXGroup)).call(axisStyle)

            seriesData.forEach(({ key, vals }, si) => {
              contentG.selectAll(`.bar-${si}`)
                .attr('x', (_, i) => (newXGroup(mdls[i]) ?? 0) + (newXBar(key) ?? 0))
                .attr('width', newXBar.bandwidth())
            })
          })

        svg.call(zoomBehavior).call(zoomBehavior.transform, zoomTransform.current)

        // Clickable legend — click to toggle series on/off
        const legendY      = H - 16
        const legendStartX = m.left + iW / 2 - (metrics.length * 90) / 2
        metrics.forEach((key, i) => {
          const lx = legendStartX + i * 90

          const legendG = svg.append('g')
            .style('cursor', 'pointer')
            .attr('opacity', hidden.has(i) ? 0.4 : 1)

          legendG.append('rect')
            .attr('x', lx).attr('y', legendY - 10)
            .attr('width', 12).attr('height', 12)
            .attr('fill', metricColors[i]).attr('rx', 2)

          legendG.append('text')
            .attr('x', lx + 16).attr('y', legendY)
            .style('font-size', '11px').style('fill', text).text(key)

          legendG.on('click', function () {
            if (hidden.has(i)) hidden.delete(i)
            else hidden.add(i)
            const isHidden = hidden.has(i)
            contentG.selectAll(`.bar-${i}`).attr('visibility', isHidden ? 'hidden' : 'visible')
            d3.select(this).attr('opacity', isHidden ? 0.4 : 1)
          })
        })

        addToolbar(container, {
          svgSel: svg, zoomBehavior,
          onReset: () => { zoomTransform.current = d3.zoomIdentity },
          title,
        })
      }

      // Title — always fixed at top
      svg.append('text').attr('x', W / 2).attr('y', 22).attr('text-anchor', 'middle')
        .style('font-size', '14px').style('fill', text).text(title)
    }

    draw()
    window.addEventListener('resize', draw)
    return () => window.removeEventListener('resize', draw)
  }, [models, accuracy, f1, auc, title, horizontal, categories, values, theme])

  return <div ref={containerRef} style={{ width: '100%', minHeight: '400px', position: 'relative' }} />
}

export default BarChart
