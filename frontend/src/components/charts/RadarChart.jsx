import { useRef, useEffect } from 'react'
import Plotly from 'plotly.js-dist-min'
import { getChartLayout, COLORS, getChartConfig } from './chartTheme'
import { useTheme } from '../../context/ThemeContext'

// Multi-model radar (scatterpolar) comparing metrics on one figure.
//   series: [{ name, values: [..] }]   metrics: ['Accuracy','Precision',...]
const PALETTE = [COLORS.accent, COLORS.purple, COLORS.success, COLORS.warning, COLORS.danger]

function RadarChart({ series = [], metrics = [], title = 'Model Comparison', rangeMin = 0 }) {
  const divRef = useRef(null)
  const { theme } = useTheme()

  useEffect(() => {
    if (!divRef.current) return

    const base = getChartLayout()
    // Close the loop by repeating the first metric/value at the end.
    const closedMetrics = metrics.length ? [...metrics, metrics[0]] : []
    const traces = series.map((s, i) => {
      const color = s.color ?? PALETTE[i % PALETTE.length]
      const vals = s.values ?? []
      return {
        type: 'scatterpolar',
        r: vals.length ? [...vals, vals[0]] : [],
        theta: closedMetrics,
        fill: 'toself',
        name: s.name,
        line: { color },
        marker: { color },
        opacity: 0.75,
      }
    })

    const gridColor = base.xaxis.gridcolor
    const layout = {
      ...base,
      title: { text: title, font: { color: base.font.color, size: 14 } },
      polar: {
        bgcolor: base.plot_bgcolor,
        radialaxis: {
          visible: true, range: [rangeMin, 1], gridcolor: gridColor,
          tickfont: { color: base.xaxis.tickfont.color }, angle: 90,
        },
        angularaxis: { gridcolor: gridColor, tickfont: { color: base.font.color } },
      },
    }

    Plotly.newPlot(divRef.current, traces, layout, getChartConfig(Plotly))

    return () => {
      if (divRef.current) Plotly.purge(divRef.current)
    }
  }, [series, metrics, title, rangeMin, theme])

  return <div ref={divRef} style={{ width: '100%', minHeight: '400px' }} />
}

export default RadarChart
