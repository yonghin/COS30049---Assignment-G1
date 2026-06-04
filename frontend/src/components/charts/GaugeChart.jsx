import { useRef, useEffect } from 'react'
import Plotly from 'plotly.js-dist-min'
import { getChartLayout, COLORS, CHART_CONFIG } from './chartTheme'
import { useTheme } from '../../context/ThemeContext'

// Spam probability gauge. spamProb === null renders an empty gauge at 0.
function GaugeChart({ spamProb = null, label }) {
  const divRef = useRef(null)
  const { theme } = useTheme()

  useEffect(() => {
    if (!divRef.current) return

    const base = getChartLayout()
    const barColor = (spamProb ?? 0) >= 0.5 ? COLORS.danger : COLORS.success
    const traces = [
      {
        type: 'indicator',
        mode: 'gauge+number',
        value: Math.round((spamProb ?? 0) * 100),
        number: { suffix: '%', font: { color: base.font.color, size: 36 } },
        title: { text: label ?? 'Spam Probability', font: { color: COLORS.muted, size: 14 } },
        gauge: {
          axis: { range: [0, 100], tickcolor: COLORS.muted },
          bar: { color: barColor },
          bgcolor: base.plot_bgcolor,
          bordercolor: base.xaxis.gridcolor,
          steps: [
            { range: [0, 50], color: 'rgba(0,204,136,0.15)' },
            { range: [50, 100], color: 'rgba(255,77,77,0.15)' },
          ],
          threshold: { line: { color: barColor, width: 3 }, thickness: 0.75, value: (spamProb ?? 0) * 100 },
        },
      },
    ]
    const layout = { ...base, height: 300, margin: { l: 30, r: 30, t: 30, b: 30 } }

    Plotly.newPlot(divRef.current, traces, layout, CHART_CONFIG)

    return () => {
      if (divRef.current) Plotly.purge(divRef.current)
    }
  }, [spamProb, label, theme])

  return <div ref={divRef} style={{ width: '100%', minHeight: '300px' }} />
}

export default GaugeChart
