import { useRef, useEffect } from 'react'
import Plotly from 'plotly.js-dist-min'
import { DARK_LAYOUT, CHART_CONFIG } from './chartTheme'

// Spam probability gauge. spamProb === null renders an empty gauge at 0.
function GaugeChart({ spamProb = null, label }) {
  const divRef = useRef(null)

  useEffect(() => {
    if (!divRef.current) return

    const barColor = (spamProb ?? 0) >= 0.5 ? '#ff4d4d' : '#00cc88'
    const traces = [
      {
        type: 'indicator',
        mode: 'gauge+number',
        value: Math.round((spamProb ?? 0) * 100),
        number: { suffix: '%', font: { color: '#e8eaf0', size: 36 } },
        title: { text: label ?? 'Spam Probability', font: { color: '#8892a4', size: 14 } },
        gauge: {
          axis: { range: [0, 100], tickcolor: '#8892a4' },
          bar: { color: barColor },
          bgcolor: '#0d1020',
          bordercolor: '#2a2d3e',
          steps: [
            { range: [0, 50], color: 'rgba(0,204,136,0.15)' },
            { range: [50, 100], color: 'rgba(255,77,77,0.15)' },
          ],
          threshold: { line: { color: barColor, width: 3 }, thickness: 0.75, value: (spamProb ?? 0) * 100 },
        },
      },
    ]
    const layout = { ...DARK_LAYOUT, height: 300, margin: { l: 30, r: 30, t: 30, b: 30 } }

    Plotly.newPlot(divRef.current, traces, layout, CHART_CONFIG)

    return () => {
      if (divRef.current) Plotly.purge(divRef.current)
    }
  }, [spamProb, label])

  return <div ref={divRef} style={{ width: '100%', minHeight: '300px' }} />
}

export default GaugeChart
