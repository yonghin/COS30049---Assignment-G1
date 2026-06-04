import { useRef, useEffect } from 'react'
import Plotly from 'plotly.js-dist-min'
import { getChartLayout, getChartConfig } from './chartTheme'
import { useTheme } from '../../context/ThemeContext'

// Confusion matrix heatmap. matrix = [[TN, FP], [FN, TP]], labels e.g. ['Ham','Spam'].
function Heatmap({ matrix = [], labels = [], title = 'Confusion Matrix' }) {
  const divRef = useRef(null)
  const { theme } = useTheme()

  useEffect(() => {
    if (!divRef.current) return

    const base = getChartLayout()
    const traces = [
      {
        type: 'heatmap',
        z: matrix,
        x: labels,
        y: labels,
        colorscale: [[0, base.plot_bgcolor], [0.5, '#0099bb'], [1, '#00d4ff']],
        showscale: false,
        text: matrix.map((row) => row.map((v) => String(v))),
        texttemplate: '%{text}',
        textfont: { color: '#e8eaf0', size: 16 },
        hovertemplate: 'Actual %{y} / Predicted %{x}: %{z}<extra></extra>',
      },
    ]

    const layout = {
      ...base,
      height: 360,
      title: { text: title, font: { color: base.font.color, size: 14 } },
      xaxis: { ...base.xaxis, title: 'Predicted', type: 'category' },
      yaxis: { ...base.yaxis, title: 'Actual', type: 'category', autorange: 'reversed' },
    }

    Plotly.newPlot(divRef.current, traces, layout, getChartConfig(Plotly))

    return () => {
      if (divRef.current) Plotly.purge(divRef.current)
    }
  }, [matrix, labels, title, theme])

  return <div ref={divRef} style={{ width: '100%', minHeight: '360px' }} />
}

export default Heatmap
