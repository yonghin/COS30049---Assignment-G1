import { useRef, useEffect } from 'react'
import Plotly from 'plotly.js-dist-min'
import { DARK_LAYOUT, CHART_CONFIG } from './chartTheme'

// Confusion matrix heatmap. matrix = [[TN, FP], [FN, TP]], labels e.g. ['Ham','Spam'].
function Heatmap({ matrix = [], labels = [], title = 'Confusion Matrix' }) {
  const divRef = useRef(null)

  useEffect(() => {
    if (!divRef.current) return

    const traces = [
      {
        type: 'heatmap',
        z: matrix,
        x: labels,
        y: labels,
        colorscale: [[0, '#1a1d2e'], [0.5, '#0099bb'], [1, '#00d4ff']],
        showscale: false,
        text: matrix.map((row) => row.map((v) => String(v))),
        texttemplate: '%{text}',
        textfont: { color: '#e8eaf0', size: 16 },
        hovertemplate: 'Actual %{y} / Predicted %{x}: %{z}<extra></extra>',
      },
    ]

    const layout = {
      ...DARK_LAYOUT,
      height: 360,
      title: { text: title, font: { color: '#e8eaf0', size: 14 } },
      xaxis: { ...DARK_LAYOUT.xaxis, title: 'Predicted' },
      yaxis: { ...DARK_LAYOUT.yaxis, title: 'Actual', autorange: 'reversed' },
    }

    Plotly.newPlot(divRef.current, traces, layout, CHART_CONFIG)

    return () => {
      if (divRef.current) Plotly.purge(divRef.current)
    }
  }, [matrix, labels, title])

  return <div ref={divRef} style={{ width: '100%', minHeight: '360px' }} />
}

export default Heatmap
