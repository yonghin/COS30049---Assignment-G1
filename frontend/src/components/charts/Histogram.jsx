import { useRef, useEffect } from 'react'
import Plotly from 'plotly.js-dist-min'
import { getChartLayout, COLORS, CHART_CONFIG } from './chartTheme'
import { useTheme } from '../../context/ThemeContext'

// Distribution histogram for a single numeric series (e.g. prediction confidence).
function Histogram({
  values = [],
  title = 'Confidence Distribution',
  xLabel = 'Confidence',
  color = COLORS.accent,
  nbins = 20,
}) {
  const divRef = useRef(null)
  const { theme } = useTheme()

  useEffect(() => {
    if (!divRef.current) return

    const base = getChartLayout()
    const traces = [
      {
        type: 'histogram',
        x: values,
        nbinsx: nbins,
        marker: { color, line: { color: base.plot_bgcolor, width: 1 } },
        opacity: 0.85,
      },
    ]
    const layout = {
      ...base,
      title: { text: title, font: { color: base.font.color, size: 14 } },
      bargap: 0.05,
      xaxis: { ...base.xaxis, title: xLabel },
      yaxis: { ...base.yaxis, title: 'Count' },
    }

    Plotly.newPlot(divRef.current, traces, layout, CHART_CONFIG)

    return () => {
      if (divRef.current) Plotly.purge(divRef.current)
    }
  }, [values, title, xLabel, color, nbins, theme])

  return <div ref={divRef} style={{ width: '100%', minHeight: '320px' }} />
}

export default Histogram
