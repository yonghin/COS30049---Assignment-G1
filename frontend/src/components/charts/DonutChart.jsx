import { useRef, useEffect } from 'react'
import Plotly from 'plotly.js-dist-min'
import { getChartLayout, COLORS, getChartConfig } from './chartTheme'
import { useTheme } from '../../context/ThemeContext'

// Donut (pie with a hole) for class proportions.
//   labels: ['Ham','Spam']  values: [80, 20]  colors: optional override
const DEFAULT_COLORS = [COLORS.success, COLORS.danger, COLORS.warning, COLORS.accent, COLORS.purple]

function DonutChart({ labels = [], values = [], colors, title = 'Class Distribution' }) {
  const divRef = useRef(null)
  const { theme } = useTheme()

  useEffect(() => {
    if (!divRef.current) return

    const base = getChartLayout()
    const traces = [
      {
        type: 'pie',
        labels,
        values,
        hole: 0.5,
        marker: {
          colors: colors ?? DEFAULT_COLORS.slice(0, labels.length),
          line: { color: base.plot_bgcolor, width: 2 },
        },
        textfont: { color: theme === 'light' ? '#111827' : '#ffffff' },
        hovertemplate: '%{label}: %{value} (%{percent})<extra></extra>',
      },
    ]
    const layout = {
      ...base,
      height: 320,
      title: { text: title, font: { color: base.font.color, size: 14 } },
      showlegend: true,
    }

    Plotly.newPlot(divRef.current, traces, layout, getChartConfig(Plotly))

    return () => {
      if (divRef.current) Plotly.purge(divRef.current)
    }
  }, [labels, values, colors, title, theme])

  return <div ref={divRef} style={{ width: '100%', minHeight: '320px' }} />
}

export default DonutChart
