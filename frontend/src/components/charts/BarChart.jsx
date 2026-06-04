import { useRef, useEffect } from 'react'
import Plotly from 'plotly.js-dist-min'
import { getChartLayout, COLORS, getChartConfig } from './chartTheme'
import { useTheme } from '../../context/ThemeContext'

const H_PALETTE = [COLORS.accent, COLORS.purple, COLORS.success, COLORS.warning, COLORS.danger]

// Grouped bar chart for model metric comparison (accuracy / f1 / auc),
// or a single horizontal bar trace when `horizontal` + `values`/`categories` are given.
function BarChart({
  models,
  accuracy,
  f1,
  auc,
  title = 'Model Performance',
  horizontal = false,
  categories,
  values,
}) {
  const divRef = useRef(null)
  const { theme } = useTheme()

  useEffect(() => {
    if (!divRef.current) return

    const base = getChartLayout()
    let traces
    let layout

    if (horizontal) {
      traces = [
        {
          x: values ?? [],
          y: categories ?? [],
          type: 'bar',
          orientation: 'h',
          marker: { color: (values ?? []).map((_, i) => H_PALETTE[i % H_PALETTE.length]) },
        },
      ]
      layout = {
        ...base,
        title: { text: title, font: { color: base.font.color, size: 14 } },
        margin: { ...base.margin, l: 160 },
        yaxis: { ...base.yaxis, automargin: true },
      }
    } else {
      traces = [
        { x: models, y: accuracy, name: 'Accuracy', type: 'bar', marker: { color: COLORS.accent } },
        { x: models, y: f1, name: 'F1 Score', type: 'bar', marker: { color: COLORS.purple } },
        { x: models, y: auc, name: 'AUC', type: 'bar', marker: { color: COLORS.success } },
      ]
      layout = {
        ...base,
        title: { text: title, font: { color: base.font.color, size: 14 } },
        barmode: 'group',
        xaxis: { ...base.xaxis, type: 'category' },
        yaxis: { ...base.yaxis, range: [0, 1.05] },
      }
    }

    Plotly.newPlot(divRef.current, traces, layout, getChartConfig(Plotly))

    return () => {
      if (divRef.current) Plotly.purge(divRef.current)
    }
  }, [models, accuracy, f1, auc, title, horizontal, categories, values, theme])

  return <div ref={divRef} style={{ width: '100%', minHeight: '400px' }} />
}

export default BarChart
