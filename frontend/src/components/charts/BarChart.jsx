import { useRef, useEffect } from 'react'
import Plotly from 'plotly.js-dist-min'
import { DARK_LAYOUT, CHART_CONFIG } from './chartTheme'

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

  useEffect(() => {
    if (!divRef.current) return

    let traces
    let layout

    if (horizontal) {
      traces = [
        {
          x: values ?? [],
          y: categories ?? [],
          type: 'bar',
          orientation: 'h',
          marker: { color: '#00d4ff' },
        },
      ]
      layout = {
        ...DARK_LAYOUT,
        title: { text: title, font: { color: '#e8eaf0', size: 14 } },
        margin: { ...DARK_LAYOUT.margin, l: 160 },
        yaxis: { ...DARK_LAYOUT.yaxis, automargin: true },
      }
    } else {
      traces = [
        { x: models, y: accuracy, name: 'Accuracy', type: 'bar', marker: { color: '#00d4ff' } },
        { x: models, y: f1, name: 'F1 Score', type: 'bar', marker: { color: '#6c63ff' } },
        { x: models, y: auc, name: 'AUC', type: 'bar', marker: { color: '#00cc88' } },
      ]
      layout = {
        ...DARK_LAYOUT,
        title: { text: title, font: { color: '#e8eaf0', size: 14 } },
        barmode: 'group',
        xaxis: { ...DARK_LAYOUT.xaxis, type: 'category' },
        yaxis: { ...DARK_LAYOUT.yaxis, range: [0, 1.05] },
      }
    }

    Plotly.newPlot(divRef.current, traces, layout, CHART_CONFIG)

    return () => {
      if (divRef.current) Plotly.purge(divRef.current)
    }
  }, [models, accuracy, f1, auc, title, horizontal, categories, values])

  return <div ref={divRef} style={{ width: '100%', minHeight: '400px' }} />
}

export default BarChart
