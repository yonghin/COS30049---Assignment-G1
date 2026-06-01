import { useRef, useEffect } from 'react'
import Plotly from 'plotly.js-dist-min'
import { DARK_LAYOUT, CHART_CONFIG } from './chartTheme'

// Dual-purpose line chart:
//  - Time series mode: pass `spamSeries` / `malwareSeries` ([{timestamp, count}])
//  - ROC mode: pass `fpr` / `tpr` / `auc` (renders curve + diagonal reference line)
function LineChart({
  spamSeries,
  malwareSeries,
  fpr,
  tpr,
  auc,
  title,
}) {
  const divRef = useRef(null)

  useEffect(() => {
    if (!divRef.current) return

    let traces
    let layout

    const isRoc = Array.isArray(fpr) && Array.isArray(tpr)

    if (isRoc) {
      traces = [
        {
          x: fpr,
          y: tpr,
          mode: 'lines',
          name: 'ROC',
          line: { color: '#00d4ff', width: 2 },
        },
        {
          x: [0, 1],
          y: [0, 1],
          mode: 'lines',
          name: 'Random',
          line: { color: '#8892a4', width: 1, dash: 'dash' },
        },
      ]
      layout = {
        ...DARK_LAYOUT,
        title: {
          text: title ?? `ROC Curve (AUC = ${auc != null ? auc.toFixed(4) : 'N/A'})`,
          font: { color: '#e8eaf0', size: 14 },
        },
        xaxis: { ...DARK_LAYOUT.xaxis, title: 'False Positive Rate', range: [0, 1] },
        yaxis: { ...DARK_LAYOUT.yaxis, title: 'True Positive Rate', range: [0, 1.02] },
      }
    } else {
      const spam = spamSeries ?? []
      const malware = malwareSeries ?? []
      traces = [
        {
          x: spam.map((p) => p.timestamp),
          y: spam.map((p) => p.count),
          mode: 'lines+markers',
          name: 'Spam',
          line: { color: '#00d4ff' },
        },
        {
          x: malware.map((p) => p.timestamp),
          y: malware.map((p) => p.count),
          mode: 'lines+markers',
          name: 'Malware',
          line: { color: '#ff4d4d' },
        },
      ]
      layout = {
        ...DARK_LAYOUT,
        title: { text: title ?? 'Live Predictions', font: { color: '#e8eaf0', size: 14 } },
        xaxis: { ...DARK_LAYOUT.xaxis, title: 'Time' },
        yaxis: { ...DARK_LAYOUT.yaxis, title: 'Count' },
      }
    }

    Plotly.newPlot(divRef.current, traces, layout, CHART_CONFIG)

    return () => {
      if (divRef.current) Plotly.purge(divRef.current)
    }
  }, [spamSeries, malwareSeries, fpr, tpr, auc, title])

  return <div ref={divRef} style={{ width: '100%', minHeight: '400px' }} />
}

export default LineChart
