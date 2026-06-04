import { useRef, useEffect } from 'react'
import Plotly from 'plotly.js-dist-min'
import { getChartLayout, COLORS, CHART_CONFIG } from './chartTheme'
import { useTheme } from '../../context/ThemeContext'

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
  const { theme } = useTheme()

  useEffect(() => {
    if (!divRef.current) return

    const base = getChartLayout()
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
          line: { color: COLORS.accent, width: 2 },
        },
        {
          x: [0, 1],
          y: [0, 1],
          mode: 'lines',
          name: 'Random',
          line: { color: COLORS.muted, width: 1, dash: 'dash' },
        },
      ]
      layout = {
        ...base,
        title: {
          text: title ?? `ROC Curve (AUC = ${auc != null ? auc.toFixed(4) : 'N/A'})`,
          font: { color: base.font.color, size: 14 },
        },
        xaxis: { ...base.xaxis, title: 'False Positive Rate', type: 'linear', range: [0, 1], tickmode: 'linear', tick0: 0, dtick: 0.2 },
        yaxis: { ...base.yaxis, title: 'True Positive Rate', type: 'linear', range: [0, 1.02], tickmode: 'linear', tick0: 0, dtick: 0.2 },
      }
    } else {
      const spam = spamSeries ?? []
      const malware = malwareSeries ?? []
      // Plotly's date parser rejects an ISO timezone offset ("...+00:00" / "...Z"),
      // so normalize to "YYYY-MM-DD HH:MM:SS" — otherwise the points never plot and
      // the x-axis collapses to a meaningless 0–0.5 range.
      const toPlotlyDate = (s) => (s ?? '').replace('T', ' ').split(/[+Z]/)[0]
      traces = [
        {
          x: spam.map((p) => toPlotlyDate(p.timestamp)),
          y: spam.map((p) => p.count),
          mode: 'lines+markers',
          name: 'Spam',
          line: { color: COLORS.accent },
        },
        {
          x: malware.map((p) => toPlotlyDate(p.timestamp)),
          y: malware.map((p) => p.count),
          mode: 'lines+markers',
          name: 'Malware',
          line: { color: COLORS.danger },
        },
      ]
      layout = {
        ...base,
        title: { text: title ?? 'Live Predictions', font: { color: base.font.color, size: 14 } },
        xaxis: { ...base.xaxis, title: 'Time', type: 'date' },
        yaxis: { ...base.yaxis, title: 'Count' },
      }
    }

    Plotly.newPlot(divRef.current, traces, layout, CHART_CONFIG)

    return () => {
      if (divRef.current) Plotly.purge(divRef.current)
    }
  }, [spamSeries, malwareSeries, fpr, tpr, auc, title, theme])

  return <div ref={divRef} style={{ width: '100%', minHeight: '400px' }} />
}

export default LineChart
