import { useRef, useEffect } from 'react'
import Plotly from 'plotly.js-dist-min'
import { getChartLayout, COLORS, getChartConfig } from './chartTheme'
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
  color,
}) {
  const divRef = useRef(null)
  const { theme } = useTheme()
  // Refs to track user zoom state for the timeseries chart.
  const isZoomedRef  = useRef(false)
  const savedRangeRef = useRef(null)
  // Ref to the attached relayout listener so we can remove it on unmount.
  const listenerRef  = useRef(null)

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
          line: { color: color ?? COLORS.accent, width: 2.5 },
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
        uirevision: 'roc',
        title: {
          text: title ?? `ROC Curve (AUC = ${auc != null ? auc.toFixed(4) : 'N/A'})`,
          font: { color: base.font.color, size: 14 },
        },
        xaxis: { ...base.xaxis, title: 'False Positive Rate', type: 'linear', range: [0, 1], tickmode: 'linear', tick0: 0, dtick: 0.2 },
        yaxis: { ...base.yaxis, title: 'True Positive Rate', type: 'linear', range: [0, 1.02], tickmode: 'linear', tick0: 0, dtick: 0.2 },
      }
    } else {
      const spam    = spamSeries    ?? []
      const malware = malwareSeries ?? []
      // Convert UTC timestamp → Malaysia time (UTC+8), then format as "YYYY-MM-DD HH:MM:SS"
      // for Plotly (which rejects ISO timezone offsets and treats bare strings as local time).
      const toPlotlyDate = (s) => {
        if (!s) return ''
        const myt = new Date(new Date(s).getTime() + 8 * 60 * 60 * 1000)
        return myt.toISOString().replace('T', ' ').split('.')[0]
      }
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

      // If the user has zoomed/panned, inject their saved range into the layout so
      // Plotly.react never auto-ranges over it. Cleared when user hits Reset Axes.
      const xAxis = isZoomedRef.current && savedRangeRef.current
        ? { ...base.xaxis, title: 'Time', type: 'date', range: savedRangeRef.current.x, autorange: false }
        : { ...base.xaxis, title: 'Time', type: 'date' }
      const yAxis = isZoomedRef.current && savedRangeRef.current
        ? { ...base.yaxis, title: 'Count', range: savedRangeRef.current.y, autorange: false }
        : { ...base.yaxis, title: 'Count' }

      layout = {
        ...base,
        uirevision: 'timeseries',
        title: { text: title ?? 'Live Predictions', font: { color: base.font.color, size: 14 } },
        xaxis: xAxis,
        yaxis: yAxis,
      }
    }

    // Plotly.react updates data without resetting user zoom/pan (uirevision keeps UI state stable).
    Plotly.react(divRef.current, traces, layout, getChartConfig(Plotly))

    // Attach the relayout listener once after Plotly has initialized the div
    // (div.on is added by Plotly — must call after Plotly.react, not before).
    if (!isRoc && !listenerRef.current) {
      const div = divRef.current
      listenerRef.current = (e) => {
        if (e['xaxis.autorange'] === true) {
          // User clicked Reset Axes — allow auto-range again.
          isZoomedRef.current  = false
          savedRangeRef.current = null
        } else if ('xaxis.range[0]' in e || 'xaxis.range' in e) {
          // User zoomed or panned — save range from Plotly's internal layout.
          isZoomedRef.current = true
          if (div._fullLayout) {
            savedRangeRef.current = {
              x: div._fullLayout.xaxis.range.slice(),
              y: div._fullLayout.yaxis.range.slice(),
            }
          }
        }
      }
      div.on('plotly_relayout', listenerRef.current)
    }
  }, [spamSeries, malwareSeries, fpr, tpr, auc, title, color, theme])

  // Purge only on unmount — not between data updates — so user zoom is preserved.
  useEffect(() => {
    const div = divRef.current
    return () => {
      if (div) {
        if (listenerRef.current) {
          try { div.removeListener('plotly_relayout', listenerRef.current) } catch (_) {}
          listenerRef.current = null
        }
        Plotly.purge(div)
      }
    }
  }, [])

  return <div ref={divRef} style={{ width: '100%', minHeight: '400px' }} />
}

export default LineChart
