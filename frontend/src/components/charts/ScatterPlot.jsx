import { useRef, useEffect } from 'react'
import Plotly from 'plotly.js-dist-min'
import { getChartLayout, COLORS, CHART_CONFIG } from './chartTheme'
import { useTheme } from '../../context/ThemeContext'

// PCA 2D malware scatter. Accepts parallel arrays:
//   pcaData: [[x, y], ...], labels: [], clusters: [], anomalies: [bool], rowIds: []
function ScatterPlot({ pcaData = [], labels = [], clusters = [], anomalies = [], rowIds = [], title = 'PCA Projection' }) {
  const divRef = useRef(null)
  const { theme } = useTheme()

  useEffect(() => {
    if (!divRef.current) return

    const points = pcaData.map((coord, i) => ({
      x: coord[0],
      y: coord[1],
      label: labels[i],
      cluster: clusters[i],
      isAnomaly: anomalies[i],
      rowId: rowIds[i] ?? i + 1,
    }))

    const benignNormal = points.filter((p) => p.label === 'BENIGN' && !p.isAnomaly)
    const malwareNormal = points.filter((p) => p.label === 'MALWARE' && !p.isAnomaly)
    const anomalyPoints = points.filter((p) => p.isAnomaly)

    const traces = [
      {
        x: benignNormal.map((p) => p.x),
        y: benignNormal.map((p) => p.y),
        mode: 'markers',
        name: 'Benign',
        type: 'scatter',
        marker: { color: COLORS.success, size: 8, opacity: 0.8 },
        text: benignNormal.map((p) => `Row ${p.rowId} | Cluster ${p.cluster}`),
      },
      {
        x: malwareNormal.map((p) => p.x),
        y: malwareNormal.map((p) => p.y),
        mode: 'markers',
        name: 'Malware',
        type: 'scatter',
        marker: { color: COLORS.danger, size: 8, opacity: 0.8 },
        text: malwareNormal.map((p) => `Row ${p.rowId} | Cluster ${p.cluster}`),
      },
      {
        x: anomalyPoints.map((p) => p.x),
        y: anomalyPoints.map((p) => p.y),
        mode: 'markers',
        name: 'Anomaly',
        type: 'scatter',
        marker: { color: COLORS.warning, symbol: 'x', size: 12, line: { width: 2, color: COLORS.warning } },
        text: anomalyPoints.map((p) => `Row ${p.rowId} | Cluster ${p.cluster}`),
      },
    ]

    const base = getChartLayout()
    const layout = {
      ...base,
      title: { text: title, font: { color: base.font.color, size: 14 } },
      xaxis: { ...base.xaxis, title: 'PC1' },
      yaxis: { ...base.yaxis, title: 'PC2' },
    }

    Plotly.newPlot(divRef.current, traces, layout, CHART_CONFIG)

    return () => {
      if (divRef.current) Plotly.purge(divRef.current)
    }
  }, [pcaData, labels, clusters, anomalies, rowIds, title, theme])

  return <div ref={divRef} style={{ width: '100%', minHeight: '400px' }} />
}

export default ScatterPlot
