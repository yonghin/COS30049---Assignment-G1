# Module: Chart Components

**Files:** `src/components/charts/BarChart.jsx`, `LineChart.jsx`, `GaugeChart.jsx`, `ScatterPlot.jsx`, `Heatmap.jsx`, `DonutChart.jsx`, `Histogram.jsx`, `RadarChart.jsx`

All chart components are built with **D3.js v7**. They receive data via props and redraw the SVG inside a `useEffect` when props or theme change.

## Shared Pattern

```jsx
const containerRef = useRef(null)
const { theme } = useTheme()

useEffect(() => {
  const container = containerRef.current
  if (!container) return

  const draw = () => {
    d3.select(container).selectAll('*').remove()
    const { bg, text, muted, border } = getThemeColors()
    const W = container.clientWidth || 600
    // ... build SVG with d3.select(container).append('svg') ...
  }

  draw()
  window.addEventListener('resize', draw)
  return () => window.removeEventListener('resize', draw)
}, [data, theme])

return <div ref={containerRef} style={{ width: '100%', position: 'relative' }} />
```

## Tasks

### `BarChart.jsx`

- [ ] Props: `models: string[]`, `accuracy: number[]`, `f1: number[]`, `auc: number[]`, `title?`
- [ ] Render three grouped bar traces: Accuracy, F1, AUC
- [ ] Hover tooltips show exact values

### `LineChart.jsx`

- [ ] Props: `spamSeries: {timestamp, count}[]`, `malwareSeries: {timestamp, count}[]`, `title?`
- [ ] Render two line traces on a shared time-axis (Spam, Malware)
- [ ] Used by Dashboard for live refresh and by ModelAnalytics for ROC curve (reuse with `fpr`/`tpr` data)

### `GaugeChart.jsx`

- [ ] Props: `spamProb: number | null`, `label?: string`
- [ ] Render semicircle gauge with D3 arc
- [ ] Color: red when `spamProb >= 0.5`, green otherwise
- [ ] Render empty/neutral state when `spamProb` is `null`

### `ScatterPlot.jsx`

- [ ] Props: `pcaData: number[][]`, `labels: string[]`, `clusters: number[]`, `anomalies: boolean[]`, `rowIds: number[]`, `title?`
- [ ] Render three groups:
  - BENIGN points (green circle)
  - MALWARE points (red circle)
  - Anomaly markers (✕ text) for rows where `anomalies[i] === true`
- [ ] Tooltip shows row ID on hover
- [ ] Supports D3 zoom and pan

### `Heatmap.jsx`

- [ ] Props: `matrix: number[][]`, `labels: string[]`, `title?`
- [ ] Render D3 scaleBand heatmap with colour scale
- [ ] Annotate each cell with its count value
- [ ] `labels` used for both x-axis (predicted) and y-axis (actual)

### `DonutChart.jsx`

- [ ] Props: `labels: string[]`, `values: number[]`, `title?`
- [ ] D3 pie + arc, centre total, horizontal legend

### `Histogram.jsx`

- [ ] Props: `values: number[]`, `nbins?`, `color?`, `title?`
- [ ] D3 bin, bar rects, gridlines

### `RadarChart.jsx`

- [ ] Props: `series: {name, values}[]`, `metrics: string[]`, `title?`, `rangeMin?`
- [ ] Manual polygon grid rings + spokes + filled series polygons
