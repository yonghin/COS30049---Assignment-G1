export const DARK_LAYOUT = {
  paper_bgcolor: '#1a1d2e',
  plot_bgcolor:  '#1a1d2e',
  font: { color: '#e8eaf0', family: 'Inter, system-ui, sans-serif', size: 12 },
  xaxis: {
    gridcolor: '#2a2d3e', zerolinecolor: '#2a2d3e',
    tickfont: { color: '#8892a4' }, titlefont: { color: '#8892a4' },
  },
  yaxis: {
    gridcolor: '#2a2d3e', zerolinecolor: '#2a2d3e',
    tickfont: { color: '#8892a4' }, titlefont: { color: '#8892a4' },
  },
  legend: { bgcolor: 'transparent', font: { color: '#e8eaf0' }, orientation: 'h', y: -0.2 },
  margin: { l: 60, r: 30, t: 50, b: 60 },
  height: 400,
}

export const CHART_CONFIG = {
  responsive: true,
  displayModeBar: true,
  toImageButtonOptions: { format: 'png', scale: 2 },
  modeBarButtonsToRemove: ['sendDataToCloud'],
}
