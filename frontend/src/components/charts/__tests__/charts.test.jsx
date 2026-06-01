import { render } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import Plotly from 'plotly.js-dist-min'
import BarChart from '../BarChart'
import LineChart from '../LineChart'
import GaugeChart from '../GaugeChart'
import ScatterPlot from '../ScatterPlot'
import Heatmap from '../Heatmap'

// Mock Plotly so jsdom doesn't need a real canvas. Provide both a default export
// (charts use `import Plotly from 'plotly.js-dist-min'`) and named exports.
vi.mock('plotly.js-dist-min', () => {
  const mock = { newPlot: vi.fn(), purge: vi.fn(), react: vi.fn() }
  return { default: mock, ...mock }
})

describe('Chart components smoke tests', () => {
  it('BarChart renders container div', () => {
    const { container } = render(
      <BarChart models={['rf_spam']} accuracy={[0.98]} f1={[0.98]} auc={[0.99]} />
    )
    expect(container.querySelector('div')).toBeTruthy()
  })
  it('LineChart renders container div', () => {
    const { container } = render(<LineChart spamSeries={[]} malwareSeries={[]} />)
    expect(container.querySelector('div')).toBeTruthy()
  })
  it('GaugeChart renders container div', () => {
    const { container } = render(<GaugeChart spamProb={0.8} label="SPAM" />)
    expect(container.querySelector('div')).toBeTruthy()
  })
  it('ScatterPlot renders container div', () => {
    const { container } = render(
      <ScatterPlot pcaData={[[0, 1]]} labels={['BENIGN']} clusters={[0]} anomalies={[false]} rowIds={[1]} />
    )
    expect(container.querySelector('div')).toBeTruthy()
  })
  it('Heatmap renders container div', () => {
    const { container } = render(
      <Heatmap matrix={[[100, 5], [3, 200]]} labels={['Ham', 'Spam']} />
    )
    expect(container.querySelector('div')).toBeTruthy()
  })
  it('Plotly.newPlot called when BarChart mounts', () => {
    render(<BarChart models={['m']} accuracy={[0.9]} f1={[0.9]} auc={[0.9]} />)
    expect(Plotly.newPlot).toHaveBeenCalled()
  })
})
