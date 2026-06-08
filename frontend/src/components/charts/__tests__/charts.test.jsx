import { render } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import BarChart from '../BarChart'
import LineChart from '../LineChart'
import GaugeChart from '../GaugeChart'
import ScatterPlot from '../ScatterPlot'
import Heatmap from '../Heatmap'

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
  it('BarChart renders an SVG element via D3', () => {
    const { container } = render(<BarChart models={['m']} accuracy={[0.9]} f1={[0.9]} auc={[0.9]} />)
    expect(container.querySelector('svg')).toBeTruthy()
  })
})
