import { render } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import Plotly from 'plotly.js-dist-min'
import RadarChart from '../RadarChart'
import Histogram from '../Histogram'
import DonutChart from '../DonutChart'

// Mock Plotly so jsdom doesn't need a real canvas (default + named exports).
vi.mock('plotly.js-dist-min', () => {
  const mock = { newPlot: vi.fn(), purge: vi.fn(), react: vi.fn() }
  return { default: mock, ...mock }
})

describe('New chart components smoke tests', () => {
  it('RadarChart renders and calls newPlot', () => {
    const { container } = render(
      <RadarChart
        metrics={['Accuracy', 'Precision', 'Recall', 'F1']}
        series={[{ name: 'RF', values: [0.98, 0.97, 0.96, 0.97] }]}
      />
    )
    expect(container.querySelector('div')).toBeTruthy()
    expect(Plotly.newPlot).toHaveBeenCalled()
  })
  it('Histogram renders container div', () => {
    const { container } = render(<Histogram values={[0.1, 0.5, 0.9, 0.95]} />)
    expect(container.querySelector('div')).toBeTruthy()
  })
  it('DonutChart renders container div', () => {
    const { container } = render(<DonutChart labels={['Ham', 'Spam']} values={[80, 20]} />)
    expect(container.querySelector('div')).toBeTruthy()
  })
})
