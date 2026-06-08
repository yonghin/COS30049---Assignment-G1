import { render } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import RadarChart from '../RadarChart'
import Histogram from '../Histogram'
import DonutChart from '../DonutChart'

describe('New chart components smoke tests', () => {
  it('RadarChart renders container div', () => {
    const { container } = render(
      <RadarChart
        metrics={['Accuracy', 'Precision', 'Recall', 'F1']}
        series={[{ name: 'RF', values: [0.98, 0.97, 0.96, 0.97] }]}
      />
    )
    expect(container.querySelector('div')).toBeTruthy()
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
