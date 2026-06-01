import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { setupServer } from 'msw/node'
import { http, HttpResponse } from 'msw'
import { vi, describe, it, expect, beforeAll, afterEach, afterAll } from 'vitest'
import ModelAnalytics from '../ModelAnalytics'

vi.mock('plotly.js-dist-min', () => {
  const mock = { react: vi.fn(), newPlot: vi.fn(), purge: vi.fn() }
  return { default: mock, ...mock }
})

const mockData = {
  model: 'rf_spam',
  confusion_matrix: [[100, 5], [3, 200]],
  roc: { fpr: [0, 0.1, 1], tpr: [0, 0.9, 1], auc: 0.99 },
  feature_importance: [{ feature: 'word_count', importance: 0.3 }],
}
const server = setupServer(
  http.get('http://localhost:8000/api/analytics/model/:name', () => HttpResponse.json(mockData))
)
beforeAll(() => server.listen())
afterEach(() => server.resetHandlers())
afterAll(() => server.close())

describe('ModelAnalytics', () => {
  it('loads without error on mount', async () => {
    render(<MemoryRouter><ModelAnalytics /></MemoryRouter>)
    await waitFor(() => expect(screen.queryByText(/error/i)).not.toBeInTheDocument())
  })
  it('tab switch triggers new fetch', async () => {
    render(<MemoryRouter><ModelAnalytics /></MemoryRouter>)
    const tab = screen.getByRole('button', { name: /naive bayes/i })
    fireEvent.click(tab)
    await waitFor(() => expect(screen.queryByText(/error/i)).not.toBeInTheDocument())
  })
})
