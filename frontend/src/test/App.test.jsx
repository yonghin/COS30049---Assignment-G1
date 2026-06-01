import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { vi } from 'vitest'
import { setupServer } from 'msw/node'
import { http, HttpResponse } from 'msw'
import { beforeAll, afterEach, afterAll } from 'vitest'
import Dashboard from '../pages/Dashboard'

// Dashboard pulls in chart components that import plotly.js, which can't load in
// the node test environment — mock it.
vi.mock('plotly.js-dist-min', () => {
  const mock = { react: vi.fn(), newPlot: vi.fn(), purge: vi.fn() }
  return { default: mock, ...mock }
})

const server = setupServer(
  http.get('http://localhost:8000/api/models', () => HttpResponse.json({ models: [] })),
  http.get('http://localhost:8000/api/predictions/history', () =>
    HttpResponse.json({ spam_series: [], malware_series: [] })
  )
)
beforeAll(() => server.listen())
afterEach(() => server.resetHandlers())
afterAll(() => server.close())

test('Dashboard renders heading region', () => {
  render(<MemoryRouter><Dashboard /></MemoryRouter>)
  expect(screen.getByText(/Recent Activity/i)).toBeInTheDocument()
})
