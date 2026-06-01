import { render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { setupServer } from 'msw/node'
import { http, HttpResponse } from 'msw'
import { vi, describe, it, expect, beforeAll, afterEach, afterAll } from 'vitest'
import Dashboard from '../Dashboard'

vi.mock('plotly.js-dist-min', () => {
  const mock = { react: vi.fn(), newPlot: vi.fn(), purge: vi.fn() }
  return { default: mock, ...mock }
})

const server = setupServer(
  http.get('http://localhost:8000/api/models', () =>
    HttpResponse.json({ models: [{ name: 'rf_spam', task: 'Spam Detection', accuracy: 0.98, f1: 0.98, auc: 0.99 }] })
  ),
  http.get('http://localhost:8000/api/predictions/history', () =>
    HttpResponse.json({ spam_series: [], malware_series: [] })
  )
)

beforeAll(() => server.listen())
afterEach(() => server.resetHandlers())
afterAll(() => server.close())

describe('Dashboard', () => {
  it('renders model card after fetch', async () => {
    render(<MemoryRouter><Dashboard /></MemoryRouter>)
    await waitFor(() => expect(screen.getByText(/rf_spam/i)).toBeInTheDocument())
  })
})
