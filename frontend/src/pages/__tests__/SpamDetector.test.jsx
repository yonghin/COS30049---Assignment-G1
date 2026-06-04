import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { setupServer } from 'msw/node'
import { http, HttpResponse } from 'msw'
import { vi, describe, it, expect, beforeAll, afterEach, afterAll } from 'vitest'
import SpamDetector from '../SpamDetector'

vi.mock('plotly.js-dist-min', () => {
  const mock = { react: vi.fn(), newPlot: vi.fn(), purge: vi.fn() }
  return { default: mock, ...mock }
})

const server = setupServer(
  http.post('http://localhost:8000/api/spam/predict', () =>
    HttpResponse.json({ label: 'SPAM', spam_prob: 0.99, ham_prob: 0.01, confidence: 0.99, model_used: 'rf_spam', timestamp: 'T' })
  )
)
beforeAll(() => server.listen())
afterEach(() => server.resetHandlers())
afterAll(() => server.close())

describe('SpamDetector', () => {
  it('shows validation error for short text', () => {
    render(<MemoryRouter><SpamDetector /></MemoryRouter>)
    fireEvent.click(screen.getByRole('button', { name: /analyze/i }))
    expect(screen.getByText(/at least 3 characters/i)).toBeInTheDocument()
  })
  it('shows SPAM result after prediction', async () => {
    render(<MemoryRouter><SpamDetector /></MemoryRouter>)
    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'Win a free prize now!' } })
    fireEvent.click(screen.getByRole('button', { name: 'Analyze' }))
    await waitFor(() => expect(screen.getByText('SPAM')).toBeInTheDocument())
  })
})
