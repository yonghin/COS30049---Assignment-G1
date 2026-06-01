import { describe, it, expect, beforeAll, afterAll, afterEach } from 'vitest'
import { setupServer } from 'msw/node'
import { http, HttpResponse } from 'msw'
import { predictSingle } from '../spamApi'

const server = setupServer(
  http.post('http://localhost:8000/api/spam/predict', () =>
    HttpResponse.json({ label: 'SPAM', spam_prob: 0.99, ham_prob: 0.01, confidence: 0.99, model_used: 'rf_spam', timestamp: 'T' })
  )
)
beforeAll(() => server.listen())
afterEach(() => server.resetHandlers())
afterAll(() => server.close())

describe('spamApi', () => {
  it('predictSingle returns label', async () => {
    const result = await predictSingle('Hello', 'rf_spam')
    expect(result.label).toBe('SPAM')
  })
})
