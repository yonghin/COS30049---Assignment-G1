import { describe, it, expect, beforeAll, afterAll, afterEach } from 'vitest'
import { setupServer } from 'msw/node'
import { http, HttpResponse } from 'msw'
import { predictSingle } from '../spamApi'

const server = setupServer(
  http.post('http://localhost:8000/api/spam/predict', () =>
    HttpResponse.json({ detail: 'Model unavailable' }, { status: 503 })
  )
)
beforeAll(() => server.listen())
afterEach(() => server.resetHandlers())
afterAll(() => server.close())

describe('error interceptor', () => {
  it('rejects with detail message', async () => {
    await expect(predictSingle('Hello', 'rf_spam')).rejects.toThrow('Model unavailable')
  })
})
