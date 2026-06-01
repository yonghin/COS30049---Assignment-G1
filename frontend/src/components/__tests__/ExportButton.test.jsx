import { render, screen, fireEvent } from '@testing-library/react'
import { it, expect, vi } from 'vitest'
import ExportButton from '../ExportButton'

it('creates object URL on click', () => {
  const createURL = vi.fn(() => 'blob:url')
  const revokeURL = vi.fn()
  globalThis.URL.createObjectURL = createURL
  globalThis.URL.revokeObjectURL = revokeURL
  render(<ExportButton data={[{ a: 1 }]} filename="test.csv" />)
  fireEvent.click(screen.getByRole('button'))
  expect(createURL).toHaveBeenCalled()
})
