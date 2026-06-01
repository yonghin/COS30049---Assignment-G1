import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { it, expect, vi } from 'vitest'
import FileUploadWidget from '../FileUploadWidget'

it('calls onFileSelected for valid extension', async () => {
  const fn = vi.fn()
  render(<FileUploadWidget accept=".csv" label="Upload CSV" onFileSelected={fn} />)
  const file = new File(['a,b\n1,2'], 'data.csv', { type: 'text/csv' })
  await userEvent.upload(screen.getByLabelText(/Upload CSV/i), file)
  expect(fn).toHaveBeenCalledWith(file)
})
