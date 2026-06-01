import { render, screen, fireEvent } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import ErrorBanner from '../ErrorBanner'

describe('ErrorBanner', () => {
  it('renders null when message is null', () => {
    const { container } = render(<ErrorBanner message={null} onDismiss={() => {}} />)
    expect(container.firstChild).toBeNull()
  })
  it('shows message when provided', () => {
    render(<ErrorBanner message="Something went wrong" onDismiss={() => {}} />)
    expect(screen.getByText('Something went wrong')).toBeInTheDocument()
  })
  it('calls onDismiss on button click', () => {
    const fn = vi.fn()
    render(<ErrorBanner message="Error" onDismiss={fn} />)
    fireEvent.click(screen.getByRole('button'))
    expect(fn).toHaveBeenCalled()
  })
})
