import Collapse from '@mui/material/Collapse'
import Alert from '@mui/material/Alert'

function ErrorBanner({ message, onDismiss }) {
  const open = message !== null && message !== undefined
  return (
    <Collapse in={open} sx={{ mb: open ? 2 : 0 }}>
      <Alert severity="error" onClose={onDismiss}>
        {message}
      </Alert>
    </Collapse>
  )
}

export default ErrorBanner