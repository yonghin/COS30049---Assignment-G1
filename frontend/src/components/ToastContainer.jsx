import Snackbar from '@mui/material/Snackbar'
import Alert from '@mui/material/Alert'
import Button from '@mui/material/Button'
import Slide from '@mui/material/Slide'

// Toasts are stacked at the top-right. Each maps its type to an Alert severity.
const SEVERITY = { success: 'success', error: 'error', info: 'info' }

// Slide in/out from the right, matching the original toast animation.
function SlideLeft(props) {
  return <Slide {...props} direction="left" />
}

function ToastContainer({ toasts = [], onDismiss, onDismissNow }) {
  if (toasts.length === 0) return null
  return (
    <Snackbar
      open
      anchorOrigin={{ vertical: 'top', horizontal: 'right' }}
      TransitionComponent={SlideLeft}
      sx={{
        top: { xs: 80, sm: 80 },
        right: { xs: 16, sm: 20 },
        left: 'auto',
      }}
    >
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10, width: '100%', maxWidth: 380 }}>
        {toasts.map((t) => (
          <Slide key={t.id} direction="left" in={!t.leaving} mountOnEnter unmountOnExit>
            <Alert
              severity={SEVERITY[t.type] ?? 'info'}
              variant="outlined"
              onClose={() => onDismiss(t.id)}
              sx={{ bgcolor: 'background.paper', boxShadow: 3, alignItems: 'center', width: '100%' }}
              action={
                t.onUndo ? (
                  <Button
                    color="inherit"
                    size="small"
                    onClick={() => {
                      // Run the undo callback, then remove THIS toast instantly
                      // so the follow-up toast (e.g. History restored) appears cleanly.
                      t.onUndo()
                      onDismissNow(t.id)
                    }}
                    sx={{ fontWeight: 700, textTransform: 'none', whiteSpace: 'nowrap' }}
                  >
                    Undo{t.secondsLeft != null ? ` (${t.secondsLeft}s)` : ''}
                  </Button>
                ) : undefined
              }
            >
              {t.message}
            </Alert>
          </Slide>
        ))}
      </div>
    </Snackbar>
  )
}

export default ToastContainer