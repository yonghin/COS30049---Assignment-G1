import Box from '@mui/material/Box'
import CircularProgress from '@mui/material/CircularProgress'
import Typography from '@mui/material/Typography'

function ProgressIndicator({ visible = false, label }) {
  if (!visible) return null
  return (
    <Box
      role="status"
      aria-live="polite"
      sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 1, py: 2.5 }}
    >
      <CircularProgress size={32} thickness={4} />
      {label && <Typography sx={{ fontSize: 13, color: 'text.secondary' }}>{label}</Typography>}
    </Box>
  )
}

export default ProgressIndicator