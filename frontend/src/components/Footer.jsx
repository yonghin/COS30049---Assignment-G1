import { Link as RouterLink } from 'react-router-dom'
import Box from '@mui/material/Box'
import Container from '@mui/material/Container'
import Typography from '@mui/material/Typography'
import Link from '@mui/material/Link'

const LINKS = [
  { to: '/dashboard', label: 'Dashboard' },
  { to: '/spam', label: 'Spam Detector' },
  { to: '/malware', label: 'Malware Detector' },
  { to: '/analytics', label: 'Model Analytics' },
  { to: '/history', label: 'History' },
]

function Footer() {
  return (
    <Box
      component="footer"
      sx={{ bgcolor: 'background.default', borderTop: 1, borderColor: 'divider', mt: 6 }}
    >
      <Container
        maxWidth="xl"
        sx={{
          py: 3,
          display: 'flex',
          flexWrap: 'wrap',
          gap: 2,
          alignItems: 'center',
          justifyContent: 'space-between',
        }}
      >
        <Box sx={{ display: 'flex', flexDirection: 'column' }}>
          <Typography sx={{ fontWeight: 700, color: 'text.primary' }}>NTCyber AI</Typography>
          <Typography sx={{ fontSize: 12, color: 'text.secondary' }}>Protect. Detect. Analyze.</Typography>
        </Box>

        <Box component="nav" aria-label="Footer" sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
          {LINKS.map((l) => (
            <Link
              key={l.to}
              component={RouterLink}
              to={l.to}
              underline="hover"
              sx={{ fontSize: 14, color: 'text.secondary', '&:hover': { color: 'primary.main' } }}
            >
              {l.label}
            </Link>
          ))}
        </Box>

        <Typography sx={{ fontSize: 13, color: 'text.secondary' }}>COS30049, Group 1</Typography>
      </Container>
    </Box>
  )
}

export default Footer