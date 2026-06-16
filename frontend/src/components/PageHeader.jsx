import { Link as RouterLink } from 'react-router-dom'
import Box from '@mui/material/Box'
import Typography from '@mui/material/Typography'
import Breadcrumbs from '@mui/material/Breadcrumbs'
import Link from '@mui/material/Link'

// Consistent page title + subtitle + breadcrumb trail across pages.
// crumbs: optional [{ label, to }]; the current page (title) is appended automatically.
function PageHeader({ title, subtitle, crumbs = [{ label: 'Home', to: '/dashboard' }] }) {
  return (
    <Box component="header" sx={{ mb: 3, animation: 'fadeSlideUp 0.35s ease-out' }}>
      <Breadcrumbs separator="/" sx={{ mb: 1, fontSize: 13 }}>
        {crumbs.map((c) => (
          <Link
            key={c.to}
            component={RouterLink}
            to={c.to}
            underline="hover"
            sx={{ color: 'text.secondary' }}
          >
            {c.label}
          </Link>
        ))}
        <Typography sx={{ color: 'text.primary', fontSize: 13, fontWeight: 600 }}>{title}</Typography>
      </Breadcrumbs>
      <Typography sx={{ fontSize: 24, fontWeight: 700, color: 'text.primary', letterSpacing: '-0.01em' }}>
        {title}
      </Typography>
      {subtitle && (
        <Typography sx={{ fontSize: 13, color: 'text.secondary', mt: 0.5, maxWidth: 720, lineHeight: 1.6 }}>
          {subtitle}
        </Typography>
      )}
    </Box>
  )
}

export default PageHeader