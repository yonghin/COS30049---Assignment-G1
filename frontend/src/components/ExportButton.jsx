import Button from '@mui/material/Button'
import FileDownloadIcon from '@mui/icons-material/FileDownload'

export function objectsToCsv(rows) {
  if (!rows || rows.length === 0) return ''
  const headers = Object.keys(rows[0])
  const escape = (val) => {
    const s = val === null || val === undefined ? '' : String(val)
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s
  }
  const lines = [headers.join(',')]
  for (const row of rows) {
    lines.push(headers.map((h) => escape(row[h])).join(','))
  }
  return lines.join('\n')
}

function ExportButton({ data = [], filename = 'export.csv', label = 'Export CSV' }) {
  const handleClick = () => {
    const csv = objectsToCsv(data)
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  return (
    <Button
      variant="outlined"
      startIcon={<FileDownloadIcon />}
      onClick={handleClick}
      disabled={!data || data.length === 0}
      sx={{ textTransform: 'none', fontWeight: 600 }}
    >
      {label}
    </Button>
  )
}

export default ExportButton