import { useRef, useState } from 'react'
import Box from '@mui/material/Box'
import Paper from '@mui/material/Paper'
import Typography from '@mui/material/Typography'
import CloudUploadIcon from '@mui/icons-material/CloudUpload'

function FileUploadWidget({ accept = '', label = 'Upload File', onFileSelected }) {
  const inputRef = useRef(null)
  const [dragOver, setDragOver] = useState(false)
  const [fileName, setFileName] = useState(null)

  const allowed = accept
    .split(',')
    .map((s) => s.trim().toLowerCase())
    .filter(Boolean)

  const isValid = (file) => {
    if (allowed.length === 0) return true
    const name = file.name.toLowerCase()
    return allowed.some((ext) => name.endsWith(ext))
  }

  const handleFile = (file) => {
    if (!file) return
    if (isValid(file)) {
      setFileName(file.name)
      onFileSelected?.(file)
    }
  }

  const handleChange = (e) => handleFile(e.target.files?.[0])

  const handleDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    handleFile(e.dataTransfer.files?.[0])
  }

  return (
    <Paper
      component="label"
      elevation={0}
      onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 1,
        p: 4,
        cursor: 'pointer',
        textAlign: 'center',
        border: '2px dashed',
        borderColor: dragOver ? 'primary.main' : 'divider',
        borderRadius: 3,
        bgcolor: dragOver ? 'action.hover' : 'background.paper',
        transition: 'border-color 0.2s, background-color 0.2s',
        '&:hover': { borderColor: 'primary.main' },
      }}
    >
      <CloudUploadIcon sx={{ fontSize: 32, color: 'text.secondary' }} />
      <Typography sx={{ fontWeight: 600, color: 'text.primary' }}>{label}</Typography>
      <Typography sx={{ fontSize: 12, color: 'text.secondary' }}>
        {fileName ? fileName : `Accepted: ${accept || 'any'}`}
      </Typography>
      <Box
        component="input"
        ref={inputRef}
        type="file"
        accept={accept}
        onChange={handleChange}
        sx={{ display: 'none' }}
      />
    </Paper>
  )
}

export default FileUploadWidget