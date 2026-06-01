import { useRef, useState } from 'react'
import styles from './FileUploadWidget.module.css'

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
    <label
      className={dragOver ? `${styles.dropzone} ${styles.dragover}` : styles.dropzone}
      onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
    >
      <div className={styles.icon}>⬆</div>
      <div className={styles.label}>{label}</div>
      <div className={styles.hint}>{fileName ? fileName : `Accepted: ${accept || 'any'}`}</div>
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        className={styles.hiddenInput}
        onChange={handleChange}
      />
    </label>
  )
}

export default FileUploadWidget
