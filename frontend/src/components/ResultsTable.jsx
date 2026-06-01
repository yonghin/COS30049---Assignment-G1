import styles from './ResultsTable.module.css'

function formatCell(value) {
  if (typeof value === 'boolean') return value ? '✓' : '✗'
  if (typeof value === 'number') return Number.isInteger(value) ? value : value.toFixed(4)
  return value
}

function ResultsTable({ columns, rows }) {
  // Derive columns from first row if not explicitly provided
  const cols = columns ?? (rows && rows.length > 0 ? Object.keys(rows[0]) : [])

  if (!rows || rows.length === 0) {
    return <div className={styles.empty}>No data to display.</div>
  }

  return (
    <div className={styles.container}>
      <table className={styles.table}>
        <thead>
          <tr>
            {cols.map((c) => (
              <th key={c}>{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i}>
              {cols.map((c) => (
                <td key={c}>{formatCell(row[c])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default ResultsTable
