import { useState, useMemo, useEffect } from 'react'
import styles from './ResultsTable.module.css'

function formatCell(value) {
  if (typeof value === 'boolean') return value ? '✓' : '✗'
  if (typeof value === 'number') return Number.isInteger(value) ? value : value.toFixed(4)
  return value
}

// Backward-compatible: existing callers pass { columns, rows }. New optional props:
//   searchable (default true)  — text box filtering across all cells
//   sortable   (default true)  — click headers to sort asc/desc
//   filterColumn               — column key to expose a category dropdown for
//   pageSize                   — number of rows per page (omit = no pagination)
function ResultsTable({
  columns,
  rows,
  searchable = true,
  sortable = true,
  filterColumn,
  pageSize,
}) {
  const cols = columns ?? (rows && rows.length > 0 ? Object.keys(rows[0]) : [])

  const [query, setQuery] = useState('')
  const [sort, setSort] = useState({ key: null, dir: 'asc' })
  const [filterValue, setFilterValue] = useState('all')
  const [page, setPage] = useState(1)

  const filterOptions = useMemo(() => {
    if (!filterColumn || !rows) return []
    return Array.from(new Set(rows.map((r) => r[filterColumn]))).filter((v) => v !== undefined)
  }, [filterColumn, rows])

  const filtered = useMemo(() => {
    let data = rows ?? []

    if (filterColumn && filterValue !== 'all') {
      data = data.filter((r) => String(r[filterColumn]) === filterValue)
    }

    if (searchable && query.trim()) {
      const q = query.trim().toLowerCase()
      data = data.filter((r) =>
        cols.some((c) => String(r[c] ?? '').toLowerCase().includes(q))
      )
    }

    if (sortable && sort.key) {
      const dir = sort.dir === 'asc' ? 1 : -1
      data = [...data].sort((a, b) => {
        const av = a[sort.key]
        const bv = b[sort.key]
        if (typeof av === 'number' && typeof bv === 'number') return (av - bv) * dir
        return String(av ?? '').localeCompare(String(bv ?? '')) * dir
      })
    }

    return data
  }, [rows, cols, query, sort, searchable, sortable, filterColumn, filterValue])

  // Reset to page 1 when filters/search change
  useEffect(() => { setPage(1) }, [query, filterValue, sort])

  const totalPages = pageSize ? Math.max(1, Math.ceil(filtered.length / pageSize)) : 1
  const view = pageSize ? filtered.slice((page - 1) * pageSize, page * pageSize) : filtered

  if (!rows || rows.length === 0) {
    return <div className={styles.empty}>No data to display.</div>
  }

  const toggleSort = (key) => {
    if (!sortable) return
    setSort((s) => (s.key === key ? { key, dir: s.dir === 'asc' ? 'desc' : 'asc' } : { key, dir: 'asc' }))
  }

  const showToolbar = searchable || (filterColumn && filterOptions.length > 0)

  return (
    <div className={styles.wrap}>
      {showToolbar && (
        <div className={styles.toolbar}>
          {searchable && (
            <input
              className={styles.search}
              type="text"
              placeholder="Search…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              aria-label="Search table"
            />
          )}
          {filterColumn && filterOptions.length > 0 && (
            <select
              className={styles.filter}
              value={filterValue}
              onChange={(e) => setFilterValue(e.target.value)}
              aria-label={`Filter by ${filterColumn}`}
            >
              <option value="all">All {filterColumn}</option>
              {filterOptions.map((o) => (
                <option key={String(o)} value={String(o)}>{String(o)}</option>
              ))}
            </select>
          )}
          <span className={styles.count}>{filtered.length} of {rows.length}</span>
        </div>
      )}

      <div className={styles.container}>
        <table className={styles.table}>
          <thead>
            <tr>
              {cols.map((c) => (
                <th
                  key={c}
                  onClick={() => toggleSort(c)}
                  className={sortable ? styles.sortable : undefined}
                >
                  {c}
                  {sort.key === c && <span className={styles.arrow}>{sort.dir === 'asc' ? ' ▲' : ' ▼'}</span>}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {view.map((row, i) => (
              <tr key={i}>
                {cols.map((c) => (
                  <td key={c}>{formatCell(row[c])}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
        {view.length === 0 && <div className={styles.empty}>No rows match your filters.</div>}
      </div>

      {pageSize && totalPages > 1 && (
        <div className={styles.pagination}>
          <button
            className={styles.pageBtn}
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            disabled={page === 1}
          >
            &#8249;
          </button>
          {Array.from({ length: totalPages }, (_, i) => i + 1).map((p) => (
            <button
              key={p}
              className={`${styles.pageBtn} ${p === page ? styles.pageBtnActive : ''}`}
              onClick={() => setPage(p)}
            >
              {p}
            </button>
          ))}
          <button
            className={styles.pageBtn}
            onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
            disabled={page === totalPages}
          >
            &#8250;
          </button>
        </div>
      )}
    </div>
  )
}

export default ResultsTable
