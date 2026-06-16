import { useState, useMemo, useEffect } from 'react'
import Box from '@mui/material/Box'
import Paper from '@mui/material/Paper'
import Typography from '@mui/material/Typography'
import TextField from '@mui/material/TextField'
import MenuItem from '@mui/material/MenuItem'
import Checkbox from '@mui/material/Checkbox'
import Pagination from '@mui/material/Pagination'
import TableContainer from '@mui/material/TableContainer'
import Table from '@mui/material/Table'
import TableHead from '@mui/material/TableHead'
import TableBody from '@mui/material/TableBody'
import TableRow from '@mui/material/TableRow'
import TableCell from '@mui/material/TableCell'
import TableSortLabel from '@mui/material/TableSortLabel'

function formatCell(value) {
  if (typeof value === 'boolean') return value ? '✓' : '✗'
  if (typeof value === 'number') return Number.isInteger(value) ? value : value.toFixed(4)
  return value
}

// Backward-compatible: existing callers pass { columns, rows }. Optional props:
//   searchable (default true)  - text box filtering across all cells
//   sortable   (default true)  - click headers to sort asc/desc
//   filterColumn               - column key to expose a category dropdown for
//   pageSize                   - number of rows per page (omit = no pagination)
function ResultsTable({
  columns,
  rows,
  searchable = true,
  sortable = true,
  filterColumn,
  pageSize,
  selectable = false,
  selectedKeys,
  keyField = '_id',
  onSelectionChange,
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

  // Reset to page 1 when filters/search change.
  useEffect(() => { setPage(1) }, [query, filterValue, sort])

  // Selection helpers (only active when selectable=true).
  const allFilteredKeys = useMemo(
    () => (selectable ? filtered.map((r) => r[keyField]).filter(Boolean) : []),
    [selectable, filtered, keyField]
  )
  const allSelected = selectable && allFilteredKeys.length > 0 && allFilteredKeys.every((k) => selectedKeys?.has(k))
  const someSelected = selectable && !allSelected && allFilteredKeys.some((k) => selectedKeys?.has(k))

  const toggleAll = () => {
    if (!onSelectionChange) return
    const next = new Set(selectedKeys)
    if (allSelected) allFilteredKeys.forEach((k) => next.delete(k))
    else allFilteredKeys.forEach((k) => next.add(k))
    onSelectionChange(next)
  }

  const toggleRow = (key) => {
    if (!onSelectionChange) return
    const next = new Set(selectedKeys)
    if (next.has(key)) next.delete(key)
    else next.add(key)
    onSelectionChange(next)
  }

  const totalPages = pageSize ? Math.max(1, Math.ceil(filtered.length / pageSize)) : 1
  const view = pageSize ? filtered.slice((page - 1) * pageSize, page * pageSize) : filtered

  if (!rows || rows.length === 0) {
    return (
      <Typography sx={{ color: 'text.secondary', textAlign: 'center', py: 3, fontSize: 13 }}>
        No data to display.
      </Typography>
    )
  }

  const toggleSort = (key) => {
    if (!sortable) return
    setSort((s) => (s.key === key ? { key, dir: s.dir === 'asc' ? 'desc' : 'asc' } : { key, dir: 'asc' }))
  }

  const showToolbar = searchable || (filterColumn && filterOptions.length > 0)

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.25 }}>
      {showToolbar && (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.25, flexWrap: 'wrap' }}>
          {searchable && (
            <TextField
              size="small"
              placeholder="Search..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              aria-label="Search table"
              sx={{ flex: 1, minWidth: 160, maxWidth: 280 }}
            />
          )}
          {filterColumn && filterOptions.length > 0 && (
            <TextField
              select
              size="small"
              value={filterValue}
              onChange={(e) => setFilterValue(e.target.value)}
              aria-label={`Filter by ${filterColumn}`}
              sx={{ minWidth: 140 }}
            >
              <MenuItem value="all">All {filterColumn}</MenuItem>
              {filterOptions.map((o) => (
                <MenuItem key={String(o)} value={String(o)}>{String(o)}</MenuItem>
              ))}
            </TextField>
          )}
          <Typography sx={{ color: 'text.secondary', fontSize: 12, ml: 'auto' }}>
            {filtered.length} of {rows.length}
          </Typography>
        </Box>
      )}

      <TableContainer component={Paper} elevation={0} sx={{ border: 1, borderColor: 'divider', borderRadius: 2 }}>
        <Table size="small">
          <TableHead sx={{ bgcolor: (theme) => (theme.palette.mode === 'dark' ? '#252a3e' : '#cfe0f5') }}>
            <TableRow>
              {selectable && (
                <TableCell padding="checkbox" sx={{ bgcolor: 'inherit' }}>
                  <Checkbox
                    size="small"
                    checked={allSelected}
                    indeterminate={someSelected}
                    onChange={toggleAll}
                    inputProps={{ 'aria-label': 'Select all' }}
                  />
                </TableCell>
              )}
              {cols.map((c) => (
                <TableCell
                  key={c}
                  sx={{
                    bgcolor: 'inherit',
                    px: 1,
                    color: (theme) => (theme.palette.mode === 'dark' ? '#9ca3af' : '#3e5575'),
                    textTransform: 'uppercase',
                    fontSize: 11,
                    fontWeight: 700,
                    letterSpacing: '0.06em',
                    whiteSpace: 'nowrap',
                  }}
                  sortDirection={sort.key === c ? sort.dir : false}
                >
                  {sortable ? (
                    <TableSortLabel
                      active={sort.key === c}
                      direction={sort.key === c ? sort.dir : 'asc'}
                      onClick={() => toggleSort(c)}
                    >
                      {c}
                    </TableSortLabel>
                  ) : (
                    c
                  )}
                </TableCell>
              ))}
            </TableRow>
          </TableHead>
          <TableBody>
            {view.map((row, i) => {
              const rowKey = row[keyField]
              const isChecked = selectable && selectedKeys?.has(rowKey)
              return (
                <TableRow
                  key={i}
                  hover
                  selected={!!isChecked}
                  sx={{
                    '&:nth-of-type(odd)': {
                      bgcolor: (theme) =>
                        theme.palette.mode === 'dark'
                          ? 'rgba(255,255,255,0.025)'
                          : 'rgba(0,0,0,0.018)',
                    },
                    '& td': {
                      borderBottom: 1,
                      borderColor: 'divider',
                    },
                  }}
                >
                  {selectable && (
                    <TableCell padding="checkbox">
                      <Checkbox
                        size="small"
                        checked={!!isChecked}
                        onChange={() => toggleRow(rowKey)}
                        inputProps={{ 'aria-label': 'Select row' }}
                      />
                    </TableCell>
                  )}
                  {cols.map((c) => (
                    <TableCell key={c} sx={{ px: 1, whiteSpace: 'nowrap' }}>
                      {formatCell(row[c])}
                    </TableCell>
                  ))}
                </TableRow>
              )
            })}
          </TableBody>
        </Table>
        {view.length === 0 && (
          <Typography sx={{ color: 'text.secondary', textAlign: 'center', py: 3, fontSize: 13 }}>
            No rows match your filters.
          </Typography>
        )}
      </TableContainer>

      {pageSize && totalPages > 1 && (
        <Box sx={{ display: 'flex', justifyContent: 'center', mt: 0.5 }}>
          <Pagination
            count={totalPages}
            page={page}
            onChange={(e, p) => setPage(p)}
            size="small"
            color="primary"
          />
        </Box>
      )}
    </Box>
  )
}

export default ResultsTable