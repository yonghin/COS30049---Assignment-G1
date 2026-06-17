import { useState, useEffect, useMemo } from 'react'
import Box from '@mui/material/Box'
import Container from '@mui/material/Container'
import Paper from '@mui/material/Paper'
import Typography from '@mui/material/Typography'
import Button from '@mui/material/Button'
import Dialog from '@mui/material/Dialog'
import DialogTitle from '@mui/material/DialogTitle'
import DialogContent from '@mui/material/DialogContent'
import DialogContentText from '@mui/material/DialogContentText'
import DialogActions from '@mui/material/DialogActions'
import PageHeader from '../components/PageHeader'
import ResultsTable from '../components/ResultsTable'
import ExportButton from '../components/ExportButton'
import DonutChart from '../components/charts/DonutChart'
import { getHistory, restoreHistory, deleteHistoryItems, subscribe } from '../utils/historyStore'
import { modelLabel } from '../constants/modelNames'
import { useToast } from '../context/ToastContext'

function History() {
  const toast = useToast()
  const [items, setItems] = useState(() => getHistory())
  const [selectedIds, setSelectedIds] = useState(new Set())
  const [showConfirm, setShowConfirm] = useState(false)

  useEffect(() => subscribe(setItems), [])

  const spamCount = items.filter((i) => i.kind === 'spam').length
  const malwareCount = items.filter((i) => i.kind === 'malware').length

  const rows = useMemo(
    () =>
      items.map((i) => ({
        _id: i.id,
        time: new Date(i.ts).toLocaleString('en-MY', {
          timeZone: 'Asia/Kuala_Lumpur',
          year: 'numeric', month: '2-digit', day: '2-digit',
          hour: '2-digit', minute: '2-digit', second: '2-digit',
          hour12: false,
        }),
        kind: i.kind,
        model: modelLabel(i.model),
        label: i.label,
        confidence: i.confidence != null ? `${(i.confidence * 100).toFixed(1)}%` : '-',
        summary: i.summary ?? '',
      })),
    [items]
  )

  const handleConfirmDelete = () => {
    const ids = [...selectedIds]
    const snapshot = [...items]
    deleteHistoryItems(ids)
    setSelectedIds(new Set())
    setShowConfirm(false)
    const label = `${ids.length} ${ids.length === 1 ? 'entry' : 'entries'} deleted`
    toast.infoWithUndo(label, () => {
      restoreHistory(snapshot)
      toast.success('History restored')
    })
  }

  // KPI card colours mapped from the old CSS variables.
  const KPIS = [
    { label: 'Total', value: items.length, color: 'secondary.main' },
    { label: 'Spam runs', value: spamCount, color: 'primary.main' },
    { label: 'Malware runs', value: malwareCount, color: 'error.main' },
  ]

  return (
    <Container maxWidth="xl" sx={{ py: 3, animation: 'fadeSlideUp 0.4s ease-out both' }}>
      <PageHeader
        title="Prediction History"
        subtitle="Every prediction you make is saved locally in your browser (localStorage). It survives reloads and never leaves this device."
      />

      {/* Delete confirmation dialog */}
      <Dialog open={showConfirm} onClose={() => setShowConfirm(false)}>
        <DialogTitle>Delete entries?</DialogTitle>
        <DialogContent>
          <DialogContentText>
            You are about to delete {selectedIds.size} prediction
            record{selectedIds.size !== 1 ? 's' : ''}. You can undo within 10 seconds.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowConfirm(false)}>Cancel</Button>
          <Button color="error" variant="contained" onClick={handleConfirmDelete}>Delete</Button>
        </DialogActions>
      </Dialog>

      {items.length === 0 ? (
        <Paper
          elevation={0}
          sx={{
            bgcolor: 'background.paper',
            border: '1px dashed',
            borderColor: 'divider',
            borderRadius: 4,
            py: 8,
            px: 3,
            textAlign: 'center',
          }}
        >
          <Box component="span" aria-hidden="true" sx={{ fontSize: 48, lineHeight: 1 }}>🗂️</Box>
          <Typography sx={{ fontSize: 18, color: 'text.primary', mt: 2 }}>No predictions yet</Typography>
          <Typography sx={{ fontSize: 13, color: 'text.secondary', mt: 1 }}>
            Run a scan on the Spam Detector or Malware Detector and it will appear here.
          </Typography>
        </Paper>
      ) : (
        <>
          <Box
            sx={{
              display: 'grid',
              gridTemplateColumns: { xs: 'repeat(3, 1fr)', md: 'repeat(3, 1fr)' },
              gap: 2,
              mb: 2.5,
            }}
          >
            {KPIS.map((k, idx) => (
              <Paper
                key={k.label}
                elevation={0}
                sx={{
                  bgcolor: 'background.paper',
                  border: 1,
                  borderColor: 'divider',
                  borderRadius: 3,
                  p: 2.25,
                  animation: 'fadeSlideUp 0.4s ease-out both',
                  animationDelay: `${idx * 0.08}s`,
                }}
              >
                <Typography sx={{ fontSize: 11, color: 'text.secondary', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                  {k.label}
                </Typography>
                <Typography sx={{ fontSize: 30, fontWeight: 700, color: k.color, mt: 0.5 }}>
                  {k.value}
                </Typography>
              </Paper>
            ))}
          </Box>

          <Box
            sx={{
              display: 'grid',
              gridTemplateColumns: { xs: '1fr', md: '1fr 2fr' },
              gap: 2,
              mb: 2.5,
              alignItems: 'stretch',
            }}
          >
            <Paper elevation={0} sx={{ bgcolor: 'background.paper', border: 1, borderColor: 'divider', borderRadius: 3, p: 2.5, minWidth: 0 }}>
              <DonutChart
                labels={['Spam', 'Malware']}
                values={[spamCount, malwareCount]}
                colors={['#00d4ff', '#ff4d4d']}
                title="Activity by type"
              />
            </Paper>
            <Paper elevation={0} sx={{ bgcolor: 'background.paper', border: 1, borderColor: 'divider', borderRadius: 3, p: 2.5, minWidth: 0 }}>
              <ResultsTable
                columns={['time', 'kind', 'model', 'label', 'confidence', 'summary']}
                rows={rows}
                filterColumn="kind"
                pageSize={10}
                selectable
                selectedKeys={selectedIds}
                keyField="_id"
                onSelectionChange={setSelectedIds}
              />
            </Paper>
          </Box>

          <Box sx={{ display: 'flex', justifyContent: 'flex-end', gap: 1.5, flexWrap: 'wrap' }}>
            <ExportButton data={rows} filename="prediction_history.csv" />
            <Button
              variant="outlined"
              color="error"
              onClick={() => setShowConfirm(true)}
              disabled={selectedIds.size === 0}
              sx={{ textTransform: 'none', fontWeight: 600 }}
            >
              {selectedIds.size > 0 ? `Delete selected (${selectedIds.size})` : 'Delete selected'}
            </Button>
          </Box>
        </>
      )}
    </Container>
  )
}

export default History