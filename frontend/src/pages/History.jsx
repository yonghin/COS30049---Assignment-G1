import { useState, useEffect, useMemo } from 'react'
import PageHeader from '../components/PageHeader'
import ResultsTable from '../components/ResultsTable'
import ExportButton from '../components/ExportButton'
import DonutChart from '../components/charts/DonutChart'
import { getHistory, restoreHistory, deleteHistoryItems, subscribe } from '../utils/historyStore'
import { modelLabel } from '../constants/modelNames'
import { useToast } from '../context/ToastContext'
import styles from './History.module.css'

function History() {
  const toast = useToast()
  const [items, setItems]         = useState(() => getHistory())
  const [selectedIds, setSelectedIds] = useState(new Set())
  const [showConfirm, setShowConfirm] = useState(false)

  useEffect(() => subscribe(setItems), [])

  const spamCount    = items.filter((i) => i.kind === 'spam').length
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
        confidence: i.confidence != null ? `${(i.confidence * 100).toFixed(1)}%` : '—',
        summary: i.summary ?? '',
      })),
    [items]
  )

  const handleConfirmDelete = () => {
    const ids      = [...selectedIds]
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

  return (
    <div className={styles.page}>
      <PageHeader
        title="Prediction History"
        subtitle="Every prediction you make is saved locally in your browser (localStorage). It survives reloads and never leaves this device."
      />

      {/* Confirmation modal */}
      {showConfirm && (
        <div className={styles.modalOverlay} onClick={() => setShowConfirm(false)}>
          <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
            <h3 className={styles.modalTitle}>Delete entries?</h3>
            <p className={styles.modalBody}>
              You're about to delete <strong>{selectedIds.size}</strong> prediction
              record{selectedIds.size !== 1 ? 's' : ''}. You can undo within 10 seconds.
            </p>
            <div className={styles.modalActions}>
              <button className={styles.modalCancel} onClick={() => setShowConfirm(false)}>
                Cancel
              </button>
              <button className={styles.modalConfirm} onClick={handleConfirmDelete}>
                Delete
              </button>
            </div>
          </div>
        </div>
      )}

      {items.length === 0 ? (
        <div className={styles.emptyState}>
          <span className={styles.emptyIcon} aria-hidden="true">🗂️</span>
          <h2 className={styles.emptyTitle}>No predictions yet</h2>
          <p className={styles.emptyText}>
            Run a scan on the Spam Detector or Malware Detector and it will appear here.
          </p>
        </div>
      ) : (
        <>
          <div className={styles.kpiGrid}>
            <div className={styles.kpiCard}>
              <div className={styles.kpiLabel}>Total</div>
              <div className={`${styles.kpiValue} ${styles.total}`}>{items.length}</div>
            </div>
            <div className={styles.kpiCard}>
              <div className={styles.kpiLabel}>Spam runs</div>
              <div className={`${styles.kpiValue} ${styles.spam}`}>{spamCount}</div>
            </div>
            <div className={styles.kpiCard}>
              <div className={styles.kpiLabel}>Malware runs</div>
              <div className={`${styles.kpiValue} ${styles.malware}`}>{malwareCount}</div>
            </div>
          </div>

          <div className={styles.row}>
            <div className={styles.chartCard}>
              <DonutChart
                labels={['Spam', 'Malware']}
                values={[spamCount, malwareCount]}
                colors={['#00d4ff', '#ff4d4d']}
                title="Activity by type"
              />
            </div>
            <div className={styles.tableCard}>
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
            </div>
          </div>

          <div className={styles.actions}>
            <ExportButton data={rows} filename="prediction_history.csv" />
            <button
              className={styles.deleteSelected}
              onClick={() => setShowConfirm(true)}
              disabled={selectedIds.size === 0}
            >
              {selectedIds.size > 0
                ? `Delete selected (${selectedIds.size})`
                : 'Delete selected'}
            </button>
          </div>
        </>
      )}
    </div>
  )
}

export default History
