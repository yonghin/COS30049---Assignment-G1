import { useState, useEffect, useMemo } from 'react'
import PageHeader from '../components/PageHeader'
import ResultsTable from '../components/ResultsTable'
import ExportButton from '../components/ExportButton'
import DonutChart from '../components/charts/DonutChart'
import { getHistory, clearHistory, subscribe } from '../utils/historyStore'
import { modelLabel } from '../constants/modelNames'
import { useToast } from '../context/ToastContext'
import styles from './History.module.css'

function History() {
  const toast = useToast()
  const [items, setItems] = useState(() => getHistory())

  // Keep in sync with predictions made on other pages (same tab).
  useEffect(() => subscribe(setItems), [])

  const spamCount = items.filter((i) => i.kind === 'spam').length
  const malwareCount = items.filter((i) => i.kind === 'malware').length

  const rows = useMemo(
    () =>
      items.map((i) => ({
        time: new Date(i.ts).toLocaleString(),
        kind: i.kind,
        model: modelLabel(i.model),
        label: i.label,
        confidence: i.confidence != null ? `${(i.confidence * 100).toFixed(1)}%` : '—',
        summary: i.summary ?? '',
      })),
    [items]
  )

  const handleClear = () => {
    clearHistory()
    toast.info('History cleared')
  }

  return (
    <div className={styles.page}>
      <PageHeader
        title="Prediction History"
        subtitle="Every prediction you make is saved locally in your browser (localStorage). It survives reloads and never leaves this device."
      />

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
              <div className={styles.kpiValue}>{items.length}</div>
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
              />
            </div>
          </div>

          <div className={styles.actions}>
            <ExportButton data={rows} filename="prediction_history.csv" />
            <button className={styles.clear} onClick={handleClear}>Clear history</button>
          </div>
        </>
      )}
    </div>
  )
}

export default History
