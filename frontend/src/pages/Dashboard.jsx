import { useState, useEffect, useRef } from 'react'
import NavBar from '../components/NavBar'
import ErrorBanner from '../components/ErrorBanner'
import ProgressIndicator from '../components/ProgressIndicator'
import ResultsTable from '../components/ResultsTable'
import ExportButton from '../components/ExportButton'
import BarChart from '../components/charts/BarChart'
import LineChart from '../components/charts/LineChart'
import { getModels, getHistory } from '../api/historyApi'
import styles from './Dashboard.module.css'

function Dashboard() {
  const [models, setModels] = useState([])
  const [spamSeries, setSpamSeries] = useState([])
  const [malwareSeries, setMalwareSeries] = useState([])
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(true)
  const intervalRef = useRef(null)

  useEffect(() => {
    let mounted = true

    const loadModels = async () => {
      try {
        const data = await getModels()
        if (mounted) setModels(data.models ?? [])
      } catch (e) {
        if (mounted) setError(e.message)
      } finally {
        if (mounted) setLoading(false)
      }
    }

    const loadHistory = async () => {
      try {
        const data = await getHistory()
        if (mounted) {
          setSpamSeries(data.spam_series ?? [])
          setMalwareSeries(data.malware_series ?? [])
        }
      } catch (e) {
        if (mounted) setError(e.message)
      }
    }

    loadModels()
    loadHistory()
    intervalRef.current = setInterval(loadHistory, 5000)

    return () => {
      mounted = false
      if (intervalRef.current) clearInterval(intervalRef.current)
    }
  }, [])

  const activityRows = [
    ...spamSeries.map((p) => ({ timestamp: p.timestamp, task: 'spam', count: p.count })),
    ...malwareSeries.map((p) => ({ timestamp: p.timestamp, task: 'malware', count: p.count })),
  ]
    .sort((a, b) => (a.timestamp < b.timestamp ? 1 : -1))
    .slice(0, 10)

  return (
    <>
      <NavBar />
      <div className={styles.page}>
        <ErrorBanner message={error} onDismiss={() => setError(null)} />

        <ProgressIndicator visible={loading} label="Loading models..." />

        <div className={styles.statsGrid}>
          {models.map((m) => (
            <div key={m.name} className={styles.card}>
              <div className={styles.statLabel}>{m.name}</div>
              <div className={styles.statValue}>{(m.accuracy * 100).toFixed(2)}%</div>
              <div className={styles.statLabel}>F1 {(m.f1 * 100).toFixed(2)}% · {m.task}</div>
            </div>
          ))}
        </div>

        <div className={styles.chartsRow}>
          <div className={styles.card}>
            <BarChart
              models={models.map((m) => m.name)}
              accuracy={models.map((m) => m.accuracy)}
              f1={models.map((m) => m.f1)}
              auc={models.map((m) => m.auc)}
              title="Model Performance"
            />
          </div>
          <div className={styles.card}>
            <LineChart spamSeries={spamSeries} malwareSeries={malwareSeries} title="Live Predictions" />
          </div>
        </div>

        <div className={styles.card}>
          <h3 className={styles.sectionTitle}>Recent Activity</h3>
          <ResultsTable
            columns={['timestamp', 'task', 'count']}
            rows={activityRows}
          />
          <div className={styles.exportRow}>
            <ExportButton data={activityRows} filename="recent_activity.csv" />
          </div>
        </div>
      </div>
    </>
  )
}

export default Dashboard
