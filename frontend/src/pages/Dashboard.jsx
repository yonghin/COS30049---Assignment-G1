import { useState, useEffect, useRef, useMemo } from 'react'
import { Link } from 'react-router-dom'
import ErrorBanner from '../components/ErrorBanner'
import ProgressIndicator from '../components/ProgressIndicator'
import ResultsTable from '../components/ResultsTable'
import ExportButton from '../components/ExportButton'
import BarChart from '../components/charts/BarChart'
import LineChart from '../components/charts/LineChart'
import RadarChart from '../components/charts/RadarChart'
import DonutChart from '../components/charts/DonutChart'
import { getModels, getHistory } from '../api/historyApi'
import { modelLabel } from '../constants/modelNames'
import { useCountUp } from '../hooks/useCountUp'
import styles from './Dashboard.module.css'

const FEATURES = [
  { to: '/spam', icon: '🛡️', title: 'Spam Detector', desc: 'Classify messages as spam or ham with three trained models.' },
  { to: '/malware', icon: '🐞', title: 'Malware Detector', desc: 'Score CSV feature sets and surface anomalous samples.' },
  { to: '/analytics', icon: '📊', title: 'Model Analytics', desc: 'Inspect confusion matrices, ROC curves and feature importance.' },
]

// Plain-language description of each model (no About page, so it lives here).
const MODEL_INFO = {
  rf_spam:                  { category: 'Spam classifier',   detects: 'Spam via engineered text features',       color: '#00d4ff' },
  nb_spam:                  { category: 'Spam classifier',   detects: 'Spam via TF-IDF token probabilities',     color: '#00cc88' },
  lr_spam:                  { category: 'Spam classifier',   detects: 'Spam via TF-IDF linear weights',          color: '#ffb347' },
  logistic_regression_spam: { category: 'Spam classifier',   detects: 'Spam via TF-IDF linear weights',          color: '#ffb347' },
  svm_malware:              { category: 'Malware classifier', detects: 'Malware vs benign memory samples',        color: '#ff4d4d' },
  kmeans_malware:           { category: 'Malware clustering', detects: 'Clusters malware samples',               color: '#6c63ff' },
  dbscan_malware:           { category: 'Anomaly detection',  detects: 'Flags anomalous outlier samples',        color: '#ffb347' },
}

const RADAR_METRICS = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
const DONUT_LABELS = ['Spam', 'Malware']
const DONUT_COLORS = ['#00d4ff', '#ff4d4d']

// Single animated counter — separate component so each can own a useCountUp hook.
function KpiCounter({ label, value, accent }) {
  const display = useCountUp(value)
  return (
    <div className={styles.kpiCard}>
      <div className={styles.kpiValue} style={accent ? { color: accent } : undefined}>
        {display.toLocaleString()}
      </div>
      <div className={styles.kpiLabel}>{label}</div>
    </div>
  )
}

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

  const sumSpam = spamSeries.reduce((acc, p) => acc + (p.count ?? 0), 0)
  const sumMalware = malwareSeries.reduce((acc, p) => acc + (p.count ?? 0), 0)

  const fmtMYT = (ts) => new Date(ts).toLocaleString('en-MY', { timeZone: 'Asia/Kuala_Lumpur' })

  const activityRows = [
    ...spamSeries.map((p) => ({ timestamp: fmtMYT(p.timestamp), task: 'spam', count: p.count })),
    ...malwareSeries.map((p) => ({ timestamp: fmtMYT(p.timestamp), task: 'malware', count: p.count })),
  ]
    .sort((a, b) => (a.timestamp < b.timestamp ? 1 : -1))
    .slice(0, 10)

  // Memoized chart props — prevent charts from re-rendering on every 5-second poll tick.
  const radarSeries = useMemo(
    () => models
      .filter((m) => m.accuracy != null && m.f1 != null && m.auc != null)
      .map((m) => ({
        name: modelLabel(m.name),
        values: [
          m.accuracy,
          m.precision != null ? m.precision : m.accuracy,
          m.recall != null ? m.recall : m.accuracy,
          m.f1,
          m.auc,
        ],
      })),
    [models]
  )
  const barLabels   = useMemo(() => models.map((m) => modelLabel(m.name)), [models])
  const barAccuracy = useMemo(() => models.map((m) => m.accuracy), [models])
  const barF1       = useMemo(() => models.map((m) => m.f1), [models])
  const barAuc      = useMemo(() => models.map((m) => m.auc), [models])
  const donutValues = useMemo(() => [sumSpam, sumMalware], [sumSpam, sumMalware])

  return (
    <div className={styles.page}>
      <section className={styles.hero}>
        <div className={styles.heroInner}>
          <h1 className={styles.heroTitle}>NTCyber AI</h1>
          <p className={styles.heroTagline}>Protect. Detect. Analyze.</p>
          <p className={styles.heroBlurb}>
            A machine-learning platform for spam classification, malware screening and model
            analytics — all in one place.
          </p>
          <div className={styles.heroActions}>
            {FEATURES.map((f) => (
              <Link key={f.to} to={f.to} className={styles.heroBtn}>
                <span aria-hidden="true">{f.icon}</span> {f.title}
              </Link>
            ))}
          </div>
        </div>
        <span className={styles.scrollCue} aria-hidden="true">⌄</span>
      </section>

      <div className={styles.body}>
        <ErrorBanner message={error} onDismiss={() => setError(null)} />

        <div className={styles.kpiGrid}>
          <KpiCounter label="Total predictions" value={sumSpam + sumMalware} accent="var(--purple)" />
          <KpiCounter label="Spam predictions" value={sumSpam} accent="var(--accent)" />
          <KpiCounter label="Malware predictions" value={sumMalware} accent="var(--danger)" />
          <KpiCounter label="Models loaded" value={models.length} accent="var(--success)" />
        </div>

        <ProgressIndicator visible={loading} label="Loading models..." />

        <h3 className={styles.sectionTitle}>Models</h3>
        <div className={styles.modelGrid}>
          {models.map((m) => {
            const info = MODEL_INFO[m.name] ?? { category: m.task, detects: m.task }
            const cardColor = info.color ?? 'var(--accent)'
            return (
              <div key={m.name} className={styles.modelCard} style={{ '--card-accent': cardColor }}>
                <div className={styles.modelHead}>
                  <span className={styles.modelName}>{modelLabel(m.name)}</span>
                  <span className={styles.modelAcc}>{(m.accuracy * 100).toFixed(1)}%</span>
                </div>
                <div className={styles.modelType}>{info.category}</div>
                <div className={styles.modelDesc}>{info.detects}</div>
              </div>
            )
          })}
        </div>

        <div className={styles.chartsRow}>
          <div className={styles.card}>
            <RadarChart
              metrics={RADAR_METRICS}
              series={radarSeries}
              title="Model Comparison"
              rangeMin={0.95}
            />
          </div>
          <div className={styles.card}>
            <DonutChart
              labels={DONUT_LABELS}
              values={donutValues}
              colors={DONUT_COLORS}
              title="Predictions by Type"
            />
          </div>
        </div>

        <div className={styles.chartsRow}>
          <div className={styles.card}>
            <BarChart
              models={barLabels}
              accuracy={barAccuracy}
              f1={barF1}
              auc={barAuc}
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
            filterColumn="task"
          />
          <div className={styles.exportRow}>
            <ExportButton data={activityRows} filename="recent_activity.csv" />
          </div>
        </div>
      </div>
    </div>
  )
}

export default Dashboard
