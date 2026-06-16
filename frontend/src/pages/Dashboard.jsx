import { useState, useEffect, useRef, useMemo } from 'react'
import { Link as RouterLink } from 'react-router-dom'
import Box from '@mui/material/Box'
import Container from '@mui/material/Container'
import Typography from '@mui/material/Typography'
import Button from '@mui/material/Button'
import Card from '@mui/material/Card'
import Paper from '@mui/material/Paper'
import ErrorBanner from '../components/ErrorBanner'
import ProgressIndicator from '../components/ProgressIndicator'
import ResultsTable from '../components/ResultsTable'
import ExportButton from '../components/ExportButton'
import BarChart from '../components/charts/BarChart'
import LineChart from '../components/charts/LineChart'
import RadarChart from '../components/charts/RadarChart'
import DonutChart from '../components/charts/DonutChart'
import { getModels, getHistory } from '../api/historyApi'
import { getHistory as getLocalHistory, subscribe } from '../utils/historyStore'
import { modelLabel } from '../constants/modelNames'
import { useCountUp } from '../hooks/useCountUp'

const FEATURES = [
  { to: '/spam', icon: '🛡️', title: 'Spam Detector', desc: 'Classify messages as spam or ham with three trained models.' },
  { to: '/malware', icon: '🐞', title: 'Malware Detector', desc: 'Score CSV feature sets and surface anomalous samples.' },
  { to: '/analytics', icon: '📊', title: 'Model Analytics', desc: 'Inspect confusion matrices, ROC curves and feature importance.' },
]

const MODEL_INFO = {
  rf_spam:                  { category: 'Spam classifier',   detects: 'Spam via engineered text features',   color: '#00d4ff' },
  nb_spam:                  { category: 'Spam classifier',   detects: 'Spam via TF-IDF token probabilities', color: '#00cc88' },
  lr_spam:                  { category: 'Spam classifier',   detects: 'Spam via TF-IDF linear weights',       color: '#ffb347' },
  logistic_regression_spam: { category: 'Spam classifier',   detects: 'Spam via TF-IDF linear weights',       color: '#ffb347' },
  svm_malware:              { category: 'Malware classifier', detects: 'Malware vs benign memory samples',     color: '#ff4d4d' },
  kmeans_malware:           { category: 'Malware clustering', detects: 'Clusters malware samples',             color: '#6c63ff' },
  dbscan_malware:           { category: 'Anomaly detection',  detects: 'Flags anomalous outlier samples',      color: '#ffb347' },
}

const RADAR_METRICS = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
const DONUT_LABELS = ['Spam', 'Malware']
const DONUT_COLORS = ['#00d4ff', '#ff4d4d']

// Single animated KPI counter. Each owns its own useCountUp hook.
// `index` drives a staggered entry animation via animation-delay.
function KpiCounter({ label, value, accent, index = 0 }) {
  const display = useCountUp(value)
  return (
    <Paper
      elevation={0}
      sx={{
        bgcolor: 'background.paper',
        border: 1,
        borderColor: 'divider',
        borderRadius: 3,
        p: 2.5,
        textAlign: 'center',
        animation: 'fadeSlideUp 0.45s ease-out both',
        animationDelay: `${index * 0.07}s`,
      }}
    >
      <Typography sx={{ fontSize: 34, fontWeight: 800, color: accent || 'primary.main', letterSpacing: '-0.02em' }}>
        {display.toLocaleString()}
      </Typography>
      <Typography sx={{ fontSize: 11, color: 'text.secondary', textTransform: 'uppercase', letterSpacing: '0.05em', mt: 0.5 }}>
        {label}
      </Typography>
    </Paper>
  )
}

function Dashboard() {
  const [models, setModels] = useState([])
  const [spamSeries, setSpamSeries] = useState([])
  const [malwareSeries, setMalwareSeries] = useState([])
  const [localHistory, setLocalHistory] = useState(() => getLocalHistory())
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

  useEffect(() => subscribe(setLocalHistory), [])

  const sumSpam = spamSeries.reduce((acc, p) => acc + (p.count ?? 0), 0)
  const sumMalware = malwareSeries.reduce((acc, p) => acc + (p.count ?? 0), 0)

  const fmtMYT = (ts) => new Date(ts).toLocaleString('en-MY', {
    timeZone: 'Asia/Kuala_Lumpur',
    year: 'numeric', month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit', second: '2-digit',
    hour12: false,
  })

  const activityRows = localHistory
    .slice(0, 10)
    .map((i) => ({
      time:  fmtMYT(i.ts),
      task:  i.kind,
      label: i.label ?? '-',
    }))

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
    <Box>
      {/* Hero */}
      <Box
        sx={{
          position: 'relative',
          minHeight: 'calc(100vh - 60px)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          textAlign: 'center',
          px: 3,
          py: 5,
          borderBottom: 1,
          borderColor: 'divider',
          background: (theme) =>
            theme.palette.mode === 'dark'
              ? 'radial-gradient(1200px 600px at 50% -10%, rgba(0,212,255,0.14), transparent 60%), linear-gradient(135deg, rgba(0,212,255,0.08), rgba(108,99,255,0.08))'
              : 'radial-gradient(1200px 600px at 50% -10%, rgba(0,102,204,0.12), transparent 60%), linear-gradient(135deg, rgba(0,102,204,0.07), rgba(91,84,230,0.07))',
        }}
      >
        <Box sx={{ maxWidth: 760, animation: 'heroIn 0.6s ease-out' }}>
          <Typography sx={{ fontSize: { xs: 32, sm: 44, md: 64 }, fontWeight: 800, color: 'text.primary', letterSpacing: '-0.03em' }}>
            NTCyber AI
          </Typography>
          <Typography sx={{ fontSize: { xs: 16, md: 22 }, fontWeight: 600, color: 'primary.main', mt: 1.5 }}>
            Protect. Detect. Analyze.
          </Typography>
          <Typography sx={{ fontSize: { xs: 14, md: 16 }, color: 'text.secondary', mt: 2, maxWidth: 600, mx: 'auto', lineHeight: 1.7 }}>
            A machine-learning platform for spam classification, malware screening and model
            analytics, all in one place.
          </Typography>
          <Box sx={{ display: 'flex', gap: 1.5, justifyContent: 'center', flexWrap: 'wrap', mt: 3.5 }}>
            {FEATURES.map((f) => (
              <Button
                key={f.to}
                component={RouterLink}
                to={f.to}
                variant="outlined"
                sx={{
                  bgcolor: 'background.paper',
                  borderColor: 'divider',
                  borderRadius: 2.5,
                  px: 2.5,
                  py: 1.5,
                  fontWeight: 600,
                  color: 'text.primary',
                  textTransform: 'none',
                  '&:hover': { borderColor: 'primary.main', transform: 'translateY(-2px)' },
                }}
              >
                <Box component="span" aria-hidden="true" sx={{ mr: 1 }}>{f.icon}</Box> {f.title}
              </Button>
            ))}
          </Box>
        </Box>
        {/* Bouncing scroll cue at the bottom of the hero */}
        <Box
          component="span"
          aria-hidden="true"
          sx={{
            position: 'absolute',
            bottom: 24,
            left: '50%',
            transform: 'translateX(-50%)',
            fontSize: 28,
            color: 'text.secondary',
            animation: 'bounce 1.6s infinite',
          }}
        >
          ⌄
        </Box>
      </Box>

      {/* Body */}
      <Container maxWidth="xl" sx={{ py: 3 }}>
        <ErrorBanner message={error} onDismiss={() => setError(null)} />

        <Box
          sx={{
            display: 'grid',
            gridTemplateColumns: { xs: 'repeat(2, 1fr)', md: 'repeat(4, 1fr)' },
            gap: 2,
            mb: 3,
          }}
        >
          <KpiCounter index={0} label="Total predictions" value={sumSpam + sumMalware} accent="secondary.main" />
          <KpiCounter index={1} label="Spam predictions" value={sumSpam} accent="primary.main" />
          <KpiCounter index={2} label="Malware predictions" value={sumMalware} accent="error.main" />
          <KpiCounter index={3} label="Models loaded" value={models.length} accent="success.main" />
        </Box>

        <ProgressIndicator visible={loading} label="Loading models..." />

        <Typography sx={{ fontSize: 14, color: 'text.primary', mb: 1.5 }}>Models</Typography>
        <Box
          sx={{
            display: 'grid',
            gridTemplateColumns: { xs: '1fr', md: 'repeat(2, 1fr)' },
            gap: 2,
            mb: 3,
          }}
        >
          {models.map((m) => {
            const info = MODEL_INFO[m.name] ?? { category: m.task, detects: m.task }
            const cardColor = info.color ?? '#00d4ff'
            return (
              <Card
                key={m.name}
                elevation={0}
                sx={{
                  bgcolor: 'background.paper',
                  border: 1,
                  borderColor: 'divider',
                  borderLeft: '4px solid',
                  borderLeftColor: cardColor,
                  borderRadius: 3,
                  p: 2.25,
                  transition: 'transform 0.2s, box-shadow 0.2s',
                  animation: 'scaleIn 0.4s ease-out both',
                  '&:hover': { boxShadow: 4, transform: 'translateY(-3px)' },
                }}
              >
                <Box sx={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between' }}>
                  <Typography sx={{ fontSize: 15, fontWeight: 700, color: 'text.primary' }}>{modelLabel(m.name)}</Typography>
                  <Typography sx={{ fontSize: 16, fontWeight: 700, color: cardColor }}>{(m.accuracy * 100).toFixed(1)}%</Typography>
                </Box>
                <Typography sx={{ fontSize: 12, color: cardColor, mt: 0.25, fontWeight: 600 }}>{info.category}</Typography>
                <Typography sx={{ fontSize: 13, color: 'text.secondary', mt: 0.75, lineHeight: 1.5 }}>{info.detects}</Typography>
              </Card>
            )
          })}
        </Box>

        <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' }, gap: 2, mb: 3 }}>
          <ChartCard>
            <RadarChart metrics={RADAR_METRICS} series={radarSeries} title="Model Comparison" rangeMin={0.95} />
          </ChartCard>
          <ChartCard>
            <DonutChart labels={DONUT_LABELS} values={donutValues} colors={DONUT_COLORS} title="Predictions by Type" />
          </ChartCard>
        </Box>

        <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' }, gap: 2, mb: 3 }}>
          <ChartCard>
            <BarChart models={barLabels} accuracy={barAccuracy} f1={barF1} auc={barAuc} title="Model Performance" />
          </ChartCard>
          <ChartCard>
            <LineChart spamSeries={spamSeries} malwareSeries={malwareSeries} title="Live Predictions" />
          </ChartCard>
        </Box>

        <ChartCard>
          <Typography sx={{ fontSize: 14, color: 'text.primary', mb: 1.5 }}>Recent Activity</Typography>
          <ResultsTable columns={['time', 'task', 'label']} rows={activityRows} filterColumn="task" />
          <Box sx={{ mt: 2, display: 'flex', justifyContent: 'flex-end' }}>
            <ExportButton data={activityRows} filename="recent_activity.csv" />
          </Box>
        </ChartCard>
      </Container>
    </Box>
  )
}

// Reusable card wrapper for charts / tables - replaces the old .card class.
function ChartCard({ children }) {
  return (
    <Paper
      elevation={0}
      sx={{
        bgcolor: 'background.paper',
        border: 1,
        borderColor: 'divider',
        borderRadius: 3,
        p: 2.5,
        animation: 'fadeIn 0.5s ease-out both',
      }}
    >
      {children}
    </Paper>
  )
}

export default Dashboard