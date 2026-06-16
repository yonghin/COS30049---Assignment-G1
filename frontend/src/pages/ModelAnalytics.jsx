import { useState, useEffect } from 'react'
import Box from '@mui/material/Box'
import Container from '@mui/material/Container'
import Paper from '@mui/material/Paper'
import Typography from '@mui/material/Typography'
import Tabs from '@mui/material/Tabs'
import Tab from '@mui/material/Tab'
import PageHeader from '../components/PageHeader'
import ErrorBanner from '../components/ErrorBanner'
import ProgressIndicator from '../components/ProgressIndicator'
import Heatmap from '../components/charts/Heatmap'
import LineChart from '../components/charts/LineChart'
import BarChart from '../components/charts/BarChart'
import RadarChart from '../components/charts/RadarChart'
import { getModelAnalytics } from '../api/analyticsApi'

const METRIC_COLORS = {
  accuracy: 'primary.main',
  precision: 'secondary.main',
  recall: 'success.main',
  f1: 'warning.main',
}

// Derive headline metrics from a confusion matrix [[TN, FP], [FN, TP]].
function metricsFromConfusion(cm, auc) {
  if (!Array.isArray(cm) || cm.length < 2) return null
  const [[tn, fp], [fn, tp]] = cm
  const total = tn + fp + fn + tp
  const safe = (num, den) => (den > 0 ? num / den : 0)
  const accuracy = safe(tn + tp, total)
  const precision = safe(tp, tp + fp)
  const recall = safe(tp, tp + fn)
  const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0
  return {
    metrics: ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC'],
    values: [accuracy, precision, recall, f1, auc ?? 0],
    cards: { accuracy, precision, recall, f1 },
  }
}

const TABS = [
  { key: 'rf_spam',                  label: 'RF Spam',             cmLabels: ['Ham', 'Spam'],       color: '#00d4ff' },
  { key: 'nb_spam',                  label: 'Naive Bayes',         cmLabels: ['Ham', 'Spam'],       color: '#00cc88' },
  { key: 'logistic_regression_spam', label: 'Logistic Regression', cmLabels: ['Ham', 'Spam'],       color: '#ffb347' },
  { key: 'svm_malware',              label: 'SVM Malware',         cmLabels: ['Benign', 'Malware'], color: '#ff4d4d' },
]

// Reusable bordered card wrapper, replaces the old .card class.
function Card({ children, sx }) {
  return (
    <Paper
      elevation={0}
      sx={{
        bgcolor: 'background.paper',
        border: 1,
        borderColor: 'divider',
        borderRadius: 3,
        p: 3,
        mb: 2.5,
        animation: 'scaleIn 0.4s ease-out both',
        ...sx,
      }}
    >
      {children}
    </Paper>
  )
}

function ModelAnalytics() {
  const [active, setActive] = useState('rf_spam')
  const [cache, setCache] = useState({})
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    let mounted = true
    if (cache[active]) return

    const load = async () => {
      setLoading(true)
      setError(null)
      try {
        const data = await getModelAnalytics(active)
        if (mounted) setCache((prev) => ({ ...prev, [active]: data }))
      } catch (e) {
        if (mounted) setError(e.message)
      } finally {
        if (mounted) setLoading(false)
      }
    }

    load()
    return () => { mounted = false }
  }, [active, cache])

  const tab = TABS.find((t) => t.key === active)
  const data = cache[active]
  const metrics = data ? metricsFromConfusion(data.confusion_matrix, data.roc?.auc) : null

  return (
    <Container maxWidth="xl" sx={{ py: 3, animation: 'fadeIn 0.4s ease-out both' }}>
      <PageHeader
        title="Model Analytics"
        subtitle="Inspect confusion matrices, ROC curves, metric profiles and feature importance per model."
      />
      <ErrorBanner message={error} onDismiss={() => setError(null)} />

      <Tabs
        value={active}
        onChange={(e, v) => setActive(v)}
        variant="scrollable"
        scrollButtons="auto"
        sx={{ borderBottom: 1, borderColor: 'divider', mb: 2.5 }}
      >
        {TABS.map((t) => (
          <Tab key={t.key} value={t.key} label={t.label} sx={{ textTransform: 'none' }} />
        ))}
      </Tabs>

      <ProgressIndicator visible={loading} label="Loading analytics..." />

      {data && (
        <Box key={active}>
          {metrics && (
            <Box
              sx={{
                display: 'grid',
                gridTemplateColumns: { xs: 'repeat(2, 1fr)', md: 'repeat(4, 1fr)' },
                gap: 2,
                mb: 2.5,
              }}
            >
              {Object.entries(metrics.cards).map(([k, v], idx) => (
                <Paper
                  key={k}
                  elevation={0}
                  sx={{
                    bgcolor: 'background.paper',
                    border: 1,
                    borderColor: 'divider',
                    borderRadius: 3,
                    p: 2,
                    textAlign: 'center',
                    animation: 'fadeSlideUp 0.4s ease-out both',
                    animationDelay: `${idx * 0.07}s`,
                  }}
                >
                  <Typography sx={{ fontSize: 11, color: 'text.secondary', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                    {k}
                  </Typography>
                  <Typography sx={{ fontSize: 24, fontWeight: 700, color: METRIC_COLORS[k], mt: 0.5 }}>
                    {(v * 100).toFixed(1)}%
                  </Typography>
                </Paper>
              ))}
            </Box>
          )}

          <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' }, gap: 2.5, mb: 0 }}>
            <Card sx={{ mb: 0 }}>
              <Heatmap matrix={data.confusion_matrix} labels={tab.cmLabels} title="Confusion Matrix" />
            </Card>
            <Card sx={{ mb: 0 }}>
              <LineChart fpr={data.roc.fpr} tpr={data.roc.tpr} auc={data.roc.auc} color={tab.color} />
            </Card>
          </Box>

          {metrics && (
            <Card sx={{ mt: 2.5 }}>
              <RadarChart
                metrics={metrics.metrics}
                series={[{ name: tab.label, values: metrics.values, color: tab.color }]}
                title={`${tab.label} metric`}
                rangeMin={0.7}
              />
            </Card>
          )}

          <Card>
            <Typography sx={{ fontSize: 14, color: 'text.primary', mb: 1.5 }}>Feature Importance</Typography>
            {data.feature_importance ? (
              <BarChart
                horizontal
                categories={data.feature_importance.map((f) => f.feature).reverse()}
                values={data.feature_importance.map((f) => f.importance).reverse()}
                title="Top Features"
              />
            ) : (
              <Typography sx={{ color: 'text.secondary', textAlign: 'center', py: 5, fontSize: 13 }}>
                Not available for this model
              </Typography>
            )}
          </Card>
        </Box>
      )}
    </Container>
  )
}

export default ModelAnalytics