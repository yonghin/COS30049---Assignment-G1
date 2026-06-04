import { useState, useEffect } from 'react'
import PageHeader from '../components/PageHeader'
import ErrorBanner from '../components/ErrorBanner'
import ProgressIndicator from '../components/ProgressIndicator'
import Heatmap from '../components/charts/Heatmap'
import LineChart from '../components/charts/LineChart'
import BarChart from '../components/charts/BarChart'
import RadarChart from '../components/charts/RadarChart'
import { getModelAnalytics } from '../api/analyticsApi'
import styles from './ModelAnalytics.module.css'

const METRIC_COLORS = {
  accuracy: 'var(--accent)',
  precision: 'var(--purple)',
  recall: 'var(--success)',
  f1: 'var(--warning)',
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
  { key: 'rf_spam',                  label: 'RF Spam',            cmLabels: ['Ham', 'Spam'],       color: '#00d4ff' },
  { key: 'nb_spam',                  label: 'Naive Bayes',        cmLabels: ['Ham', 'Spam'],       color: '#00cc88' },
  { key: 'logistic_regression_spam', label: 'Logistic Regression',cmLabels: ['Ham', 'Spam'],       color: '#ffb347' },
  { key: 'svm_malware',              label: 'SVM Malware',        cmLabels: ['Benign', 'Malware'], color: '#ff4d4d' },
]

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
    <div className={styles.page}>
      <PageHeader
        title="Model Analytics"
        subtitle="Inspect confusion matrices, ROC curves, metric profiles and feature importance per model."
      />
      <ErrorBanner message={error} onDismiss={() => setError(null)} />

      <div className={styles.tabs}>
          {TABS.map((t) => (
            <button
              key={t.key}
              className={t.key === active ? `${styles.tab} ${styles.activeTab}` : styles.tab}
              onClick={() => setActive(t.key)}
            >
              {t.label}
            </button>
          ))}
        </div>

        <ProgressIndicator visible={loading} label="Loading analytics..." />

        {data && (
          <>
            {metrics && (
              <div className={styles.metricCards}>
                {Object.entries(metrics.cards).map(([k, v]) => (
                  <div key={k} className={styles.metricCard}>
                    <div className={styles.metricLabel}>{k}</div>
                    <div className={styles.metricValue} style={{ color: METRIC_COLORS[k] }}>{(v * 100).toFixed(1)}%</div>
                  </div>
                ))}
              </div>
            )}

            <div className={styles.topRow}>
              <div className={styles.card}>
                <Heatmap matrix={data.confusion_matrix} labels={tab.cmLabels} title="Confusion Matrix" />
              </div>
              <div className={styles.card}>
                <LineChart fpr={data.roc.fpr} tpr={data.roc.tpr} auc={data.roc.auc} color={tab.color} />
              </div>
            </div>

            {metrics && (
              <div className={styles.card}>
                <RadarChart
                  metrics={metrics.metrics}
                  series={[{ name: tab.label, values: metrics.values, color: tab.color }]}
                  title={`${tab.label} — metric profile`}
                />
              </div>
            )}

            <div className={styles.card}>
              <h3 className={styles.sectionTitle}>Feature Importance</h3>
              {data.feature_importance ? (
                <BarChart
                  horizontal
                  categories={data.feature_importance.map((f) => f.feature).reverse()}
                  values={data.feature_importance.map((f) => f.importance).reverse()}
                  title="Top Features"
                />
              ) : (
                <div className={styles.noData}>Not available for this model</div>
              )}
            </div>
          </>
        )}
      </div>
  )
}

export default ModelAnalytics
