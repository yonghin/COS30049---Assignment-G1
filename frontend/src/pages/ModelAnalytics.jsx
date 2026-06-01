import { useState, useEffect } from 'react'
import NavBar from '../components/NavBar'
import ErrorBanner from '../components/ErrorBanner'
import ProgressIndicator from '../components/ProgressIndicator'
import Heatmap from '../components/charts/Heatmap'
import LineChart from '../components/charts/LineChart'
import BarChart from '../components/charts/BarChart'
import { getModelAnalytics } from '../api/analyticsApi'
import styles from './ModelAnalytics.module.css'

const TABS = [
  { key: 'rf_spam', label: 'RF Spam', cmLabels: ['Ham', 'Spam'] },
  { key: 'nb_spam', label: 'Naive Bayes', cmLabels: ['Ham', 'Spam'] },
  { key: 'logistic_regression_spam', label: 'Logistic Regression', cmLabels: ['Ham', 'Spam'] },
  { key: 'svm_malware', label: 'SVM Malware', cmLabels: ['Benign', 'Malware'] },
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

  return (
    <>
      <NavBar />
      <div className={styles.page}>
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
            <div className={styles.topRow}>
              <div className={styles.card}>
                <Heatmap matrix={data.confusion_matrix} labels={tab.cmLabels} title="Confusion Matrix" />
              </div>
              <div className={styles.card}>
                <LineChart fpr={data.roc.fpr} tpr={data.roc.tpr} auc={data.roc.auc} />
              </div>
            </div>

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
    </>
  )
}

export default ModelAnalytics
