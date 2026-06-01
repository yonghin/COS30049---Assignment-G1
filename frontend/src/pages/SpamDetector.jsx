import { useState } from 'react'
import NavBar from '../components/NavBar'
import ErrorBanner from '../components/ErrorBanner'
import ProgressIndicator from '../components/ProgressIndicator'
import ResultsTable from '../components/ResultsTable'
import ExportButton from '../components/ExportButton'
import FileUploadWidget from '../components/FileUploadWidget'
import GaugeChart from '../components/charts/GaugeChart'
import { predictSingle, predictBatch } from '../api/spamApi'
import styles from './SpamDetector.module.css'

const MODELS = [
  { value: 'rf_spam', label: 'Random Forest' },
  { value: 'nb_spam', label: 'Naive Bayes' },
  { value: 'logistic_regression_spam', label: 'Logistic Regression' },
]

function SpamDetector() {
  const [tab, setTab] = useState('single')
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  // Single
  const [text, setText] = useState('')
  const [model, setModel] = useState('rf_spam')
  const [result, setResult] = useState(null)

  // Batch
  const [batchModel, setBatchModel] = useState('rf_spam')
  const [file, setFile] = useState(null)
  const [batch, setBatch] = useState(null)

  const analyzeSingle = async () => {
    setError(null)
    if (text.trim().length < 3) {
      setError('Message must be at least 3 characters.')
      return
    }
    setLoading(true)
    try {
      const data = await predictSingle(text, model)
      setResult(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const analyzeBatch = async () => {
    setError(null)
    if (!file) {
      setError('Please upload a .txt or .csv file first.')
      return
    }
    setLoading(true)
    try {
      const data = await predictBatch(file, batchModel)
      setBatch(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <>
      <NavBar />
      <div className={styles.page}>
        <ErrorBanner message={error} onDismiss={() => setError(null)} />

        <div className={styles.tabs}>
          <button
            className={tab === 'single' ? `${styles.tab} ${styles.activeTab}` : styles.tab}
            onClick={() => setTab('single')}
          >
            Single Message
          </button>
          <button
            className={tab === 'batch' ? `${styles.tab} ${styles.activeTab}` : styles.tab}
            onClick={() => setTab('batch')}
          >
            Batch Upload
          </button>
        </div>

        {tab === 'single' && (
          <div className={styles.card}>
            <div className={styles.singleLayout}>
              <div>
                <textarea
                  className={styles.textarea}
                  placeholder="Paste a message to classify..."
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                />
                <select className={styles.select} value={model} onChange={(e) => setModel(e.target.value)}>
                  {MODELS.map((m) => (
                    <option key={m.value} value={m.value}>{m.label}</option>
                  ))}
                </select>
                <button className={styles.primary} onClick={analyzeSingle} disabled={loading}>
                  Analyze
                </button>
              </div>
              <div className={styles.resultPanel}>
                <GaugeChart spamProb={result ? result.spam_prob : null} label="Spam Probability" />
                {result && (
                  <div className={styles.resultMeta}>
                    <span className={result.label === 'SPAM' ? `${styles.resultChip} ${styles.spam}` : `${styles.resultChip} ${styles.ham}`}>
                      {result.label}
                    </span>
                    <span className={styles.confidence}>
                      Confidence: {(result.confidence * 100).toFixed(2)}%
                    </span>
                  </div>
                )}
              </div>
            </div>
            <ProgressIndicator visible={loading} />
          </div>
        )}

        {tab === 'batch' && (
          <div className={styles.card}>
            <FileUploadWidget accept=".txt,.csv" label="Upload .txt or .csv" onFileSelected={setFile} />
            <div className={styles.batchControls}>
              <select className={styles.select} value={batchModel} onChange={(e) => setBatchModel(e.target.value)}>
                {MODELS.map((m) => (
                  <option key={m.value} value={m.value}>{m.label}</option>
                ))}
              </select>
              <button className={styles.primary} onClick={analyzeBatch} disabled={loading}>
                Analyze
              </button>
            </div>
            <ProgressIndicator visible={loading} />
            {batch && (
              <>
                <div className={styles.statsGrid}>
                  <div className={styles.statCard}>
                    <div className={styles.statLabel}>Total</div>
                    <div className={styles.statValue}>{batch.total}</div>
                  </div>
                  <div className={styles.statCard}>
                    <div className={styles.statLabel}>Spam</div>
                    <div className={`${styles.statValue} ${styles.spamText}`}>{batch.spam_count}</div>
                  </div>
                  <div className={styles.statCard}>
                    <div className={styles.statLabel}>Ham</div>
                    <div className={`${styles.statValue} ${styles.hamText}`}>{batch.ham_count}</div>
                  </div>
                </div>
                <ResultsTable
                  columns={['row', 'text', 'label', 'spam_prob']}
                  rows={batch.results}
                />
                <div className={styles.exportRow}>
                  <ExportButton data={batch.results} filename="spam_batch_results.csv" />
                </div>
              </>
            )}
          </div>
        )}
      </div>
    </>
  )
}

export default SpamDetector
