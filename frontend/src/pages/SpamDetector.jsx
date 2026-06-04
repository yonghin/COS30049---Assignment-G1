import { useState } from 'react'
import PageHeader from '../components/PageHeader'
import ErrorBanner from '../components/ErrorBanner'
import ProgressIndicator from '../components/ProgressIndicator'
import ResultsTable from '../components/ResultsTable'
import ExportButton from '../components/ExportButton'
import FileUploadWidget from '../components/FileUploadWidget'
import GaugeChart from '../components/charts/GaugeChart'
import Histogram from '../components/charts/Histogram'
import DonutChart from '../components/charts/DonutChart'
import KeywordHighlight from '../components/KeywordHighlight'
import { predictSingle, predictBatch } from '../api/spamApi'
import { modelLabel } from '../constants/modelNames'
import { useToast } from '../context/ToastContext'
import { recordPrediction } from '../utils/historyStore'
import styles from './SpamDetector.module.css'

const MODELS = [
  { value: 'rf_spam', label: modelLabel('rf_spam') },
  { value: 'nb_spam', label: modelLabel('nb_spam') },
  { value: 'logistic_regression_spam', label: modelLabel('logistic_regression_spam') },
]

// Enhancement 3 — hardcoded frontend sample presets (no backend change).
const SINGLE_SAMPLES = {
  ham: 'Hey, are we still on for lunch tomorrow at noon? Let me know if the time works for you.',
  spam: 'WINNER!! You have been SELECTED to receive a FREE $1000 gift card. Click http://claim-now.biz to claim your prize NOW. Reply STOP to opt out.',
}

const BATCH_SAMPLES = {
  ham: [
    'Can you send me the meeting notes from yesterday?',
    'Thanks for dinner last night, it was great catching up.',
    'Reminder: the dentist appointment is on Thursday at 3pm.',
    'I left your book on the kitchen table, grab it when you can.',
  ],
  spam: [
    'Congratulations! You won a FREE cruise. Call 0900 123456 now to claim.',
    'URGENT: Your account has been suspended. Verify at http://secure-login.ru immediately.',
    'Get CHEAP meds online, 80% DISCOUNT, no prescription needed. Order today!',
    'You have been awarded a $5000 cash prize. Text WIN to 80085 to collect.',
  ],
}

function SpamDetector() {
  const toast = useToast()
  const [tab, setTab] = useState('single')
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  // Single
  const [text, setText] = useState('')
  const [model, setModel] = useState('rf_spam')
  const [result, setResult] = useState(null)
  const [analyzed, setAnalyzed] = useState('')
  const [compare, setCompare] = useState(null)
  const [comparing, setComparing] = useState(false)

  // Batch
  const [batchModel, setBatchModel] = useState('rf_spam')
  const [file, setFile] = useState(null)
  const [fileLabel, setFileLabel] = useState(null)
  const [batch, setBatch] = useState(null)

  const analyzeSingle = async () => {
    setError(null)
    if (text.trim().length < 3) {
      setError('Message must be at least 3 characters.')
      return
    }
    setLoading(true)
    setCompare(null)
    try {
      const data = await predictSingle(text, model)
      setResult(data)
      setAnalyzed(text)
      recordPrediction({
        kind: 'spam',
        model: data.model_used,
        label: data.label,
        confidence: data.confidence,
        summary: text.slice(0, 80),
      })
      toast.success(`Classified as ${data.label} (${(data.confidence * 100).toFixed(1)}%)`)
    } catch (e) {
      setError(e.message)
      toast.error(e.message)
    } finally {
      setLoading(false)
    }
  }

  const compareModels = async () => {
    setError(null)
    if (text.trim().length < 3) {
      setError('Message must be at least 3 characters.')
      return
    }
    setComparing(true)
    try {
      const rows = await Promise.all(
        MODELS.map(async (m) => ({ model: m.value, ...(await predictSingle(text, m.value)) }))
      )
      setCompare(rows)
      setAnalyzed(text)
    } catch (e) {
      setError(e.message)
    } finally {
      setComparing(false)
    }
  }

  const loadSingleSample = (kind) => {
    setText(SINGLE_SAMPLES[kind])
    setResult(null)
    setCompare(null)
    setError(null)
  }

  const loadBatchSample = (kind) => {
    const body = BATCH_SAMPLES[kind].join('\n')
    const sampleFile = new File([body], `sample_${kind}.txt`, { type: 'text/plain' })
    setFile(sampleFile)
    setFileLabel(`sample_${kind}.txt (loaded)`)
    setBatch(null)
    setError(null)
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
      recordPrediction({
        kind: 'spam',
        model: batchModel,
        label: `Batch (${data.spam_count}/${data.total} spam)`,
        confidence: data.total ? data.spam_count / data.total : 0,
        summary: `Batch of ${data.total} messages`,
      })
      toast.success(`Analyzed ${data.total} messages · ${data.spam_count} spam`)
    } catch (e) {
      setError(e.message)
      toast.error(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className={styles.page}>
      <PageHeader
        title="Spam Detector"
        subtitle="Classify a single message or a batch file as spam or ham using three trained models."
      />
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
                <div className={styles.sampleRow}>
                  <span className={styles.sampleHint}>Try an example:</span>
                  <button className={styles.sample} onClick={() => loadSingleSample('ham')}>Ham ✉</button>
                  <button className={`${styles.sample} ${styles.sampleSpam}`} onClick={() => loadSingleSample('spam')}>Spam ⚠</button>
                </div>
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
                <button className={styles.secondary} onClick={compareModels} disabled={comparing}>
                  Compare all 3 models
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
                      Confidence: {(result.confidence * 100).toFixed(2)}% · {modelLabel(result.model_used)}
                    </span>
                  </div>
                )}
                {result && (
                  <div className={styles.breakdown}>
                    <div className={styles.breakdownTitle}>Ham vs Spam</div>
                    <div className={styles.breakRow}>
                      <span className={styles.breakLabel}>Ham</span>
                      <div className={styles.breakTrack}>
                        <div className={`${styles.breakFill} ${styles.hamFill}`} style={{ width: `${result.ham_prob * 100}%` }} />
                      </div>
                      <span className={styles.breakPct}>{(result.ham_prob * 100).toFixed(1)}%</span>
                    </div>
                    <div className={styles.breakRow}>
                      <span className={styles.breakLabel}>Spam</span>
                      <div className={styles.breakTrack}>
                        <div className={`${styles.breakFill} ${styles.spamFill}`} style={{ width: `${result.spam_prob * 100}%` }} />
                      </div>
                      <span className={styles.breakPct}>{(result.spam_prob * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {result && <KeywordHighlight text={analyzed} />}

            {compare && (
              <div className={styles.compareWrap}>
                <h3 className={styles.compareTitle}>Model comparison</h3>
                <table className={styles.compareTable}>
                  <thead>
                    <tr><th>Model</th><th>Label</th><th>Spam probability</th></tr>
                  </thead>
                  <tbody>
                    {compare.map((c) => (
                      <tr key={c.model}>
                        <td>{modelLabel(c.model)}</td>
                        <td>
                          <span className={c.label === 'SPAM' ? styles.spamText : styles.hamText}>{c.label}</span>
                        </td>
                        <td>{(c.spam_prob * 100).toFixed(2)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            <ProgressIndicator visible={loading || comparing} />
          </div>
        )}

        {tab === 'batch' && (
          <div className={styles.card}>
            <FileUploadWidget accept=".txt,.csv" label="Upload .txt or .csv" onFileSelected={(f) => { setFile(f); setFileLabel(f.name) }} />
            <div className={styles.sampleRow}>
              <span className={styles.sampleHint}>Load sample:</span>
              <button className={styles.sample} onClick={() => loadBatchSample('ham')}>Ham set ✉</button>
              <button className={`${styles.sample} ${styles.sampleSpam}`} onClick={() => loadBatchSample('spam')}>Spam set ⚠</button>
              {fileLabel && <span className={styles.fileLabel}>{fileLabel}</span>}
            </div>
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
                <div className={styles.chartsRow}>
                  <div className={styles.chartCard}>
                    <DonutChart
                      labels={['Ham', 'Spam']}
                      values={[batch.ham_count, batch.spam_count]}
                      title="Ham vs Spam"
                    />
                  </div>
                  <div className={styles.chartCard}>
                    <Histogram
                      values={batch.results.map((r) => r.spam_prob)}
                      title="Spam Probability Distribution"
                      xLabel="Spam probability"
                    />
                  </div>
                </div>
                <ResultsTable
                  columns={['row', 'text', 'label', 'spam_prob']}
                  rows={batch.results}
                  filterColumn="label"
                />
                <div className={styles.exportRow}>
                  <ExportButton data={batch.results} filename="spam_batch_results.csv" />
                </div>
              </>
            )}
          </div>
        )}
      </div>
  )
}

export default SpamDetector
