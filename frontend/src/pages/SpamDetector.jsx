import { useState } from 'react'
import Box from '@mui/material/Box'
import Container from '@mui/material/Container'
import Paper from '@mui/material/Paper'
import Typography from '@mui/material/Typography'
import Button from '@mui/material/Button'
import TextField from '@mui/material/TextField'
import MenuItem from '@mui/material/MenuItem'
import Tabs from '@mui/material/Tabs'
import Tab from '@mui/material/Tab'
import Chip from '@mui/material/Chip'
import LinearProgress from '@mui/material/LinearProgress'
import Table from '@mui/material/Table'
import TableHead from '@mui/material/TableHead'
import TableBody from '@mui/material/TableBody'
import TableRow from '@mui/material/TableRow'
import TableCell from '@mui/material/TableCell'
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

const MODELS = [
  { value: 'nb_spam', label: modelLabel('nb_spam') },
  { value: 'logistic_regression_spam', label: modelLabel('logistic_regression_spam') },
  { value: 'rf_spam', label: modelLabel('rf_spam') },
]

// Enhancement 3: hardcoded frontend sample presets (no backend change).
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

// Reusable bordered card wrapper, replaces the old .card class.
function Card({ children, sx }) {
  return (
    <Paper
      elevation={0}
      sx={{ bgcolor: 'background.paper', border: 1, borderColor: 'divider', borderRadius: 3, p: 3, mb: 2.5, ...sx }}
    >
      {children}
    </Paper>
  )
}

// Single Ham/Spam probability bar.
function BreakdownRow({ label, pct, color }) {
  return (
    <Box sx={{ display: 'grid', gridTemplateColumns: '44px 1fr 52px', alignItems: 'center', gap: 1.25, mb: 1 }}>
      <Typography sx={{ fontSize: 13, color: 'text.primary' }}>{label}</Typography>
      <Box sx={{ bgcolor: 'action.hover', border: 1, borderColor: 'divider', borderRadius: 1.5, height: 14, overflow: 'hidden' }}>
        <Box sx={{ height: '100%', width: `${pct}%`, bgcolor: color, transition: 'width 0.3s' }} />
      </Box>
      <Typography sx={{ fontSize: 13, color: 'text.secondary', textAlign: 'right' }}>{pct.toFixed(1)}%</Typography>
    </Box>
  )
}

function SpamDetector() {
  const toast = useToast()
  const [tab, setTab] = useState('single')
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  // Single
  const [text, setText] = useState('')
  const [model, setModel] = useState('nb_spam')
  const [result, setResult] = useState(null)
  const [analyzed, setAnalyzed] = useState('')
  const [compare, setCompare] = useState(null)
  const [comparing, setComparing] = useState(false)

  // Batch
  const [batchModel, setBatchModel] = useState('nb_spam')
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
      toast.success(`Analyzed ${data.total} messages, ${data.spam_count} spam`)
    } catch (e) {
      setError(e.message)
      toast.error(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <Container maxWidth="xl" sx={{ py: 3 }}>
      <PageHeader
        title="Spam Detector"
        subtitle="Classify a single message or a batch file as spam or ham using three trained models."
      />
      <ErrorBanner message={error} onDismiss={() => setError(null)} />

      <Tabs value={tab} onChange={(e, v) => setTab(v)} sx={{ borderBottom: 1, borderColor: 'divider', mb: 2.5 }}>
        <Tab value="single" label="Single Message" sx={{ textTransform: 'none' }} />
        <Tab value="batch" label="Batch Upload" sx={{ textTransform: 'none' }} />
      </Tabs>

      {tab === 'single' && (
        <Card>
          <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '2fr 3fr' }, gap: 3 }}>
            {/* Left: input */}
            <Box>
              <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', flexWrap: 'wrap', mb: 2 }}>
                <Typography sx={{ fontSize: 12, color: 'text.secondary' }}>Try an example:</Typography>
                <Button
                  size="small"
                  variant="outlined"
                  onClick={() => loadSingleSample('ham')}
                  sx={{
                    textTransform: 'none',
                    borderRadius: 5,
                    color: 'text.primary',
                    borderColor: 'divider',
                    '&:hover': { borderColor: 'primary.main', color: 'primary.main' },
                  }}
                >
                  Ham
                </Button>
                <Button
                  size="small"
                  variant="outlined"
                  onClick={() => loadSingleSample('spam')}
                  sx={{
                    textTransform: 'none',
                    borderRadius: 5,
                    color: 'text.primary',
                    borderColor: 'divider',
                    '&:hover': { borderColor: 'error.main', color: 'error.main' },
                  }}
                >
                  Spam
                </Button>
              </Box>
              <TextField
                fullWidth
                multiline
                minRows={5}
                placeholder="Paste a message to classify..."
                value={text}
                onChange={(e) => setText(e.target.value)}
                sx={{ mb: 1.5 }}
              />
              <TextField
                select
                fullWidth
                value={model}
                onChange={(e) => setModel(e.target.value)}
                sx={{ mb: 1.5 }}
              >
                {MODELS.map((m) => (
                  <MenuItem key={m.value} value={m.value}>{m.label}</MenuItem>
                ))}
              </TextField>
              <Button
                fullWidth
                variant="contained"
                onClick={analyzeSingle}
                disabled={loading}
                sx={{ textTransform: 'none', fontWeight: 600, mb: 1 }}
              >
                Analyze
              </Button>
              <Button
                fullWidth
                variant="outlined"
                onClick={compareModels}
                disabled={comparing}
                sx={{ textTransform: 'none', fontWeight: 600 }}
              >
                Compare all 3 models
              </Button>
            </Box>

            {/* Right: result */}
            <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
              <GaugeChart spamProb={result ? result.spam_prob : null} label="Spam Probability" />
              {result && (
                <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 1 }}>
                  <Chip
                    label={result.label}
                    color={result.label === 'SPAM' ? 'error' : 'success'}
                    variant="outlined"
                    sx={{ fontWeight: 700, fontSize: 16, px: 1 }}
                  />
                  <Typography sx={{ color: 'text.secondary', fontSize: 13 }}>
                    Confidence: {(result.confidence * 100).toFixed(2)}%, {modelLabel(result.model_used)}
                  </Typography>
                </Box>
              )}
              {result && (
                <Box sx={{ width: '100%' }}>
                  <Typography sx={{ fontSize: 12, color: 'text.secondary', textTransform: 'uppercase', letterSpacing: '0.05em', mb: 1 }}>
                    Ham vs Spam
                  </Typography>
                  <BreakdownRow label="Ham" pct={result.ham_prob * 100} color="success.main" />
                  <BreakdownRow label="Spam" pct={result.spam_prob * 100} color="error.main" />
                </Box>
              )}
            </Box>
          </Box>

          {result && <KeywordHighlight text={analyzed} />}

          {compare && (
            <Box sx={{ mt: 2.5 }}>
              <Typography sx={{ fontSize: 14, color: 'text.primary', mb: 1.5 }}>Model comparison</Typography>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Model</TableCell>
                    <TableCell>Label</TableCell>
                    <TableCell>Spam probability</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {compare.map((c) => (
                    <TableRow key={c.model}>
                      <TableCell>{modelLabel(c.model)}</TableCell>
                      <TableCell>
                        <Typography component="span" sx={{ color: c.label === 'SPAM' ? 'error.main' : 'success.main', fontWeight: 600 }}>
                          {c.label}
                        </Typography>
                      </TableCell>
                      <TableCell>{(c.spam_prob * 100).toFixed(2)}%</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </Box>
          )}

          {(loading || comparing) && <LinearProgress sx={{ mt: 2 }} />}
        </Card>
      )}

      {tab === 'batch' && (
        <Card>
          <FileUploadWidget
            accept=".txt,.csv"
            label="Upload .txt or .csv"
            onFileSelected={(f) => { setFile(f); setFileLabel(f.name) }}
          />
          <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', flexWrap: 'wrap', my: 2.5 }}>
            <Typography sx={{ fontSize: 12, color: 'text.secondary' }}>Load sample:</Typography>
            <Chip label="Ham set" size="small" variant="outlined" onClick={() => loadBatchSample('ham')} />
            <Chip label="Spam set" size="small" variant="outlined" color="error" onClick={() => loadBatchSample('spam')} />
            {fileLabel && <Typography sx={{ color: 'text.secondary', fontSize: 13 }}>{fileLabel}</Typography>}
          </Box>
          <Box sx={{ display: 'flex', gap: 1.5, alignItems: 'center', my: 2, flexWrap: 'wrap' }}>
            <TextField
              select
              value={batchModel}
              onChange={(e) => setBatchModel(e.target.value)}
              sx={{ flex: 1, minWidth: 200 }}
            >
              {MODELS.map((m) => (
                <MenuItem key={m.value} value={m.value}>{m.label}</MenuItem>
              ))}
            </TextField>
            <Button variant="contained" onClick={analyzeBatch} disabled={loading} sx={{ textTransform: 'none', fontWeight: 600 }}>
              Analyze
            </Button>
          </Box>
          <ProgressIndicator visible={loading} />

          {batch && (
            <>
              <Box sx={{ display: 'grid', gridTemplateColumns: { xs: 'repeat(2, 1fr)', md: 'repeat(3, 1fr)' }, gap: 2, my: 2.5 }}>
                {[
                  { label: 'Total', value: batch.total, color: 'primary.main' },
                  { label: 'Spam', value: batch.spam_count, color: 'error.main' },
                  { label: 'Ham', value: batch.ham_count, color: 'success.main' },
                ].map((s) => (
                  <Paper key={s.label} elevation={0} sx={{ bgcolor: 'action.hover', border: 1, borderColor: 'divider', borderRadius: 3, p: 2 }}>
                    <Typography sx={{ fontSize: 11, color: 'text.secondary', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                      {s.label}
                    </Typography>
                    <Typography sx={{ fontSize: 28, fontWeight: 700, color: s.color }}>{s.value}</Typography>
                  </Paper>
                ))}
              </Box>
              <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' }, gap: 2, my: 2.5 }}>
                <Paper elevation={0} sx={{ bgcolor: 'action.hover', border: 1, borderColor: 'divider', borderRadius: 3, p: 2 }}>
                  <DonutChart labels={['Ham', 'Spam']} values={[batch.ham_count, batch.spam_count]} title="Ham vs Spam" />
                </Paper>
                <Paper elevation={0} sx={{ bgcolor: 'action.hover', border: 1, borderColor: 'divider', borderRadius: 3, p: 2 }}>
                  <Histogram
                    values={batch.results.map((r) => r.spam_prob)}
                    title="Spam Probability Distribution"
                    xLabel="Spam probability"
                  />
                </Paper>
              </Box>
              <ResultsTable
                columns={['row', 'text', 'label', 'spam_prob']}
                rows={batch.results}
                filterColumn="label"
              />
              <Box sx={{ mt: 2, display: 'flex', justifyContent: 'flex-end' }}>
                <ExportButton data={batch.results} filename="spam_batch_results.csv" />
              </Box>
            </>
          )}
        </Card>
      )}
    </Container>
  )
}

export default SpamDetector