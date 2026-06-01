import client from './client'

export const predictSingle = async (text, model = 'rf_spam') =>
  (await client.post('/api/spam/predict', { text, model })).data

export const predictBatch = async (file, model = 'rf_spam') => {
  const form = new FormData(); form.append('file', file); form.append('model', model)
  return (await client.post('/api/spam/predict/batch', form)).data
}
