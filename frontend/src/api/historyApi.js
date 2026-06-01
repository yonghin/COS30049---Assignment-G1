import client from './client'

export const getHistory = async (since = null) =>
  (await client.get('/api/predictions/history', { params: since ? { since } : {} })).data

export const clearHistory = async () => (await client.delete('/api/predictions/history')).data

export const getModels = async () => (await client.get('/api/models')).data
