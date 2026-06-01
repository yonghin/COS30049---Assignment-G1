import client from './client'

export const getModelAnalytics = async (modelName) =>
  (await client.get(`/api/analytics/model/${modelName}`)).data
