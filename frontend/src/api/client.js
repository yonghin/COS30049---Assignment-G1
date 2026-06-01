import axios from 'axios'

const client = axios.create({ baseURL: 'http://localhost:8000', timeout: 30000 })

client.interceptors.response.use(
  (response) => response,
  (error) => {
    const message = error.response?.data?.detail ?? error.message ?? 'An unexpected error occurred.'
    return Promise.reject(new Error(message))
  }
)

export default client
