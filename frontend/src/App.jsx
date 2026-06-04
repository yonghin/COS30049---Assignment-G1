import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import SpamDetector from './pages/SpamDetector'
import MalwareDetector from './pages/MalwareDetector'
import ModelAnalytics from './pages/ModelAnalytics'
import History from './pages/History'

function App() {
  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<Navigate to="/dashboard" />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/spam" element={<SpamDetector />} />
          <Route path="/malware" element={<MalwareDetector />} />
          <Route path="/analytics" element={<ModelAnalytics />} />
          <Route path="/history" element={<History />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  )
}

export default App
