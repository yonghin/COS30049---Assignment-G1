import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Dashboard from './pages/Dashboard'
import SpamDetector from './pages/SpamDetector'
import MalwareDetector from './pages/MalwareDetector'
import ModelAnalytics from './pages/ModelAnalytics'

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Navigate to="/dashboard" />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/spam" element={<SpamDetector />} />
        <Route path="/malware" element={<MalwareDetector />} />
        <Route path="/analytics" element={<ModelAnalytics />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App
