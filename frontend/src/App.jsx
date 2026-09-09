import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Analyzer from './pages/Analyzer';
import Batch from './pages/Batch';
import History from './pages/History';
import HistoryDetail from './pages/HistoryDetail';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<Analyzer />} />
          <Route path="/batch" element={<Batch />} />
          <Route path="/history" element={<History />} />
          <Route path="/history/:id" element={<HistoryDetail />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
