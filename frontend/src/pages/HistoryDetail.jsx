import { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { getAnalysisDetail } from '../api/client';
import ResultCard from '../components/ResultCard';
import EmotionChart from '../components/EmotionChart';
import SentimentGauge from '../components/SentimentGauge';
import './HistoryDetail.css';

export default function HistoryDetail() {
  const { id } = useParams();
  const [analysis, setAnalysis] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchDetail();
  }, [id]);

  const fetchDetail = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await getAnalysisDetail(id);
      setAnalysis(data);
    } catch (err) {
      console.error('Failed to load analysis:', err);
      if (err.response?.status === 404) {
        setError('Analysis not found.');
      } else if (err.code === 'ERR_NETWORK') {
        setError('Cannot connect to backend.');
      } else {
        setError('Failed to load analysis details.');
      }
    } finally {
      setIsLoading(false);
    }
  };

  const formatDate = (dateStr) => {
    return new Date(dateStr).toLocaleString('en-US', {
      weekday: 'short', year: 'numeric', month: 'long', day: 'numeric',
      hour: '2-digit', minute: '2-digit', second: '2-digit',
    });
  };

  if (isLoading) {
    return (
      <div className="detail-page" id="history-detail-page">
        <div className="detail-loading">
          <div className="spinner" style={{ width: 40, height: 40 }}></div>
          <p>Loading analysis...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="detail-page" id="history-detail-page">
        <div className="detail-error glass-card">
          <span className="empty-state-icon">❌</span>
          <h3>{error}</h3>
          <Link to="/history" className="btn btn-primary" style={{ marginTop: 16 }}>
            ← Back to History
          </Link>
        </div>
      </div>
    );
  }

  if (!analysis) return null;

  // Transform to match ResultCard's expected format
  const resultData = {
    id: analysis.id,
    detected_lang: analysis.detected_lang,
    sentiment: {
      label: analysis.sentiment_label,
      score: analysis.sentiment_score,
    },
    emotions: analysis.emotion_scores || {},
    dominant_emotion: analysis.dominant_emotion,
  };

  return (
    <div className="detail-page" id="history-detail-page">
      <div className="detail-breadcrumb">
        <Link to="/history" className="breadcrumb-link">← History</Link>
        <span className="breadcrumb-sep">/</span>
        <span className="breadcrumb-current">Analysis #{analysis.id}</span>
      </div>

      <div className="page-header">
        <h1 className="page-title">Analysis #{analysis.id}</h1>
        <p className="page-subtitle">
          {formatDate(analysis.created_at)}
          {analysis.source === 'batch' && ' · From batch upload'}
        </p>
      </div>

      {/* Original text */}
      <div className="detail-original glass-card animate-fade-in-up">
        <h3 className="detail-section-title">Original Text</h3>
        <p className="detail-text">{analysis.input_text}</p>
      </div>

      {/* Results */}
      <div className="detail-results">
        <div className="detail-results-grid">
          <div className="detail-results-left">
            <div className="glass-card animate-fade-in-up">
              <SentimentGauge
                label={analysis.sentiment_label}
                score={analysis.sentiment_score}
              />
            </div>
          </div>
          <div className="detail-results-right">
            <EmotionChart emotions={analysis.emotion_scores} chartType="radar" />
          </div>
        </div>

        <ResultCard result={resultData} />
      </div>
    </div>
  );
}
