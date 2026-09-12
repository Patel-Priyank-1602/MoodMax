import { useState } from 'react';
import TextInput from '../components/TextInput';
import ResultCard from '../components/ResultCard';
import EmotionChart from '../components/EmotionChart';
import SentimentGauge from '../components/SentimentGauge';
import { analyzeText } from '../api/client';
import './Analyzer.css';

export default function Analyzer() {
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleAnalyze = async (text) => {
    setIsLoading(true);
    setError(null);

    try {
      const data = await analyzeText(text);
      setResult(data);
    } catch (err) {
      console.error('Analysis failed:', err);
      if (err.response?.data?.detail) {
        setError(err.response.data.detail);
      } else if (err.code === 'ERR_NETWORK') {
        setError('Cannot connect to backend. Make sure the server is running on port 8000.');
      } else {
        setError('Analysis failed. Please try again.');
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleDownloadCsv = () => {
    if (!result) return;

    const emotionKeys = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'neutral'];
    const headers = [
      'Text',
      'Sentiment',
      'Confidence',
      'Dominant Emotion',
      'Language',
      ...emotionKeys.map((e) => `Emotion_${e.charAt(0).toUpperCase() + e.slice(1)}`),
    ];

    const escapeCsv = (str) => {
      if (str == null) return '""';
      const text = String(str).replace(/"/g, '""');
      return `"${text}"`;
    };

    const scores = result.emotions || {};
    const row = [
      escapeCsv(result.input_text),
      escapeCsv(result.sentiment?.label || ''),
      result.sentiment?.score != null ? (result.sentiment.score * 100).toFixed(1) + '%' : '',
      escapeCsv(result.dominant_emotion || ''),
      escapeCsv((result.detected_lang || '').toUpperCase()),
      ...emotionKeys.map((e) => (scores[e] != null ? (scores[e] * 100).toFixed(1) + '%' : '')),
    ].join(',');

    const csvContent = '\uFEFF' + [headers.join(','), row].join('\r\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', `moodmax_analysis.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="analyzer-page" id="analyzer-page">
      <div className="page-header">
        <h1 className="page-title">Analyze <span className="title-accent">Text</span></h1>
        <p className="page-subtitle">
          Paste any social media text to get instant sentiment and emotion analysis.
          Supports 100+ languages.
        </p>
      </div>

      <TextInput onAnalyze={handleAnalyze} isLoading={isLoading} />

      {isLoading && (
        <div className="analyzer-loading animate-fade-in">
          <div className="loading-bar"></div>
          <p className="loading-text">Analyzing with AI models...</p>
        </div>
      )}

      {error && (
        <div className="analyzer-error card animate-fade-in-up">
          <svg className="error-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10"/>
            <line x1="12" y1="8" x2="12" y2="12"/>
            <line x1="12" y1="16" x2="12.01" y2="16"/>
          </svg>
          <span className="error-text">{error}</span>
        </div>
      )}

      {result && !isLoading && (
        <div className="analyzer-results stagger-children">
          {/* Sentiment + Result Card side by side */}
          <div className="results-grid">
            <div className="results-left">
              <div className="glass-card animate-fade-in-up">
                <SentimentGauge
                  label={result.sentiment.label}
                  score={result.sentiment.score}
                />
              </div>

              {/* Quick stats with CSV download button */}
              <div className="quick-stats glass-card animate-fade-in-up">
                <div className="quick-stats-row">
                  <div className="stat-item">
                    <span className="stat-value">{result.detected_lang?.toUpperCase() || 'EN'}</span>
                    <span className="stat-label">Language</span>
                  </div>
                  <div className="stat-divider"></div>
                  <div className="stat-item">
                    <span className="stat-value highlight-green">
                      {Object.entries(result.emotions || {})
                        .filter(([, s]) => s > 0.1).length}
                    </span>
                    <span className="stat-label">Active Emotions</span>
                  </div>
                </div>
                <div className="quick-stats-divider-h"></div>
                <div className="quick-stats-actions">
                  <button
                    type="button"
                    className="btn btn-download-csv w-full"
                    onClick={handleDownloadCsv}
                    id="download-single-csv-button"
                  >
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                      <polyline points="7 10 12 15 17 10"/>
                      <line x1="12" y1="15" x2="12" y2="3"/>
                    </svg>
                    <span>Download CSV</span>
                  </button>
                </div>
              </div>
            </div>

            <div className="results-right">
              <EmotionChart emotions={result.emotions} chartType="radar" />
            </div>
          </div>

          <ResultCard result={result} />
        </div>
      )}
    </div>
  );
}
