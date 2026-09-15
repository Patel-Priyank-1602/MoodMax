import { useState } from 'react';
import TextInput from '../components/TextInput';
import ResultCard from '../components/ResultCard';
import EmotionChart from '../components/EmotionChart';
import SentimentGauge from '../components/SentimentGauge';
import { analyzeText, analyzeAspects } from '../api/client';
import './Analyzer.css';

const EMOTION_COLORS = {
  joy: '#F59E0B',
  sadness: '#3B82F6',
  anger: '#EF4444',
  fear: '#8B5CF6',
  surprise: '#EC4899',
  disgust: '#10B981',
  neutral: '#94A3B8',
};

export default function Analyzer() {
  const [result, setResult] = useState(null);
  const [aspectsData, setAspectsData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleAnalyze = async (text) => {
    setIsLoading(true);
    setError(null);
    setAspectsData(null);

    try {
      const [data, aspects] = await Promise.all([
        analyzeText(text),
        analyzeAspects(text).catch((err) => {
          console.warn('Aspect analysis failed/skipped:', err);
          return null;
        }),
      ]);
      setResult(data);
      setAspectsData(aspects);
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

          {aspectsData && aspectsData.has_multiple_aspects && (
            <div className="absa-section card animate-fade-in-up" id="absa-results">
              <div className="absa-header">
                <div className="absa-title-group">
                  <span className="absa-icon-wrap">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#267F4B" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/>
                      <polyline points="3.27 6.96 12 12.01 20.73 6.96"/>
                      <line x1="12" y1="22.08" x2="12" y2="12"/>
                    </svg>
                  </span>
                  <div>
                    <h3 className="absa-title">Aspect-Based Breakdown (ABSA)</h3>
                    <p className="absa-subtitle">
                      Detected {aspectsData.aspects.length} distinct targets with localized emotional tone
                    </p>
                  </div>
                </div>
                <div className="absa-overall-badge">
                  <span className="absa-overall-label">Synthesized:</span>
                  <span className={`sentiment-badge sentiment-${aspectsData.overall_sentiment.toLowerCase()}`}>
                    {aspectsData.overall_sentiment}
                  </span>
                </div>
              </div>

              <div className="absa-grid">
                {aspectsData.aspects.map((item, idx) => (
                  <div key={idx} className="absa-card">
                    <div className="absa-card-header">
                      <span className="absa-target-pill">{item.aspect}</span>
                      <span className={`sentiment-badge sentiment-${item.sentiment_label.toLowerCase()}`}>
                        {item.sentiment_label} ({(item.sentiment_score * 100).toFixed(0)}%)
                      </span>
                    </div>
                    <blockquote className="absa-clause">
                      "{item.clause_text}"
                    </blockquote>
                    <div className="absa-card-footer">
                      <span className="absa-emotion-tag">
                        <span
                          className="emotion-dot"
                          style={{
                            background: EMOTION_COLORS[item.dominant_emotion?.toLowerCase()] || '#267F4B',
                          }}
                        />
                        {item.dominant_emotion ? item.dominant_emotion.charAt(0).toUpperCase() + item.dominant_emotion.slice(1) : 'Neutral'}
                      </span>
                      {item.correction_applied && (
                        <span className="absa-corrected-tag" title={item.correction_reason}>
                          ⚡ Corrected
                        </span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
