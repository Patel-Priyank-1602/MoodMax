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

  return (
    <div className="analyzer-page" id="analyzer-page">
      <div className="page-header">
        <h1 className="page-title">Analyze Text</h1>
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
        <div className="analyzer-error glass-card animate-fade-in-up">
          <span className="error-icon">⚠️</span>
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

              {/* Quick stats */}
              <div className="quick-stats glass-card animate-fade-in-up">
                <div className="stat-item">
                  <span className="stat-value">{result.detected_lang?.toUpperCase()}</span>
                  <span className="stat-label">Language</span>
                </div>
                <div className="stat-divider"></div>
                <div className="stat-item">
                  <span className="stat-value">
                    {Object.entries(result.emotions)
                      .filter(([, s]) => s > 0.1).length}
                  </span>
                  <span className="stat-label">Active Emotions</span>
                </div>
                <div className="stat-divider"></div>
                <div className="stat-item">
                  <span className="stat-value">#{result.id}</span>
                  <span className="stat-label">Analysis ID</span>
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
