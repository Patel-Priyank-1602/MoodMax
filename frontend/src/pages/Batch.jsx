import { useState, useEffect, useCallback } from 'react';
import FileUploader from '../components/FileUploader';
import { analyzeBatch, getBatchStatus, getAnalyticsSummary } from '../api/client';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
  PieChart, Pie, Legend,
} from 'recharts';
import './Batch.css';

const SENTIMENT_COLORS = {
  Positive: '#00b894',
  Negative: '#ff7675',
  Neutral: '#636e72',
};

const EMOTION_COLORS = {
  joy: '#fdcb6e', sadness: '#74b9ff', anger: '#ff7675',
  fear: '#a29bfe', surprise: '#fd79a8', disgust: '#00b894', neutral: '#636e72',
};

export default function Batch() {
  const [isLoading, setIsLoading] = useState(false);
  const [batchJob, setBatchJob] = useState(null);
  const [summary, setSummary] = useState(null);
  const [error, setError] = useState(null);

  const pollStatus = useCallback(async (jobId) => {
    try {
      const status = await getBatchStatus(jobId);
      setBatchJob(status);

      if (status.status === 'done') {
        // Fetch analytics summary
        const analytics = await getAnalyticsSummary(jobId);
        setSummary(analytics);
        setIsLoading(false);
      } else if (status.status === 'failed') {
        setError('Batch job failed. Check the backend logs.');
        setIsLoading(false);
      } else {
        // Still processing, poll again
        setTimeout(() => pollStatus(jobId), 2000);
      }
    } catch (err) {
      setError('Failed to check batch status.');
      setIsLoading(false);
    }
  }, []);

  const handleUpload = async (file) => {
    setIsLoading(true);
    setError(null);
    setSummary(null);
    setBatchJob(null);

    try {
      const result = await analyzeBatch(file);
      setBatchJob(result);
      // Start polling
      pollStatus(result.batch_job_id);
    } catch (err) {
      console.error('Batch upload failed:', err);
      if (err.response?.data?.detail) {
        setError(err.response.data.detail);
      } else if (err.code === 'ERR_NETWORK') {
        setError('Cannot connect to backend. Make sure the server is running.');
      } else {
        setError('Batch upload failed. Please try again.');
      }
      setIsLoading(false);
    }
  };

  const sentimentData = summary ? Object.entries(summary.sentiment_breakdown).map(([name, value]) => ({
    name, value, fill: SENTIMENT_COLORS[name] || '#636e72',
  })) : [];

  const emotionData = summary ? Object.entries(summary.emotion_breakdown)
    .map(([name, value]) => ({
      name: name.charAt(0).toUpperCase() + name.slice(1),
      value,
      fill: EMOTION_COLORS[name] || '#6c5ce7',
    }))
    .sort((a, b) => b.value - a.value) : [];

  return (
    <div className="batch-page" id="batch-page">
      <div className="page-header">
        <h1 className="page-title">Batch Analysis</h1>
        <p className="page-subtitle">
          Upload a CSV file with a "text" column for bulk sentiment & emotion analysis.
        </p>
      </div>

      <FileUploader onUpload={handleUpload} isLoading={isLoading} />

      {error && (
        <div className="batch-error glass-card animate-fade-in-up">
          <span>⚠️</span>
          <span>{error}</span>
        </div>
      )}

      {batchJob && (
        <div className="batch-status glass-card animate-fade-in-up">
          <div className="batch-status-header">
            <h3 className="batch-status-title">
              Batch Job #{batchJob.batch_job_id || batchJob.id}
            </h3>
            <span className={`badge badge-${batchJob.status === 'done' ? 'positive' : batchJob.status === 'failed' ? 'negative' : 'neutral'}`}>
              {batchJob.status}
            </span>
          </div>
          <div className="batch-status-details">
            <div className="batch-stat">
              <span className="batch-stat-value">{batchJob.total_items}</span>
              <span className="batch-stat-label">Total Items</span>
            </div>
            {isLoading && (
              <div className="batch-progress">
                <div className="loading-bar"></div>
                <span className="batch-progress-text">Processing...</span>
              </div>
            )}
          </div>
        </div>
      )}

      {summary && (
        <div className="batch-dashboard stagger-children">
          {/* Summary stats */}
          <div className="dashboard-stats animate-fade-in-up">
            <div className="dashboard-stat glass-card">
              <span className="dashboard-stat-icon">📊</span>
              <span className="dashboard-stat-value">{summary.total_analyses}</span>
              <span className="dashboard-stat-label">Total Analyzed</span>
            </div>
            <div className="dashboard-stat glass-card">
              <span className="dashboard-stat-icon">😊</span>
              <span className="dashboard-stat-value">{summary.sentiment_breakdown.Positive || 0}</span>
              <span className="dashboard-stat-label">Positive</span>
            </div>
            <div className="dashboard-stat glass-card">
              <span className="dashboard-stat-icon">😔</span>
              <span className="dashboard-stat-value">{summary.sentiment_breakdown.Negative || 0}</span>
              <span className="dashboard-stat-label">Negative</span>
            </div>
            <div className="dashboard-stat glass-card">
              <span className="dashboard-stat-icon">🌍</span>
              <span className="dashboard-stat-value">{Object.keys(summary.top_languages).length}</span>
              <span className="dashboard-stat-label">Languages</span>
            </div>
          </div>

          {/* Charts */}
          <div className="dashboard-charts">
            <div className="glass-card chart-card animate-fade-in-up">
              <h3 className="chart-card-title">Sentiment Breakdown</h3>
              <ResponsiveContainer width="100%" height={280}>
                <PieChart>
                  <Pie
                    data={sentimentData}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={100}
                    paddingAngle={4}
                    dataKey="value"
                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                  >
                    {sentimentData.map((entry, index) => (
                      <Cell key={index} fill={entry.fill} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>

            <div className="glass-card chart-card animate-fade-in-up">
              <h3 className="chart-card-title">Emotion Distribution</h3>
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={emotionData} margin={{ left: -10, right: 10 }}>
                  <XAxis dataKey="name" tick={{ fill: '#9d9db5', fontSize: 11 }} />
                  <YAxis tick={{ fill: '#5a5a7a', fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{
                      background: 'rgba(26, 26, 46, 0.9)',
                      border: '1px solid rgba(108, 92, 231, 0.2)',
                      borderRadius: '8px',
                      color: '#e8e8f0',
                    }}
                  />
                  <Bar dataKey="value" radius={[6, 6, 0, 0]} barSize={32}>
                    {emotionData.map((entry, index) => (
                      <Cell key={index} fill={entry.fill} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Language breakdown */}
          {Object.keys(summary.top_languages).length > 0 && (
            <div className="glass-card lang-card animate-fade-in-up">
              <h3 className="chart-card-title">Languages Detected</h3>
              <div className="lang-grid">
                {Object.entries(summary.top_languages).map(([lang, count]) => (
                  <div key={lang} className="lang-item">
                    <span className="lang-code">{lang.toUpperCase()}</span>
                    <span className="lang-count">{count}</span>
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
