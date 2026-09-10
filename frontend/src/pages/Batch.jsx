import { useState, useCallback } from 'react';
import FileUploader from '../components/FileUploader';
import { analyzeBatch, getBatchStatus, getAnalyticsSummary } from '../api/client';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
  PieChart, Pie,
} from 'recharts';
import './Batch.css';

const SENTIMENT_COLORS = {
  Positive: '#10B981',
  Negative: '#EF4444',
  Neutral: '#94A3B8',
};

const EMOTION_COLORS = {
  joy: '#F59E0B',
  sadness: '#3B82F6',
  anger: '#EF4444',
  fear: '#8B5CF6',
  surprise: '#EC4899',
  disgust: '#10B981',
  neutral: '#94A3B8',
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
        const analytics = await getAnalyticsSummary(jobId);
        setSummary(analytics);
        setIsLoading(false);
      } else if (status.status === 'failed') {
        setError('Batch job failed. Check the backend logs.');
        setIsLoading(false);
      } else {
        setTimeout(() => pollStatus(jobId), 2000);
      }
    } catch {
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
    name, value, fill: SENTIMENT_COLORS[name] || '#94A3B8',
  })) : [];

  const emotionData = summary ? Object.entries(summary.emotion_breakdown)
    .map(([name, value]) => ({
      name: name.charAt(0).toUpperCase() + name.slice(1),
      value,
      fill: EMOTION_COLORS[name.toLowerCase()] || '#267F4B',
    }))
    .sort((a, b) => b.value - a.value) : [];

  return (
    <div className="batch-page" id="batch-page">
      <div className="page-header">
        <h1 className="page-title">Batch <span className="title-accent">Analysis</span></h1>
        <p className="page-subtitle">
          Upload a CSV file with a "text" column for bulk sentiment & emotion analysis.
        </p>
      </div>

      <FileUploader onUpload={handleUpload} isLoading={isLoading} />

      {error && (
        <div className="batch-error card animate-fade-in-up">
          <svg className="error-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10"/>
            <line x1="12" y1="8" x2="12" y2="12"/>
            <line x1="12" y1="16" x2="12.01" y2="16"/>
          </svg>
          <span>{error}</span>
        </div>
      )}

      {batchJob && (
        <div className="batch-status card animate-fade-in-up">
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
            <div className="dashboard-stat card stat-total">
              <span className="dashboard-stat-icon">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#267F4B" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="18" y1="20" x2="18" y2="10"/>
                  <line x1="12" y1="20" x2="12" y2="4"/>
                  <line x1="6" y1="20" x2="6" y2="14"/>
                </svg>
              </span>
              <span className="dashboard-stat-value stat-val-total">{summary.total_analyses}</span>
              <span className="dashboard-stat-label">Total Analyzed</span>
            </div>
            <div className="dashboard-stat card stat-pos">
              <span className="dashboard-stat-icon">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#10B981" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/>
                  <polyline points="17 6 23 6 23 12"/>
                </svg>
              </span>
              <span className="dashboard-stat-value stat-val-pos">{summary.sentiment_breakdown.Positive || 0}</span>
              <span className="dashboard-stat-label">Positive</span>
            </div>
            <div className="dashboard-stat card stat-neg">
              <span className="dashboard-stat-icon">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#EF4444" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="23 18 13.5 8.5 8.5 13.5 1 6"/>
                  <polyline points="17 18 23 18 23 12"/>
                </svg>
              </span>
              <span className="dashboard-stat-value stat-val-neg">{summary.sentiment_breakdown.Negative || 0}</span>
              <span className="dashboard-stat-label">Negative</span>
            </div>
            <div className="dashboard-stat card stat-lang">
              <span className="dashboard-stat-icon">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#8B5CF6" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="12" cy="12" r="10"/>
                  <line x1="2" y1="12" x2="22" y2="12"/>
                  <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/>
                </svg>
              </span>
              <span className="dashboard-stat-value stat-val-lang">{Object.keys(summary.top_languages).length}</span>
              <span className="dashboard-stat-label">Languages</span>
            </div>
          </div>

          {/* Charts */}
          <div className="dashboard-charts">
            <div className="card chart-card animate-fade-in-up">
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
                  <Tooltip
                    contentStyle={{
                      background: '#FFFFFF',
                      border: '1px solid #E5E7F2',
                      borderRadius: '10px',
                      boxShadow: '0 4px 20px rgba(30, 41, 59, 0.08)',
                      color: '#172033',
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>

            <div className="card chart-card animate-fade-in-up">
              <h3 className="chart-card-title">Emotion Distribution</h3>
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={emotionData} margin={{ left: -10, right: 10 }}>
                  <XAxis dataKey="name" tick={{ fill: '#64748B', fontSize: 11, fontWeight: 500 }} />
                  <YAxis tick={{ fill: '#64748B', fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{
                      background: '#FFFFFF',
                      border: '1px solid #E5E7F2',
                      borderRadius: '10px',
                      boxShadow: '0 4px 20px rgba(30, 41, 59, 0.08)',
                      color: '#172033',
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
            <div className="card lang-card animate-fade-in-up">
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
