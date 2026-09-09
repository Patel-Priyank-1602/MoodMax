/**
 * MoodMax API Client
 * Handles all backend communication.
 */

import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 60000, // 60s for batch processing
});

/**
 * POST /api/analyze — Analyze a single text
 */
export async function analyzeText(text) {
  const response = await api.post('/api/analyze', { text });
  return response.data;
}

/**
 * POST /api/analyze/batch — Upload CSV for batch analysis
 */
export async function analyzeBatch(file) {
  const formData = new FormData();
  formData.append('file', file);
  const response = await api.post('/api/analyze/batch', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
}

/**
 * GET /api/batch/:id — Get batch job status
 */
export async function getBatchStatus(batchJobId) {
  const response = await api.get(`/api/batch/${batchJobId}`);
  return response.data;
}

/**
 * GET /api/history — Get paginated analysis history
 */
export async function getHistory({ limit = 20, offset = 0, search = '' } = {}) {
  const params = { limit, offset };
  if (search) params.search = search;
  const response = await api.get('/api/history', { params });
  return response.data;
}

/**
 * GET /api/history/:id — Get single analysis detail
 */
export async function getAnalysisDetail(id) {
  const response = await api.get(`/api/history/${id}`);
  return response.data;
}

/**
 * GET /api/analytics/summary — Get analytics dashboard data
 */
export async function getAnalyticsSummary(batchJobId = null) {
  const params = {};
  if (batchJobId) params.batch_job_id = batchJobId;
  const response = await api.get('/api/analytics/summary', { params });
  return response.data;
}

/**
 * GET /health — Backend health check
 */
export async function healthCheck() {
  const response = await api.get('/health');
  return response.data;
}

export default api;
