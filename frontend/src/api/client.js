/**
 * MoodMax API Client
 * Pure stateless backend communication.
 */

import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 120000, // 120s for larger batch processing
});

/**
 * POST /api/analyze — Analyze a single text
 */
export async function analyzeText(text) {
  const response = await api.post('/api/analyze', { text });
  return response.data;
}

/**
 * POST /api/analyze/batch — Upload CSV for direct batch analysis
 * Returns { filename, summary, results } directly
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
 * GET /health — Backend health check
 */
export async function healthCheck() {
  const response = await api.get('/health');
  return response.data;
}

export default api;
