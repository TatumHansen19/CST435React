// frontend/src/services/api.js
import axios from 'axios';

// Use env var if set, otherwise call relative /api (rewritten by Vercel)
const API_BASE_URL =
  (typeof import.meta !== 'undefined' && import.meta.env?.VITE_API_BASE) ||
  process.env.REACT_APP_API_BASE ||
  '/api';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: { 'Content-Type': 'application/json' },
});

async function tryGet(pathA, pathB) {
  try {
    const r = await apiClient.get(pathA);
    return r.data;
  } catch (e) {
    if (!pathB) throw e;
    const r2 = await apiClient.get(pathB); // fallback
    return r2.data;
  }
}

export const api = {
  // Health check — try '/' then '/health'
  healthCheck: async () => {
    return tryGet('/', '/health');
  },

  // Model info — try '/model/info' then '/model-info'
  getModelInfo: async () => {
    return tryGet('/model/info', '/model-info');
  },

  // Generate text
  generateText: async (seedText, numWords = 50, temperature = 1.0) => {
    const res = await apiClient.post('/generate', {
      seed_text: seedText,
      num_words: numWords,
      temperature,
    });
    return res.data;
  },

  // Stats
  getStats: async () => {
    return tryGet('/stats');
  },

  // Training plot
  getTrainingPlot: async () => {
    const res = await apiClient.get('/visualizations/training', { responseType: 'blob' });
    return URL.createObjectURL(res.data);
  },
};

export default api;
