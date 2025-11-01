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

export const api = {
  // Health check — your backend’s health is served at root (“/”).
  // With baseURL '/api', this calls '/api/' which Vercel will rewrite to Railway '/'.
  healthCheck: async () => {
    const res = await apiClient.get('/');
    return res.data;
  },

  // Model info
  getModelInfo: async () => {
    const res = await apiClient.get('/model-info');
    return res.data;
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
    const res = await apiClient.get('/stats');
    return res.data;
  },

  // Training plot image
  getTrainingPlot: async () => {
    const res = await apiClient.get('/visualizations/training', { responseType: 'blob' });
    return URL.createObjectURL(res.data);
  },
};

export default api;
