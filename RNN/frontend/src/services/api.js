// frontend/src/api.js
import axios from 'axios';

const API_BASE_URL =
  import.meta.env?.VITE_API_BASE ||
  process.env.REACT_APP_API_BASE ||
  '/api';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const api = {
  // Health check
  healthCheck: async () => {
    const response = await apiClient.get('/health');
    return response.data;
  },

  // Get model info
  getModelInfo: async () => {
    const response = await apiClient.get('/model-info');
    return response.data;
  },

  // Generate text
  generateText: async (seedText, numWords = 50, temperature = 1.0) => {
    const response = await apiClient.post('/generate', {
      seed_text: seedText,
      num_words: numWords,
      temperature: temperature,
    });
    return response.data;
  },

  // Get stats
  getStats: async () => {
    const response = await apiClient.get('/stats');
    return response.data;
  },

  // Get training plot visualization
  getTrainingPlot: async () => {
    const response = await apiClient.get('/visualizations/training', {
      responseType: 'blob',
    });
    return URL.createObjectURL(response.data);
  },
};

export default api;
