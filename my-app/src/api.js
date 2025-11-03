// my-app/src/api.js
import axios from "axios";

// Ensure API_BASE ends without trailing slash
const API_BASE = (import.meta.env.VITE_API_BASE ?? "http://localhost:8000/api").replace(/\/+$/, "");

export const http = axios.create({
  baseURL: API_BASE,
  withCredentials: true,
});

// ✅ remove /api prefix from all endpoints below

export async function healthCheck() {
  const { data } = await http.get("/health");
  return data;
}

export async function getModelInfo() {
  const { data } = await http.get("/model-info");
  return data;
}

export async function generateText(payload) {
  const { data } = await http.post("/generate", payload);
  return data;
}
