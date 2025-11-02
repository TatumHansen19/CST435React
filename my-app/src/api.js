// my-app/src/api.js
import axios from "axios";

const API_BASE = (import.meta?.env?.VITE_API_BASE ?? "http://localhost:8000").replace(/\/+$/, "");

export const http = axios.create({
  baseURL: API_BASE,
  withCredentials: true,
});

export async function healthCheck() {
  const { data } = await http.get("/api/health");
  return data;
}

export async function getModelInfo() {
  const { data } = await http.get("/api/model-info");
  return data;
}

export async function generateText(payload) {
  const { data } = await http.post("/api/generate", payload);
  return data;
}
