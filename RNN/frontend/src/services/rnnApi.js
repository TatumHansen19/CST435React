// RNN/frontend/src/services/rnnApi.js
const base = (process.env.REACT_APP_API_BASE || "/api").replace(/\/+$/, "");

async function get(path) {
  const res = await fetch(`${base}${path}`, {
    method: "GET",
    headers: { "Content-Type": "application/json" },
  });
  if (!res.ok) throw new Error(`GET ${path} failed: ${res.status}`);
  return res.json();
}

async function post(path, body) {
  const res = await fetch(`${base}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`POST ${path} failed: ${res.status}`);
  return res.json();
}

export const rnnApi = {
  health: () => get("/api/health"),
  generate: (payload) => post("/api/generate", payload),
};
