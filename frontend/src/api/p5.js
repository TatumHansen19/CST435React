async function http(method, path, body) {
  const res = await fetch(`/api/p5${path}`, {
    method,
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined
  });
  if (!res.ok) {
    let msg = `HTTP ${res.status}`;
    try {
      const j = await res.json();
      msg = j.error || j.detail || JSON.stringify(j);
    } catch {}
    throw new Error(msg);
  }
  return res.json();
}

export function health() {
  return http("GET", "/health");
}

export function modelInfo() {
  return http("GET", "/model-info");
}

export function generate(payload) {
  return http("POST", "/generate", payload);
}
