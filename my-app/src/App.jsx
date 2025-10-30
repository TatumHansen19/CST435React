import React from "react";
export default function App() {
  const [msg, setMsg] = React.useState("...loading");
  React.useEffect(() => {
    fetch("/api/hello").then(r=>r.json()).then(d=>setMsg(d.message)).catch(()=>setMsg("API unreachable"));
  }, []);
  return (
    <main style={{ padding: 24, fontFamily: "system-ui" }}>
      <h1>Vercel (React) + Railway API ✅</h1>
      <p>Message from API: <b>{msg}</b></p>
      <p><a href="/api/health" target="_blank" rel="noreferrer">Health check</a></p>
    </main>
  );
}
