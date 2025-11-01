// RNN/backend/server.js
const express = require("express");
const cors = require("cors");

const app = express();
app.use(express.json());

// Allow your Vercel app + local dev
app.use(
  cors({
    origin: [
      "https://cst-435-react-b7fbctswi-tatums-projects-965c11b1.vercel.app",
      "https://cst-435-react.vercel.app",
      "http://localhost:5173"
    ],
    methods: ["GET", "POST", "OPTIONS"],
    allowedHeaders: ["Content-Type", "Authorization"]
  })
);

// ---- ROUTES ----

// Health (frontend probes '/'; we also expose '/health')
app.get("/", (req, res) =>
  res.json({ status: "healthy", service: "rnn-api" })
);
app.get("/health", (req, res) =>
  res.json({ status: "healthy", service: "rnn-api" })
);

// Model info — support BOTH spellings
app.get("/model/info", (req, res) =>
  res.json({ name: "RNN-LSTM", backend: "node", vocab_size: 30000 })
);
app.get("/model-info", (req, res) =>
  res.json({ name: "RNN-LSTM", backend: "node", vocab_size: 30000 })
);

// Generate — stub (replace with your real generator when ready)
app.post("/generate", (req, res) => {
  const { seed_text = "", num_words = 20, temperature = 1.0 } = req.body || {};
  const text = `${seed_text} ...generated (${num_words} words @ temp=${temperature})`;
  res.json({ text });
});

// Stats
app.get("/stats", (req, res) => {
  res.json({ uptime_s: Math.round(process.uptime()) });
});

// Tiny 1x1 PNG for /visualizations/training (stub)
app.get("/visualizations/training", (req, res) => {
  const png = Buffer.from(
    "89504E470D0A1A0A0000000D49484452000000010000000108060000001F15C4890000000A49444154789C6360000002000154A24F920000000049454E44AE426082",
    "hex"
  );
  res.setHeader("Content-Type", "image/png");
  res.send(png);
});

const PORT = process.env.PORT || 8080; // Railway provides PORT
app.listen(PORT, () => console.log(`API listening on :${PORT}`));
