import express from "express";
import cors from "cors";

const app = express();
const PORT = process.env.PORT || 8080;

app.use(cors({
  origin: [
    "https://cst-435-react.vercel.app",
    "https://cst-435-react-b7fbctswi-tatums-projects-965c11b1.vercel.app",
    "http://localhost:5173"
  ],
  methods: ["GET","POST","OPTIONS"],
  allowedHeaders: ["Content-Type","Authorization"]
}));
app.use(express.json());

// --- HEALTH ---
app.get("/api/", (req, res) => res.json({ status: "healthy", service: "rnn-api" }));
app.get("/api",  (req, res) => res.json({ status: "healthy", service: "rnn-api" })); // <— added
app.get("/api/health", (req, res) => res.json({ status: "healthy", service: "rnn-api" }));

// --- MODEL INFO (support both styles your UI might use) ---
app.get("/api/model-info", (req, res) => {
  res.json({ name: "RNN-LSTM", backend: "node", vocab_size: 30000 });
});
app.get("/api/model/info", (req, res) => {
  res.json({ name: "RNN-LSTM", backend: "node", vocab_size: 30000 });
});

// --- GENERATE ---
app.post("/api/generate", (req, res) => {
  const { seed_text = "", num_words = 20, temperature = 1.0 } = req.body || {};
  const text = `${seed_text} ...generated (${num_words} @ temp=${temperature})`;
  res.json({ text });
});

// --- OPTIONAL: stats & tiny placeholder plot ---
app.get("/api/stats", (req, res) => res.json({ uptime_s: Math.round(process.uptime()) }));
app.get("/api/visualizations/training", (req, res) => {
  const png = Buffer.from(
    "89504E470D0A1A0A0000000D49484452000000010000000108060000001F15C4890000000A49444154789C6360000002000154A24F920000000049454E44AE426082",
    "hex"
  );
  res.setHeader("Content-Type", "image/png");
  res.send(png);
});

// Optional hello endpoint you had:
app.get("/api/hello", (req, res) => res.json({ message: "Hello from Railway API!" }));

app.listen(PORT, () => console.log(`API listening on :${PORT}`));
