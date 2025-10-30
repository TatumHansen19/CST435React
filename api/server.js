import express from "express";
import cors from "cors";

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

// keep the /api prefix to match Vercel rewrite
app.get("/api/health", (req, res) => res.json({ ok: true, ts: Date.now() }));
app.get("/api/hello", (req, res) => res.json({ message: "Hello from Railway API!" }));

app.listen(PORT, () => console.log(`API listening on :${PORT}`));
