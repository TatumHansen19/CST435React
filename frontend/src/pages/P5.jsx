import { useState } from "react";
import { modelInfo, generate } from "../api/p5.js";

export default function P5_RNN() {
  const [seedText, setSeedText] = useState("the");
  const [numWords, setNumWords] = useState(20);
  const [temperature, setTemperature] = useState(0.9);
  const [topK, setTopK] = useState(40);
  const [topP, setTopP] = useState(0.9);
  const [useBeam, setUseBeam] = useState(true);
  const [beamWidth, setBeamWidth] = useState(3);
  const [output, setOutput] = useState("");
  const [loading, setLoading] = useState(false);
  const [info, setInfo] = useState(null);
  const [err, setErr] = useState("");

  async function fetchInfo() {
    setErr("");
    try {
      const i = await modelInfo();
      setInfo(i);
    } catch (e) {
      setErr(e.message || "Failed to fetch model info");
    }
  }

  async function onGenerate(e) {
    e.preventDefault();
    setLoading(true);
    setErr("");
    setOutput("");
    try {
      const res = await generate({
        seed_text: seedText,
        num_words: Number(numWords),
        temperature: Number(temperature),
        top_k: Number(topK),
        top_p: Number(topP),
        use_beam_search: Boolean(useBeam),
        beam_width: Number(beamWidth)
      });
      setOutput(res.generated_text);
    } catch (e) {
      setErr(e.message || "Generation failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="panel">
      <h1>Project 5 — RNN Text Generator</h1>
      <div className="row">
        <form className="form" onSubmit={onGenerate}>
          <label>Seed Text</label>
          <textarea value={seedText} onChange={(e) => setSeedText(e.target.value)} rows={4} />

          <div className="grid-2">
            <div>
              <label>Words</label>
              <input type="number" min="1" max="500" value={numWords} onChange={e => setNumWords(e.target.value)} />
            </div>
            <div>
              <label>Temperature</label>
              <input type="number" step="0.05" min="0.1" max="2.0" value={temperature} onChange={e => setTemperature(e.target.value)} />
            </div>
          </div>

          <div className="grid-2">
            <div>
              <label>Top-K</label>
              <input type="number" min="0" max="200" value={topK} onChange={e => setTopK(e.target.value)} />
            </div>
            <div>
              <label>Top-P</label>
              <input type="number" step="0.01" min="0" max="1" value={topP} onChange={e => setTopP(e.target.value)} />
            </div>
          </div>

          <div className="grid-2">
            <div className="checkbox">
              <input id="beam" type="checkbox" checked={useBeam} onChange={e => setUseBeam(e.target.checked)} />
              <label htmlFor="beam">Use Beam Search</label>
            </div>
            <div>
              <label>Beam Width</label>
              <input type="number" min="1" max="10" value={beamWidth} onChange={e => setBeamWidth(e.target.value)} />
            </div>
          </div>

          <button type="submit" disabled={loading}>{loading ? "Generating..." : "Generate"}</button>
          <button type="button" onClick={fetchInfo} className="secondary">Model Info</button>
        </form>

        <div className="output">
          <h3>Output</h3>
          {err && <div className="error">{err}</div>}
          <pre className="mono">{output || "—"}</pre>

          {info && (
            <>
              <h3>Model Info</h3>
              <pre className="mono">
                {JSON.stringify(info, null, 2)}
              </pre>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
