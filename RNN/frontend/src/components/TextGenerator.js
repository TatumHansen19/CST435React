import React, { useState } from 'react';
import { rnnApi as api } from "../../services/rnnApi.js";
import './TextGenerator.css';

const TextGenerator = () => {
  const [seedText, setSeedText] = useState('the');
  const [numWords, setNumWords] = useState(20);
  const [temperature, setTemperature] = useState(1.0);
  const [generatedText, setGeneratedText] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleGenerate = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setGeneratedText('');

    try {
      const result = await api.generateText(seedText, numWords, temperature);
      setGeneratedText(result.generated_text);
    } catch (err) {
      setError(`Error: ${err.message || 'Failed to generate text'}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="text-generator">
      <h2>🎯 Text Generator</h2>
      
      <form onSubmit={handleGenerate}>
        <div className="form-group">
          <label htmlFor="seedText">Seed Text:</label>
          <input
            id="seedText"
            type="text"
            value={seedText}
            onChange={(e) => setSeedText(e.target.value)}
            placeholder="Enter starting text..."
            maxLength={500}
          />
        </div>

        <div className="form-row">
          <div className="form-group">
            <label htmlFor="numWords">Words to Generate:</label>
            <input
              id="numWords"
              type="number"
              min="1"
              max="500"
              value={numWords}
              onChange={(e) => setNumWords(parseInt(e.target.value))}
            />
          </div>

          <div className="form-group">
            <label htmlFor="temperature">Temperature:</label>
            <div className="temperature-input">
              <input
                id="temperature"
                type="range"
                min="0.1"
                max="2.0"
                step="0.1"
                value={temperature}
                onChange={(e) => setTemperature(parseFloat(e.target.value))}
              />
              <span className="temp-value">{temperature.toFixed(1)}</span>
            </div>
            <small>
              {temperature < 0.7 ? '🧊 Conservative' : temperature < 1.3 ? '⚖️ Balanced' : '🔥 Creative'}
            </small>
          </div>
        </div>

        <button type="submit" disabled={loading}>
          {loading ? 'Generating...' : '✨ Generate Text'}
        </button>
      </form>

      {error && <div className="error-message">{error}</div>}

      {generatedText && (
        <div className="output">
          <h3>Generated Output:</h3>
          <div className="generated-text">
            <p>{generatedText}</p>
          </div>
          <button
            onClick={() => navigator.clipboard.writeText(generatedText)}
            className="copy-button"
          >
            📋 Copy to Clipboard
          </button>
        </div>
      )}
    </div>
  );
};

export default TextGenerator;
