import React, { useState, useEffect } from 'react';
import TextGenerator from './components/TextGenerator';
import ModelInfo from './components/ModelInfo';
import { rnnApi as api } from './services/rnnApi';

import './App.css';

function App() {
  const [isConnected, setIsConnected] = useState(false);
  const [connectionError, setConnectionError] = useState('');

  useEffect(() => {
    // Check API connection on app load
    checkConnection();
    // Recheck every 30 seconds
    const interval = setInterval(checkConnection, 30000);
    return () => clearInterval(interval);
  }, []);

  const checkConnection = async () => {
    try {
      await api.healthCheck();
      setIsConnected(true);
      setConnectionError('');
    } catch (error) {
      setIsConnected(false);
      setConnectionError('Cannot connect to backend API. Make sure the server is reachable via /api');

    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-content">
          <h1>🧠 RNN Text Generator</h1>
          <p>Generate creative text using LSTM Neural Networks</p>
        </div>
        <div className={`status-indicator ${isConnected ? 'connected' : 'disconnected'}`}>
          <span className="status-dot"></span>
          <span className="status-text">{isConnected ? 'Connected' : 'Disconnected'}</span>
        </div>
      </header>

      <main className="app-main">
        {connectionError && (
          <div className="connection-alert">
            <span className="alert-icon">⚠️</span>
            <div className="alert-content">
              <strong>Connection Issue</strong>
              <p>{connectionError}</p>
              <button onClick={checkConnection}>Retry Connection</button>
            </div>
          </div>
        )}

        <div className="content-grid">
          <section className="main-section">
            <TextGenerator />
          </section>

          <aside className="info-section">
            <ModelInfo />
          </aside>
        </div>
      </main>

      <footer className="app-footer">
        <p>RNN Text Generator • CST-435 • Powered by TensorFlow & FastAPI</p>
      </footer>
    </div>
  );
}

export default App;
