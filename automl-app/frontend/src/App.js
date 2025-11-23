import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Papa from 'papaparse';
import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css'; // Make sure to import the new CSS!
import Leaderboard from './Leaderboard';

const MODELS_CONFIG = {
  "Classification": ["Logistic Regression", "Random Forest", "SVM", "KNN", "XGBoost"],
  "Regression": ["Linear Regression", "Ridge", "Lasso", "Random Forest", "XGBoost", "Decision Tree"]
};

function App() {
  const [darkMode, setDarkMode] = useState(() => localStorage.getItem('theme') === 'dark');
  const [file, setFile] = useState(null);
  const [target, setTarget] = useState('');
  const [columns, setColumns] = useState([]);
  const [dataStats, setDataStats] = useState(null);
  const [detectedTaskType, setDetectedTaskType] = useState("Classification");
  const [selectedModels, setSelectedModels] = useState(MODELS_CONFIG["Classification"]);
  const [taskId, setTaskId] = useState(null);
  const [completedTaskId, setCompletedTaskId] = useState(null); // Store task ID even after completion
  const [progress, setProgress] = useState(0);
  const [logs, setLogs] = useState([]);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    document.documentElement.setAttribute('data-bs-theme', darkMode ? 'dark' : 'light');
    localStorage.setItem('theme', darkMode ? 'dark' : 'light');
  }, [darkMode]);

  // --- POLLING & FILE HANDLING (Keep your existing logic here) ---
  // I am hiding the logic functions for brevity, paste your existing handleFileChange/useEffect/Submit here
  // ... [PASTE LOGIC HERE] ...

  // --- COPY-PASTE LOGIC START ---
  useEffect(() => {
    let interval = null;
    if (taskId) {
      interval = setInterval(async () => {
        try {
          const res = await axios.get(`http://127.0.0.1:5000/status/${taskId}`);
          const data = res.data;
          setProgress(data.progress);
          setLogs(data.logs || []);

          if (data.status === 'completed') {
            setResults(data.results);
            setCompletedTaskId(taskId); // Save task ID for code download
            setLoading(false);
            setTaskId(null); // Stop polling
          } else if (data.status === 'failed') {
            setError(data.error);
            setLoading(false);
            setTaskId(null);
          }
        } catch (err) {
          console.error("Polling Error:", err);
        }
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [taskId]);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile); setColumns([]); setTarget(''); setDataStats(null);
    if (selectedFile) {
      let sizeString = selectedFile.size / 1024 < 1024 ? `${(selectedFile.size / 1024).toFixed(1)} KB` : `${(selectedFile.size / (1024 * 1024)).toFixed(2)} MB`;
      Papa.parse(selectedFile, {
        header: true, preview: 1,
        complete: (res) => {
          if (res.meta && res.meta.fields) {
            setColumns(res.meta.fields);
            setDataStats({ rows: "Calculating...", cols: res.meta.fields.length, size: sizeString });
            // Fast Row Estimation Logic
            if (selectedFile.size < 50 * 1024) {
              Papa.parse(selectedFile, { header: true, complete: (r) => setDataStats(p => ({ ...p, rows: r.data.length.toLocaleString() })) });
            } else {
              const reader = new FileReader();
              reader.onload = (ev) => {
                const lines = ev.target.result.split('\n').length;
                const estimated = Math.floor(selectedFile.size / (50 * 1024 / lines));
                setDataStats(p => ({ ...p, rows: `~${estimated.toLocaleString()}` }));
              };
              reader.readAsText(selectedFile.slice(0, 50 * 1024));
            }
          }
        }
      });
    }
  };

  const handleModelToggle = (m) => setSelectedModels(prev => prev.includes(m) ? prev.filter(x => x !== m) : [...prev, m]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file || !target) return;
    const formData = new FormData();
    formData.append('file', file);
    formData.append('target', target);
    formData.append('models', JSON.stringify(selectedModels));

    setLoading(true);
    setError('');
    setResults(null);
    setCompletedTaskId(null); // Clear previous task ID
    setLogs(["🚀 Initializing Upload..."]);
    setProgress(0);

    try {
      const res = await axios.post('http://127.0.0.1:5000/upload', formData);
      setTaskId(res.data.task_id); // Start Polling
    } catch (err) {
      setError(err.response?.data?.error || 'Upload Failed');
      setLoading(false);
    }
  };

  useEffect(() => { setSelectedModels(MODELS_CONFIG[detectedTaskType]); }, [detectedTaskType]);
  // --- COPY-PASTE LOGIC END ---


  return (
    <div className="container py-5">
      {/* HEADER */}
      <div className="d-flex justify-content-between align-items-center mb-5">
        <div>
          <h1 className="fw-bold mb-0" style={{
            background: 'linear-gradient(90deg, #667eea 0%, #764ba2 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent'
          }}>
            AutoML <span className="fw-light text-muted" style={{ WebkitTextFillColor: darkMode ? '#ccc' : '#555' }}>Comparator</span>
          </h1>
          <p className="text-muted mb-0">Upload CSV, Select Target, Relax.</p>
        </div>

        {/* MODERN TOGGLE */}
        <div onClick={() => setDarkMode(!darkMode)} className="glass-card d-flex align-items-center justify-content-center"
          style={{ width: '50px', height: '50px', cursor: 'pointer' }}>
          <span style={{ fontSize: '1.5rem' }}>{darkMode ? '🌙' : '☀️'}</span>
        </div>
      </div>

      {/* MAIN INPUT CARD */}
      <div className="glass-card p-4 p-md-5 mb-5 animate__animated animate__fadeInUp">
        <form onSubmit={handleSubmit}>
          <div className="row g-4">
            {/* 1. File Upload */}
            <div className="col-md-6">
              <label className="form-label fw-bold text-uppercase small text-muted tracking-wide">1. Dataset Source</label>
              <div className="input-group">
                <input type="file" className="form-control" accept=".csv" onChange={handleFileChange} />
              </div>
            </div>

            {/* 2. Target Selection */}
            <div className="col-md-6">
              <label className="form-label fw-bold text-uppercase small text-muted tracking-wide">2. Target Variable</label>
              <input type="text" className="form-control" list="cols" value={target} onChange={e => setTarget(e.target.value)} disabled={!file} placeholder="Which column to predict?" />
              <datalist id="cols">{columns.map(c => <option key={c} value={c} />)}</datalist>
            </div>
          </div>

          {/* STATS BADGES */}
          {dataStats && (
            <div className="d-flex gap-3 mt-4 justify-content-start">
              <div className="px-3 py-2 rounded-3 bg-primary bg-opacity-10 border border-primary text-primary">
                <small className="d-block text-uppercase opacity-75" style={{ fontSize: '0.7rem' }}>Rows</small>
                <strong className="font-monospace">{dataStats.rows}</strong>
              </div>
              <div className="px-3 py-2 rounded-3 bg-success bg-opacity-10 border border-success text-success">
                <small className="d-block text-uppercase opacity-75" style={{ fontSize: '0.7rem' }}>Columns</small>
                <strong className="font-monospace">{dataStats.cols}</strong>
              </div>
              <div className="px-3 py-2 rounded-3 bg-info bg-opacity-10 border border-info text-info">
                <small className="d-block text-uppercase opacity-75" style={{ fontSize: '0.7rem' }}>Size</small>
                <strong className="font-monospace">{dataStats.size}</strong>
              </div>
            </div>
          )}

          <hr className="my-4 opacity-25" />

          {/* 3. MODEL SELECTION */}
          <div className="mb-4">
            <div className="d-flex justify-content-between align-items-center mb-3">
              <label className="form-label fw-bold text-uppercase small text-muted tracking-wide mb-0">3. Model Zoo</label>
              <select className="form-select w-auto form-select-sm py-1" value={detectedTaskType} onChange={(e) => setDetectedTaskType(e.target.value)}>
                <option value="Classification">🎯 Classification</option>
                <option value="Regression">📈 Regression</option>
              </select>
            </div>

            <div className="d-flex flex-wrap gap-2">
              {MODELS_CONFIG[detectedTaskType].map(m => (
                <div key={m} onClick={() => handleModelToggle(m)}
                  className={`px-3 py-2 rounded-pill border cursor-pointer transition-all ${selectedModels.includes(m)
                    ? 'bg-primary text-white border-primary shadow-sm'
                    : 'bg-transparent text-muted border-secondary'
                    }`}
                  style={{ cursor: 'pointer', fontSize: '0.9rem', transition: '0.2s' }}>
                  {selectedModels.includes(m) && <span className="me-2">✓</span>}
                  {m}
                </div>
              ))}
            </div>
          </div>

          <button type="submit" className="btn btn-primary w-100 py-3 shadow-lg" disabled={loading || !file || !target}>
            {loading ? (
              <span><span className="spinner-border spinner-border-sm me-2"></span> Processing Pipeline...</span>
            ) : (
              <span className="h5 mb-0">🚀 Launch Experiments</span>
            )}
          </button>
        </form>

        {/* LOGS & PROGRESS */}
        {loading && (
          <div className="mt-4 animate__animated animate__fadeIn">
            <div className="progress" style={{ height: '6px', borderRadius: '10px', backgroundColor: 'rgba(255,255,255,0.1)' }}>
              <div className="progress-bar bg-gradient-primary" style={{ width: `${progress}%`, transition: 'width 0.5s ease' }}></div>
            </div>
            <div className="mt-3 p-3 rounded bg-black bg-opacity-25 font-monospace small text-muted" style={{ maxHeight: '120px', overflowY: 'auto' }}>
              {logs.map((l, i) => <div key={i} className="mb-1"> {l}</div>)}
              <div className="text-primary blink">_</div>
            </div>
          </div>
        )}

        {error && <div className="alert alert-danger mt-3 rounded-3 border-0 shadow-sm">{error}</div>}
      </div>

      {results && <Leaderboard results={results} darkMode={darkMode} taskId={completedTaskId} />}
    </div>
  );
}

export default App;