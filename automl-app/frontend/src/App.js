import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Papa from 'papaparse';
import 'bootstrap/dist/css/bootstrap.min.css';
import Leaderboard from './Leaderboard';

// Models for specific tasks (Teammate's Feature)
const MODELS_CONFIG = {
  "Classification": ["Logistic Regression", "Random Forest", "SVM", "KNN", "XGBoost"],
  "Regression": ["Linear Regression", "Ridge", "Lasso", "Random Forest", "XGBoost", "Decision Tree"]
};

function App() {
  const [darkMode, setDarkMode] = useState(() => localStorage.getItem('theme') === 'dark');
  const [file, setFile] = useState(null);
  const [target, setTarget] = useState('');
  const [columns, setColumns] = useState([]);

  // Stats & Task State
  const [dataStats, setDataStats] = useState(null);
  const [detectedTaskType, setDetectedTaskType] = useState("Classification"); // Default
  const [selectedModels, setSelectedModels] = useState(MODELS_CONFIG["Classification"]);

  // Async Task State (Polling)
  const [taskId, setTaskId] = useState(null);
  const [progress, setProgress] = useState(0);
  const [logs, setLogs] = useState([]);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // Theme Effect
  useEffect(() => {
    document.documentElement.setAttribute('data-bs-theme', darkMode ? 'dark' : 'light');
    localStorage.setItem('theme', darkMode ? 'dark' : 'light');
  }, [darkMode]);

  // --- POLLING LOGIC (Crucial for new Backend) ---
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

  // --- FILE HANDLING ---
  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    setColumns([]);
    setDataStats(null);
    setTarget('');

    if (selectedFile) {
      let fileSize = selectedFile.size / 1024;
      let sizeString = fileSize < 1024 ? `${fileSize.toFixed(1)} KB` : `${(fileSize / 1024).toFixed(2)} MB`;

      Papa.parse(selectedFile, {
        header: true,
        preview: 500, // Preview first 500 rows for fast detection
        skipEmptyLines: true,
        complete: (res) => {
          if (res.meta && res.meta.fields) {
            setColumns(res.meta.fields);
            setDataStats({
              rows: "Calculating...", // Full count requires full parse, simplified here
              cols: res.meta.fields.length,
              size: sizeString
            });
          }
          // Auto-detect Task Type based on target (if target was selected immediately)
          // Real logic happens when target changes below
        }
      });
    }
  };

  // --- TASK DETECTION LOGIC ---
  useEffect(() => {
    // When target changes, we need to guess if it's Regression or Classification
    // Note: Since we don't have full data in state to save RAM, we rely on Backend to confirm.
    // But we can do a simple UI switch based on user preference if needed.
    // For now, reset models to default of current task type.
    setSelectedModels(MODELS_CONFIG[detectedTaskType]);
  }, [detectedTaskType]);

  const handleModelToggle = (model) => {
    setSelectedModels(prev =>
      prev.includes(model) ? prev.filter(m => m !== model) : [...prev, model]
    );
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file || !target) { setError('Please provide file and target column.'); return; }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('target', target);
    formData.append('models', JSON.stringify(selectedModels));

    setLoading(true);
    setError('');
    setResults(null);
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

  return (
    <div className="container mt-5 mb-5">
      <div className="d-flex justify-content-between align-items-center mb-4">
        <h1 className="mb-0">🤖 AutoML Model Comparator</h1>

        {/* YOUR TOGGLE SWITCH */}
        <div onClick={() => setDarkMode(!darkMode)} style={{ width: '60px', height: '30px', backgroundColor: darkMode ? '#6610f2' : '#ccc', borderRadius: '30px', position: 'relative', cursor: 'pointer', transition: 'background-color 0.3s ease' }}>
          <div style={{ width: '26px', height: '26px', backgroundColor: '#fff', borderRadius: '50%', position: 'absolute', top: '2px', left: darkMode ? '32px' : '2px', transition: 'left 0.3s cubic-bezier(0.68, -0.55, 0.27, 1.55)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '14px' }}>
            {darkMode ? '🌙' : '☀️'}
          </div>
        </div>
      </div>

      <div className="card p-4 shadow-sm mb-4">
        <form onSubmit={handleSubmit}>
          <div className="row">
            <div className="col-md-6 mb-3">
              <label className="form-label fw-bold">1. Upload Dataset (CSV)</label>
              <input type="file" className="form-control" accept=".csv" onChange={handleFileChange} />
              {/* YOUR STATS BADGE */}
              {dataStats && (
                <div className={`mt-2 p-2 rounded border d-inline-block ${darkMode ? 'bg-secondary bg-opacity-25 border-secondary' : 'bg-light border'}`}>
                  <small className="fw-bold text-success">📊 Data Stats: </small>
                  <span className="ms-2 small">{dataStats.cols} Columns <span className="mx-2 text-muted">|</span> {dataStats.size}</span>
                </div>
              )}
            </div>
            <div className="col-md-6 mb-3">
              <label className="form-label fw-bold">2. Target Column</label>
              <input type="text" className="form-control" list="cols" value={target} onChange={e => setTarget(e.target.value)} disabled={!file} placeholder="Type or select..." />
              <datalist id="cols">{columns.map(c => <option key={c} value={c} />)}</datalist>
            </div>
          </div>

          {/* DYNAMIC MODEL SELECTION */}
          <div className="mb-3">
            <div className="d-flex justify-content-between align-items-center mb-2">
              <label className="form-label fw-bold">3. Select Models</label>
              <select className="form-select w-auto form-select-sm" value={detectedTaskType} onChange={(e) => setDetectedTaskType(e.target.value)}>
                <option value="Classification">Classification</option>
                <option value="Regression">Regression</option>
              </select>
            </div>
            <div className={`card card-body ${darkMode ? 'bg-secondary bg-opacity-10' : 'bg-light'}`}>
              <div className="d-flex flex-wrap gap-3">
                {MODELS_CONFIG[detectedTaskType].map(m => (
                  <div className="form-check" key={m}>
                    <input className="form-check-input" type="checkbox" checked={selectedModels.includes(m)} onChange={() => handleModelToggle(m)} />
                    <label className="form-check-label" style={{ cursor: 'pointer' }} onClick={() => handleModelToggle(m)}>{m}</label>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <button type="submit" className="btn btn-primary w-100" disabled={loading || !file || !target}>
            {loading ? `⏳ Pipeline Running... ${progress}%` : '🚀 Launch Pipeline'}
          </button>
        </form>

        {/* LIVE LOGS */}
        {loading && (
          <div className="mt-4">
            <div className="progress" style={{ height: '25px' }}>
              <div className="progress-bar progress-bar-striped progress-bar-animated bg-success" style={{ width: `${progress}%` }}>{progress}%</div>
            </div>
            <div className={`mt-2 p-2 rounded font-monospace small ${darkMode ? 'bg-dark text-light' : 'bg-light text-dark'}`} style={{ maxHeight: '100px', overflowY: 'auto' }}>
              {logs.map((l, i) => <div key={i}>{l}</div>)}
            </div>
          </div>
        )}

        {error && <div className="alert alert-danger mt-3">{error}</div>}
      </div>

      {results && <Leaderboard results={results} darkMode={darkMode} />}
    </div>
  );
}

export default App;