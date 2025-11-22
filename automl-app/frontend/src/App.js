import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Papa from 'papaparse';
import 'bootstrap/dist/css/bootstrap.min.css';
import Leaderboard from './Leaderboard';

// Models for specific tasks
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
  const [detectedTaskType, setDetectedTaskType] = useState("Classification");
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

  // --- POLLING LOGIC ---
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
            setTaskId(null);
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

  // --- FILE HANDLING (UPDATED FOR BETTER STATS) ---
  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    setColumns([]);
    setDataStats(null);
    setTarget('');

    if (selectedFile) {
      let fileSize = selectedFile.size / 1024;
      let sizeString = fileSize < 1024 ? `${fileSize.toFixed(1)} KB` : `${(fileSize / 1024).toFixed(2)} MB`;

      // 1. FAST PREVIEW: Get Columns & Init Stats immediately
      Papa.parse(selectedFile, {
        header: true,
        preview: 10, // Only parse first 10 rows for headers
        skipEmptyLines: true,
        complete: (res) => {
          if (res.meta && res.meta.fields) {
            setColumns(res.meta.fields);
            setDataStats({
              rows: "⏳", // Placeholder while worker counts
              cols: res.meta.fields.length,
              size: sizeString
            });
          }
        }
      });

      // 2. BACKGROUND WORKER: Count actual rows (Non-blocking)
      let rowCount = 0;
      Papa.parse(selectedFile, {
        worker: true, // Uses a separate thread
        header: true,
        skipEmptyLines: true,
        step: () => {
          rowCount++;
        },
        complete: () => {
          // Update state with final count
          setDataStats(prev => ({ ...prev, rows: rowCount.toLocaleString() }));
        }
      });
    }
  };

  // --- TASK DETECTION LOGIC ---
  useEffect(() => {
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
      setTaskId(res.data.task_id);
    } catch (err) {
      setError(err.response?.data?.error || 'Upload Failed');
      setLoading(false);
    }
  };

  return (
    <div className="container mt-5 mb-5">
      <div className="d-flex justify-content-between align-items-center mb-4">
        <h1 className="mb-0">🤖 AutoML Model Comparator</h1>

        <div onClick={() => setDarkMode(!darkMode)} style={{ width: '60px', height: '30px', backgroundColor: darkMode ? '#6610f2' : '#ccc', borderRadius: '30px', position: 'relative', cursor: 'pointer', transition: 'background-color 0.3s ease' }}>
          <div style={{ width: '26px', height: '26px', backgroundColor: '#fff', borderRadius: '50%', position: 'absolute', top: '2px', left: darkMode ? '32px' : '2px', transition: 'left 0.3s cubic-bezier(0.68, -0.55, 0.27, 1.55)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '14px' }}>
            {darkMode ? '🌙' : '☀️'}
          </div>
        </div>
      </div>

      <div className="card p-4 shadow-sm mb-4">
        <form onSubmit={handleSubmit}>

          {/* 1. FILE UPLOAD & STATS */}
          <div className="row mb-3">
            <div className="col-md-6">
              <label className="form-label fw-bold">1. Upload Dataset (CSV)</label>
              <input type="file" className="form-control" accept=".csv" onChange={handleFileChange} />
            </div>
            <div className="col-md-6">
              <label className="form-label fw-bold">2. Target Column</label>
              <input type="text" className="form-control" list="cols" value={target} onChange={e => setTarget(e.target.value)} disabled={!file} placeholder="Type or select..." />
              <datalist id="cols">{columns.map(c => <option key={c} value={c} />)}</datalist>
            </div>
          </div>

          {/* BETTER STATS GRID */}
          {dataStats && (
            <div className={`mb-4 p-3 rounded border ${darkMode ? 'bg-secondary bg-opacity-10 border-secondary' : 'bg-light border-light'}`}>
              <div className="d-flex justify-content-around text-center">
                <div>
                  <div className="text-muted small fw-bold text-uppercase">Rows</div>
                  <div className="fs-5 fw-bold text-primary font-monospace">{dataStats.rows}</div>
                </div>
                <div className="border-end border-secondary opacity-25"></div>
                <div>
                  <div className="text-muted small fw-bold text-uppercase">Columns</div>
                  <div className="fs-5 fw-bold text-success font-monospace">{dataStats.cols}</div>
                </div>
                <div className="border-end border-secondary opacity-25"></div>
                <div>
                  <div className="text-muted small fw-bold text-uppercase">Size</div>
                  <div className="fs-5 fw-bold text-info font-monospace">{dataStats.size}</div>
                </div>
              </div>
            </div>
          )}

          {/* 3. MODEL SELECTION */}
          <div className="mb-3">
            <div className="d-flex justify-content-between align-items-center mb-2">
              <label className="form-label fw-bold">3. Select Models</label>
              <select className="form-select w-auto form-select-sm" value={detectedTaskType} onChange={(e) => setDetectedTaskType(e.target.value)}>
                <option value="Classification">Classification</option>
                <option value="Regression">Regression</option>
              </select>
            </div>
            <div className={`card card-body ${darkMode ? 'bg-secondary bg-opacity-10' : 'bg-white border'}`}>
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

          <button type="submit" className="btn btn-primary w-100 btn-lg" disabled={loading || !file || !target}>
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