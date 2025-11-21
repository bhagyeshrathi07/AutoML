import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Papa from 'papaparse';
import 'bootstrap/dist/css/bootstrap.min.css';
import Leaderboard from './Leaderboard';

// Define models for specific tasks
const MODELS_CONFIG = {
    "Classification": ["Logistic Regression", "Random Forest", "SVM", "KNN", "XGBoost"],
    "Regression": ["Linear Regression", "Ridge", "Lasso", "Random Forest", "XGBoost", "Decision Tree"]
};

function App() {
  const [darkMode, setDarkMode] = useState(true);
  const [file, setFile] = useState(null);
  const [fileData, setFileData] = useState([]);
  const [target, setTarget] = useState('');
  const [columns, setColumns] = useState([]);
  
  // Task State
  const [detectedTaskType, setDetectedTaskType] = useState(null);
  const [selectedModels, setSelectedModels] = useState([]);

  // Async Task State
  const [taskId, setTaskId] = useState(null);
  const [progress, setProgress] = useState(0);
  const [logs, setLogs] = useState([]);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // Theme Toggle
  useEffect(() => {
    document.documentElement.setAttribute('data-bs-theme', darkMode ? 'dark' : 'light');
  }, [darkMode]);

  // --- TASK DETECTION LOGIC ---
  useEffect(() => {
    if (!target || fileData.length === 0) {
        setDetectedTaskType(null);
        return;
    }

    // Analyze the target column in the preview data
    const targetValues = fileData.map(row => row[target]).filter(val => val !== null && val !== undefined && val !== "");
    
    // 1. Check if numeric
    const isNumeric = targetValues.every(val => !isNaN(parseFloat(val)) && isFinite(val));
    
    // 2. Check Cardinality (Unique values)
    const uniqueValues = new Set(targetValues).size;
    
    // HEURISTIC: If Numeric AND > 20 unique values -> Regression. Otherwise Classification.
    let type = "Classification";
    if (isNumeric && uniqueValues > 20) {
        type = "Regression";
    }

    setDetectedTaskType(type);
    setSelectedModels(MODELS_CONFIG[type]);
    
  }, [target, fileData]);

  // --- POLLING LOGIC ---
  useEffect(() => {
    let interval = null;
    if (taskId && loading) {
        interval = setInterval(async () => {
            try {
                const res = await axios.get(`http://127.0.0.1:5000/status/${taskId}`);
                const data = res.data;
                
                setProgress(data.progress);
                setLogs(data.logs || []);

                if (data.status === 'completed') {
                    setResults(data.results);
                    setLoading(false);
                    // We keep taskId set so Leaderboard can use it for downloads
                } else if (data.status === 'failed') {
                    setError(data.error);
                    setLoading(false);
                    setTaskId(null);
                }
            } catch (err) {
                console.error("Polling Error:", err);
            }
        }, 2000);
    }
    return () => clearInterval(interval);
  }, [taskId, loading]);

  // Handle File Selection
  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    setColumns([]);
    setFileData([]);
    setTarget('');
    setDetectedTaskType(null);

    if (selectedFile) {
      Papa.parse(selectedFile, {
        header: true,
        preview: 500,
        skipEmptyLines: true,
        complete: (res) => {
            if (res.meta && res.meta.fields) setColumns(res.meta.fields);
            if (res.data) setFileData(res.data);
        }
      });
    }
  };

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
    formData.append('task_type_override', detectedTaskType); 

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
        <h1>🤖 AutoML Pro <span className="badge bg-success fs-6">Async</span></h1>
        <button className="btn btn-outline-secondary btn-sm" onClick={() => setDarkMode(!darkMode)}>
           {darkMode ? '☀️ Light Mode' : '🌙 Dark Mode'}
        </button>
      </div>

      <div className="card p-4 shadow-sm mb-4">
        <form onSubmit={handleSubmit}>
          <div className="row">
            <div className="col-md-6 mb-3">
                <label className="form-label fw-bold">1. Upload Dataset (CSV)</label>
                <input type="file" className="form-control" accept=".csv" onChange={handleFileChange} />
            </div>
            <div className="col-md-6 mb-3">
                <label className="form-label fw-bold">2. Target Column</label>
                <input type="text" className="form-control" list="cols" 
                       value={target} onChange={e=>setTarget(e.target.value)} 
                       disabled={!file}
                       placeholder="Type column name..." />
                <datalist id="cols">{columns.map(c => <option key={c} value={c}/>)}</datalist>
            </div>
          </div>

          {/* DYNAMIC MODEL SELECTION */}
          {detectedTaskType && (
              <div className="mb-3 animate__animated animate__fadeIn">
                <div className="d-flex justify-content-between align-items-center mb-2">
                    <label className="form-label fw-bold">3. Select Models</label>
                    <span className={`badge ${detectedTaskType === 'Regression' ? 'bg-warning text-dark' : 'bg-info text-dark'}`}>
                        Detected: {detectedTaskType}
                    </span>
                </div>
                <div className="card card-body bg-opacity-10 bg-secondary">
                    <div className="d-flex flex-wrap gap-3">
                        {MODELS_CONFIG[detectedTaskType].map(m => (
                            <div className="form-check" key={m}>
                                <input className="form-check-input" type="checkbox" checked={selectedModels.includes(m)} 
                                    onChange={() => handleModelToggle(m)}/>
                                <label className="form-check-label">{m}</label>
                            </div>
                        ))}
                    </div>
                </div>
              </div>
          )}

          <button type="submit" className="btn btn-primary w-100" disabled={loading || !file || !target || !detectedTaskType}>
             {loading ? '⏳ Pipeline Running...' : '🚀 Launch Pipeline'}
          </button>
        </form>

        {/* PROGRESS BAR */}
        {loading && (
            <div className="mt-4">
                <div className="progress" style={{height: '25px'}}>
                    <div className="progress-bar progress-bar-striped progress-bar-animated bg-success" 
                         style={{width: `${progress}%`}}>{progress}%</div>
                </div>
                <div className="mt-2 p-2 bg-dark text-light rounded font-monospace small" style={{maxHeight: '100px', overflowY: 'auto'}}>
                    {logs.map((l, i) => <div key={i}>{l}</div>)}
                </div>
            </div>
        )}

        {error && <div className="alert alert-danger mt-3">{error}</div>}
      </div>

      {/* RESULTS SECTION */}
      {results && (
          <Leaderboard results={results} darkMode={darkMode} taskId={taskId} />
      )}
    </div>
  );
}

export default App;