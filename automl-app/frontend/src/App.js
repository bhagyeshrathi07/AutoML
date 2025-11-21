import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Papa from 'papaparse';
import 'bootstrap/dist/css/bootstrap.min.css';
import Leaderboard from './Leaderboard';

const AVAILABLE_MODELS = [
  "Logistic Regression",
  "Random Forest",
  "SVM",
  "KNN",
  "XGBoost"
];

function App() {
  const [darkMode, setDarkMode] = useState(() => {
    return localStorage.getItem('theme') === 'dark';
  });

  const [file, setFile] = useState(null);
  const [target, setTarget] = useState('');
  const [columns, setColumns] = useState([]);
  const [selectedModels, setSelectedModels] = useState(AVAILABLE_MODELS);

  // --- NEW STATE FOR DATA STATS ---
  const [dataStats, setDataStats] = useState(null); // { rows, cols, size }
  // --------------------------------

  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    if (darkMode) {
      document.documentElement.setAttribute('data-bs-theme', 'dark');
      localStorage.setItem('theme', 'dark');
    } else {
      document.documentElement.setAttribute('data-bs-theme', 'light');
      localStorage.setItem('theme', 'light');
    }
  }, [darkMode]);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);

    setColumns([]);
    setDataStats(null); // Reset stats

    if (selectedFile) {
      // Calculate simple file size string (e.g., "4.5 MB")
      let fileSize = selectedFile.size / 1024; // in KB
      let sizeString = fileSize < 1024
        ? `${fileSize.toFixed(1)} KB`
        : `${(fileSize / 1024).toFixed(2)} MB`;

      // Parse file to get Rows/Cols
      Papa.parse(selectedFile, {
        header: true,
        skipEmptyLines: true, // Don't count empty lines as rows
        complete: (results) => {
          // Save columns for dropdown
          if (results.meta && results.meta.fields) {
            setColumns(results.meta.fields);

            // Update Stats
            setDataStats({
              rows: results.data.length,
              cols: results.meta.fields.length,
              size: sizeString
            });
          }
        },
        error: (err) => {
          console.error("Error parsing CSV:", err);
          setError("Failed to read CSV file.");
        }
      });
    }
  };

  const handleModelChange = (modelName) => {
    if (selectedModels.includes(modelName)) {
      setSelectedModels(selectedModels.filter(m => m !== modelName));
    } else {
      setSelectedModels([...selectedModels, modelName]);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file || !target) {
      setError('Please provide file and target column.');
      return;
    }
    if (selectedModels.length === 0) {
      setError('Please select at least one model.');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('target', target);
    formData.append('models', JSON.stringify(selectedModels));

    setLoading(true);
    setError('');
    setResults(null);

    try {
      const response = await axios.post('http://127.0.0.1:5000/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setResults(response.data);
    } catch (err) {
      console.error(err);
      setError('Error: ' + (err.response?.data?.error || 'Failed to process dataset'));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container mt-5 mb-5">
      <div className="d-flex justify-content-between align-items-center mb-4">
        <h1 className="mb-0">🤖 AutoML Model Comparator</h1>

        {/* Dark Mode Toggle */}
        <div
          onClick={() => setDarkMode(!darkMode)}
          style={{
            width: '60px', height: '30px',
            backgroundColor: darkMode ? '#6610f2' : '#ccc',
            borderRadius: '30px', position: 'relative', cursor: 'pointer',
            transition: 'background-color 0.3s ease'
          }}
        >
          <div style={{
            width: '26px', height: '26px', backgroundColor: '#fff',
            borderRadius: '50%', position: 'absolute', top: '2px',
            left: darkMode ? '32px' : '2px',
            transition: 'left 0.3s cubic-bezier(0.68, -0.55, 0.27, 1.55)',
            display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '14px'
          }}>
            {darkMode ? '🌙' : '☀️'}
          </div>
        </div>
      </div>

      <div className="card p-4 shadow-sm">
        <form onSubmit={handleSubmit}>

          {/* 1. File Upload */}
          <div className="mb-3">
            <label className="form-label fw-bold">1. Upload Dataset (CSV)</label>
            <input
              type="file"
              className="form-control"
              accept=".csv"
              onChange={handleFileChange}
            />

            {/* --- NEW: DATA SIZE INDICATOR --- */}
            {dataStats && (
              <div className={`mt-2 p-2 rounded border d-inline-block ${darkMode ? 'bg-secondary bg-opacity-25 border-secondary' : 'bg-light border'}`}>
                <small className="fw-bold text-success">
                  📊 Data Stats:
                </small>
                <span className="ms-2 small">
                  {dataStats.rows.toLocaleString()} Rows
                  <span className="mx-2 text-muted">|</span>
                  {dataStats.cols} Columns
                  <span className="mx-2 text-muted">|</span>
                  {dataStats.size}
                </span>
              </div>
            )}
            {/* -------------------------------- */}
          </div>

          {/* 2. Target Column */}
          <div className="mb-3">
            <label className="form-label fw-bold">2. Target Column Name</label>
            <input
              type="text"
              className="form-control"
              placeholder="Select from list or type manually..."
              value={target}
              onChange={(e) => setTarget(e.target.value)}
              list="columnOptions"
            />
            <datalist id="columnOptions">
              {columns.map((col, index) => (
                <option key={index} value={col} />
              ))}
            </datalist>
          </div>

          {/* 3. Model Selection */}
          <div className="mb-4">
            <label className="form-label fw-bold">3. Choose Models to Train</label>
            <div className="d-flex flex-wrap gap-3">
              {AVAILABLE_MODELS.map((model) => (
                <div className="form-check" key={model}>
                  <input
                    className="form-check-input"
                    type="checkbox"
                    id={model}
                    checked={selectedModels.includes(model)}
                    onChange={() => handleModelChange(model)}
                    style={{ cursor: 'pointer' }}
                  />
                  <label className="form-check-label" htmlFor={model} style={{ cursor: 'pointer' }}>
                    {model}
                  </label>
                </div>
              ))}
            </div>
          </div>

          <button type="submit" className="btn btn-primary w-100" disabled={loading}>
            {loading ? (
              <span><span className="spinner-border spinner-border-sm me-2" />Training Models...</span>
            ) : (
              '🚀 Launch Pipeline'
            )}
          </button>
        </form>

        {error && <div className="alert alert-danger mt-3">{error}</div>}
      </div>

      {results && <Leaderboard results={results} darkMode={darkMode} />}
    </div>
  );
}

export default App;