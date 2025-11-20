import React, { useState } from 'react';
import axios from 'axios';
import 'bootstrap/dist/css/bootstrap.min.css';
import Leaderboard from './Leaderboard';

// Define the list of available models exactly as they match the Backend keys
const AVAILABLE_MODELS = [
  "Logistic Regression",
  "Random Forest",
  "SVM",
  "KNN",
  "XGBoost"
];

function App() {
  const [file, setFile] = useState(null);
  const [target, setTarget] = useState('');
  // Default: All models selected
  const [selectedModels, setSelectedModels] = useState(AVAILABLE_MODELS);
  
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // Handle Checkbox Toggles
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
    // Send selected models as a JSON string
    formData.append('models', JSON.stringify(selectedModels));

    setLoading(true);
    setError('');
    setResults(null);

    try {
      // Synchronous POST: We wait right here until the server finishes
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
      <h1 className="text-center mb-4">🤖 AutoML Model Comparator</h1>
      
      <div className="card p-4 shadow-sm">
        <form onSubmit={handleSubmit}>
          
          {/* 1. File Upload */}
          <div className="mb-3">
            <label className="form-label fw-bold">1. Upload Dataset (CSV)</label>
            <input type="file" className="form-control" accept=".csv" onChange={(e) => setFile(e.target.files[0])} />
          </div>
          
          {/* 2. Target Column */}
          <div className="mb-3">
            <label className="form-label fw-bold">2. Target Column Name</label>
            <input type="text" className="form-control" placeholder="e.g., y, species, income" value={target} onChange={(e) => setTarget(e.target.value)} />
          </div>

          {/* 3. Model Selection Checkboxes */}
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
                  />
                  <label className="form-check-label" htmlFor={model}>
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

      {results && <Leaderboard results={results} />}
    </div>
  );
}

export default App;