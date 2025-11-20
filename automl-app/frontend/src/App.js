import React, { useState } from 'react';
import axios from 'axios';
import 'bootstrap/dist/css/bootstrap.min.css'; // Import Bootstrap styles
import Leaderboard from './Leaderboard';

function App() {
  const [file, setFile] = useState(null);
  const [target, setTarget] = useState('');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setResults(null); // Reset results on new file
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file || !target) {
      setError('Please provide both a file and a target column name.');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('target', target);

    setLoading(true);
    setError('');

    try {
      // Calls your Python Backend running on port 5000
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
    <div className="container mt-5">
      <h1 className="text-center mb-4">🤖 AutoML Model Comparator</h1>
      
      <div className="card p-4 shadow-sm">
        <form onSubmit={handleSubmit}>
          <div className="mb-3">
            <label className="form-label">Upload Dataset (CSV)</label>
            <input 
              type="file" 
              className="form-control" 
              accept=".csv"
              onChange={handleFileChange} 
            />
          </div>
          
          <div className="mb-3">
            <label className="form-label">Target Column Name</label>
            <input 
              type="text" 
              className="form-control" 
              placeholder="e.g., species, survived, price"
              value={target}
              onChange={(e) => setTarget(e.target.value)} 
            />
          </div>

          <button 
            type="submit" 
            className="btn btn-primary w-100" 
            disabled={loading}
          >
            {loading ? (
              <span>
                <span className="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
                Training Models (This may take time)...
              </span>
            ) : (
              '🚀 Launch Pipeline'
            )}
          </button>
        </form>

        {error && <div className="alert alert-danger mt-3">{error}</div>}
      </div>

      {/* Show Leaderboard when results arrive */}
      {results && <Leaderboard results={results} />}
    </div>
  );
}

export default App;