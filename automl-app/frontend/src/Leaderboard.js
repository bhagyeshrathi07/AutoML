import React, { useState, useEffect } from 'react';
import ConfusionMatrix from './ConfusionMatrix';
import ROCChart from './ROCChart';

const Leaderboard = ({ results, darkMode }) => {
  const [sortedResults, setSortedResults] = useState([]);
  const [sortCriteria, setSortCriteria] = useState('Accuracy');

  // Trigger sort whenever results or criteria change
  useEffect(() => {
    if (!results) return;
    sortModels(sortCriteria);
  }, [results, sortCriteria]);

  const sortModels = (criteria) => {
    const sorted = [...results].sort((a, b) => {
      // SCENARIO 1: Accuracy
      if (criteria === 'Accuracy') {
        if (b.Accuracy !== a.Accuracy) return b.Accuracy - a.Accuracy;
        return a["Training Time (s)"] - b["Training Time (s)"];
      }
      // SCENARIO 2: F1 Score
      if (criteria === 'F1') {
        if (b["F1 Score"] !== a["F1 Score"]) return b["F1 Score"] - a["F1 Score"];
        return a["Training Time (s)"] - b["Training Time (s)"];
      }
      // SCENARIO 3: Efficiency (CPU)
      if (criteria === 'CPU') {
        if (a["Max CPU (%)"] !== b["Max CPU (%)"]) return a["Max CPU (%)"] - b["Max CPU (%)"];
        return b["F1 Score"] - a["F1 Score"];
      }
      // SCENARIO 4: Memory (RAM)
      if (criteria === 'RAM') {
        if (a["Max RAM (MB)"] !== b["Max RAM (MB)"]) return a["Max RAM (MB)"] - b["Max RAM (MB)"];
        return b["F1 Score"] - a["F1 Score"];
      }
      // SCENARIO 5: Speed
      if (criteria === 'Time') {
        if (a["Training Time (s)"] !== b["Training Time (s)"]) return a["Training Time (s)"] - b["Training Time (s)"];
        return b["F1 Score"] - a["F1 Score"];
      }
      return 0;
    });
    setSortedResults(sorted);
  };

  if (!results || results.length === 0) return null;

  const bestModel = sortedResults[0];

  const getRankBadge = (index) => {
    if (index === 0) return "🥇";
    if (index === 1) return "🥈";
    if (index === 2) return "🥉";
    return index + 1;
  };

  return (
    <div className="mt-4">
      {/* HEADER & SORTING DROPDOWN */}
      <div className="d-flex justify-content-between align-items-center mb-3">
        <h3>🏆 Model Leaderboard</h3>
        
        <div className="d-flex align-items-center">
          <label className="me-2 fw-bold">Prioritize By:</label>
          <select 
            className={`form-select w-auto ${darkMode ? 'bg-dark text-white border-secondary' : ''}`}
            value={sortCriteria} 
            onChange={(e) => setSortCriteria(e.target.value)}
          >
            <option value="Accuracy">Best Accuracy (Default)</option>
            <option value="F1">Best F1 Score (Balanced)</option>
            <option value="Time">Fastest Training Time</option>
            <option value="CPU">Lowest CPU Usage</option>
            <option value="RAM">Lowest RAM Usage</option>
          </select>
        </div>
      </div>

      {/* TABLE SECTION */}
      <div className="table-responsive">
        <table className={`table table-bordered shadow-sm ${darkMode ? 'table-dark table-hover' : 'table-striped table-hover'}`}>
          <thead className={darkMode ? "table-light" : "table-dark"}>
            <tr>
              <th>Rank</th>
              <th>Model</th>
              <th className={sortCriteria === 'Accuracy' ? "bg-primary text-white" : ""}>Accuracy</th>
              <th className={sortCriteria === 'F1' ? "bg-primary text-white" : ""}>F1 Score</th>
              <th className={sortCriteria === 'Time' ? "bg-primary text-white" : ""}>Training Time</th>
              <th className={sortCriteria === 'RAM' ? "bg-primary text-white" : ""}>Max RAM</th>
              <th className={sortCriteria === 'CPU' ? "bg-primary text-white" : ""}>Max CPU</th>
            </tr>
          </thead>
          <tbody>
            {sortedResults.map((model, index) => (
              <tr key={index} className={index === 0 ? (darkMode ? "table-active border-success" : "table-success fw-bold") : ""}>
                <td className="text-center">{getRankBadge(index)}</td>
                <td>{model.Model}</td>
                <td>{(model.Accuracy * 100).toFixed(2)}%</td>
                <td>{(model["F1 Score"] * 100).toFixed(2)}%</td>
                <td>{model["Training Time (s)"]} s</td>
                <td>{model["Max RAM (MB)"]} MB</td>
                <td>{model["Max CPU (%)"]}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* VISUALIZATION SECTION (Dynamic based on Best Model) */}
      {bestModel && (
        <div className={`card mt-4 mb-4 shadow-sm ${darkMode ? 'bg-dark text-white border-secondary' : ''}`}>
          <div className="card-header fw-bold border-secondary">
            📊 Visual Analysis: {bestModel.Model}
          </div>
          <div className="card-body">
            <div className="row align-items-center">
              
              {/* Left Col: Confusion Matrix (Now smaller: 4/12 columns) */}
              <div className="col-md-4 border-end border-secondary d-flex justify-content-center">
                 <ConfusionMatrix data={bestModel.ConfusionMatrix} darkMode={darkMode} />
              </div>

              {/* Right Col: ROC Curve (Now larger: 8/12 columns) */}
              <div className="col-md-8">
                 <ROCChart 
                    data={bestModel.ROCData} 
                    auc={bestModel.AUC} 
                    darkMode={darkMode} 
                 />
              </div>
              
            </div>
            <p className="text-left mt-3text-muted small">
              *Charts update automatically when the sorting criteria changes.
            </p>
          </div>
        </div>
      )}

      {/* RECOMMENDATION & DOWNLOAD CARD */}
      <div className={`alert mt-3 ${darkMode ? 'alert-dark border-success text-white' : 'alert-success border-success'}`} style={darkMode ? {borderColor: '#198754'} : {}}>
        <div className="d-flex justify-content-between align-items-center">
            <div>
                <h5>
                    ✨ Recommended for 
                    {sortCriteria === 'Accuracy' && " Maximum Accuracy"}
                    {sortCriteria === 'F1' && " Balanced Performance (F1)"}
                    {sortCriteria === 'Time' && " High Speed"}
                    {sortCriteria === 'CPU' && " Efficiency"}
                    : {bestModel?.Model}
                </h5>
                <p className="mb-0 small opacity-75">
                  Best Hyperparameters: <code>{JSON.stringify(bestModel?.["Best Params"])}</code>
                </p>
            </div>
            
            {/* DYNAMIC DOWNLOAD BUTTON */}
            <a 
                href={`http://127.0.0.1:5000/download?model=${bestModel?.Model}`}
                className="btn btn-success fw-bold shadow-sm"
                target="_blank"
                rel="noopener noreferrer"
            >
                ⬇️ Download {bestModel?.Model}
            </a>
        </div>
      </div>
    </div> 
  );
};

export default Leaderboard;