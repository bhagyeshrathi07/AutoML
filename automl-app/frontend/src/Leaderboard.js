import React, { useState, useEffect } from 'react';

const Leaderboard = ({ results }) => {
  // State to keep track of the sorted list and current criteria
  const [sortedResults, setSortedResults] = useState([]);
  const [sortCriteria, setSortCriteria] = useState('Accuracy');

  // This effect runs whenever the 'results' prop changes or user changes sort criteria
  useEffect(() => {
    if (!results) return;
    sortModels(sortCriteria);
  }, [results, sortCriteria]);

  const sortModels = (criteria) => {
    // Create a copy of results to avoid mutating the original prop
    const sorted = [...results].sort((a, b) => {
      
      // SCENARIO 1: User wants Best Accuracy
      if (criteria === 'Accuracy') {
        // Primary: Accuracy (High to Low)
        // Secondary: Time (Low to High)
        if (b.Accuracy !== a.Accuracy) return b.Accuracy - a.Accuracy;
        return a["Training Time (s)"] - b["Training Time (s)"];
      }

      // SCENARIO 2: User wants Best F1 Score
      if (criteria === 'F1') {
        // Primary: F1 Score (High to Low)
        if (b["F1 Score"] !== a["F1 Score"]) return b["F1 Score"] - a["F1 Score"];
        // Secondary: Time (Low to High)
        return a["Training Time (s)"] - b["Training Time (s)"];
      }

      // SCENARIO 3: User wants Best Efficiency (Low CPU)
      if (criteria === 'CPU') {
        // Primary: CPU (Low to High)
        // Secondary: F1 Score (High to Low) as tie-breaker
        if (a["Max CPU (%)"] !== b["Max CPU (%)"]) return a["Max CPU (%)"] - b["Max CPU (%)"];
        return b["F1 Score"] - a["F1 Score"];
      }

      // SCENARIO 4: User wants Lowest RAM (Memory)
      if (criteria === 'RAM') {
        // Primary: RAM (Low to High)
        if (a["Max RAM (MB)"] !== b["Max RAM (MB)"]) return a["Max RAM (MB)"] - b["Max RAM (MB)"];
        return b["F1 Score"] - a["F1 Score"];
      }

      // SCENARIO 5: User wants Fastest Training
      if (criteria === 'Time') {
        // Primary: Time (Low to High)
        if (a["Training Time (s)"] !== b["Training Time (s)"]) return a["Training Time (s)"] - b["Training Time (s)"];
        return b["F1 Score"] - a["F1 Score"];
      }
      
      return 0;
    });

    setSortedResults(sorted);
  };

  if (!results || results.length === 0) return null;

  const getRankBadge = (index) => {
    if (index === 0) return "🥇";
    if (index === 1) return "🥈";
    if (index === 2) return "🥉";
    return index + 1;
  };

  return (
    <div className="mt-4">
      <div className="d-flex justify-content-between align-items-center mb-3">
        <h3>🏆 Model Leaderboard</h3>
        
        {/* SORT DROPDOWN */}
        <div className="d-flex align-items-center">
          <label className="me-2 fw-bold">Prioritize By:</label>
          <select 
            className="form-select w-auto" 
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

      <div className="table-responsive">
        <table className="table table-striped table-hover table-bordered shadow-sm">
          <thead className="table-dark">
            <tr>
              <th>Rank</th>
              <th>Model</th>
              <th className={sortCriteria === 'Accuracy' ? "bg-primary" : ""}>Accuracy</th>
              <th className={sortCriteria === 'F1' ? "bg-primary" : ""}>F1 Score</th>
              <th className={sortCriteria === 'Time' ? "bg-primary" : ""}>Training Time</th>
              <th className={sortCriteria === 'RAM' ? "bg-primary" : ""}>Max RAM</th>
              <th className={sortCriteria === 'CPU' ? "bg-primary" : ""}>Max CPU</th>
            </tr>
          </thead>
          <tbody>
            {sortedResults.map((model, index) => (
              <tr key={index} className={index === 0 ? "table-success fw-bold" : ""}>
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

      {/* Dynamic Recommendation Card */}
      <div className="alert alert-success mt-3 border-success">
        <h5>
            ✨ Recommended for 
            {sortCriteria === 'Accuracy' && " Maximum Accuracy"}
            {sortCriteria === 'F1' && " Balanced Performance (F1)"}
            {sortCriteria === 'Time' && " High Speed"}
            {sortCriteria === 'CPU' && " Efficiency"}
            : {sortedResults[0]?.Model}
        </h5>
        <p className="mb-0 text-muted small">
          Best Hyperparameters: <code>{JSON.stringify(sortedResults[0]?.["Best Params"])}</code>
        </p>
      </div>
    </div>
  );
};

export default Leaderboard;