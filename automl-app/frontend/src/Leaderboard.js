import React, { useState, useEffect } from 'react';
import ConfusionMatrix from './ConfusionMatrix';
import ROCChart from './ROCChart';
import RegressionChart from './RegressionChart';

const Leaderboard = ({ results, darkMode, taskId }) => {
  // --- 1. HOOKS MUST BE AT THE TOP (Before any return) ---

  // Determine default sort key safely (even if results are empty initially)
  const defaultSort = (results && results.length > 0 && results[0]["Task Type"] === "Regression")
    ? 'R2 Score'
    : 'Accuracy';

  const [sortKey, setSortKey] = useState(defaultSort);
  const [sortedResults, setSortedResults] = useState([]);

  // Effect: Update sort key if the dataset changes (e.g. User switches from Classification to Regression)
  useEffect(() => {
    if (results && results.length > 0) {
      const type = results[0]["Task Type"];
      setSortKey(type === "Classification" ? 'Accuracy' : 'R2 Score');
    }
  }, [results]);

  // Effect: Sort the data whenever results or sortKey changes
  useEffect(() => {
    if (!results || results.length === 0) return;

    const sorted = [...results].sort((a, b) => {
      const valA = a[sortKey] !== undefined ? a[sortKey] : -Infinity;
      const valB = b[sortKey] !== undefined ? b[sortKey] : -Infinity;

      // Metrics where LOWER is better (Ascending Sort)
      if (['RMSE', 'MAE', 'Training Time (s)', 'Max RAM (MB)', 'Max CPU (%)'].includes(sortKey)) {
        return valA - valB;
      }
      // Metrics where HIGHER is better (Descending Sort)
      return valB - valA;
    });
    setSortedResults(sorted);
  }, [results, sortKey]);

  // --- 2. EARLY RETURN IS NOW SAFE (After Hooks) ---
  if (!results || results.length === 0 || sortedResults.length === 0) return null;

  // --- 3. RENDER HELPERS ---
  const bestModel = sortedResults[0];
  const taskType = bestModel["Task Type"] || "Classification";
  const isClassification = taskType === "Classification";

  const SortableHeader = ({ label, metricKey }) => {
    const isActive = sortKey === metricKey;
    return (
      <th
        className={`align-middle ${isActive ? 'bg-primary text-white' : ''}`}
        style={{ cursor: 'pointer', minWidth: '120px' }}
        onClick={() => setSortKey(metricKey)}
      >
        <div className="d-flex justify-content-between align-items-center">
          <span>{label}</span>
          {isActive && <span>{['RMSE', 'MAE', 'Training Time (s)', 'Max RAM (MB)', 'Max CPU (%)'].includes(metricKey) ? '▲' : '▼'}</span>}
        </div>
      </th>
    );
  };

  return (
    <div className="animate__animated animate__fadeIn mt-4">
      {/* HEADER */}
      <div className="d-flex align-items-center justify-content-between mb-3">
        <h3 className="m-0">🏆 Leaderboard <span className={`badge ms-2 fs-6 ${isClassification ? 'bg-primary' : 'bg-warning text-dark'}`}>{taskType}</span></h3>
        <div className="d-flex align-items-center">
          <label className="me-2 fw-bold">Sort By:</label>
          <select className={`form-select w-auto ${darkMode ? 'bg-dark text-white border-secondary' : ''}`} value={sortKey} onChange={(e) => setSortKey(e.target.value)}>
            {isClassification ? (
              <>
                <option value="Accuracy">Accuracy</option>
                <option value="F1 Score">F1 Score</option>
              </>
            ) : (
              <>
                <option value="R2 Score">R2 Score</option>
                <option value="RMSE">RMSE (Lower is better)</option>
              </>
            )}
            <option value="Training Time (s)">Time (Fastest)</option>
            <option value="Max RAM (MB)">RAM (Efficiency)</option> {/* ADDED THIS LINE */}
            <option value="Max CPU (%)">CPU (Efficiency)</option>
          </select>
        </div>
      </div>

      {/* TABLE */}
      <div className="table-responsive shadow-sm rounded">
        <table className={`table table-bordered align-middle mb-0 ${darkMode ? 'table-dark table-hover' : 'table-striped'}`}>
          <thead className={darkMode ? 'table-secondary text-dark' : 'table-light'}>
            <tr>
              <th className="text-center" style={{ width: '60px' }}>Rank</th>
              <th>Model</th>

              {/* DYNAMIC HEADERS */}
              {isClassification ? (
                <>
                  <SortableHeader label="Accuracy" metricKey="Accuracy" />
                  <SortableHeader label="F1 Score" metricKey="F1 Score" />
                </>
              ) : (
                <>
                  <SortableHeader label="R2 Score" metricKey="R2 Score" />
                  <SortableHeader label="MAE" metricKey="MAE" />
                  <SortableHeader label="RMSE" metricKey="RMSE" />
                </>
              )}

              <SortableHeader label="Time (s)" metricKey="Training Time (s)" />
              <SortableHeader label="RAM (MB)" metricKey="Max RAM (MB)" />
              <SortableHeader label="CPU (%)" metricKey="Max CPU (%)" />
            </tr>
          </thead>
          <tbody>
            {sortedResults.map((r, i) => (
              <tr key={i} className={i === 0 ? (darkMode ? 'table-active border-success' : 'table-success fw-bold border-2 border-success') : ''}>
                <td className="text-center h5">{i === 0 ? '🥇' : i + 1}</td>
                <td>
                  {r.Model}
                  {i === 0 && <span className="badge bg-success ms-2">WINNER</span>}
                </td>

                {isClassification ? (
                  <>
                    <td className={sortKey === 'Accuracy' ? 'fw-bold text-decoration-underline' : ''}>{(r.Accuracy * 100).toFixed(2)}%</td>
                    <td className={sortKey === 'F1 Score' ? 'fw-bold text-decoration-underline' : ''}>{(r["F1 Score"] * 100).toFixed(2)}%</td>
                  </>
                ) : (
                  <>
                    <td className={sortKey === 'R2 Score' ? 'fw-bold text-decoration-underline' : ''}>{r["R2 Score"]}</td>
                    <td className={sortKey === 'MAE' ? 'fw-bold text-decoration-underline' : ''}>{r["MAE"]}</td>
                    <td className={sortKey === 'RMSE' ? 'fw-bold text-decoration-underline' : ''}>{r["RMSE"]}</td>
                  </>
                )}

                <td>{r["Training Time (s)"]}</td>
                <td className={sortKey === 'Max RAM (MB)' ? 'fw-bold text-decoration-underline' : ''}>{r["Max RAM (MB)"]}</td>
                <td>{r["Max CPU (%)"]}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* DETAIL CARD */}
      {bestModel && (
        <div className={`card mt-4 shadow border-success ${darkMode ? 'bg-dark text-white' : ''}`}>
          <div className="card-header bg-success text-white fw-bold d-flex justify-content-between align-items-center">
            <span>✨ Recommended Model: {bestModel.Model}</span>
            <span className="badge bg-white text-success">Rank #1</span>
          </div>
          <div className="card-body">
            <div className="row">
              <div className="col-md-7">
                <h6 className="text-success">⚙️ Best Hyperparameters</h6>
                <pre className="p-3 rounded border border-secondary" style={{ backgroundColor: darkMode ? '#2c3034' : '#f8f9fa', fontSize: '0.85rem', maxHeight: '150px', overflowY: 'auto' }}>
                  {JSON.stringify(bestModel["Best Params"], null, 2)}
                </pre>
              </div>
              <div className="col-md-5 d-flex flex-column gap-2 justify-content-center">
                <h6 className="fw-bold">📥 Actions</h6>
                <a href={`http://127.0.0.1:5000/download?model=${bestModel.Model}`}
                  className="btn btn-success w-100 shadow-sm">
                  📦 Download Model (.pkl)
                </a>

                {/* NEW CODE BUTTON */}
                <a href={`http://127.0.0.1:5000/download-code?model=${bestModel.Model}&task_id=${taskId}`}
                  className="btn btn-outline-primary w-100">
                  📜 Generate Python Script
                </a>
              </div>
            </div>

            {/* VISUALIZATIONS */}
            <div className="row mt-4 pt-4 border-top border-secondary">
              <div className="col-12 mb-2 fw-bold text-muted">📊 Visual Analysis ({bestModel.Model})</div>
              {isClassification ? (
                <>
                  <div className="col-md-4 border-end border-secondary d-flex justify-content-center">
                    <ConfusionMatrix data={bestModel.ConfusionMatrix} darkMode={darkMode} />
                  </div>
                  <div className="col-md-8">
                    <ROCChart data={bestModel.ROCData} auc={bestModel.AUC} darkMode={darkMode} />
                  </div>
                </>
              ) : (
                <div className="col-12">
                  <RegressionChart data={bestModel.ScatterData} r2={bestModel["R2 Score"]} darkMode={darkMode} />
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Leaderboard;