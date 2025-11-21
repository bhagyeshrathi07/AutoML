import React, { useState } from 'react';
import ConfusionMatrix from './ConfusionMatrix';
import ROCChart from './ROCChart';
import RegressionChart from './RegressionChart';

const Leaderboard = ({ results, darkMode, taskId }) => {
  if (!results || results.length === 0) return null;

  const taskType = results[0]["Task Type"];
  const isClassification = taskType === "Classification";
  
  // Default sort: Accuracy for Classif, R2 for Reg
  const [sortKey, setSortKey] = useState(isClassification ? 'Accuracy' : 'R2 Score');

  // Sorting Logic
  const sorted = [...results].sort((a, b) => {
      const valA = a[sortKey] !== undefined ? a[sortKey] : -Infinity;
      const valB = b[sortKey] !== undefined ? b[sortKey] : -Infinity;
      
      // Metrics where LOWER is better (Ascending Sort)
      if (['RMSE', 'MAE', 'Training Time (s)', 'Max RAM (MB)'].includes(sortKey)) {
          return valA - valB;
      }
      // Metrics where HIGHER is better (Descending Sort)
      return valB - valA;
  });

  const bestModel = sorted[0];

  // --- HELPER FOR SORT BUTTONS ---
  const SortableHeader = ({ label, metricKey }) => {
      const isActive = sortKey === metricKey;
      const isAscendingMetric = ['RMSE', 'MAE', 'Training Time (s)', 'Max RAM (MB)'].includes(metricKey);
      
      return (
          <th className={`align-middle ${isActive ? 'table-active border-primary' : ''}`} 
              style={{ cursor: 'pointer', minWidth: '130px' }}
              onClick={() => setSortKey(metricKey)}>
              
              <div className="d-flex justify-content-between align-items-center">
                  <span className="fw-bold me-2">{label}</span>
                  <button className={`btn btn-sm ${isActive ? 'btn-primary' : 'btn-outline-secondary'} border-0 py-0 px-1`}>
                      {isActive ? (
                          isAscendingMetric ? '▲' : '▼'
                      ) : (
                          <span style={{ opacity: 0.3 }}>⇅</span>
                      )}
                  </button>
              </div>
          </th>
      );
  };

  return (
    <div className="animate__animated animate__fadeIn">
      <div className="d-flex align-items-center justify-content-between mb-3">
        <h3 className="m-0">🏆 Leaderboard <span className={`badge ${isClassification ? 'bg-primary' : 'bg-warning text-dark'}`}>{taskType}</span></h3>
        <small className="text-muted">Click column headers to sort</small>
      </div>
      
      {/* TABLE */}
      <div className="table-responsive shadow-sm rounded">
        <table className={`table table-bordered align-middle mb-0 ${darkMode ? 'table-dark table-hover' : 'table-striped'}`}>
          <thead className={darkMode ? 'table-secondary text-dark' : 'table-light'}>
            <tr>
              <th className="text-center" style={{width: '60px'}}>Rank</th>
              <th>Model</th>
              
              {/* DYNAMIC SORTABLE HEADERS */}
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
            </tr>
          </thead>
          <tbody>
            {sorted.map((r, i) => (
              <tr key={i} className={i === 0 ? 'table-success fw-bold border-2 border-success' : ''}>
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
                
                <td className={sortKey === 'Training Time (s)' ? 'fw-bold' : ''}>{r["Training Time (s)"]}</td>
                <td className={sortKey === 'Max RAM (MB)' ? 'fw-bold' : ''}>{r["Max RAM (MB)"]}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* DETAIL CARD */}
      <div className={`card mt-4 shadow border-success ${darkMode ? 'bg-dark text-white' : ''}`}>
        <div className="card-header bg-success text-white fw-bold d-flex justify-content-between align-items-center">
            <span>✨ Best Model Details: {bestModel.Model}</span>
            <span className="badge bg-white text-success">Rank #1</span>
        </div>
        <div className="card-body">
            <div className="row">
                <div className="col-md-8">
                    <h6 className="text-success">⚙️ Hyperparameters</h6>
                    <pre className="p-3 rounded border border-secondary" style={{ backgroundColor: darkMode ? '#2c3034' : '#f8f9fa', fontSize: '0.85rem' }}>
                        {JSON.stringify(bestModel["Best Params"], null, 2)}
                    </pre>
                </div>
                <div className="col-md-4 d-flex flex-column gap-3 justify-content-center">
                    <h6 className="fw-bold">📥 Export</h6>
                    <a href={`http://127.0.0.1:5000/download-model?model=${bestModel.Model}`} 
                       className="btn btn-success w-100 py-2 shadow-sm">
                       📦 Download Model (.pkl)
                    </a>
                    
                    <a href={`http://127.0.0.1:5000/download-code?model=${bestModel.Model}&task_id=${taskId}`} 
                       className="btn btn-outline-primary w-100 py-2">
                       📜 Download Python Code (.py)
                    </a>
                </div>
            </div>

            {/* VISUALIZATIONS */}
            <div className="row mt-4 pt-4 border-top border-secondary">
                {isClassification ? (
                    <>
                        <div className="col-md-6 text-center">
                            <h6 className="fw-bold mb-3">Confusion Matrix</h6>
                            <div className="d-flex justify-content-center">
                                <ConfusionMatrix data={bestModel.ConfusionMatrix} darkMode={darkMode} />
                            </div>
                        </div>
                        <div className="col-md-6">
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
    </div>
  );
};

export default Leaderboard;