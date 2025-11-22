import React from 'react';

const ConfusionMatrix = ({ data, darkMode }) => {
  // If no data (e.g. Regression model), don't render anything or render a fallback
  if (!data || !Array.isArray(data)) {
    return (
      <div className="d-flex align-items-center justify-content-center h-100 text-muted">
        <small>Not applicable for Regression</small>
      </div>
    );
  }

  // data structure: [[TN, FP], [FN, TP]]
  const matrix = data;

  // Define styles based on Dark Mode
  const boxStyle = {
    padding: '20px',
    borderRadius: '8px',
    textAlign: 'center',
    backgroundColor: darkMode ? 'rgba(255, 255, 255, 0.05)' : '#f8f9fa',
    border: darkMode ? '1px solid #444' : '1px solid #dee2e6',
    minWidth: '80px'
  };

  const labelStyle = {
    fontWeight: 'bold',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    color: darkMode ? '#ccc' : '#333'
  };

  return (
    <div className="d-flex flex-column align-items-center">
      <h6 className="mb-3">Confusion Matrix</h6>

      <div style={{
        display: 'grid',
        gridTemplateColumns: 'auto 1fr 1fr', // 3 Columns: Label | Box | Box
        gap: '10px',
        alignItems: 'center'
      }}>

        {/* ROW 1: Top Labels */}
        <div></div> {/* Top-Left Corner (Empty) */}
        <div style={labelStyle}>Pred: 0</div>
        <div style={labelStyle}>Pred: 1</div>

        {/* ROW 2: Actual 0 */}
        <div style={labelStyle}>Actual: 0</div>
        <div style={boxStyle}>
          <h4 className="m-0 text-success">{matrix[0][0]}</h4>
          <small className="text-muted">TN</small>
        </div>
        <div style={boxStyle}>
          <h4 className="m-0 text-danger">{matrix[0][1]}</h4>
          <small className="text-muted">FP</small>
        </div>

        {/* ROW 3: Actual 1 */}
        <div style={labelStyle}>Actual: 1</div>
        <div style={boxStyle}>
          <h4 className="m-0 text-danger">{matrix[1][0]}</h4>
          <small className="text-muted">FN</small>
        </div>
        <div style={boxStyle}>
          <h4 className="m-0 text-success">{matrix[1][1]}</h4>
          <small className="text-muted">TP</small>
        </div>

      </div>
    </div>
  );
};

export default ConfusionMatrix;