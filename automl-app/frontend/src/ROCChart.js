import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine
} from 'recharts';

const ROCChart = ({ data, auc, darkMode }) => {
  // 1. Update Fallback Height
  if (!data || data.length === 0) {
    return (
      <div className="d-flex align-items-center justify-content-center" style={{ height: 400 }}>
        <p className="text-muted">No ROC Data (Binary Classification Only)</p>
      </div>
    );
  }

  const axisColor = darkMode ? "#ccc" : "#333";
  const gridColor = darkMode ? "#444" : "#e0e0e0";
  const tooltipBg = darkMode ? "#333" : "#fff";
  const tooltipBorder = darkMode ? "#555" : "#ccc";

  return (
    // 2. Main Chart Height set to 400px
    <div style={{ width: '100%', height: 400 }}>
      <h6 className="text-center mb-2">
        ROC Curve <span className="badge bg-primary ms-2">AUC: {auc}</span>
      </h6>
      
      <ResponsiveContainer width="100%" height="100%">
        <LineChart 
          data={data} 
          // Ensure bottom margin is enough for labels
          margin={{ top: 10, right: 30, left: 20, bottom: 30 }} 
        >
          <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />
          
          <XAxis 
            dataKey="x" 
            type="number" 
            domain={[0, 1]} 
            tick={{ fill: axisColor, fontSize: 12 }}
            stroke={axisColor}
            label={{ 
              value: 'False Positive Rate', 
              position: 'insideBottom', 
              offset: -20,
              fill: axisColor,
              fontSize: 13
            }} 
          />
          
          <YAxis 
            type="number" 
            domain={[0, 1]} 
            tick={{ fill: axisColor, fontSize: 12 }}
            stroke={axisColor}
            label={{ 
              value: 'True Positive Rate', 
              angle: -90, 
              position: 'insideLeft', 
              fill: axisColor,
              fontSize: 13
            }} 
          />
          
          <Tooltip 
            contentStyle={{ 
              backgroundColor: tooltipBg, 
              borderColor: tooltipBorder, 
              borderRadius: '8px',
              color: axisColor 
            }}
            formatter={(value) => value.toFixed(3)}
            labelFormatter={(label) => `FPR: ${label}`}
          />
          
          <ReferenceLine 
            segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} 
            stroke="#dc3545" 
            strokeDasharray="5 5" 
            label={{ position: 'center', value: 'Random Guess', fill: '#dc3545', fontSize: 12, dy: -10 }}
          />
          
          <Line 
            type="monotone" 
            dataKey="y" 
            stroke="#0d6efd" 
            strokeWidth={3} 
            dot={false} 
            activeDot={{ r: 6 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default ROCChart;