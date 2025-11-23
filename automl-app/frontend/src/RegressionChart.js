import React from 'react';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine
} from 'recharts';

const RegressionChart = ({ data, r2, darkMode }) => {
  // 1. Fallback if no data
  if (!data || data.length === 0) {
    return (
      <div className="d-flex align-items-center justify-content-center" style={{ height: 400 }}>
        <p className="text-muted">No Scatter Data Available</p>
      </div>
    );
  }

  // Theme Colors
  const axisColor = darkMode ? "#ccc" : "#333";
  const gridColor = darkMode ? "#444" : "#e0e0e0";
  const tooltipBg = darkMode ? "#333" : "#fff";
  const tooltipBorder = darkMode ? "#555" : "#ccc";

  // 2. Helper to format large numbers (e.g., 5000000 -> 5M)
  const formatNumber = (num) => {
    if (Math.abs(num) >= 1000000000) return (num / 1000000000).toFixed(1) + 'B';
    if (Math.abs(num) >= 1000000) return (num / 1000000).toFixed(1) + 'M';
    if (Math.abs(num) >= 1000) return (num / 1000).toFixed(1) + 'k';
    return num.toFixed(2);
  };

  // 3. Smart Domain Calculation (Adds 5% padding so points aren't on the edge)
  const allVals = data.map(d => d.actual).concat(data.map(d => d.predicted));
  const minVal = Math.min(...allVals);
  const maxVal = Math.max(...allVals);
  const padding = (maxVal - minVal) * 0.05; // 5% buffer

  return (
    <div style={{ width: '100%', height: 400 }}>
      <h6 className="text-center mb-2">
        Actual vs. Predicted <span className="badge bg-warning text-dark ms-2">R²: {r2}</span>
      </h6>

      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 20, right: 30, bottom: 20, left: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={gridColor} opacity={0.4} />

          {/* X AXIS (Actual) */}
          <XAxis
            type="number"
            dataKey="actual"
            name="Actual"
            domain={[minVal - padding, maxVal + padding]}
            tickFormatter={formatNumber}
            tick={{ fill: axisColor, fontSize: 11 }}
            stroke={axisColor}
            label={{
              value: 'Actual Values',
              position: 'insideBottom',
              offset: -10,
              fill: axisColor,
              fontSize: 12,
              fontWeight: 'bold'
            }}
          />

          {/* Y AXIS (Predicted) */}
          <YAxis
            type="number"
            dataKey="predicted"
            name="Predicted"
            domain={[minVal - padding, maxVal + padding]}
            tickFormatter={formatNumber}
            tick={{ fill: axisColor, fontSize: 11 }}
            stroke={axisColor}
            label={{
              value: 'Predicted Values',
              angle: -90,
              position: 'insideLeft',
              fill: axisColor,
              fontSize: 12,
              fontWeight: 'bold',
              dy: 40 // Shift label up slightly to avoid overlapping ticks
            }}
          />

          <Tooltip
            cursor={{ strokeDasharray: '3 3' }}
            contentStyle={{
              backgroundColor: tooltipBg,
              borderColor: tooltipBorder,
              borderRadius: '8px',
              color: axisColor,
              fontSize: '13px'
            }}
            formatter={(value, name) => [formatNumber(value), name === 'actual' ? 'Actual' : 'Predicted']}
            labelFormatter={() => ''} // Hide default index label
          />

          {/* Perfect Prediction Line (Diagonal) */}
          <ReferenceLine
            segment={[{ x: minVal - padding, y: minVal - padding }, { x: maxVal + padding, y: maxVal + padding }]}
            stroke="#10b981" // Green
            strokeWidth={2}
            strokeDasharray="5 5"
            label={{ value: 'Perfect Fit', position: 'insideTopLeft', fill: '#10b981', fontSize: 11 }}
          />

          <Scatter
            name="Predictions"
            data={data}
            fill="#0d6efd"
            fillOpacity={0.6}
            r={4} // Slightly larger dots
          />
        </ScatterChart>
      </ResponsiveContainer>

      <p className="text-center text-muted small mt-2" style={{ opacity: 0.7 }}>
        (Points closer to the green line = Better predictions)
      </p>
    </div>
  );
};

export default RegressionChart;