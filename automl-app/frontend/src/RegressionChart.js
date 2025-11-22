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
  // 1. Keep height consistent even if empty
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

  // Calculate domain to make the chart "zoom in" on the data
  const allVals = data.map(d => d.actual).concat(data.map(d => d.predicted));
  const minVal = Math.floor(Math.min(...allVals));
  const maxVal = Math.ceil(Math.max(...allVals));

  return (
    <div style={{ width: '100%', height: 400 }}>
      <h6 className="text-center mb-2">
        Actual vs. Predicted <span className="badge bg-warning text-dark ms-2">R²: {r2}</span>
      </h6>

      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />

          <XAxis
            type="number"
            dataKey="actual"
            name="Actual"
            domain={[minVal, maxVal]}
            tick={{ fill: axisColor, fontSize: 12 }}
            stroke={axisColor}
            label={{
              value: 'Actual Values',
              position: 'insideBottom',
              offset: -10,
              fill: axisColor,
              fontSize: 13
            }}
          />

          <YAxis
            type="number"
            dataKey="predicted"
            name="Predicted"
            domain={[minVal, maxVal]}
            tick={{ fill: axisColor, fontSize: 12 }}
            stroke={axisColor}
            label={{
              value: 'Predicted Values',
              angle: -90,
              position: 'insideLeft',
              fill: axisColor,
              fontSize: 13
            }}
          />

          <Tooltip
            cursor={{ strokeDasharray: '3 3' }}
            contentStyle={{
              backgroundColor: tooltipBg,
              borderColor: tooltipBorder,
              borderRadius: '8px',
              color: axisColor
            }}
          />

          {/* Perfect Prediction Line (Diagonal) */}
          <ReferenceLine
            segment={[{ x: minVal, y: minVal }, { x: maxVal, y: maxVal }]}
            stroke="#28a745"
            strokeDasharray="5 5"
            label={{ value: 'Perfect Fit', position: 'insideTopLeft', fill: '#28a745', fontSize: 12 }}
          />

          <Scatter name="Predictions" data={data} fill="#0d6efd" fillOpacity={0.6} />
        </ScatterChart>
      </ResponsiveContainer>

      <p className="text-center text-muted small mt-2">
        (Points on the green dashed line indicate perfect predictions)
      </p>
    </div>
  );
};

export default RegressionChart;