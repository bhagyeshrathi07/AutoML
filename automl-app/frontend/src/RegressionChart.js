import React from 'react';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Label
} from 'recharts';

const RegressionChart = ({ data, r2, darkMode }) => {
  if (!data || data.length === 0) return <p>No Scatter Data Available</p>;

  const axisColor = darkMode ? "#ccc" : "#333";
  const gridColor = darkMode ? "#444" : "#e0e0e0";

  const allVals = data.map(d => d.actual).concat(data.map(d => d.predicted));
  const minVal = Math.min(...allVals);
  const maxVal = Math.max(...allVals);

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
            stroke={axisColor} 
            tick={{ fill: axisColor }}
          >
             <Label value="Actual Values" offset={0} position="insideBottom" fill={axisColor} />
          </XAxis>
          <YAxis 
            type="number" 
            dataKey="predicted" 
            name="Predicted" 
            domain={[minVal, maxVal]}
            stroke={axisColor}
            tick={{ fill: axisColor }}
          >
              <Label value="Predicted Values" angle={-90} position="insideLeft" fill={axisColor} />
          </YAxis>
          <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: darkMode ? '#333' : '#fff' }} />
          
          {/* Perfect Prediction Line (y=x) */}
          <ReferenceLine 
            segment={[{ x: minVal, y: minVal }, { x: maxVal, y: maxVal }]} 
            stroke="red" 
            strokeDasharray="5 5" 
            label="Perfect Fit"
          />
          
          <Scatter name="Predictions" data={data} fill="#0d6efd" fillOpacity={0.6} />
        </ScatterChart>
      </ResponsiveContainer>
      <p className="text-center text-muted small mt-2">
          (Points closer to the red dashed line indicate better predictions)
      </p>
    </div>
  );
};

export default RegressionChart;