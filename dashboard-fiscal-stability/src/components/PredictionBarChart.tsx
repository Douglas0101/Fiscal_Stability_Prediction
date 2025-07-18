import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface PredictionBarChartProps {
  predictions: any[];
}

const PredictionBarChart: React.FC<PredictionBarChartProps> = ({ predictions }) => {
  // Aggregate predictions to count occurrences of each prediction value
  const predictionCounts = predictions.reduce((acc, curr) => {
    const predictionValue = curr.prediction;
    acc[predictionValue] = (acc[predictionValue] || 0) + 1;
    return acc;
  }, {});

  const chartData = Object.keys(predictionCounts).map(key => ({
    prediction: key,
    count: predictionCounts[key]
  }));

  return (
    <div className="card bg-base-100 shadow-xl mt-4">
      <div className="card-body">
        <h2 className="card-title">Distribuição das Previsões de Estabilidade Fiscal</h2>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart
            data={chartData}
            margin={{
              top: 5,
              right: 30,
              left: 20,
              bottom: 5,
            }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="prediction" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="count" fill="#8884d8" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default PredictionBarChart;