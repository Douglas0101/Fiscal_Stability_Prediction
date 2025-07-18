import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface CountryTimeSeriesChartProps {
  countryData: any[];
}

const CountryTimeSeriesChart: React.FC<CountryTimeSeriesChartProps> = ({ countryData }) => {
  // Filter data for the selected country and sort by year
  const chartData = countryData.sort((a, b) => a.year - b.year);

  return (
    <div className="card bg-base-100 shadow-xl mt-4">
      <div className="card-body">
        <h2 className="card-title">Evolução Histórica para {chartData[0]?.['Country Name']}</h2>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart
            data={chartData}
            margin={{
              top: 5,
              right: 30,
              left: 20,
              bottom: 5,
            }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="year" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="fiscal_stability_index" stroke="#8884d8" name="Índice de Estabilidade Fiscal" />
            <Line type="monotone" dataKey="GDP Growth (% Annual)" stroke="#82ca9d" name="Crescimento do PIB (% Anual)" />
            <Line type="monotone" dataKey="Public Debt (% of GDP)" stroke="#ffc658" name="Dívida Pública (% do PIB)" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default CountryTimeSeriesChart;