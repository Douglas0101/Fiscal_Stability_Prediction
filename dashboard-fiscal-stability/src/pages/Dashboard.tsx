import { useEffect, useState } from 'react';
import Papa from 'papaparse';
import FiscalStabilityMap from '../components/FiscalStabilityMap';
import PredictionBarChart from '../components/PredictionBarChart';
import CountryTimeSeriesChart from '../components/CountryTimeSeriesChart';

interface CountryData {
  'country_id': string;
  'year': number;
  'Country Name': string;
  'Inflation (CPI %)': number;
  'GDP (Current USD)': number;
  'Unemployment Rate (%)': number;
  'Public Debt (% of GDP)': number;
  'fiscal_stability_index': number;
  'GDP Growth (% Annual)': number;
}

interface PredictionData {
  'country_id': string;
  'prediction': number;
}

const availableModels = ['xgb', 'ebm'];

const Dashboard = () => {
  const [data, setData] = useState<CountryData[]>([]);
  const [predictions, setPredictions] = useState<PredictionData[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('xgb');
  const [selectedCountry, setSelectedCountry] = useState<CountryData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<Error | null>(null);

  const fetchData = async () => {
    try {
      const response = await fetch('http://localhost:8000/data');
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const csvData = await response.text();
      const parsedData = Papa.parse(csvData, { header: true, dynamicTyping: true }).data as CountryData[];
      setData(parsedData);
    } catch (error: any) {
      console.error("Error fetching data:", error);
      setError(error);
    }
  };

  const fetchPredictions = async (model: string) => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`http://localhost:8000/predictions/${model}`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const csvData = await response.text();
      const parsedData = Papa.parse(csvData, { header: true, dynamicTyping: true }).data as PredictionData[];
      setPredictions(parsedData);
    } catch (error: any) {
      console.error(`Error fetching predictions for model ${model}:`, error);
      setError(error);
      setPredictions([]); // Clear predictions on error
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  useEffect(() => {
    if (!selectedModel) return;
    fetchPredictions(selectedModel);
  }, [selectedModel]);

  const countryHistoricalData = selectedCountry ? data.filter(d => d.country_id === selectedCountry.country_id) : [];

  if (loading) {
    return <div className="flex justify-center items-center h-screen text-xl">Carregando dados...</div>;
  }

  if (error) {
    return <div className="flex justify-center items-center h-screen text-xl text-red-500">Erro ao carregar dados: {error.message}</div>;
  }

  return (
    <div className="p-4 bg-gray-100 min-h-screen">
      <div className="flex justify-between items-center mb-6 bg-white p-4 rounded-lg shadow-md">
        <h1 className="text-3xl font-extrabold text-gray-800">Dashboard de Estabilidade Fiscal</h1>
        <div className="form-control w-full max-w-xs">
          <label className="label">
            <span className="label-text text-gray-700">Selecionar Modelo</span>
          </label>
          <select
            className="select select-bordered select-primary w-full"
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
          >
            {availableModels.map((model) => (
              <option key={model} value={model}>
                {model.toUpperCase()}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
        <div className="stats shadow bg-white rounded-lg p-4">
          <div className="stat">
            <div className="stat-title text-gray-600">Total de Países</div>
            <div className="stat-value text-primary">{data.length}</div>
          </div>
        </div>
        <div className="stats shadow bg-white rounded-lg p-4">
          <div className="stat">
            <div className="stat-title text-gray-600">Previsões Geradas</div>
            <div className="stat-value text-secondary">{predictions.length}</div>
          </div>
        </div>
        <div className="stats shadow bg-white rounded-lg p-4">
          <div className="stat">
            <div className="stat-title text-gray-600">Modelo Utilizado</div>
            <div className="stat-value text-accent">{selectedModel.toUpperCase()}</div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <FiscalStabilityMap predictions={predictions} data={data} setSelectedCountry={setSelectedCountry} />
        </div>
        <div className="lg:col-span-1">
          <PredictionBarChart predictions={predictions} />
        </div>
      </div>

      {selectedCountry && (
        <div className="card bg-white shadow-xl mt-6 p-6 rounded-lg relative">
          <h2 className="text-2xl font-bold mb-4 text-gray-800">Detalhes do País: {selectedCountry['Country Name']} ({selectedCountry.year})</h2>
          <p className="text-lg mb-2">Previsão de Estabilidade Fiscal ({selectedModel.toUpperCase()}): <span className="font-semibold text-blue-600">{predictions.find(p => p.country_id === selectedCountry.country_id)?.prediction}</span></p>
          <div className="divider text-gray-500">Indicadores Chave</div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-gray-700">
            <p><strong>Inflação (CPI %):</strong> {selectedCountry['Inflation (CPI %)']}</p>
            <p><strong>GDP (Current USD):</strong> {selectedCountry['GDP (Current USD)']}</p>
            <p><strong>Unemployment Rate (%):</strong> {selectedCountry['Unemployment Rate (%)']}</p>
            <p><strong>Public Debt (% of GDP):</strong> {selectedCountry['Public Debt (% of GDP)']}</p>
          </div>
          {countryHistoricalData.length > 0 && (
            <CountryTimeSeriesChart countryData={countryHistoricalData} />
          )}
          <button className="btn btn-sm btn-ghost absolute top-4 right-4 text-gray-500 hover:text-gray-800" onClick={() => setSelectedCountry(null)}>✕</button>
        </div>
      )}
    </div>
  );
};

export default Dashboard;