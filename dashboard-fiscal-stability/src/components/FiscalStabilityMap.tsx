import React from 'react';
import { ComposableMap, Geographies, Geography, ZoomableGroup } from 'react-simple-maps';
import { scaleQuantize } from 'd3-scale';

interface FiscalStabilityMapProps {
  predictions: PredictionData[];
  data: CountryData[];
  setSelectedCountry: (country: CountryData | null) => void;
}

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

const geoUrl = 'https://raw.githubusercontent.com/deldersveld/topojson/master/world-countries.json';

const FiscalStabilityMap: React.FC<FiscalStabilityMapProps> = ({ predictions, data, setSelectedCountry }) => {
  const colorScale = scaleQuantize()
    .domain([0, 1] as number[])
    .range(['#ffedea', '#ffcec5', '#ffad9f', '#ff8a75', '#ff5533', '#e2492d', '#c93e2c', '#b0322a', '#972626'] as string[]);

  return (
    <div className="card bg-base-100 shadow-xl">
      <div className="card-body">
        <h2 className="card-title">Mapa de Estabilidade Fiscal</h2>
        <ComposableMap>
          <ZoomableGroup>
            <Geographies geography={geoUrl}>
              {({ geographies }) =>
                geographies.map((geo) => {
                  const d = predictions.find((s) => s.country_id === geo.properties.id);
                  const countryData = data.find((s) => s.country_id === geo.properties.id);
                  return (
                    <Geography
                      key={geo.rsmKey}
                      geography={geo}
                      fill={d ? colorScale(Number(d.prediction)) : '#F5F4F6'}
                      onClick={() => setSelectedCountry(countryData || null)}
                      style={{
                        default: { outline: "none" },
                        hover: { outline: "none", stroke: "#333", strokeWidth: 1 },
                        pressed: { outline: "none" },
                      }}
                    />
                  );
                })
              }
            </Geographies>
          </ZoomableGroup>
        </ComposableMap>
      </div>
    </div>
  );
};

export default FiscalStabilityMap;