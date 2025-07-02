"use client";

import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

interface FeatureData {
  [key: string]: number;
}

interface PredictionFormProps {
  features: FeatureData;
  onFeaturesChange: (features: FeatureData) => void;
  onPrediction: (prediction: number) => void;
  onError: (error: string | null) => void;
  setLoading: (loading: boolean) => void;
  loading: boolean;
}

export function PredictionForm({ features, onFeaturesChange, onPrediction, onError, setLoading, loading }: PredictionFormProps) {
  const [featureNames, setFeatureNames] = useState<string[]>([]);

  useEffect(() => {
    const fetchFeatureNames = async () => {
      try {
        const response = await fetch("http://localhost:5000/features");
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        setFeatureNames(data.feature_names);
        const initialFeatures: FeatureData = {};
        data.feature_names.forEach((name: string) => {
          initialFeatures[name] = 0.0;
        });
        onFeaturesChange(initialFeatures);
      } catch (e: any) {
        onError(`Failed to fetch feature names: ${e.message}`);
      }
    };
    fetchFeatureNames();
  }, [onError, onFeaturesChange]);

  const handleFeatureChange = (name: string, value: string) => {
    onFeaturesChange({
      ...features,
      [name]: parseFloat(value),
    });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    onError(null);
    onPrediction(null as any);

    try {
      const response = await fetch("http://localhost:5000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(features),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      onPrediction(data.prediction);
    } catch (e: any) {
      onError(`Failed to get prediction: ${e.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card className="w-full max-w-2xl">
      <CardHeader>
        <CardTitle>Insira os Dados para Predição</CardTitle>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit}>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
            {featureNames.length > 0 ? (
              featureNames.map((name) => (
                <div key={name} className="grid w-full max-w-sm items-center gap-1.5">
                  <Label htmlFor={name}>{name}</Label>
                  <Input
                    type="number"
                    id={name}
                    name={name}
                    value={features[name] || ""}
                    onChange={(e) => handleFeatureChange(name, e.target.value)}
                    step="any"
                  />
                </div>
              ))
            ) : (
              <p>Carregando features...</p>
            )}
          </div>
          <Button
            type="submit"
            className="w-full"
            disabled={loading || featureNames.length === 0}
          >
            {loading ? "Fazendo Predição..." : "Fazer Predição"}
          </Button>
        </form>
      </CardContent>
    </Card>
  );
}