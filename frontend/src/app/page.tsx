
'use client';

import { useState, useEffect } from 'react';
import { useSession, signIn } from 'next-auth/react';
import { PredictionForm } from '@/components/PredictionForm';
import { PredictionResult } from '@/components/PredictionResult';
import { ShapAnalysis } from '@/components/ShapAnalysis';
import { DataVisualization } from '@/components/DataVisualization';
import { PredictionHistory } from '@/components/PredictionHistory';
import { Button } from '@/components/ui/button';

interface FeatureData {
  [key: string]: number;
}

interface Prediction {
  id: number;
  features: { [key: string]: number };
  prediction: number;
  createdAt: string;
}

export default function Home() {
  const { data: session, status } = useSession();
  const [prediction, setPrediction] = useState<number | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [features, setFeatures] = useState<FeatureData>({});
  const [history, setHistory] = useState<Prediction[]>([]);

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const response = await fetch('/api/history');
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        setHistory(data);
      } catch (e: any) {
        console.error("Failed to fetch history:", e);
        setError(`Failed to fetch history: ${e.message}`);
      }
    };
    if (status === 'authenticated') {
      fetchHistory();
    }
  }, [status]);

  const handlePrediction = async (newPrediction: number) => {
    setPrediction(newPrediction);
    try {
      const response = await fetch('/api/history', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ features, prediction: newPrediction }),
      });
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const savedPrediction = await response.json();
      setHistory((prevHistory) => [savedPrediction, ...prevHistory]);
    } catch (e: any) {
      console.error("Failed to save prediction to history:", e);
      setError(`Failed to save prediction to history: ${e.message}`);
    }
  };

  if (status === 'loading') {
    return <p>Loading...</p>;
  }

  if (status === 'unauthenticated') {
    return (
      <div className="min-h-screen bg-background flex flex-col items-center justify-center p-4">
        <h1 className="text-4xl font-bold mb-4">Acesso Negado</h1>
        <p className="text-lg text-muted-foreground mb-8">Você precisa estar logado para acessar esta página.</p>
        <Button onClick={() => signIn()}>Login</Button>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background flex flex-col items-center justify-center p-4">
      <header className="mb-10 text-center">
        <h1 className="text-4xl font-bold">📊 Dashboard de Predição de Estabilidade Fiscal</h1>
        <p className="text-lg text-muted-foreground mt-2">Insira os valores para as features e obtenha uma predição.</p>
      </header>

      {error && (
        <div className="bg-destructive text-destructive-foreground p-4 rounded-md mb-4 w-full max-w-2xl text-center" role="alert">
          {error}
        </div>
      )}

      <main className="w-full flex flex-col items-center">
        <PredictionForm 
          features={features}
          onFeaturesChange={setFeatures}
          onPrediction={handlePrediction} 
          onError={setError} 
          setLoading={setLoading} 
          loading={loading} 
        />
        <PredictionResult prediction={prediction} />
        <DataVisualization features={features} />
        <ShapAnalysis />
        <PredictionHistory history={history} />
      </main>

      <footer className="mt-12 text-muted-foreground text-sm">
        Desenvolvido para o Projeto de Estabilidade Fiscal
      </footer>
    </div>
  );
}
