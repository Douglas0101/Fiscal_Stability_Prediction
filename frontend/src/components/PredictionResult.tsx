
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface PredictionResultProps {
  prediction: number | null;
}

export function PredictionResult({ prediction }: PredictionResultProps) {
  if (prediction === null) return null;

  return (
    <Card className="mt-8 w-full max-w-2xl">
      <CardHeader>
        <CardTitle>Resultado da Predição</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="text-center">
          <p className="text-3xl font-bold">{prediction.toFixed(4)}</p>
          <p className="text-md mt-2 text-muted-foreground">
            Um valor mais alto geralmente indica maior estabilidade fiscal.
          </p>
        </div>
      </CardContent>
    </Card>
  );
}
