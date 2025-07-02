
"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface Prediction {
  id: number;
  features: { [key: string]: number };
  prediction: number;
  createdAt: string;
}

interface PredictionHistoryProps {
  history: Prediction[];
}

export function PredictionHistory({ history }: PredictionHistoryProps) {
  return (
    <Card className="mt-8 w-full max-w-6xl">
      <CardHeader>
        <CardTitle>Histórico de Predições</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="relative w-full overflow-auto">
          <table className="w-full caption-bottom text-sm">
            <thead className="[&_tr]:border-b">
              <tr className="border-b transition-colors hover:bg-muted/50 data-[state=selected]:bg-muted">
                <th className="h-12 px-4 text-left align-middle font-medium text-muted-foreground [&:has([role=checkbox])]:pr-0 w-[100px]">
                  ID
                </th>
                <th className="h-12 px-4 text-left align-middle font-medium text-muted-foreground [&:has([role=checkbox])]:pr-0">
                  Predição
                </th>
                <th className="h-12 px-4 text-left align-middle font-medium text-muted-foreground [&:has([role=checkbox])]:pr-0">
                  Data
                </th>
              </tr>
            </thead>
            <tbody className="[&_tr:last-child]:border-0">
              {history.map((item) => (
                <tr key={item.id} className="border-b transition-colors hover:bg-muted/50 data-[state=selected]:bg-muted">
                  <td className="p-4 align-middle [&:has([role=checkbox])]:pr-0 font-medium">{item.id}</td>
                  <td className="p-4 align-middle [&:has([role=checkbox])]:pr-0">{item.prediction.toFixed(4)}</td>
                  <td className="p-4 align-middle [&:has([role=checkbox])]:pr-0">{new Date(item.createdAt).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
}
