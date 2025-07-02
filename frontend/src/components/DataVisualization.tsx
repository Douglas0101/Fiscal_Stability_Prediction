
"use client";

import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface DataVisualizationProps {
  features: { [key: string]: number };
}

export function DataVisualization({ features }: DataVisualizationProps) {
  const data = Object.entries(features).map(([name, value]) => ({ name, value }));

  return (
    <Card className="mt-8 w-full max-w-6xl">
      <CardHeader>
        <CardTitle>Visualização das Features</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={400}>
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="value" fill="#8884d8" />
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
