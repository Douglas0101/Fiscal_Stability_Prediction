
import Image from "next/image";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export function ShapAnalysis() {
  return (
    <div className="mt-12 w-full max-w-6xl">
      <h2 className="text-3xl font-bold text-center mb-6">Análise de Importância das Features (SHAP)</h2>
      <p className="text-lg text-muted-foreground text-center mb-8">
        As imagens abaixo mostram a importância de cada feature para as predições do modelo.
      </p>
      <div className="grid md:grid-cols-2 gap-8">
        <Card>
          <CardHeader>
            <CardTitle className="text-center">SHAP Summary Plot (Bar)</CardTitle>
          </CardHeader>
          <CardContent>
            <Image
              src="/shap_summary_bar.png"
              alt="SHAP Summary Plot (Bar)"
              width={600}
              height={400}
              objectFit="contain"
              className="rounded-md"
            />
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle className="text-center">SHAP Summary Plot (Beeswarm)</CardTitle>
          </CardHeader>
          <CardContent>
            <Image
              src="/shap_summary_beeswarm.png"
              alt="SHAP Summary Plot (Beeswarm)"
              width={600}
              height={400}
              objectFit="contain"
              className="rounded-md"
            />
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
