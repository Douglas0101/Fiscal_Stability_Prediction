import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno


def analisar_padroes_de_ausencia(df: pd.DataFrame,
                                 variavel_alvo: str,
                                 coluna_pais: str,
                                 coluna_ano: str):
    """
    Realiza uma Análise Exploratória de Dados (EDA) focada em diagnosticar
    os padrões de dados ausentes num DataFrame.
    """
    if df is None:
        print("DataFrame não fornecido. A análise não pode continuar.")
        return

    print("\n" + "=" * 60)
    print("INÍCIO DA ANÁLISE EXPLORATÓRIA DE DADOS AUSENTES (EDA)")
    print("=" * 60)

    # Configurações de visualização
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 7)

    # --- ANÁLISE 1: PADRÕES DE AUSÊNCIA POR PAÍS (DIMENSÃO GEOGRÁFICA) ---
    print(f"\n--- 1. Análise de Ausência por País para a variável '{variavel_alvo}' ---")

    ausencia_por_pais = df.groupby(coluna_pais)[variavel_alvo].apply(
        lambda x: x.isnull().mean() * 100).sort_values(ascending=False)
    top_15_paises = ausencia_por_pais.head(15)

    print("\nTop 15 Países com Maior Percentual de Dados Ausentes:")
    print(top_15_paises.round(2).to_string())

    # Visualização com correção do FutureWarning
    plt.figure()
    sns.barplot(x=top_15_paises.index, y=top_15_paises.values,
                hue=top_15_paises.index, palette="viridis", legend=False)
    plt.title(f"Top 15 Países por % de Dados Ausentes em '{variavel_alvo}'",
              fontsize=16)
    plt.xlabel("País", fontsize=12)
    plt.ylabel("Percentual de Ausência (%)", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    # --- ANÁLISE 2: PADRÕES DE AUSÊNCIA POR ANO (DIMENSÃO TEMPORAL) ---
    print(f"\n--- 2. Análise de Ausência por Ano para a variável '{variavel_alvo}' ---")

    ausencia_por_ano = df.groupby(coluna_ano)[variavel_alvo].apply(
        lambda x: x.isnull().mean() * 100)
    plt.figure()
    sns.lineplot(x=ausencia_por_ano.index, y=ausencia_por_ano.values,
                 marker='o', color='crimson')
    plt.title(f"Evolução da % de Dados Ausentes em '{variavel_alvo}' por Ano",
              fontsize=16)
    plt.xlabel("Ano", fontsize=12)
    plt.ylabel("Percentual de Ausência (%)", fontsize=12)
    plt.xticks(ausencia_por_ano.index.astype(int), rotation=45)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

    # --- ANÁLISE 3: CORRELAÇÃO DE AUSÊNCIA ENTRE VARIÁVEIS ---
    print("\n--- 3. Análise de Correlação de Nulidade entre Todas as Variáveis ---")
    print("A seguir, serão exibidas visualizações da biblioteca 'missingno' e "
          "alternativas robustas.")

    # a. Matriz de Ausência (Workaround para o erro de versão)
    print("\nVisualização 3a: Matriz de Nulidade")
    print("Mostra onde os dados estão em falta. Áreas claras indicam dados ausentes.")
    try:
        plt.figure(figsize=(12, 7))
        sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
        plt.title("Matriz de Nulidade do Dataset (Visualização Alternativa)",
                  fontsize=16)
        plt.show()
    except Exception as e:
        print(f"AVISO: Não foi possível gerar a matriz de nulidade alternativa: {e}")

    # b. Heatmap de Correlação de Nulidade (com try-except para robustez)
    print("\nVisualização 3b: Heatmap de Correlação de Nulidade")
    print(
        "Mostra a correlação de ausência entre colunas. Um valor de 1 significa "
        "que se o dado falta numa coluna, ele sempre falta na outra.")
    try:
        msno.heatmap(df, cmap='viridis')
        plt.title("Heatmap de Correlação de Nulidade", fontsize=16)
        plt.show()
    except Exception as e:
        print(f"AVISO: Não foi possível gerar o heatmap de nulidade do missingno: {e}")

    # c. Dendrograma (com try-except para robustez)
    print("\nVisualização 3c: Dendrograma de Nulidade")
    print("Agrupa colunas que têm padrões de ausência semelhantes.")
    try:
        msno.dendrogram(df)
        plt.title("Dendrograma de Agrupamento por Nulidade", fontsize=16)
        plt.show()
    except Exception as e:
        print(f"AVISO: Não foi possível gerar o dendrograma de nulidade do missingno: {e}")

    print("\n" + "=" * 60)
    print("ANÁLISE EXPLORATÓRIA DE DADOS AUSENTES CONCLUÍDA")
    print("=" * 60)


def sumario_interpretativo():
    """Imprime um sumário com a interpretação dos resultados esperados da análise."""
    print("\n" + "=" * 60)
    print("SUMÁRIO DA ANÁLISE DE DADOS AUSENTES")
    print("=" * 60)
    print("""
Com base na análise exploratória focada em dados ausentes, foram identificados os seguintes
padrões:
1.  **Concentração Geográfica:** A ausência de dados na variável alvo está fortemente
    concentrada num grupo específico de países.
2.  **Tendência Temporal:** O gráfico de linha temporal indica se o problema de dados faltantes
    foi mais acentuado num período específico.
3.  **Correlação de Ausência:** O heatmap de nulidade revela se a ausência da variável alvo está
    correlacionada com outras colunas.

**Próximos Passos Sugeridos:**
- Considerar a exclusão estratégica de países com mais de 90% de dados faltantes.
- Utilizar métodos de imputação multivariada (como MICE) se a correlação de ausência for alta.
- Investigar causas externas para a falta de dados em períodos específicos.
""")
