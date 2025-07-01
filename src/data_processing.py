# src/data_processing.py

import pandas as pd


def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processa os dados brutos, renomeando colunas de forma flexível,
    selecionando as essenciais e validando a presença das features críticas.
    """
    print("--- Iniciando processamento de dados ---")
    print("Colunas originais encontradas no arquivo CSV:")
    print(df.columns.tolist())

    # CORREÇÃO: Mapeamento ajustado para corresponder exatamente aos nomes do log.
    column_mapping = {
        # Nomes padrão já corretos
        'country_name': 'country_name',
        'year': 'year',
        'Unemployment Rate (%)': 'Unemployment Rate (%)',
        'Public Debt (% of GDP)': 'Public Debt (% of GDP)',

        # Mapeamentos corrigidos com base no log
        'GDP Growth (% Annual)': 'GDP Growth (%)',
        'Inflation (CPI %)': 'Inflation Rate (%)',
        'Current Account Balance (% GDP)': 'Current Account Balance (% of GDP)',

        # Mapeamento para uma possível coluna de balanço governamental
        'Government Expense (% of GDP)': 'Government Budget Balance (% of GDP)'  # Exemplo, pode precisar de cálculo
    }

    df.rename(columns=column_mapping, inplace=True)
    print("\nColunas após tentativa de mapeamento para nomes padrão:")
    print(df.columns.tolist())

    required_columns = [
        'country_name', 'year', 'Public Debt (% of GDP)', 'GDP Growth (%)',
        'Inflation Rate (%)', 'Unemployment Rate (%)',
        'Government Budget Balance (% of GDP)', 'Current Account Balance (% of GDP)'
    ]

    available_columns = [col for col in required_columns if col in df.columns]
    missing_columns = set(required_columns) - set(available_columns)

    if missing_columns:
        print(
            f"\nAVISO: As seguintes colunas padrão não foram encontradas ou mapeadas e serão ignoradas: {missing_columns}")

    critical_features = {'country_name', 'year', 'Public Debt (% of GDP)'}
    missing_critical = critical_features.intersection(missing_columns)

    if missing_critical:
        raise ValueError(
            f"Erro Crítico: Features essenciais estão faltando: {missing_critical}. Verifique o 'column_mapping'.")

    processed_df = df[available_columns].copy()
    processed_df['year'] = pd.to_numeric(processed_df['year'])

    print("\nProcessamento de dados e mapeamento de colunas concluídos com sucesso.")
    return processed_df
