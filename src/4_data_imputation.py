import pandas as pd


def imputar_dados_temporais(df: pd.DataFrame, coluna_pais: str, coluna_ano: str) -> pd.DataFrame:
    """
    Preenche os valores ausentes num DataFrame de dados em painel usando uma
    estratégia refinada em duas fases: interpolação temporal e imputação global
    por mediana.

    Args:
        df (pd.DataFrame): O DataFrame a ser imputado.
        coluna_pais (str): O nome da coluna que identifica o país.
        coluna_ano (str): O nome da coluna que identifica o ano.

    Returns:
        pd.DataFrame: O DataFrame com os dados 100% preenchidos.
    """
    print("\n" + "=" * 60)
    print("INÍCIO DA IMPUTAÇÃO DE DADOS AUSENTES (ESTRATÉGIA REFINADA)")
    print("=" * 60)

    # a. Ordenação dos Dados - Crítico para a lógica temporal
    print(f"1. Ordenando os dados por '{coluna_pais}' e '{coluna_ano}'...")
    df_imputado = df.sort_values(by=[coluna_pais, coluna_ano])

    # Identifica colunas numéricas para imputação
    colunas_numericas = df_imputado.select_dtypes(include='number').columns.tolist()
    if coluna_ano in colunas_numericas:
        colunas_numericas.remove(coluna_ano)

    print(f"\n2. Aplicando imputação às seguintes colunas numéricas: "
          f"{colunas_numericas}")

    # --- FASE 1: IMPUTAÇÃO TEMPORAL POR GRUPO (PAÍS) ---
    print("   -> Etapa 1.1: Interpolação linear para preencher lacunas internas...")
    df_imputado[colunas_numericas] = df_imputado.groupby(coluna_pais)[colunas_numericas].transform(
        lambda x: x.interpolate(method='linear', limit_direction='both')
    )

    print("   -> Etapa 1.2: Preenchimento de pontas (ffill/bfill) para garantir "
          "completude dentro dos grupos...")
    df_imputado[colunas_numericas] = df_imputado.groupby(coluna_pais)[colunas_numericas].transform(lambda x: x.ffill())
    df_imputado[colunas_numericas] = df_imputado.groupby(coluna_pais)[colunas_numericas].transform(lambda x: x.bfill())

    # Verificação intermediária
    ausentes_intermediario = df_imputado.isnull().sum().sum()
    print(f"\n   -> Células vazias restantes após a imputação temporal: "
          f"{ausentes_intermediario}")

    # --- FASE 2: IMPUTAÇÃO GLOBAL POR MEDIANA (FALLBACK DE SEGURANÇA) ---
    if ausentes_intermediario > 0:
        print("\n3. Iniciando fallback de imputação global para colunas "
              "completamente vazias em alguns grupos...")
        for col in colunas_numericas:
            if df_imputado[col].isnull().any():
                mediana_global = df_imputado[col].median()
                # --- CORREÇÃO APLICADA ---
                # Substituímos o inplace=True pela reatribuição explícita para evitar o FutureWarning.
                df_imputado[col] = df_imputado[col].fillna(mediana_global)
                print(
                    f"   -> Coluna '{col}' teve NaNs restantes preenchidos com a "
                    f"mediana global ({mediana_global:.2f}).")

    # --- Verificação Final ---
    total_ausentes_final = df_imputado.isnull().sum().sum()
    print("\n4. Verificação Final de Nulos...")
    print(f"   -> Total de células vazias após a imputação: {total_ausentes_final}")

    assert total_ausentes_final == 0, "ERRO CRÍTICO: Ainda existem dados ausentes após a imputação!"

    print("\nIMPUTAÇÃO CONCLUÍDA COM SUCESSO. O dataset está 100% completo.")
    print("=" * 60)

    return df_imputado