import pandas as pd
from typing import List, Optional


def preprocessar_e_limpar_dados(
        df: pd.DataFrame,
        variavel_alvo: str,
        coluna_pais: str,
        limiar_ausencia_pais: float = 0.9,
        anos_a_remover: Optional[List[int]] = None
) -> Optional[pd.DataFrame]:
    """
    Realiza a limpeza de um DataFrame removendo anos e países com base em
    critérios definidos.

    Args:
        df (pd.DataFrame): O DataFrame bruto a ser processado.
        variavel_alvo (str): A coluna alvo usada como referência para a
                             filtragem de países.
        coluna_pais (str): A coluna que identifica os países.
        limiar_ausencia_pais (float, optional): O limiar (0 a 1) de dados
                                                ausentes na variável alvo para
                                                remover um país. Padrão é 0.9 (90%).
        anos_a_remover (Optional[List[int]], optional): Uma lista de anos a
                                                        serem removidos do dataset.
                                                        Padrão é None.

    Returns:
        Optional[pd.DataFrame]: O DataFrame limpo, ou None se ocorrer um erro.
    """
    if df is None:
        print("DataFrame de entrada é nulo. Interrompendo o pré-processamento.")
        return None

    if anos_a_remover is None:
        anos_a_remover = []

    print("\n" + "=" * 60)
    print("INÍCIO DO PRÉ-PROCESSAMENTO E LIMPEZA DE DADOS")
    print("=" * 60)

    # --- Relatório Inicial ---
    dimensoes_antes = df.shape
    paises_antes = set(df[coluna_pais].unique())
    total_celulas_vazias_antes = df.isnull().sum().sum()
    total_celulas_antes = df.size

    df_limpo = df.copy()

    # --- a. Filtragem por Ano ---
    if anos_a_remover:
        print(f"\n1. Removendo os anos especificados: {anos_a_remover}...")
        df_limpo = df_limpo[~df_limpo['year'].isin(anos_a_remover)]
        print(f"   -> {dimensoes_antes[0] - df_limpo.shape[0]} linhas foram removidas.")

    # --- b. Filtragem por País ---
    print(f"\n2. Removendo países com mais de {limiar_ausencia_pais:.0%} de dados "
          f"ausentes em '{variavel_alvo}'...")

    # Cálculo da porcentagem de ausência por país
    ausencia_por_pais = df_limpo.groupby(coluna_pais)[variavel_alvo].apply(
        lambda x: x.isnull().mean())

    # Identificação dos países a serem removidos
    paises_a_remover = set(ausencia_por_pais[ausencia_por_pais > limiar_ausencia_pais].index)

    if paises_a_remover:
        df_limpo = df_limpo[~df_limpo[coluna_pais].isin(paises_a_remover)]
        print(f"   -> {len(paises_a_remover)} países foram identificados e removidos.")
    else:
        print("   -> Nenhum país atendeu ao critério de remoção.")

    # --- c. Relatório Final de Limpeza ---
    dimensoes_depois = df_limpo.shape
    paises_depois = set(df_limpo[coluna_pais].unique())
    total_celulas_vazias_depois = df_limpo.isnull().sum().sum()
    total_celulas_depois = df_limpo.size

    print("\n" + "=" * 60)
    print("RELATÓRIO DE LIMPEZA")
    print("=" * 60)
    print(f"Dimensões ANTES: {dimensoes_antes[0]} linhas, {dimensoes_antes[1]} colunas.")
    print(f"Dimensões DEPOIS: {dimensoes_depois[0]} linhas, {dimensoes_depois[1]} colunas.")
    print("-" * 20)
    print(f"Países ANTES: {len(paises_antes)}")
    print(f"Países DEPOIS: {len(paises_depois)}")
    print(
        f"Países Removidos ({len(paises_a_remover)}): "
        f"{sorted(list(paises_a_remover)) if paises_a_remover else 'Nenhum'}")
    print("-" * 20)
    print(f"Anos Removidos: {anos_a_remover if anos_a_remover else 'Nenhum'}")
    print("-" * 20)
    porcentagem_antes = (total_celulas_vazias_antes / total_celulas_antes) * 100 \
        if total_celulas_antes > 0 else 0
    porcentagem_depois = (total_celulas_vazias_depois / total_celulas_depois) * 100 \
        if total_celulas_depois > 0 else 0
    print(f"Células Vazias ANTES: {total_celulas_vazias_antes} ({porcentagem_antes:.2f}%)")
    print(f"Células Vazias DEPOIS: {total_celulas_vazias_depois} ({porcentagem_depois:.2f}%)")
    print("=" * 60)

    return df_limpo