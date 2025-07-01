import pytest
import pandas as pd
import os
from src import data_processing
from src import config

# Mock do caminho do arquivo de dados brutos para o teste
@pytest.fixture
def mock_raw_data_path(tmp_path):
    # Cria um arquivo CSV temporário para o teste
    test_csv_content = """country_name,country_id,year,Inflation (CPI %)
Brazil,BR,2020,5.0
Brazil,BR,2021,6.0
"""
    test_file = tmp_path / "world_bank_data_2025.csv"
    test_file.write_text(test_csv_content)
    return str(test_file)

def test_carregar_dados_sucesso(mock_raw_data_path):
    # Testa se a função carrega os dados corretamente
    df = data_processing.carregar_dados(mock_raw_data_path)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert 'country_name' in df.columns
    assert len(df) == 2

def test_carregar_dados_arquivo_nao_encontrado():
    # Testa se a função levanta FileNotFoundError para arquivo inexistente
    with pytest.raises(FileNotFoundError):
        data_processing.carregar_dados("caminho/inexistente/arquivo.csv")
