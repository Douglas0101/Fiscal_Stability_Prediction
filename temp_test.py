# temp_test.py
# Salve este ficheiro na raiz do seu projeto (ao lado de main.py)

print("--- INICIANDO TESTE DE IMPORTAÇÃO DIRETA ---")

try:
    # Primeiro, vamos garantir que o ficheiro de configuração carrega sem erros.
    print("Passo 1: Tentando importar 'settings' de 'src.config'...")
    from src.config import settings
    print("SUCESSO: 'settings' importado com sucesso.")
    print(f"DEBUG: Modo DEBUG em config.py está definido como: {settings.DEBUG}")

    # Agora, o teste principal.
    print("\nPasso 2: Tentando importar 'process_data' de 'src.data_processing'...")
    # A linha abaixo vai carregar e executar o código em data_processing.py
    from src.data_processing import process_data
    print("SUCESSO: 'process_data' importado com sucesso.")
    print("\n--- TESTE CONCLUÍDO COM SUCESSO! O PROBLEMA NÃO É DE IMPORTAÇÃO. ---")
    print("O seu ambiente está correto. Por favor, tente executar 'python main.py train-model' novamente.")


except ImportError as e:
    print("\n--- FALHA NO TESTE: Ocorreu um ImportError ---")
    print(f"ERRO: {e}")
    print("\nIsto confirma que o problema está no carregamento do módulo, provavelmente devido a cache ou um erro de sintaxe.")
    print("Por favor, execute o comando de limpeza da cache e tente rodar este script mais uma vez.")
    print("Comando de limpeza: find . -type d -name \"__pycache__\" -exec rm -r {} + && find . -type f -name \"*.pyc\" -delete")


except Exception as e:
    print(f"\n--- FALHA NO TESTE: Ocorreu um erro inesperado: {type(e).__name__} ---")
    print(f"ERRO: {e}")

