Modelo Preditivo para Análise de Estabilidade Fiscal

📖 Visão Geral

Este projeto é uma solução completa de Machine Learning para prever o índice de estabilidade fiscal de países, utilizando dados do Banco Mundial. A aplicação inclui um pipeline de ML automatizado, uma API para servir os modelos e um dashboard interativo para visualização de dados e previsões.

O objetivo é fornecer uma ferramenta robusta para análise de risco soberano, permitindo que os usuários compreendam os fatores que mais impactam a estabilidade fiscal e acessem previsões atualizadas através de uma interface moderna e intuitiva.

➡️ Acessar o Dashboard (link para a instância em execução)

✨ Funcionalidades
O projeto é dividido em um backend robusto e um frontend moderno:

Backend (API & Pipeline de ML)
Pipeline Automatizado: Orquestração completa do pipeline de ML com um único comando, incluindo processamento de dados, treinamento e versionamento de modelos.

Dois Modelos Preditivos: Utiliza XGBoost para performance e Explainable Boosting Machine (EBM) para interpretabilidade.

MLOps Robusto: Rastreamento de experimentos e versionamento de modelos com MLflow.

API de Alta Performance: Uma API construída com FastAPI para servir os modelos e fornecer previsões.

Banco de Dados: Gerenciamento de dados e persistência com SQLAlchemy.

Interpretabilidade: Geração de gráficos de importância de features e valores SHAP para entender as decisões do modelo.

Frontend (Dashboard)
Interface Moderna: Design limpo e responsivo construído com Chakra UI e Tailwind CSS.

Visualização Interativa: Gráficos dinâmicos e de alto desempenho com Recharts para explorar os dados e os resultados do modelo.

Performance: Carregamento rápido e otimizado com Vite e código fortemente tipado com TypeScript.

🛠️ Tecnologias Utilizadas

Área

Tecnologias

Backend & ML

Python, FastAPI, SQLAlchemy, Pydantic, XGBoost, InterpretML (EBM), MLflow, SHAP, Scikit-learn

Frontend

React, TypeScript, Vite, Chakra UI, Tailwind CSS, Recharts

Infra & DevOps

Docker, Docker Compose, Nginx

Banco de Dados

PostgreSQL (produção), SQLite (desenvolvimento)


Exportar para as Planilhas

🏛️ Arquitetura e Estrutura do Projeto

A aplicação é totalmente containerizada com Docker e orquestrada com Docker Compose, garantindo um ambiente de desenvolvimento e produção consistente.

/
├── .github/          # Workflows de CI/CD (ex: GitHub Actions)

├── data/             # Dados brutos, processados e para features

├── dashboard-fiscal-stability/ # Código-fonte do frontend em React

├── mlflow_docker/    # Dockerfile para o serviço do MLflow

├── mlruns_data/      # Artefatos e logs do MLflow

├── notebooks/        # Notebooks para exploração e modelagem inicial

├── src/              # Código-fonte do backend (API e pipeline)

├── tests/            # Testes unitários e de integração

├── .env.example      # Exemplo de arquivo de variáveis de ambiente

├── docker-compose.yml# Orquestração dos serviços (API, DB, MLflow, Frontend)

├── Dockerfile        # Container para a API Python

└── main.py           # Orquestrador do pipeline de ML

🚀 Como Iniciar (Ambiente Local com Docker)

Este projeto foi desenhado para ser executado com Docker. Siga os passos abaixo para iniciar todos os serviços.

Pré-requisitos

Docker

Docker Compose

Python 3.10+ (apenas para executar o script orquestrador main.py)

Instalação e Execução
Clone o repositório:

Bash

**git clone https://github.com/douglas0101/fiscal_stability_prediction.git
cd fiscal_stability_prediction
Configure as Variáveis de Ambiente:
Copie o arquivo de exemplo e preencha com suas configurações.**

Bash

**cp .env.example .env
Edite o arquivo .env se precisar alterar portas ou credenciais do banco de dados.**

**Instale as dependências do orquestrador:**

Bash

**pip install -r requirements.txt
Execute o Orquestrador do Pipeline:
Este comando irá subir todos os contêineres Docker, executar o pipeline de processamento de dados, treinar os modelos e, ao final, reiniciar a API para carregar os novos artefatos.**

Bash

**python main.py
O que esperar após a execução:
API FastAPI: Disponível em http://localhost:8000/docs.**

**Dashboard React: Disponível em http://localhost:8501.**

**MLflow UI: Disponível em http://localhost:5000.**

🕹️ Uso
Executando o Pipeline de ML Manualmente
Você pode executar partes do pipeline de forma isolada através do docker-compose exec:

Bash

# Executar o processamento de dados
**docker-compose --env-file .env exec api python src/data_processing.py**

# Treinar um modelo específico (ex: xgb)
**docker-compose --env-file .env exec api python src/train.py xgb
API Endpoints
A API expõe os seguintes endpoints principais:**

GET /api/v1/health: Verifica a saúde da aplicação.

POST /api/v1/predict: Envia dados de um país/ano para obter uma previsão de estabilidade fiscal.

GET /api/v1/countries: Lista os países disponíveis para análise.

Consulte a documentação interativa em http://localhost:8000/docs para detalhes completos.

🤝 Como Contribuir

Agradecemos o seu interesse em contribuir! Por favor, siga as seguintes diretrizes:

Crie um fork do projeto.

Crie uma nova branch para a sua funcionalidade (git checkout -b feature/minha-feature).

Faça o commit das suas alterações (git commit -m 'Adiciona nova feature').

Faça o push para a sua branch (git push origin feature/minha-feature).

Abra um Pull Request.

Certifique-se de que o seu código segue os padrões de estilo do projeto e que todos os testes passam.





