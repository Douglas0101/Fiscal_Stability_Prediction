# Dockerfile

# 1. Usar uma imagem base Python oficial
FROM python:3.10-slim

# 2. Definir variáveis de ambiente para otimizar o Python no Docker
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# 3. Definir o diretório de trabalho dentro do container
WORKDIR /app

# 4. Copiar o arquivo de dependências e instalar
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. Copiar todo o código fonte e os dados necessários para o container
# Isso garante que tanto a API quanto o treinamento tenham acesso a tudo.
COPY ./src ./src
COPY ./data ./data

# 6. Expor a porta que a API usará
EXPOSE 8000

# 7. Comando padrão para iniciar o servidor da API (Uvicorn)
# Este comando será executado quando o container da API for iniciado.
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]