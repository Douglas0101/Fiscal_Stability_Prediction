# Dockerfile

# --- Estágio 1: Builder ---
# Neste estágio, instalamos todas as dependências de forma isolada para manter a imagem final leve.
FROM python:3.11-slim AS builder

WORKDIR /app

# Copia apenas o ficheiro de requisitos primeiro para aproveitar o cache do Docker.
COPY requirements.txt .

# Instala as dependências Python de forma eficiente.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- Estágio 2: Final ---
# Este é o estágio que irá gerar a imagem de produção.
FROM python:3.11-slim

# --- CORREÇÃO APLICADA AQUI ---
# Instala a biblioteca OpenMP (libgomp1), uma dependência de sistema crucial para
# bibliotecas de ML como LightGBM e XGBoost.
# Fazemos isso em uma única camada RUN para otimizar o tamanho da imagem, limpando o cache do apt no final.
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Define o diretório de trabalho para a imagem final.
WORKDIR /app

# Cria um utilizador e grupo não-root para aumentar a segurança da aplicação.
RUN addgroup --system app && adduser --system --group app

# Define um diretório de cache para o Matplotlib que seja gravável pelo utilizador 'app'.
ENV MPLCONFIGDIR /app/.cache/matplotlib

# Copia as dependências já instaladas do estágio 'builder'.
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copia o código-fonte da sua aplicação para o contentor.
COPY ./src ./src

# Muda a posse de todos os ficheiros no diretório /app para o utilizador 'app'.
RUN chown -R app:app /app

# Muda para o utilizador não-root para a execução da aplicação.
USER app

# Expõe a porta que a aplicação irá usar dentro do contentor.
EXPOSE 8000

# Define o comando padrão para executar a aplicação quando o contentor iniciar.
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
