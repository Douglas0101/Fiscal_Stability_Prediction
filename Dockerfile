# Arquivo: Dockerfile

# Use uma imagem base do Python
FROM python:3.9-slim

# Define o diretório de trabalho dentro do contêiner
WORKDIR /app

# Define variáveis de ambiente
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Instala dependências do sistema, se necessário
# RUN apt-get update && apt-get install -y ...

# Copia o arquivo de dependências e instala os pacotes
# Isso aproveita o cache do Docker, reinstalando apenas se o requirements.txt mudar
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo o código do projeto para o diretório de trabalho
COPY . .

# Expõe a porta que a API usará dentro do contêiner
EXPOSE 8000

# Comando para iniciar a aplicação com Uvicorn
# O host 0.0.0.0 torna a API acessível de fora do contêiner
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]