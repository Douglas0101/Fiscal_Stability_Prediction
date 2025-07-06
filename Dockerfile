# Dockerfile

# A linha ARG foi removida.

FROM python:3.9-slim

RUN apt-get update && apt-get install -y git

WORKDIR /app

# --- CORREÇÃO FINAL: Fixar (hardcode) os IDs de usuário e grupo ---
RUN groupadd -g 1000 appgroup && \
    useradd --no-log-init -u 1000 -g appgroup -m appuser

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser:appgroup ./src /app/src

USER appuser

EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]