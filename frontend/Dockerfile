# --- Estágio 1: Build ---
# Esta etapa instala todas as dependências (incluindo as de desenvolvimento)
# e constrói a aplicação para produção.
FROM node:18-alpine AS builder

# Define o diretório de trabalho
WORKDIR /app

# --- PASSO 1: Copiar arquivos de projeto PRIMEIRO ---
# Copiamos os arquivos do projeto antes de instalar as dependências.
# Isso nos permite diagnosticar problemas de arquivos ausentes mais cedo.
# A cópia é explícita para evitar problemas com .dockerignore.
COPY package.json package-lock.json* ./
COPY public ./public
COPY src ./src
COPY next.config.mjs postcss.config.js tailwind.config.ts tsconfig.json ./

# --- PASSO 2: Diagnóstico de Arquivos (ESSENCIAL) ---
# Vamos verificar a estrutura de arquivos ANTES de qualquer outra coisa.
# O build falhará aqui com uma mensagem clara se o diretório 'app' não for encontrado.
RUN echo "--- Verificando a presença do diretório 'src/app' ---" && \
    if [ ! -d "src/app" ]; then \
      echo "ERRO CRÍTICO: O diretório 'src/app' não foi encontrado após a cópia." && \
      echo "Causa Provável: Um arquivo .dockerignore na pasta 'frontend' está excluindo a pasta 'src'." && \
      echo "AÇÃO: Verifique ou remova o arquivo .dockerignore da pasta 'frontend'." && \
      echo "Conteúdo atual do diretório /app (o que foi copiado):" && \
      ls -laR && \
      exit 1; \
    fi && \
    echo "--- Diagnóstico OK: Diretório 'src/app' encontrado. ---"

# --- PASSO 3: Instalar Dependências ---
# Agora que sabemos que os arquivos estão corretos, instalamos as dependências.
RUN npm ci

# --- PASSO 4: Construir a Aplicação ---
# Com os arquivos e dependências no lugar, o build deve funcionar.
RUN npx next build

# --- Estágio 2: Produção (Otimizado) ---
# Esta etapa cria a imagem final, que é otimizada para ser leve e segura.
FROM node:18-alpine AS runner

WORKDIR /app
ENV NODE_ENV=production

RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

# Copia apenas as dependências de produção
COPY --from=builder /app/package.json ./package.json
RUN npm ci --omit=dev && npm cache clean --force

# Copia os artefatos de build
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/next.config.mjs ./next.config.mjs

RUN chown -R nextjs:nodejs /app
USER nextjs

EXPOSE 3000
ENV PORT 3000

# Comando para iniciar o servidor de produção do Next.js
CMD ["npx", "next", "start"]
