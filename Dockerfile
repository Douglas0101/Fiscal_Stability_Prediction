# --- Estágio 1: Build ---
# Esta etapa instala todas as dependências (incluindo as de desenvolvimento)
# e constrói a aplicação para produção.
FROM node:18-alpine AS builder

# Define o diretório de trabalho
WORKDIR /app

# Copia os arquivos de definição de pacotes. A ordem é importante para o cache do Docker.
COPY package.json package-lock.json* ./

# Usa 'npm ci' que é mais rápido e confiável para ambientes automatizados.
# Ele instala as dependências exatas do package-lock.json
RUN npm ci

# Copia o restante do código do frontend para o contêiner
COPY . .

# CORREÇÃO DEFINITIVA: Executa o comando de build do Next.js diretamente com 'npx'.
# Isso funciona mesmo que o script "build" esteja faltando no seu package.json.
RUN npx next build

# --- Estágio 2: Produção ---
# Esta etapa cria a imagem final, que é otimizada para ser leve e segura.
FROM node:18-alpine AS runner

WORKDIR /app

# Define o ambiente para produção
ENV NODE_ENV=production

# Cria um usuário e grupo não-root para rodar a aplicação com mais segurança
RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

# Copia o package.json para instalar APENAS as dependências de produção.
# Isso resulta em uma imagem final muito menor.
COPY --from=builder /app/package.json ./package.json
RUN npm ci --omit=dev && npm cache clean --force

# Copia os artefatos de build do estágio anterior
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/next.config.mjs ./next.config.mjs

# Define o novo usuário como dono dos arquivos e como usuário padrão
RUN chown -R nextjs:nodejs /app
USER nextjs

# Expõe a porta que a aplicação vai usar
EXPOSE 3000
ENV PORT 3000

# Comando para iniciar o servidor de produção do Next.js
CMD ["npx", "next", "start"]
