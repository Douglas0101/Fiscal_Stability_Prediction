// File: frontend/next.config.mjs
// (Ficheiro Atualizado)
// Configura um proxy reverso. Todas as chamadas do frontend para /api/*
// serão redirecionadas para o backend, resolvendo problemas de CORS e de rede Docker.

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: `${process.env.BACKEND_API_URL}/:path*`,
      },
    ];
  },
};

export default nextConfig;