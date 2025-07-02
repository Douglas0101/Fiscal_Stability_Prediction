/** @type {import('next').NextConfig} */
const nextConfig = {
  // Adiciona a opção 'standalone' para otimizar o build para Docker
  output: 'standalone',
};

export default nextConfig;
