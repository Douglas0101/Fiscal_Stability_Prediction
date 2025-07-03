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
```
# File: frontend/.env.local
# (Ficheiro Novo)
# Define variáveis de ambiente para o desenvolvimento local ('npm run dev').

# URL da API backend que o proxy do Next.js usará para redirecionar as chamadas.
BACKEND_API_URL=[http://127.0.0.1:8000](http://127.0.0.1:8000)

# URL que o código do navegador usará. É um caminho relativo para que o proxy funcione.
NEXT_PUBLIC_API_URL=/api

# URL base da aplicação frontend, usado pelo NextAuth.
NEXTAUTH_URL=http://localhost:3000

```typescript

// File: frontend/src/app/api/auth/[...nextauth]/route.ts
// (Ficheiro Atualizado)
// Configura o Next-Auth para usar o provedor de 'Credentials' e delegar
// a validação para o nosso backend FastAPI, em vez de aceder ao banco de dados diretamente.

import NextAuth, { NextAuthOptions } from "next-auth";
import CredentialsProvider from "next-auth/providers/credentials";

export const authOptions: NextAuthOptions = {
  providers: [
    CredentialsProvider({
      name: "Credentials",
      credentials: {
        email: { label: "Email", type: "email" },
        password: { label: "Password", type: "password" },
      },
      async authorize(credentials) {
        if (!credentials?.email || !credentials?.password) {
          return null;
        }

        try {
          // Usa a URL pública, que será interceptada pelo proxy do Next.js.
          const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/auth/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              email: credentials.email,
              password: credentials.password,
            }),
          });

          // Se a resposta do backend não for 200 (OK), a autorização falha.
          if (!res.ok) {
            return null;
          }

          const user = await res.json();

          if (user) {
            // Retorna os dados do usuário para o Next-Auth criar a sessão.
            return {
              id: user.id.toString(),
              name: user.name,
              email: user.email,
            };
          }
          return null;
        } catch (e) {
          console.error("Authorize error:", e);
          return null;
        }
      },
    }),
  ],
  pages: {
    signIn: '/login', // Define a página de login customizada.
  },
  session: {
    strategy: 'jwt',
  },
  callbacks: {
    // Adiciona o ID do usuário ao token JWT.
    async jwt({ token, user }) {
      if (user) {
        token.id = user.id;
      }
      return token;
    },
    // Adiciona o ID do usuário à sessão do cliente.
    async session({ session, token }) {
      if (session.user) {
        // @ts-ignore
        session.user.id = token.id;
      }
      return session;
    },
  },
};

const handler = NextAuth(authOptions);

export { handler as GET, handler as POST };
