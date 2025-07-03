import NextAuth from "next-auth";
import CredentialsProvider from "next-auth/providers/credentials";
import bcrypt from "bcryptjs";
import { User } from "next-auth"; // Importa o tipo User

// --- Simulação da Base de Dados de Utilizadores ---
// No mundo real, estes dados viriam de uma base de dados (ex: PostgreSQL, MongoDB).
// A palavra-passe '123456' foi encriptada previamente com bcrypt.
const users = [
  {
    id: "1",
    name: "Utilizador Admin",
    email: "admin@example.com",
    password:
      "$2a$10$vK.ZIM2h/22.O5y25aR/..bXn2jN4g2.Xb2QfN/Yh/DEk/d.p.p8O", // Hash para '123456'
  },
];
// ----------------------------------------------------

const handler = NextAuth({
  providers: [
    CredentialsProvider({
      // O nome a ser exibido no formulário de login (opcional)
      name: "Credentials",
      credentials: {
        email: { label: "Email", type: "text", placeholder: "admin@example.com" },
        password: { label: "Password", type: "password" },
      },
      async authorize(credentials): Promise<User | null> {
        // Verifica se email e password foram fornecidos
        if (!credentials?.email || !credentials?.password) {
          return null;
        }

        // 1. Encontrar o utilizador na nossa "base de dados"
        const user = users.find((u) => u.email === credentials.email);

        // Se o utilizador não for encontrado, a autorização falha
        if (!user) {
          console.log("Utilizador não encontrado");
          return null;
        }

        // 2. Verificar se a palavra-passe está correta
        // Compara a palavra-passe fornecida com a hash armazenada de forma segura
        const isPasswordCorrect = await bcrypt.compare(
          credentials.password,
          user.password
        );

        if (!isPasswordCorrect) {
          console.log("Palavra-passe incorreta");
          return null;
        }

        // 3. Se tudo estiver correto, retorna o objeto do utilizador
        // O NextAuth irá usar isto para criar a sessão e o JWT.
        // Não inclua a password no objeto retornado!
        console.log("Autenticação bem-sucedida para:", user.email);
        return {
          id: user.id,
          name: user.name,
          email: user.email,
        };
      },
    }),
  ],
  // Configurações adicionais do NextAuth
  pages: {
    signIn: "/login", // Redireciona para a sua página de login personalizada
  },
  session: {
    strategy: "jwt", // Usa JSON Web Tokens para as sessões
  },
  callbacks: {
    // O callback 'jwt' é chamado sempre que um JWT é criado ou atualizado.
    async jwt({ token, user }) {
      // Se o objeto 'user' existir (ocorre no login), adiciona o ID ao token.
      if (user) {
        token.id = user.id;
      }
      return token;
    },
    // O callback 'session' é chamado sempre que uma sessão é acedida.
    async session({ session, token }) {
      // Adiciona o ID do token ao objeto da sessão,
      // para que possa ser acedido no lado do cliente.
      if (session.user) {
        (session.user as any).id = token.id;
      }
      return session;
    },
  },
});

export { handler as GET, handler as POST };
