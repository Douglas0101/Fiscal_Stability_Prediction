import NextAuth from "next-auth";
import CredentialsProvider from "next-auth/providers/credentials";
import bcrypt from "bcrypt";

const handler = NextAuth({
  providers: [
    CredentialsProvider({
      // O nome que será exibido no formulário de login (por exemplo, "Sign in with...")
      name: "Credentials",
      // `credentials` é usado para gerar um formulário na página de login padrão.
      // Você pode especificar quais campos devem ser enviados.
      credentials: {
        email: { label: "Email", type: "email", placeholder: "test@example.com" },
        password: { label: "Password", type: "password" },
      },
      async authorize(credentials, req) {
        // Adicione aqui a lógica para buscar o usuário de uma fonte de dados, como um banco de dados.
        // ESTE É UM EXEMPLO. SUBSTITUA PELA SUA LÓGICA DE BANCO DE DADOS.
        if (!credentials) {
            return null;
        }

        // Em um cenário real, você faria uma busca no seu banco de dados.
        // Ex: const user = await db.user.findUnique({ where: { email: credentials.email } });
        const user = {
          id: "1",
          name: "Admin User",
          email: "admin@example.com",
          // Senha original é 'password'. O hash foi gerado com: bcrypt.hashSync('password', 10)
          passwordHash: "$2b$10$E/gL3h2S.AUyU5Yg8jC1a.aJzfY/Y.H.e.g.L3h2S.AUyU5Yg8jC1",
        };

        if (!user) {
          console.log("Nenhum usuário encontrado com este email.");
          return null;
        }

        // Compara a senha enviada no formulário com o hash salvo no "banco de dados"
        const passwordsMatch = await bcrypt.compare(credentials.password, user.passwordHash);

        if (passwordsMatch) {
          // Qualquer objeto retornado aqui será salvo no token JWT.
          // Não retorne a senha ou o hash da senha.
          return { id: user.id, name: user.name, email: user.email };
        } else {
          // Se as senhas não baterem, retorne null
          console.log("Senha incorreta.");
          return null;
        }
      },
    }),
  ],
  // Adicione aqui outras configurações do NextAuth se necessário
  // pages: {
  //   signIn: '/login', // Se você tiver uma página de login customizada
  // }
});

export { handler as GET, handler as POST };
