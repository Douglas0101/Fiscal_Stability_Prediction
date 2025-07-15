
# 📊 Dashboard de Análise de Estabilidade Fiscal

Um dashboard moderno, interativo e responsivo para visualização e análise de dados sobre estabilidade fiscal e risco soberano. Construído com as tecnologias mais recentes do ecossistema React.

![Banner do Projeto](https://via.placeholder.com/1200x400.png?text=Dashboard+Fiscal+Stability)

---

## ✨ Funcionalidades

- **Visualização de Dados Dinâmica:** Gráficos interativos para análise de séries temporais, comparações e distribuições.
- **Design Moderno e Responsivo:** Interface limpa e adaptável a qualquer tamanho de tela, com temas claro e escuro.
- **Performance Otimizada:** Carregamento rápido com lazy-loading de componentes e rotas.
- **API-Driven:** Arquitetura desacoplada para consumir dados de qualquer back-end REST.

---

## 🚀 Tecnologias Utilizadas

Este projeto foi construído utilizando um stack moderno de front-end:

- **Framework:** [React 18](https://reactjs.org/)
- **Build Tool:** [Vite](https://vitejs.dev/)
- **Linguagem:** [TypeScript](https://www.typescriptlang.org/)
- **Estilização:** [Tailwind CSS](https://tailwindcss.com/)
- **Componentes UI:** [Chakra UI](https://chakra-ui.com/)
- **Roteamento:** [React Router](https://reactrouter.com/)
- **Requisições HTTP:** [Axios](https://axios-http.com/)
- **Gráficos:** [Recharts](https://recharts.org/)
- **Animações:** [Framer Motion](https://www.framer.com/motion/)
- **Testes:** [Jest](https://jestjs.io/) & [React Testing Library](https://testing-library.com/)

---

## 🏁 Início Rápido

Siga os passos abaixo para configurar e executar o projeto em seu ambiente local.

### Pré-requisitos

- [Node.js](https://nodejs.org/en/) (versão 18 ou superior)
- [npm](https://www.npmjs.com/) (geralmente vem com o Node.js)

### Instalação

1.  **Clone o repositório:**
    ```bash
    git clone https://github.com/seu-usuario/dashboard-fiscal-stability.git
    ```

2.  **Navegue até o diretório do projeto:**
    ```bash
    cd dashboard-fiscal-stability
    ```

3.  **Instale as dependências:**
    ```bash
    npm install
    ```

### Executando a Aplicação

Para iniciar o servidor de desenvolvimento, execute:

```bash
npm run dev
```

A aplicação estará disponível em `http://localhost:5173`.

---

## 📜 Scripts Disponíveis

- `npm run dev`: Inicia o servidor de desenvolvimento com Hot-Reload.
- `npm run build`: Compila e otimiza a aplicação para produção na pasta `dist/`.
- `npm run lint`: Executa o ESLint para analisar o código em busca de problemas.
- `npm run preview`: Inicia um servidor local para visualizar a build de produção.

---

## 📂 Estrutura de Pastas

O projeto segue uma arquitetura modular para facilitar a manutenção e escalabilidade.

```
/src
├── /components    # Componentes reutilizáveis (Botões, Inputs, Gráficos)
├── /contexts      # Contextos React para gerenciamento de estado global
├── /hooks         # Hooks customizados com lógica de negócio
├── /pages         # Componentes que representam as páginas da aplicação
├── /services      # Funções para comunicação com APIs externas
├── /styles        # Arquivos de estilização globais e temas
├── /utils         # Funções utilitárias
└── App.tsx        # Componente raiz com a configuração de rotas
```

---

## 🤝 Contribuindo

Contribuições são sempre bem-vindas! Se você deseja melhorar este projeto, siga os passos abaixo:

1.  **Faça um Fork** do projeto.
2.  **Crie uma nova branch** para sua feature: `git checkout -b feature/sua-feature-incrivel`.
3.  **Faça o commit** de suas alterações: `git commit -m 'feat: Adiciona sua feature incrível'`.
4.  **Faça o push** para a sua branch: `git push origin feature/sua-feature-incrivel`.
5.  **Abra um Pull Request**.

Por favor, certifique-se de que seu código segue as convenções do projeto e que todos os testes estão passando.

---

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.
