**Dashboard de Análise de Risco Fiscal**

**Este projeto contém o frontend para o Dashboard de Análise de Risco Fiscal, uma aplicação moderna construída com React, Vite e TypeScript para fornecer uma interface interativa e de alto desempenho para os modelos de Machine Learning da nossa API.**

**✨ Funcionalidades**

**Interface Moderna: Design limpo e responsivo construído com Chakra UI e Tailwind CSS.**

**Temas: Suporte para modos claro, escuro e de alto contraste.**

**Visualização de Dados: Gráficos interativos e dinâmicos com Recharts.**

**Performance: Carregamento rápido com Vite e lazy-loading de componentes.**

**Qualidade de Código: Código fortemente tipado com TypeScript.**

🚀 Como Iniciar (Ambiente Local)
Para executar este projeto fora do ambiente Docker, siga os passos abaixo.

**Pré-requisitos**

**Node.js (v18 ou superior)**

**npm ou yarn**

**Instalação**
**Clone o repositório (se aplicável) e navegue até a pasta do projeto.**

**Instale as dependências necessárias:**

**Bash**

**npm install**
**Crie um ficheiro .env.local na raiz do projeto para configurar as variáveis de ambiente:**

**VITE_API_BASE_URL=http://localhost:8000**

**Inicie o servidor de desenvolvimento:**

**Bash**

**npm run dev**
**A aplicação estará disponível em http://localhost:5173 (ou outra porta indicada pelo Vite).**

**scripts Disponíveis**
**npm run dev: Inicia o servidor de desenvolvimento com Hot-Reload.**

**npm run build: Compila a aplicação TypeScript e gera os ficheiros de produção na pasta /dist.**

**npm run lint: Executa o linter para verificar a qualidade do código.**

**npm run preview: Inicia um servidor local para visualizar a versão de produção.**

**🏛️ Estrutura de Pastas**
**O projeto segue uma arquitetura modular para facilitar a manutenção e escalabilidade.**

/src

├── /assets         # Imagens, fontes e outros ficheiros estáticos

├── /components     # Componentes React reutilizáveis (ex: botões, gráficos)
│   ├── /charts     # Componentes específicos de gráficos
│   └── /layout     # Componentes de layout (Header, Sidebar, etc.)

├── /contexts       # Contextos React para gerenciamento de estado global

├── /hooks          # Hooks customizados com lógica de negócio

├── /pages          # Componentes que representam as páginas da aplicação

├── /services       # Funções para comunicação com APIs externas (Axios)

├── /styles         # Ficheiros de CSS global e configuração de temas

├── /types          # Definições de tipos TypeScript

└── App.tsx         # Componente raiz da aplicação

└── main.tsx        # Ponto de entrada da aplicação

## 🤝 Como Contribuir

#### Agradecemos o seu interesse em contribuir! Por favor, siga as seguintes diretrizes:

#### Crie um fork do projeto.

**Crie uma nova branch para a sua funcionalidade (git checkout -b feature/minha-feature).**

**Faça o commit das suas alterações (git commit -m 'Adiciona nova feature').**

**Faça o push para a sua branch (git push origin feature/minha-feature).**

**Abra um Pull Request.**

**Certifique-se de que o seu código segue os padrões de estilo do projeto e que todos os testes passam.**






