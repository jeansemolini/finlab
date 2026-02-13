# 📊 FinLab - Financial AI Analysis Platform

FinLab é uma plataforma de análise financeira inteligente que utiliza IA, embeddings vetoriais e RAG (Retrieval-Augmented Generation) para fornecer insights profundos sobre empresas cotadas em bolsa.

## 🎯 Características Principais

- **Análise Multidimensional**: Combina 3 análises independentes (Fundamental, Momentum e Sentimento)
- **RAG (Retrieval-Augmented Generation)**: Respostas baseadas em dados reais do SEC e notícias
- **Busca Vetorial**: Utiliza Qdrant com embeddings densos, esparsos e ColBERT
- **Estruturação de Dados**: Converte respostas LLM em schemas Pydantic validados
- **API FastAPI**: Endpoints RESTful para integração
- **Frontend React**: Interface web interativa

## 🏗️ Arquitetura

```
┌─────────────────────────────────────────────────────────────────┐
│                     FINLAB - Financial Analysis                 │
└─────────────────────────────────────────────────────────────────┘
                                ↓
        ┌───────────────────────┬───────────────────────┐
        ↓                       ↓                       ↓
    ┌────────┐            ┌──────────┐           ┌──────────┐
    │ EDGAR  │            │  QDRANT  │           │  GROQ    │
    │ SEC    │            │ Vector DB│           │  LLM     │
    │ Filings│            │  Search  │           │  Models  │
    └────────┘            └──────────┘           └──────────┘
        ↓                       ↓                       ↓
    ┌─────────────────────────────────────────────────────────┐
    │              INGESTION & PROCESSING LAYER               │
    └─────────────────────────────────────────────────────────┘
        ↓
    ┌─────────────────────────────────────────────────────────┐
    │                 API LAYER (FastAPI)                      │
    │  ┌──────────┬──────────┬──────────┬──────────┐           │
    │  │ Search   │ RAG      │ Agent    │ Config   │           │
    │  │ Endpoint │ Endpoint │ Endpoint │ Settings │           │
    │  └──────────┴──────────┴──────────┴──────────┘           │
    └─────────────────────────────────────────────────────────┘
```

## 📁 Estrutura do Projeto

```
finlab/
├── 📄 README.md                          # Este arquivo
├── 📄 pyproject.toml                     # Dependências (uv)
├── 📄 .env                               # Variáveis de ambiente
│
├── 📁 ingestion/                         # Pipeline de dados
│  ├── ingestion.py                       # Fetch SEC + chunking semântico
│  ├── news_ingestion.py                  # Fetch notícias
│  ├── create_collection.py               # Setup Qdrant
│  ├── test-query.py                      # Teste de busca
│  └── 📁 utils/
│     ├── edgar_client.py                 # SEC filings client
│     ├── semantic_chunker.py             # Chunking inteligente
│     ├── news_client.py                  # Notícias client
│     └── simple_chunker.py               # Chunking simples
│
├── 📁 api/                               # FastAPI Backend
│  ├── main.py                            # FastAPI app root
│  ├── .env                               # Config local
│  │
│  ├── 📁 config/
│  │  ├── settings.py                     # Pydantic BaseSettings
│  │  └── prompts.py                      # LLM prompts
│  │
│  ├── 📁 models/                         # Pydantic schemas
│  │  ├── search.py                       # SearchRequest/Response
│  │  ├── rag.py                          # RAGRequest/Response
│  │  └── agent.py                        # AgentRequest/Response
│  │
│  ├── 📁 services/                       # Business logic
│  │  ├── search.py                       # SearchService (RRF fusion)
│  │  ├── embeddings.py                   # EmbeddingService
│  │  ├── rag.py                          # RAGService
│  │  └── agent.py                        # AgentService (3-way analysis)
│  │
│  └── 📁 routers/                        # FastAPI endpoints
│     ├── search.py                       # GET /search
│     ├── rag.py                          # POST /rag
│     └── agent.py                        # POST /agent
│
├── 📁 evaluations/                       # Testes e avaliações
│  ├── level-1-unit-tests.py              # Testes básicos
│  ├── level-2-integration-tests.py       # Testes de integração
│  ├── level-3-human-annotation.py        # Avaliação humana (Langfuse)
│  └── 📁 test_cases/
│     ├── apple_test.json
│     ├── ibm_test.json
│     ├── no_company_test.json
│     └── natural_language_test.json
│
├── 📁 guardrails/                        # Validação com Guardrails
│  ├── guardrails-demo-1.py               # Exemplo profanidade
│  ├── guardrails-demo-2.py               # Exemplo RAG validado
│  └── guardrails-demo-3.py               # Exemplo agent validado
│
└── 📁 finlab-front/                      # React Frontend
   ├── package.json
   ├── src/
   │  ├── components/
   │  ├── pages/
   │  └── App.jsx
   └── public/
```

## 🚀 Instalação e Setup

### Pré-requisitos

- Python 3.12+
- Node.js 18+
- `uv` package manager
- Conta Qdrant Cloud (ou instância local)
- Chaves de API: Groq, EDGAR, OpenAI (opcional)

### 1. Clone e Setup do Backend

```bash
cd finlab
uv sync --upgrade
```

### 2. Configurar Variáveis de Ambiente

Crie `.env` na raiz do projeto:

```env
# QDRANT - Vector Database
QDRANT_URL="https://[seu-cluster].us-east-1-1.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."

# LLM APIs
GROQ_API_KEY="gsk_xxxxxxxxxxxx"
OPENAI_API_KEY="sk-proj-xxxxxxxxxxxx"

# Opcional
GOOGLE_API_KEY="AIzaSyxxxxxx"
```

### 3. Criar Collection Qdrant

```bash
python ingestion/create_collection.py
```

### 4. Ingerir Dados (SEC Filings)

```bash
python ingestion/ingestion.py
```

Isso irá:
- Fetch 10-K e 10-Q da AAPL do EDGAR
- Chunking semântico com HDBSCAN
- Gerar embeddings (Dense + Sparse + ColBERT)
- Fazer upload para Qdrant

### 5. Ingerir Dados (Notícias)

```bash
python ingestion/news_ingestion.py
```

### 6. Iniciar API

```bash
cd api
source ../.venv/bin/activate
uvicorn main:app --reload
```

A API estará disponível em `http://localhost:8000`

### 7. Setup Frontend

```bash
cd finlab-front
npm install
npm run dev
```

Frontend em `http://localhost:3000`

## 🔌 API Endpoints

### 1. **POST /search** - Busca Vetorial

Busca híbrida com RRF (Reciprocal Rank Fusion)

```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Apple business model",
    "limit": 3
  }'
```

**Response:**
```json
{
  "results": [
    {
      "score": 0.95,
      "text": "Apple Inc. designs, manufactures...",
      "metadata": {
        "source": "10-K",
        "ticker": "AAPL"
      }
    }
  ]
}
```

### 2. **POST /rag** - Retrieval-Augmented Generation

Responde perguntas baseado em documentos recuperados

```bash
curl -X POST "http://localhost:8000/rag" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are Apple'\''s main risks?",
    "limit": 3
  }'
```

**Response:**
```json
{
  "query": "What are Apple's main risks?",
  "answer": "According to Apple's 10-K filing, os principais riscos incluem: dependência de fornecedores, flutuações cambiais, competição...",
  "metadata": [
    {"score": 0.92, "source": "10-K Item 1A"}
  ]
}
```

### 3. **POST /agent** - Análise Multidimensional

Combina 3 análises (Fundamental, Momentum, Sentimento) + recomendação final

```bash
curl -X POST "http://localhost:8000/agent" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How is Apple doing today?",
    "limit": 3
  }'
```

**Response:**
```json
{
  "query": "How is Apple doing today?",
  "ticker": "AAPL",
  "fundamental_analysis": {
    "overall_investment_thesis": "Apple maintains strong competitive position with robust financials...",
    "investment_grade": "A",
    "confidence_score": 0.92,
    "key_strengths": ["Brand power", "Ecosystem lock-in", "Strong cash flow"],
    "key_concerns": ["China exposure", "Regulatory risks", "Market saturation"],
    "recommendation": "buy"
  },
  "momentum_analysis": {
    "overall_momentum": "positive",
    "momentum_strength": "strong",
    "key_momentum_drivers": ["Services growth", "Margin expansion"],
    "momentum_risks": ["Market slowdown", "Supply chain disruption"],
    "short_term_outlook": "bullish",
    "momentum_score": 8.5
  },
  "sentiment_analysis": {
    "sentiment_score": 8,
    "sentiment_direction": "Positive",
    "key_news_themes": ["Product innovation", "Q4 earnings beat"],
    "recent_catalysts": ["Vision Pro launch", "New AI features"],
    "market_outlook": "Strong demand for Apple products continues"
  },
  "final_recommendation": {
    "action": "BUY",
    "confidence": 0.88,
    "rationale": "Strong fundamentals combined with positive momentum and market sentiment...",
    "key_risks": ["Regulatory pressure", "Economic slowdown"],
    "key_opportunities": ["Emerging markets", "Services expansion"],
    "time_horizon": "Medium-term"
  }
}
```

## 📊 Fluxos de Dados

### Fluxo de Ingestion

```
EdgarClient (10-K/10-Q)
    ↓
SemanticChunker (HDBSCAN grouping)
    ↓
EmbeddingGeneration (Dense + Sparse + ColBERT)
    ↓
Qdrant Upload
```

### Fluxo de Agent (POST /agent)

```
1. Extract Ticker from Query
    ↓
2. Parallel Analysis (asyncio.gather):
   ├─ _analyze_fundamental()
   │  ├─ Search: FUNDAMENTAL_QUERIES
   │  └─ LLM: FUNDAMENTAL_PROMPT → FundamentalAnalysis
   ├─ _analyze_momentum()
   │  ├─ Search: MOMENTUM_QUERIES
   │  └─ LLM: MOMENTUM_PROMPT → MomentumAnalysis
   └─ _analyze_sentiment()
      ├─ Search: SENTIMENT_QUERY + News
      └─ LLM: SENTIMENT_PROMPT → SentimentAnalysis
    ↓
3. Aggregation:
   └─ LLM: AGGREGATION_PROMPT → FinalRecommendation
    ↓
4. Return AgentResponse
```

## 🧪 Testes

### Testes Unitários (Level 1)

```bash
python evaluations/level-1-unit-tests.py
```

Testa:
- Extração de ticker (static mapping + LLM fallback)
- Queries naturais em linguagem

### Testes de Integração (Level 2)

```bash
python evaluations/level-2-integration-tests.py
```

Testa:
- Pipeline completo end-to-end
- Validação de schemas

### Avaliação Humana (Level 3)

```bash
python evaluations/level-3-human-annotation.py
```

Integra com Langfuse para rastreamento e avaliação

## 🛡️ Validação com Guardrails

### Exemplos

```bash
# Profanidade em PT-BR
python guardrails/guardrails-demo-1.py

# RAG com validação
python guardrails/guardrails-demo-2.py

# Agent com validação
python guardrails/guardrails-demo-3.py
```

## 📈 Modelos e Tecnologias

| Componente | Modelo/Tecnologia |
|-----------|------------------|
| Dense Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384 dims) |
| Sparse Embeddings | `Qdrant/bm25` |
| ColBERT | `colbert-ir/colbertv2.0` (128 dims, multivector) |
| LLM | `llama-3.1-8b-instant` (Groq) |
| Vector DB | Qdrant Cloud |
| Chunking | HDBSCAN (semântico) |
| Framework API | FastAPI |
| Frontend | React + Vite |

## 🔑 Variáveis de Ambiente Detalhadas

```env
# QDRANT
QDRANT_URL=https://[cluster-id].us-east-1-1.aws.cloud.qdrant.io:6333
QDRANT_API_KEY=[bearer-token]

# LLM Providers
GROQ_API_KEY=[groq-api-key]
OPENAI_API_KEY=[openai-api-key]
GOOGLE_API_KEY=[google-api-key]

# Collection Config (default values)
COLLECTION_NAME=financial
DENSE_MODEL=sentence-transformers/all-MiniLM-L6-v2
SPARSE_MODEL=Qdrant/bm25
COLBERT_MODEL=colbert-ir/colbertv2.0
GROQ_MODEL=llama-3.1-8b-instant
```

## 🐛 Troubleshooting

### Erro: "Address already in use" (porta 8000)

```bash
kill -9 $(lsof -ti:8000)
```

### Erro: Qdrant connection refused

Verifique:
1. URL do Qdrant está correta com porta 6333
2. API Key está válida
3. Rede tem acesso ao cluster Qdrant

### Erro: RECORD file missing em dependências

```bash
rm -rf .venv
uv sync --upgrade
```

### Embeddings não encontrados

Execute ingestion:
```bash
python ingestion/ingestion.py
```

## 📚 Prompts LLM

Os prompts são configuráveis em `api/config/prompts.py`:

- `RAG_PROMPT`: Responde perguntas sobre documentos
- `FUNDAMENTAL_PROMPT`: Análise fundamentalista (Grade A-D)
- `MOMENTUM_PROMPT`: Análise de momentum (0-10)
- `SENTIMENT_PROMPT`: Análise de sentimento (1-10)
- `AGGREGATION_PROMPT`: Agregação final (BUY/HOLD/SELL)

## 🚢 Deploy

### Backend (Uvicorn)

```bash
# Production
cd api
source ../.venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Frontend (Build)

```bash
cd finlab-front
npm run build
npm run preview
```

## 📖 Documentação Interativa

API docs automática:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🤝 Contribuindo

1. Fork o projeto
2. Crie uma branch (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -m 'Add nova-feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Abra um Pull Request

## 📜 Licença

MIT

## 📞 Contato

Jean Semolini  
Email: jean.maiko@hotmail.com

---

**Última atualização**: 13 de fevereiro de 2026
**Versão**: 0.1.0
