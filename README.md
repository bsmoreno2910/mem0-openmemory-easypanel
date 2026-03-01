# Mem0 + OpenMemory UI - Docker Compose Setup

Este é um setup completo do **Mem0** com a **UI nativa (OpenMemory)** usando Docker Compose.

## 📋 Componentes

- **Mem0 API**: FastAPI backend para gerenciar memórias
- **OpenMemory UI**: Interface web nativa do Mem0
- **Qdrant**: Banco de dados vetorial para armazenamento de memórias

## 🚀 Quick Start

### 1. Preparar os arquivos

```bash
# Criar diretório do projeto
mkdir mem0-project
cd mem0-project

# Copiar os arquivos
cp mem0_docker_compose.yaml docker-compose.yaml
mkdir mem0-api
cp mem0_Dockerfile mem0-api/Dockerfile
cp mem0_app.py mem0-api/app.py
cp mem0_requirements.txt mem0-api/requirements.txt
cp mem0_.env.example .env
```

### 2. Configurar variáveis de ambiente

Edite o arquivo `.env` com suas credenciais:

```bash
nano .env
```

**Variáveis obrigatórias:**
- `OPENAI_API_KEY`: Sua chave da OpenAI
- `QDRANT_API_KEY`: Chave de segurança do Qdrant
- `QDRANT_HOST`: Host do Qdrant (padrão: qdrant)
- `QDRANT_PORT`: Porta do Qdrant (padrão: 6333)

### 3. Iniciar os serviços

```bash
# Iniciar com Docker Compose
docker-compose up -d

# Verificar status
docker-compose ps

# Ver logs
docker-compose logs -f
```

## 🌐 Acessar os serviços

- **OpenMemory UI**: http://localhost:3000
- **Mem0 API**: http://localhost:8000
- **API Docs (Swagger)**: http://localhost:8000/docs
- **Qdrant Dashboard**: http://localhost:6333/dashboard

## 📡 Endpoints da API

### Health Check
```bash
curl http://localhost:8000/health
```

### Adicionar Memória
```bash
curl -X POST http://localhost:8000/memory/add \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Meu nome é João"}],
    "user_id": "user123"
  }'
```

### Buscar Memória
```bash
curl -X POST http://localhost:8000/memory/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Como é meu nome?",
    "user_id": "user123"
  }'
```

### Obter Todas as Memórias
```bash
curl http://localhost:8000/memory/all?user_id=user123
```

### Atualizar Memória
```bash
curl -X PUT http://localhost:8000/memory/update \
  -H "Content-Type: application/json" \
  -d '{
    "memory_id": "mem_123",
    "data": {"updated_field": "new_value"},
    "user_id": "user123"
  }'
```

### Deletar Memória
```bash
curl -X DELETE http://localhost:8000/memory/delete \
  -H "Content-Type: application/json" \
  -d '{
    "memory_id": "mem_123",
    "user_id": "user123"
  }'
```

## 🛑 Parar os serviços

```bash
docker-compose down

# Remover volumes (cuidado!)
docker-compose down -v
```

## 📊 Estrutura de Diretórios

```
mem0-project/
├── docker-compose.yaml
├── .env
├── .env.example
├── mem0-api/
│   ├── Dockerfile
│   ├── app.py
│   └── requirements.txt
└── README.md
```

## 🔧 Configuração no Easypanel

Se você estiver usando **Easypanel**, você pode:

1. Criar um novo serviço do tipo **Compose**
2. Fazer upload do arquivo `docker-compose.yaml`
3. Configurar as variáveis de ambiente
4. Fazer o deploy

## 📝 Variáveis de Ambiente

| Variável | Descrição | Padrão |
|----------|-----------|--------|
| `OPENAI_API_KEY` | Chave da API OpenAI | - |
| `LLM_MODEL` | Modelo LLM a usar | gpt-4o-mini |
| `EMBEDDING_MODEL` | Modelo de embedding | text-embedding-3-large |
| `VECTOR_STORE` | Tipo de vector store | qdrant |
| `QDRANT_HOST` | Host do Qdrant | qdrant |
| `QDRANT_PORT` | Porta do Qdrant | 6333 |
| `QDRANT_API_KEY` | Chave de API do Qdrant | - |

## 🐛 Troubleshooting

### Mem0 API não conecta ao Qdrant
```bash
# Verificar se Qdrant está rodando
docker-compose logs qdrant

# Testar conexão
curl http://localhost:6333/health
```

### OpenMemory UI não conecta à API
```bash
# Verificar se a API está respondendo
curl http://localhost:8000/health

# Verificar logs da API
docker-compose logs mem0-api
```

### Erro de dependências Python
```bash
# Reconstruir a imagem
docker-compose build --no-cache mem0-api
docker-compose up -d
```

## 📚 Referências

- [Mem0 Documentation](https://docs.mem0.ai)
- [OpenMemory UI](https://hub.docker.com/r/mem0/openmemory-ui)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

## 📄 Licença

Este projeto usa componentes de código aberto. Consulte as licenças individuais.

---

**Criado com ❤️ para Mem0**
