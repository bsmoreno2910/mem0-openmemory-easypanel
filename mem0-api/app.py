from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from mem0 import Memory
import os
import logging

# FlashRank reranker (inicializado lazy para não bloquear o startup)
_flashrank_ranker = None
def get_flashrank():
    global _flashrank_ranker
    if _flashrank_ranker is None:
        try:
            from flashrank import Ranker
            _flashrank_ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/app/.cache/flashrank")
            logger.info("✅ FlashRank reranker loaded")
        except Exception as e:
            logger.warning(f"⚠️ FlashRank not available: {e}")
    return _flashrank_ranker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Mem0 API",
    description="Memory layer API for AI applications",
    version="1.0.0"
)

# Enable CORS for the UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store config for health check
qdrant_host = os.getenv("QDRANT_HOST", "localhost")
qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
qdrant_api_key = os.getenv("QDRANT_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

# Configure Mem0
try:
    config = {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "url": f"http://{qdrant_host}:{qdrant_port}",
                "api_key": qdrant_api_key,
            },
        },
        "llm": {
            "provider": "openai",
            "config": {
                "api_key": openai_api_key,
                "model": llm_model,
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "api_key": openai_api_key,
                "model": embedding_model,
            },
        },
    }

    memory = Memory.from_config(config)
    logger.info("✅ Mem0 initialized successfully")
    mem0_initialized = True
    mem0_error = None
except Exception as e:
    logger.error(f"❌ Failed to initialize Mem0: {str(e)}")
    memory = None
    mem0_initialized = False
    mem0_error = str(e)

# Request/Response Models
class AddMemoryRequest(BaseModel):
    messages: list
    user_id: str = "default"

class SearchMemoryRequest(BaseModel):
    query: str
    user_id: str = "default"
    limit: int = 10
    top_n: int = 5

class UpdateMemoryRequest(BaseModel):
    memory_id: str
    data: dict
    user_id: str = "default"

class DeleteMemoryRequest(BaseModel):
    memory_id: str
    user_id: str = "default"

# Endpoints
@app.get("/health")
def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "mem0-api",
        "version": "1.0.0",
        "mem0_initialized": mem0_initialized
    }

@app.get("/health/detailed")
def health_detailed():
    """Detailed health check — verifica conexão com Qdrant e OpenAI"""
    import httpx

    result = {
        "service": "mem0-api",
        "version": "1.0.0",
        "mem0": {
            "initialized": mem0_initialized,
            "error": mem0_error
        },
        "qdrant": {
            "host": qdrant_host,
            "port": qdrant_port,
            "status": "unknown",
            "collections": []
        },
        "openai": {
            "model": llm_model,
            "embedding_model": embedding_model,
            "api_key_set": bool(openai_api_key and openai_api_key.startswith("sk-")),
            "status": "unknown"
        }
    }

    # Testar conexão com Qdrant
    try:
        qdrant_url = f"http://{qdrant_host}:{qdrant_port}/collections"  # HTTP direto, sem SSL
        headers = {}
        if qdrant_api_key:
            headers["api-key"] = qdrant_api_key
        response = httpx.get(qdrant_url, headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            collections = [c["name"] for c in data.get("result", {}).get("collections", [])]
            result["qdrant"]["status"] = "✅ connected"
            result["qdrant"]["collections"] = collections
            result["qdrant"]["collections_count"] = len(collections)
        else:
            result["qdrant"]["status"] = f"❌ error (HTTP {response.status_code})"
    except Exception as e:
        result["qdrant"]["status"] = f"❌ unreachable: {str(e)}"

    # Testar conexão com OpenAI
    try:
        openai_response = httpx.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {openai_api_key}"},
            timeout=5
        )
        if openai_response.status_code == 200:
            result["openai"]["status"] = "✅ connected"
        elif openai_response.status_code == 401:
            result["openai"]["status"] = "❌ invalid API key"
        else:
            result["openai"]["status"] = f"❌ error (HTTP {openai_response.status_code})"
    except Exception as e:
        result["openai"]["status"] = f"❌ unreachable: {str(e)}"

    # Status geral
    qdrant_ok = "✅" in result["qdrant"]["status"]
    openai_ok = "✅" in result["openai"]["status"]
    result["overall_status"] = "✅ healthy" if (mem0_initialized and qdrant_ok and openai_ok) else "⚠️ degraded"

    return result

@app.post("/memory/add")
def add_memory(request: AddMemoryRequest):
    """Add new memory"""
    if not memory:
        raise HTTPException(status_code=500, detail=f"Mem0 not initialized: {mem0_error}")

    try:
        result = memory.add(request.messages, user_id=request.user_id)
        logger.info(f"✅ Memory added for user: {request.user_id}")
        return {"success": True, "result": result}
    except Exception as e:
        logger.error(f"❌ Error adding memory: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memory/search")
def search_memory(request: SearchMemoryRequest):
    """Search memories with optional FlashRank reranking"""
    if not memory:
        raise HTTPException(status_code=500, detail=f"Mem0 not initialized: {mem0_error}")

    try:
        result = memory.search(query=request.query, user_id=request.user_id, limit=request.limit)

        # Aplicar FlashRank reranking se disponível
        ranker = get_flashrank()
        if ranker:
            try:
                from flashrank import RerankRequest
                memories = result.get("results", result) if isinstance(result, dict) else result
                if memories and isinstance(memories, list):
                    passages = [{"id": i, "text": m.get("memory", str(m))} for i, m in enumerate(memories)]
                    rerank_req = RerankRequest(query=request.query, passages=passages)
                    reranked = ranker.rerank(rerank_req)
                    # Reordenar memórias conforme score do FlashRank
                    reranked_sorted = sorted(reranked, key=lambda x: x.get("score", 0), reverse=True)
                    top_ids = [r["id"] for r in reranked_sorted[:request.top_n]]
                    reranked_memories = [memories[i] for i in top_ids if i < len(memories)]
                    # Adicionar score do reranker
                    for i, mem in enumerate(reranked_memories):
                        mem["rerank_score"] = reranked_sorted[i].get("score", 0)
                    if isinstance(result, dict):
                        result["results"] = reranked_memories
                        result["reranked"] = True
                    else:
                        result = reranked_memories
                    logger.info(f"✅ FlashRank reranking applied: {len(memories)} -> {len(reranked_memories)} results")
            except Exception as re:
                logger.warning(f"⚠️ FlashRank reranking failed, returning original results: {re}")

        logger.info(f"✅ Memory search completed for user: {request.user_id}")
        return {"success": True, "result": result}
    except Exception as e:
        logger.error(f"❌ Error searching memory: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memory/all")
def get_all_memories(user_id: str = "default"):
    """Get all memories for a user"""
    if not memory:
        raise HTTPException(status_code=500, detail=f"Mem0 not initialized: {mem0_error}")

    try:
        result = memory.get_all(user_id=user_id)
        logger.info(f"✅ Retrieved all memories for user: {user_id}")
        return {"success": True, "result": result}
    except Exception as e:
        logger.error(f"❌ Error retrieving memories: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/memory/update")
def update_memory(request: UpdateMemoryRequest):
    """Update existing memory"""
    if not memory:
        raise HTTPException(status_code=500, detail=f"Mem0 not initialized: {mem0_error}")

    try:
        result = memory.update(request.memory_id, data=request.data, user_id=request.user_id)
        logger.info(f"✅ Memory updated: {request.memory_id}")
        return {"success": True, "result": result}
    except Exception as e:
        logger.error(f"❌ Error updating memory: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/memory/delete")
def delete_memory(request: DeleteMemoryRequest):
    """Delete memory"""
    if not memory:
        raise HTTPException(status_code=500, detail=f"Mem0 not initialized: {mem0_error}")

    try:
        result = memory.delete(request.memory_id, user_id=request.user_id)
        logger.info(f"✅ Memory deleted: {request.memory_id}")
        return {"success": True, "result": result}
    except Exception as e:
        logger.error(f"❌ Error deleting memory: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/users")
def list_users():
    """Lista todos os user_ids que possuem memórias no Qdrant"""
    import httpx
    try:
        # Busca todos os pontos da coleção mem0 no Qdrant e extrai user_ids únicos
        headers = {}
        if qdrant_api_key:
            headers["api-key"] = qdrant_api_key

        # Primeiro verifica se a coleção existe
        col_res = httpx.get(
            f"http://{qdrant_host}:{qdrant_port}/collections",
            headers=headers, timeout=5
        )
        collections = [c["name"] for c in col_res.json().get("result", {}).get("collections", [])]

        user_ids = set()

        for collection in collections:
            offset = None
            while True:
                params = {"limit": 100, "with_payload": "true"}
                if offset:
                    params["offset"] = offset

                scroll_res = httpx.post(
                    f"http://{qdrant_host}:{qdrant_port}/collections/{collection}/points/scroll",
                    headers={**headers, "Content-Type": "application/json"},
                    json={"limit": 100, "with_payload": True, "offset": offset},
                    timeout=10
                )
                data = scroll_res.json().get("result", {})
                points = data.get("points", [])

                for point in points:
                    payload = point.get("payload", {})
                    uid = payload.get("user_id") or payload.get("userId") or payload.get("user")
                    if uid:
                        user_ids.add(uid)

                next_offset = data.get("next_page_offset")
                if not next_offset or not points:
                    break
                offset = next_offset

        users = sorted(list(user_ids))
        return {"success": True, "users": users, "total": len(users)}

    except Exception as e:
        logger.error(f"❌ Error listing users: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/memory/delete/{memory_id}")
def delete_memory_by_id(memory_id: str):
    """Delete memory by ID (via URL param)"""
    if not memory:
        raise HTTPException(status_code=500, detail=f"Mem0 not initialized: {mem0_error}")
    try:
        result = memory.delete(memory_id)
        logger.info(f"✅ Memory deleted: {memory_id}")
        return {"success": True, "result": result}
    except Exception as e:
        logger.error(f"❌ Error deleting memory: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "Mem0 API is running",
        "docs": "/docs",
        "health": "/health",
        "health_detailed": "/health/detailed"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
