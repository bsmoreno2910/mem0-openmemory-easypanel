from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from mem0 import Memory
import os
import logging

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

# Configure Mem0
try:
    config = {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "host": os.getenv("QDRANT_HOST", "localhost"),
                "port": int(os.getenv("QDRANT_PORT", 6333)),
                "api_key": os.getenv("QDRANT_API_KEY"),
                "https": False,
                "prefer_grpc": False,
            },
        },
        "llm": {
            "provider": "openai",
            "config": {
                "api_key": os.getenv("OPENAI_API_KEY"),
                "model": os.getenv("LLM_MODEL", "gpt-4o-mini"),
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "api_key": os.getenv("OPENAI_API_KEY"),
                "model": os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
            },
        },
    }
    
    memory = Memory.from_config(config)
    logger.info("✅ Mem0 initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize Mem0: {str(e)}")
    memory = None

# Request/Response Models
class AddMemoryRequest(BaseModel):
    messages: list
    user_id: str = "default"

class SearchMemoryRequest(BaseModel):
    query: str
    user_id: str = "default"

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
        "version": "1.0.0"
    }

@app.post("/memory/add")
def add_memory(request: AddMemoryRequest):
    """Add new memory"""
    if not memory:
        raise HTTPException(status_code=500, detail="Mem0 not initialized")
    
    try:
        result = memory.add(request.messages, user_id=request.user_id)
        logger.info(f"✅ Memory added for user: {request.user_id}")
        return {"success": True, "result": result}
    except Exception as e:
        logger.error(f"❌ Error adding memory: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memory/search")
def search_memory(request: SearchMemoryRequest):
    """Search memories"""
    if not memory:
        raise HTTPException(status_code=500, detail="Mem0 not initialized")
    
    try:
        result = memory.search(query=request.query, user_id=request.user_id)
        logger.info(f"✅ Memory search completed for user: {request.user_id}")
        return {"success": True, "result": result}
    except Exception as e:
        logger.error(f"❌ Error searching memory: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memory/all")
def get_all_memories(user_id: str = "default"):
    """Get all memories for a user"""
    if not memory:
        raise HTTPException(status_code=500, detail="Mem0 not initialized")
    
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
        raise HTTPException(status_code=500, detail="Mem0 not initialized")
    
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
        raise HTTPException(status_code=500, detail="Mem0 not initialized")
    
    try:
        result = memory.delete(request.memory_id, user_id=request.user_id)
        logger.info(f"✅ Memory deleted: {request.memory_id}")
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
        "health": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
