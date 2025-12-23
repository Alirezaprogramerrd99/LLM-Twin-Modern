from fastapi import FastAPI, Depends, Query
from pydantic import BaseModel, HttpUrl
from typing import List, Tuple

from llm_engineering.application.settings import get_settings, Settings
from llm_engineering.application.log_setup import setup_logging
from llm_engineering.application.services.hello_service import HelloService
from llm_engineering.application.services.rag_service import RAGService
from fastapi import HTTPException
from loguru import logger

import hashlib
from llm_engineering.application.services.web_loader import WebLoaderService

# Configure logging once
setup_logging()

app = FastAPI(title="LLM Handbook Ground-Up (Services + Logging + Mini RAG)")

# --- Dependencies ---
def settings_dep() -> Settings:
    return get_settings()

def hello_service_dep(settings: Settings = Depends(settings_dep)) -> HelloService:
    return HelloService(settings=settings)

# Build a single in-memory RAGService instance for the app lifetime
_rag_service: RAGService | None = None
def rag_service_dep(settings: Settings = Depends(settings_dep)) -> RAGService:
    # global variable to hold the RAGService instance.
    global _rag_service
    # Lazy initialization; build on first request
    if _rag_service is None:
         # build the RAGService using the provided settings (builder pattern with classmethod)
        _rag_service = RAGService.build(settings)
    return _rag_service


# --- Existing endpoints ---
@app.get("/", tags=["meta"])
def root(settings: Settings = Depends(settings_dep)):
    return {
        "ok": True,
        "app_name": settings.app_name,
        "environment": settings.app_env,
        "debug": settings.debug,
    }

@app.get("/greet", tags=["demo"])
def greet(
    name: str | None = Query(default=None, description="Your name"),
    svc: HelloService = Depends(hello_service_dep),
):
    return svc.greet(name)

# --- New: RAG endpoints ---
class IndexItem(BaseModel):
    id: str
    text: str

@app.post("/index", tags=["rag"])
def index(items: List[IndexItem], rag: RAGService = Depends(rag_service_dep)):
    n = rag.index([(it.id, it.text) for it in items])
    return {"indexed": n}

@app.get("/search", tags=["rag"])
def search(q: str, k: int = 5, rag: RAGService = Depends(rag_service_dep)):
    """q = query string, k = top-k results"""
    results = rag.search(q, k=k)
    # results: List[(id, score, raw_text)]
    
    # returns a list of dicts containing id, score, and text, got from results which is a list of 3-element tuples
    return [
        {"id": doc_id, "score": round(score, 4), "text": raw}
        for (doc_id, score, raw) in results
    ]


@app.get("/debug/mongo")
def debug_mongo(rag: RAGService = Depends(rag_service_dep)):
    if rag.doc_store is None:
        return {"error": "Mongo store not enabled"}

    docs = list(rag.doc_store.collection.find({}, {"_id": 1, "text": 1}))
    return {"count": len(docs), "docs": docs}


#--- New: RAG ask endpoint ---
# Full RAG step: retrieve + generate answer
# Example: /ask?q=What+do+cats+eat?&k=3
# Returns: { "answer": "...", "sources": [ ... ] }
# tags=["rag"] puts this endpoint in the "rag" group in the Swagger UI
# ... means that the parameter is required.
@app.get("/ask", tags=["rag"])
def ask(
    q: str = Query(..., description="User question"),
    k: int = Query(3, ge=1, le=10),
    rag: RAGService = Depends(rag_service_dep),
):
    # calling the RAG pipeline's ask method
    result = rag.ask(q, k=k)
    if not result:
        raise HTTPException(status_code=500, detail="RAGService.ask returned no result")
    return result


@app.get("/history", tags=["meta"])
def history(
    limit: int = Query(10, ge=1, le=100),
    rag: RAGService = Depends(rag_service_dep),
):
    if rag.history_store is None:
        raise HTTPException(status_code=503, detail="History store not configured (use_mongo=False?)")

    docs = rag.history_store.recent(limit=limit)
    # Convert ObjectId and datetime to strings for JSON
    cleaned = []
    for d in docs:
        d = dict(d)
        d["_id"] = str(d.get("_id", ""))
        if "created_at" in d:
            d["created_at"] = d["created_at"].isoformat()
        cleaned.append(d)
        
    return {"count": len(cleaned), "items": cleaned}

# should be added to schemas
class IngestUrlRequest(BaseModel):
    url: HttpUrl


@app.post("/ingest/url", tags=["ingest"])
def ingest_url(body: IngestUrlRequest, rag: RAGService = Depends(rag_service_dep)):
    url = str(body.url)

    loader = WebLoaderService()
    try:
        title, text = loader.fetch(url)
    except Exception as e:
        # Log full traceback in server logs
        logger.error("Ingest failed for url='{}': {}\n{}", url, e, traceback.format_exc())
        # Return readable error to client
        raise HTTPException(status_code=422, detail=f"Ingest failed: {type(e).__name__}: {e}")

    doc_id = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    indexed = rag.index([(doc_id, text)])

    return {"ok": True, "doc_id": doc_id, "title": title, "indexed_chunks": indexed, "url": url}
