import time
import json
import faiss
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from contextlib import asynccontextmanager
from typing import List

EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
FAISS_INDEX_FILE = "index_data/faiss.index"
METADATA_FILE = "index_data/metadata.jsonl"
INITIAL_RETRIEVAL_K = 10

resources = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\nLoading resources...")
    print("Loading embedding model...")
    resources["embed_model"] = SentenceTransformer(EMBED_MODEL_NAME)
    print("Loading cross-encoder reranker...")
    resources["reranker"] = CrossEncoder(RERANK_MODEL_NAME)
    print("Loading FAISS index...")
    try:
        index = faiss.read_index(FAISS_INDEX_FILE)
        if hasattr(index, "hnsw"):
            index.hnsw.efSearch = 128
        resources["index"] = index
    except Exception as e:
        print("Failed loading index:", e)
        resources["index"] = None
    print("Loading metadata...")
    metadata = []
    try:
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            for line in f:
                metadata.append(json.loads(line))
    except Exception as e:
        print("Failed loading metadata:", e)
    resources["metadata"] = metadata
    print("Resources loaded successfully.\n")
    yield
    resources.clear()

app = FastAPI(lifespan=lifespan)

class QueryRequest(BaseModel):
    query: str
    k: int = 3

class SearchResult(BaseModel):
    score: float
    text: str
    pdf_name: str
    page: int

class QueryResponse(BaseModel):
    results: List[SearchResult]
    latency_ms: float

@app.post("/search", response_model=QueryResponse)
async def search(req: QueryRequest):
    t0 = time.perf_counter()
    if resources["index"] is None:
        raise HTTPException(status_code=500, detail="Index not loaded")
    embed_model = resources["embed_model"]
    reranker = resources["reranker"]
    index = resources["index"]
    metadata = resources["metadata"]
    query_embedding = embed_model.encode(["query: " + req.query], normalize_embeddings=True)
    distances, indices = index.search(query_embedding, INITIAL_RETRIEVAL_K)
    candidates = []
    candidate_texts = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1 or idx >= len(metadata):
            continue
        meta = metadata[idx]
        text = meta["text"]
        candidates.append({
            "faiss_score": float(dist),
            "text": text,
            "pdf_name": meta["pdf_name"],
            "page": meta["page"],
        })
        candidate_texts.append(text)
    if len(candidate_texts) > 0:
        pairs = [(req.query, text) for text in candidate_texts]
        rerank_scores = reranker.predict(pairs)
        for i in range(len(candidates)):
            candidates[i]["rerank_score"] = float(rerank_scores[i])
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    final_results = []
    for c in candidates[:req.k]:
        final_results.append(
            SearchResult(
                score=c["rerank_score"],
                text=c["text"][:200] + "...",
                pdf_name=c["pdf_name"],
                page=c["page"]
            )
        )
    t1 = time.perf_counter()
    latency = (t1 - t0) * 1000
    return QueryResponse(results=final_results, latency_ms=round(latency, 2))

@app.get("/health")
def health():
    return {"status": "ok"}
