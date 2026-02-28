import time
import json
import faiss
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from contextlib import asynccontextmanager
from typing import List, Tuple

EMBED_MODEL_NAME = "BAAI/bge-large-en-v1.5"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
FAISS_INDEX_FILE = "index_data/faiss.index"
METADATA_FILE = "index_data/metadata.jsonl"
INITIAL_RETRIEVAL_K = 80
RERANK_BATCH_SIZE = 128
RERANK_PREFILTER_TOP = 25

resources = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device detected: {device}")
    resources["device"] = device
    resources["embed_model"] = SentenceTransformer(EMBED_MODEL_NAME, device=device)
    reranker = CrossEncoder(
        RERANK_MODEL_NAME,
        device=device,
        max_length=512
    )
    reranker.model.eval()
    if device == "cuda":
        reranker.model = reranker.model.to("cuda")
        try:
            reranker.model.half()
        except:
            pass
    resources["reranker"] = reranker
    try:
        index = faiss.read_index(FAISS_INDEX_FILE)
        if hasattr(index, "hnsw"):
            index.hnsw.efSearch = 512
        resources["index"] = index
    except Exception:
        resources["index"] = None
    metadata = []
    try:
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            for line in f:
                metadata.append(json.loads(line))
    except Exception:
        pass
    resources["metadata"] = metadata
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
    total_ms: float
    embed_ms: float
    retrieve_ms: float
    rerank_ms: float
    rerank_token_ms: float
    rerank_forward_ms: float

def make_pairs(query: str, texts: List[str]) -> List[Tuple[str,str]]:
    return [(query, t) for t in texts]

def gpu_rerank(reranker, query: str, texts: List[str], batch_size: int = 64, max_length: int = 256):
    tokenizer = reranker.tokenizer
    model = reranker.model
    device = next(model.parameters()).device if len(list(model.parameters()))>0 else torch.device("cpu")
    scores = []
    token_time = 0.0
    forward_time = 0.0
    model.eval()
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            pairs = [(query, t) for t in batch_texts]
            t_tok_start = time.perf_counter()
            enc = tokenizer(pairs, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            t_tok_end = time.perf_counter()
            token_time += (t_tok_end - t_tok_start)
            for k, v in enc.items():
                enc[k] = v.to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_fwd_start = time.perf_counter()
            outputs = model(**enc)
            t_fwd_end = time.perf_counter()
            forward_time += (t_fwd_end - t_fwd_start)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            if logits.dim() > 1:
                batch_scores = logits.squeeze(-1).detach().cpu().numpy().tolist()
            else:
                batch_scores = logits.detach().cpu().numpy().tolist()
            if isinstance(batch_scores, float):
                batch_scores = [batch_scores]
            scores.extend(batch_scores)
    return scores, token_time * 1000.0, forward_time * 1000.0

@app.post("/search", response_model=QueryResponse)
async def search(req: QueryRequest):
    if resources.get("index", None) is None:
        raise HTTPException(status_code=500, detail="Index not loaded")
    t0 = time.perf_counter()
    device = resources["device"]
    embed_model = resources["embed_model"]
    reranker = resources["reranker"]
    index = resources["index"]
    metadata = resources["metadata"]
    t_embed_start = time.perf_counter()
    query_embedding = embed_model.encode(["query: " + req.query], normalize_embeddings=True)
    t_embed_end = time.perf_counter()
    t_retrieve_start = time.perf_counter()
    D, I = index.search(query_embedding, INITIAL_RETRIEVAL_K)
    candidates = []
    candidate_texts = []
    candidate_indices = []
    for dist, idx in zip(D[0], I[0]):
        if idx == -1 or idx >= len(metadata):
            continue
        meta = metadata[idx]
        text = meta["text"]
        candidates.append({
            "faiss_score": float(dist),
            "text": text,
            "pdf_name": meta["pdf_name"],
            "page": meta["page"],
            "idx": idx
        })
        candidate_texts.append(text)
        candidate_indices.append(idx)
    t_retrieve_end = time.perf_counter()
    t_rerank_start = time.perf_counter()
    final_candidates = []
    tok_ms = 0.0
    fwd_ms = 0.0
    rerank_token_ms = 0.0
    rerank_forward_ms = 0.0
    if candidate_texts:
        prefilter_texts = candidate_texts[:RERANK_PREFILTER_TOP]
        prefilter_candidates = candidates[:len(prefilter_texts)]
        rerank_scores, tok_ms, fwd_ms = gpu_rerank(
            reranker,
            req.query,
            prefilter_texts,
            batch_size=RERANK_BATCH_SIZE,
            max_length=256
        )
        rerank_token_ms = float(tok_ms)
        rerank_forward_ms = float(fwd_ms)
        for i, cand in enumerate(prefilter_candidates):
            cand["rerank_score"] = float(rerank_scores[i]) if i < len(rerank_scores) else float(cand.get("faiss_score", 0.0))
        prefilter_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        final_candidates = prefilter_candidates
    try:
        model_dev = next(reranker.model.parameters()).device
        if model_dev.type == "cuda":
            torch.cuda.synchronize()
    except Exception:
        pass
    t_rerank_end = time.perf_counter()
    try:
        model_dev = next(reranker.model.parameters()).device
        if model_dev.type == "cuda":
            torch.cuda.synchronize()
            used_mb = torch.cuda.memory_allocated(0) / 1024**2
            print(f"RERANK DEBUG: token_ms={tok_ms:.1f} fwd_ms={fwd_ms:.1f} gpu_used_mb={used_mb:.1f}")
        else:
            print(f"RERANK DEBUG (cpu): token_ms={tok_ms:.1f} fwd_ms={fwd_ms:.1f}")
    except Exception as e:
        print("RERANK DEBUG EX:", e)
    results = []
    for c in final_candidates[:req.k]:
        results.append(SearchResult(
            score=c.get("rerank_score", c.get("faiss_score", 0.0)),
            text=(c["text"][:800] + "...") if len(c["text"])>800 else c["text"],
            pdf_name=c["pdf_name"],
            page=c["page"]
        ))
    t1 = time.perf_counter()
    total_ms = (t1 - t0) * 1000
    embed_ms = (t_embed_end - t_embed_start) * 1000
    retrieve_ms = (t_retrieve_end - t_retrieve_start) * 1000
    rerank_ms = (t_rerank_end - t_rerank_start) * 1000
    return QueryResponse(
        results=results,
        total_ms=round(total_ms,2),
        embed_ms=round(embed_ms,2),
        retrieve_ms=round(retrieve_ms,2),
        rerank_ms=round(rerank_ms,2),
        rerank_token_ms=round(rerank_token_ms,2),
        rerank_forward_ms=round(rerank_forward_ms,2),
    )

@app.get("/health")
def health():
    return {"status":"ok","device": resources.get("device","unknown")}