# Architecture & System Design

This document outlines the engineering decisions, system limits, and trade-offs made while building the real estate document intelligence prototype.

## 1. Technical Decisions & Justifications

We bypassed some of the basic suggested tools in favor of a highly optimized, production-leaning stack to prioritize both latency and retrieval accuracy.

* **PDF Processing (`PyMuPDF` + `Tesseract`)**: `PyMuPDF` was selected over `PyPDF2` because it is significantly faster and supports bounding-box table extraction. `Tesseract OCR` was added as a fallback to ensure text is captured even from image-heavy scanned property brochures.
* **Embeddings (`BAAI/bge-base-en-v1.5`)**: Selected over the suggested `all-MiniLM-L6-v2` because it consistently achieves higher accuracy on retrieval benchmarks, better capturing complex real estate semantics.
* **Vector Database (`FAISS` with HNSW Index)**: `FAISS` was used with a Hierarchical Navigable Small World (HNSW) index. HNSW provides incredibly fast Approximate Nearest Neighbor (ANN) search, trading a negligible amount of accuracy for a massive reduction in query latency.
* **Two-Stage Retrieval Pipeline**: Instead of relying solely on the embedding model, the system retrieves the top candidates using FAISS (fast but less precise) and reranks them using a Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) to return the top 3 (highly precise but computationally heavy).
* **Backend (`FastAPI`)**: Used for its asynchronous capabilities and speed. The models and FAISS index are loaded entirely into RAM during the app's startup lifecycle to eliminate cold-start disk I/O on user queries.

---

## 2. System Behavior Under Scale

To build with scalability and latency awareness, it is critical to understand the system's operational limits:

* **What happens as PDFs grow larger?**
    As document sizes increase, the ingestion time scales linearly due to the CPU-intensive nature of chunking and embedding. The resulting FAISS index size will also grow, consuming more server RAM. 
* **What would break first in production?**
    **Memory Exhaustion (OOM).** Because the `FAISS IndexHNSWFlat` index and the Transformer models are loaded entirely into the server's RAM for speed, scaling to millions of pages will eventually crash the server. At that scale, the architecture must migrate to a distributed vector database (like Pinecone or Qdrant).
* **Where are the bottlenecks?**
    * *Ingestion Phase:* The OCR fallback is heavily CPU-bound and is the slowest step when processing image-heavy PDFs. 
    * *Query Phase:* If not using GPU, the Cross-Encoder reranker requires thousands of floating-point operations. If the number of candidates retrieved by FAISS is increased to improve recall, the CPU load will spike and latency will severely degrade.

---

## 3. Observations & trade-offs (re-ranking vs speed)

* **Prefilter size (dense K)**
    * ↑K → much better recall (reranker can only improve candidates it sees) but increased retrieval time and more tokens to rerank.
    * Practical sweet spot: **K=80** gives strong recall with modest latency increase when reranker is GPU-enabled.
* **HNSW efSearch**
    * ↑efSearch → higher recall from FAISS at query time (+ms). Setting **efSearch=512** is a low latency/high recall win on modern CPUs/GPUs.
* **Reranker depth (prefilter top-N)**
    * Deeper rerank (**25–40**) gives big Top-1 gains but increases rerank time linearly.
    * Using GPU + batching (**batch_size=64–128**) reduces wall time; measure tokenization vs forward time to decide if pre-tokenization is necessary.
* **Embedding model size**
    * Larger models (**bge-large**) improve semantic matching but increase embedding time slightly (query embedding cost is small).
* **Hybrid retrieval (Dense + BM25)**
    * Adds ~negligible latency but improves exact match recall for numbers and names — high payoff.
* **Net effect**
    * With above tuning (**K≈80, efSearch≈512, prefilter≈25, GPU reranker**) you can reach high Recall@3 and good Top-1 while keeping average latency in the low hundreds ms. Always measure avg + P95.
 ---
 
## 4. Concrete steps taken to improve robustness & reduce hallucinations

* **Chunk quality**
    * Sentence-aware chunking with 20–30% overlap; drop tiny chunks; strip repeated headers/footers to avoid noisy high-score boilerplate.
* **OCR + table handling**
    * Use page OCR only when text is sparse; extract tables and convert to markdown to preserve structured facts (reduces LLM hallucination on tabular facts).
* **Hybrid retrieval**
    * Combine FAISS dense retrieval with BM25; fuse scores to ensure exact-match signals (IDs, numbers) rank well before rerank.
* **Cross-encoder reranking on GPU**
    * Prefilter top-N candidates and rescore with CrossEncoder to improve Top-1 ranking; measure tokenization vs forward time and batch appropriately.
* **Verification and rule checks**
    * Numeric/entity sanity checks post-generation (regex compare numbers, check RERA patterns, verify names against retrieved chunks).
* **Conservative fallback**
    * If cross-encoder score or evidence coverage is low, return retrieved snippets and “insufficient evidence” instead of a generated answer.
* **Negative-query tests**
    * Run synthetic negative queries and track false positive rate; use as monitoring signal to tighten thresholds.
* **Evaluation & monitoring**
    * Automated metrics (Recall@K, MRR, nDCG, hallucination rate) and stage-wise latency (embed / retrieve / rerank / generate) recorded per query. Use P95 latency for SLOs.
* **Advanced: pre-tokenize chunks**
    * For large rerank budgets, prestore tokenized chunk tensors to avoid tokenization CPU bottleneck at query time.
---
## 5. Caching Strategy (Query-Time Optimization)

To reduce latency and improve responsiveness, caching can be applied during querying.

### 1. Can we cache embeddings for repeated queries?
**Yes.**
If the same query is asked multiple times, we can store its embedding vector and reuse it instead of recomputing it.

**Benefit:**
* **Removes embedding computation time.**
* **Reduces average latency significantly for repeated queries.**
* **Very low memory overhead.**
---

### 2. Can we cache Top-K retrieval results?
**Yes (recommended).**
We can cache the top-K retrieved chunk IDs and scores for a given query.

**Benefit:**
* **Skips FAISS + BM25 retrieval step.**
* **Only reranking (if enabled) needs to run.**
* **Improves both average latency and P95 latency.**

---

> Caching is a low-risk, high-impact optimization that improves system scalability without affecting retrieval correctness.
