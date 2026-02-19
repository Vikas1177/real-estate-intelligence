# Architecture & System Design Notes

This document outlines the engineering decisions, system limits, and trade-offs made while building the real estate document intelligence prototype.

## 1. Technical Decisions & Justifications

We bypassed some of the basic suggested tools in favor of a highly optimized, production-leaning stack to prioritize both latency and retrieval accuracy.

* **PDF Processing (`PyMuPDF` + `Tesseract`)**: `PyMuPDF` was selected over `PyPDF2` because it is significantly faster and supports bounding-box table extraction. `Tesseract OCR` was added as a fallback to ensure text is captured even from image-heavy scanned property brochures.
* **Embeddings (`BAAI/bge-base-en-v1.5`)**: Selected over the suggested `all-MiniLM-L6-v2` because it consistently achieves higher accuracy on retrieval benchmarks, better capturing complex real estate semantics.
* **Vector Database (`FAISS` with HNSW Index)**: `FAISS` was used with a Hierarchical Navigable Small World (HNSW) index. HNSW provides incredibly fast Approximate Nearest Neighbor (ANN) search, trading a negligible amount of accuracy for a massive reduction in query latency.
* **Two-Stage Retrieval Pipeline**: Instead of relying solely on the embedding model, the system retrieves the top 30 candidates using FAISS (fast but less precise) and reranks them using a Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) to return the top 3 (highly precise but computationally heavy).
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
    * *Query Phase:* The Cross-Encoder reranker requires thousands of floating-point operations. If the number of candidates retrieved by FAISS is increased beyond 30 to improve recall, the CPU load will spike and latency will severely degrade.

---

## 3. Challenges & Trade-offs

* **Handling Large PDFs**: Processing massive documents is computationally expensive. For this prototype, it is recommended to limit uploads to PDFs with <50 pages to keep ingestion times reasonable on consumer hardware.
* **Balancing Chunk Size**: Striking the right balance between accuracy and speed required a hybrid chunking strategy. The system targets 300-word chunks but respects sentence boundaries to prevent cutting off crucial context. A 75-word overlap was implemented to ensure semantic continuity between adjacent chunks. 
* **Domain-Specific Jargon**: Real estate documents contain highly specific terminology (e.g., "UP RERA", "IGBC Platinum"). While the current model performs well, fine-tuning the embedding model on Indian real estate data remains an optional stretch goal to push Top-1 accuracy even higher.
