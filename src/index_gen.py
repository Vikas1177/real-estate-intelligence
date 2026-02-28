import os
import json
import time
from pathlib import Path
from typing import List
from tqdm import tqdm
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pytesseract
import argparse

from processing import extract_text_pages_smart, chunk_text, CHUNK_WORDS, CHUNK_OVERLAP_WORDS

EMBED_MODEL_NAME = "BAAI/bge-large-en-v1.5"
EMBED_BATCH = 64
FAISS_INDEX_FILE = "faiss.index"
METADATA_FILE = "metadata.jsonl"
EMBEDDINGS_FILE = "embeddings.npy"
RERANKER_INFO_FILE = "reranker_info.json"
HNSW_M = 32
HNSW_EFCONSTRUCTION = 200
HNSW_EFSEARCH = 512
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

def ingest_pdfs(pdf_paths: List[str], out_dir: str = "."):
    pipeline_start = time.perf_counter()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = out_dir / METADATA_FILE
    embeddings_path = out_dir / EMBEDDINGS_FILE
    index_path = out_dir / FAISS_INDEX_FILE
    reranker_info_path = out_dir / RERANKER_INFO_FILE
    print(" Step 1: Extracting Pages ")
    all_pages = []
    pages_processed = 0
    t0 = time.perf_counter()
    for pdf_path in pdf_paths:
        print(f"Processing {pdf_path}...")
        pdf_name = Path(pdf_path).name
        pages = extract_text_pages_smart(pdf_path)
        for p in pages:
            pages_processed += 1
            all_pages.append({"pdf_name": pdf_name, **p})
    t1 = time.perf_counter()
    extract_time = t1 - t0
    print(f"Step 1 Complete: {len(all_pages)} pages in {extract_time:.2f}s")
    print("\n Step 2: Chunking Text ")
    t0 = time.perf_counter()
    all_chunks = []
    for p in all_pages:
        chunks = chunk_text(p['text'], p['pdf_name'], p['page_num'], target_words=CHUNK_WORDS, overlap_words=CHUNK_OVERLAP_WORDS)
        for c in chunks:
            c['ocr_used'] = p['ocr_used']
            c['ocr_confidence'] = p['ocr_confidence']
        all_chunks.extend(chunks)
    t1 = time.perf_counter()
    chunk_time = t1 - t0
    print(f"Step 2 Complete: Created {len(all_chunks)} chunks in {chunk_time:.2f}s")
    if not all_chunks:
        print("No text found in docs.")
        return
    print("\n Step 3: Saving Metadata ")
    t0 = time.perf_counter()
    with open(metadata_path, "w", encoding="utf-8") as f:
        for c in all_chunks:
            meta = {
                "pdf_name": c["pdf_name"],
                "page": c["page"],
                "chunk_id": c["chunk_id"],
                "text": c["text"],
                "ocr_used": c.get("ocr_used", False),
                "ocr_confidence": c.get("ocr_confidence", 0.0)
            }
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    t1 = time.perf_counter()
    meta_time = t1 - t0
    print(f"Step 3 Complete: Metadata saved in {meta_time:.2f}s to {metadata_path}")
    print(f"\n Step 4: Computing Embeddings ({EMBED_MODEL_NAME}) ")
    t0 = time.perf_counter()
    model = SentenceTransformer(EMBED_MODEL_NAME)
    texts = ["passage: " + c["text"] for c in all_chunks]
    embeddings = []
    for i in tqdm(range(0, len(texts), EMBED_BATCH)):
        batch = texts[i:i+EMBED_BATCH]
        emb = model.encode(batch, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        embeddings.append(emb)
    embeddings = np.vstack(embeddings).astype("float32")
    faiss.normalize_L2(embeddings)
    np.save(embeddings_path, embeddings)
    t1 = time.perf_counter()
    embed_time = t1 - t0
    print(f"Step 4 Complete: {len(embeddings)} vectors computed in {embed_time:.2f}s")
    print("\n Step 5: Building FAISS HNSW Index ")
    t0 = time.perf_counter()
    d = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(d, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = HNSW_EFCONSTRUCTION
    try:
        index.hnsw.efSearch = HNSW_EFSEARCH
    except Exception:
        pass
    index.add(embeddings)
    faiss.write_index(index, str(index_path))
    t1 = time.perf_counter()
    index_time = t1 - t0
    print(f"Step 5 Complete: Index built in {index_time:.2f}s and saved to {index_path}")
    print("\n Step 6: Writing reranker info ")
    t0 = time.perf_counter()
    reranker_info = {
        "cross_encoder_model": CROSS_ENCODER_MODEL,
        "note": "Load this cross-encoder at query-time and use it to rescore top-K FAISS hits"
    }
    with open(reranker_info_path, "w", encoding="utf-8") as f:
        json.dump(reranker_info, f, ensure_ascii=False, indent=2)
    t1 = time.perf_counter()
    reranker_time = t1 - t0
    print(f"Reranker info saved to {reranker_info_path} (time: {reranker_time:.2f}s)")
    total_time = time.perf_counter() - pipeline_start
    print(f"INGESTION COMPLETE")
    print(f"Total Time:   {total_time:.2f}s")
    print(f"  - Extraction: {extract_time:.2f}s")
    print(f"  - Chunking:   {chunk_time:.2f}s")
    print(f"  - Metadata:   {meta_time:.2f}s")
    print(f"  - Embedding:  {embed_time:.2f}s")
    print(f"  - Indexing:   {index_time:.2f}s")
    print(f"  - Reranker write: {reranker_time:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_dir", required=True)
    parser.add_argument("--out_dir", default="./index_data")
    parser.add_argument("--tesseract_cmd", default="")
    args = parser.parse_args()
    if args.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract_cmd
    pdf_dir = Path(args.pdf_dir)
    pdf_paths = [str(p) for p in pdf_dir.glob("*.pdf")]
    if pdf_paths:
        ingest_pdfs(pdf_paths, out_dir=args.out_dir)
    else:
        print("No PDFs found.")