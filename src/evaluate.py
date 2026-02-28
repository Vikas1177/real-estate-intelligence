import requests
import json
import math
import csv
import statistics
from typing import List, Dict

API_URL = "http://localhost:8000/search"

TEST_SET = [
    ("What is the total super built-up area of Max House, Okhla?", 2),
    ("How many tenant floors are there in Max House?", 2),
    ("What is the typical floor plate size at Max House?", 2),
    ("What is the green rating of Max House?", 2),
    ("How far is Max House, Okhla from the Okhla NSIC Metro Station?", 3),
    ("How far is Max House from IGI Airport?", 3),
    ("Is Max House within walking distance of a metro station?", 3),
    ("What façade material is used in Max House?", 6),
    ("What is the floor-to-ceiling height at Max House?", 7),
    ("Does Max House use double-glazed windows?", 7),
    ("What air treatment technology is used in Max House?", 10),
    ("Is Max House LEED certified?", 10),
    ("Does Max House incorporate biophilic design principles?", 10),
]

K_EVAL = [1, 3]
SAVE_CSV = "evaluation_results.csv"


def recall_at_k(expected_page: int, retrieved_pages: List[int], k: int) -> int:
    return 1 if expected_page in retrieved_pages[:k] else 0


def reciprocal_rank(expected_page: int, retrieved_pages: List[int]) -> float:
    for i, p in enumerate(retrieved_pages, start=1):
        if p == expected_page:
            return 1.0 / i
    return 0.0


def dcg_at_k(relevant_pages: set, retrieved_pages: List[int], k: int) -> float:
    dcg = 0.0
    for i, p in enumerate(retrieved_pages[:k], start=1):
        rel = 1.0 if p in relevant_pages else 0.0
        if rel > 0.0:
            dcg += (2 ** rel - 1) / math.log2(i + 1)
    return dcg


def idcg_at_k(relevant_count: int, k: int) -> float:
    idcg = 0.0
    rels = min(relevant_count, k)
    for i in range(1, rels + 1):
        idcg += (2 ** 1 - 1) / math.log2(i + 1)
    return idcg


def ndcg_at_k(relevant_pages: set, retrieved_pages: List[int], k: int) -> float:
    idcg = idcg_at_k(len(relevant_pages), k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(relevant_pages, retrieved_pages, k) / idcg


def evaluate_query(query: str, expected_page: int, k: int = 3) -> Dict:
    resp = requests.post(API_URL, json={"query": query, "k": k}, timeout=60)
    data = resp.json()

    latency_total = data.get("total_ms", None)
    embed_ms = data.get("embed_ms", None)
    retrieve_ms = data.get("retrieve_ms", None)
    rerank_ms = data.get("rerank_ms", None)

    results = data.get("results", [])
    retrieved_pages = [r.get("page", -1) for r in results]

    rr = reciprocal_rank(expected_page, retrieved_pages)
    recalls = {f"recall@{kk}": recall_at_k(expected_page, retrieved_pages, kk) for kk in K_EVAL}
    ndcgs = {f"ndcg@{kk}": ndcg_at_k({expected_page}, retrieved_pages, kk) for kk in K_EVAL}
    top1 = 1 if (retrieved_pages and retrieved_pages[0] == expected_page) else 0
    top3 = 1 if expected_page in retrieved_pages[:3] else 0

    return {
        "query": query,
        "expected_page": expected_page,
        "latency_ms": latency_total,
        "embed_ms": embed_ms,
        "retrieve_ms": retrieve_ms,
        "rerank_ms": rerank_ms,
        "recalls": recalls,
        "ndcgs": ndcgs,
        "reciprocal_rank": rr,
        "top1": top1,
        "top3": top3,
        "retrieved_pages": retrieved_pages
    }


def run_evaluation():
    rows = []
    latencies = []
    recall_counts = {f"recall@{k}": 0 for k in K_EVAL}
    ndcg_sums = {f"ndcg@{k}": 0.0 for k in K_EVAL}
    top1_count = 0
    top3_count = 0
    rr_list = []

    for query, expected_page in TEST_SET:
        res = evaluate_query(query, expected_page, k=3)
        rows.append(res)

        if res["latency_ms"] is not None:
            latencies.append(res["latency_ms"])

        for k in K_EVAL:
            recall_counts[f"recall@{k}"] += res["recalls"][f"recall@{k}"]
            ndcg_sums[f"ndcg@{k}"] += res["ndcgs"][f"ndcg@{k}"]

        top1_count += res["top1"]
        top3_count += res["top3"]
        rr_list.append(res["reciprocal_rank"])

    n = len(TEST_SET)

    avg_latency = statistics.mean(latencies) if latencies else None
    p95_latency = (sorted(latencies)[int(math.ceil(0.95 * len(latencies))) - 1]
                   if latencies else None)
    recall_at = {k: (recall_counts[f"recall@{k}"] / n) for k in K_EVAL}
    ndcg_at = {k: (ndcg_sums[f"ndcg@{k}"] / n) for k in K_EVAL}
    top1_acc = top1_count / n
    top3_acc = top3_count / n
    mrr = statistics.mean(rr_list) if rr_list else 0.0

    with open(SAVE_CSV, "w", newline='', encoding="utf-8") as csvfile:
        fieldnames = ["query", "expected_page", "latency_ms", "top1", "top3", "reciprocal_rank",
                      "retrieved_pages", "recalls", "ndcgs"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "query": r["query"],
                "expected_page": r["expected_page"],
                "latency_ms": r.get("latency_ms"),
                "top1": r.get("top1"),
                "top3": r.get("top3"),
                "reciprocal_rank": r.get("reciprocal_rank"),
                "retrieved_pages": ";".join([str(p) for p in r.get("retrieved_pages", [])]),
                "recalls": json.dumps(r.get("recalls", {})),
                "ndcgs": json.dumps(r.get("ndcgs", {}))
            })

    print("\nEVALUATION SUMMARY")
    print(f"Queries evaluated: {n}")
    print(f"Average latency (ms): {avg_latency:.2f}" if avg_latency is not None else "Average latency (ms): None")
    print(f"P95 latency (ms): {p95_latency:.2f}" if p95_latency is not None else "P95 latency (ms): None")
    for k in K_EVAL:
        print(f"Recall@{k}: {recall_at[k]*100:.2f}%")
    print(f"Top-1 Accuracy: {top1_acc*100:.2f}%")
    print(f"Top-3 Accuracy: {top3_acc*100:.2f}%")
    print(f"MRR: {mrr:.4f}")
    for k in K_EVAL:
        print(f"nDCG@{k}: {ndcg_at[k]:.4f}")
    print(f"Per-query results saved to: {SAVE_CSV}")


if __name__ == "__main__":
    run_evaluation()