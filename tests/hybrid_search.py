import psycopg
import re
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

DATABASE_URL = "postgresql://ai_user:ai_password@127.0.0.1:5432/rag_engine"

model = SentenceTransformer("all-MiniLM-L6-v2")

def simple_tokeniser(text: str) -> list[str:]:
    return re.findall(r"\w+", text.lower())

def get_all_chunks():
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("""
            SELECT id, chunk_index, content 
            FROM document_chunks
            ORDER BY chunk_index ASC;
            """)

            return cur.fetchall()

def dense_search(query: str, top_k: int):
    query_vector = model.encode(query).tolist()

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("""
            SELECT id, chunk_index, content,
            (embedding <=> %s::vector) AS cosine_distance
            FROM document_chunks
            ORDER BY cosine_distance ASC
            LIMIT %s;
            """, (query_vector, top_k))

            return cur.fetchall()

def sparse_search(query: str, corpus_rows: int, top_k: int):
    tokenised_query = simple_tokeniser(query)
    tokenised_corpus = [simple_tokeniser(row[2]) for row in corpus_rows]
    bm25 = BM25Okapi(tokenised_corpus)
    scores = bm25.get_scores(tokenised_query)

    ranked = sorted(zip(corpus_rows, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]

def reciprocal_rank_fusion(dense_results: list, sparse_results: list, k:int = 60):
    """
    Merges dense and sparse rankings using Reciprocal Rank Fusion.
    dense_results: list of tuples (id, chunk_index, content, distance)
    sparse_results: list of tuples ((id, chunk_index, content), score)
    """
    rrf_map = {}

    # 1. Score dense vector ranks
    for rank_idx, row in enumerate(dense_results, start=1):
        doc_id, chunk_idx, content, _ = row
        if doc_id not in rrf_map:
            rrf_map[doc_id] = {"chunk_index": chunk_idx, "content":content, "rrf_score":0.0}

        rrf_map[doc_id]["rrf_score"] += 1.0 / (k+rank_idx)

    # 2. Score sparse BM25 ranks
    for rank_idx, (row, score) in enumerate(sparse_results, start=1):
        # Only consider BM25 results that have a positive score
        if score <= 0.0:
            continue

        doc_id, chunk_idx, content = row
        if doc_id not in rrf_map:
            rrf_map[doc_id] = {"chunk_index": chunk_idx, "content": content, "rrf_score": 0.0}
            
        # Add 1 / (k + rank)
        rrf_map[doc_id]["rrf_score"] += 1.0 / (k + rank_idx)

    sorted_candidates = sorted(rrf_map.values(), key=lambda x: x["rrf_score"], reverse=True)
    return sorted_candidates

def run_hybrid_search(query:str):
    print(f"\n===")
    print(f"QUERY: '{query}'")
    print(f"===")

    # A. Fetch all documents for BM25
    all_rows = get_all_chunks()

    # B. Run Dense and Sparse
    dense_hits = dense_search(query, top_k=3)
    sparse_hits = sparse_search(query, all_rows, top_k=3)

    # C. Fuse ranks
    fused_results = reciprocal_rank_fusion(dense_hits, sparse_hits,  k = 60)
                                           
    print("\n--- Top Hybrid Results (Fused via RRF) ---")
    for rank, item in enumerate(fused_results[:2], start=1):
        print(f"Rank #{rank} | Chunk #{item['chunk_index']} | Combined RRF Score: {item['rrf_score']:.6f}")
        print(f"Content: \"{item['content'][:110]}...\"\n")

if __name__ == "__main__":
    # Test 1: Conceptual Query (Vector should dominate)
    run_hybrid_search("How long do we have to alert authorities after a system crash?")

    # Test 2: Specific Exact Reference (BM25 should dominate)
    run_hybrid_search("What does Section 206 state?")

