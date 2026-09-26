import re
import psycopg
from typing import List
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

# Database URI
DATABASE_URL = "postgresql://ai_user:ai_password@127.0.0.1:5432/rag_engine"

print("** Loading Bi-encoder (all-MiniLM-L6-v2)..**")
bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")

print("** Loading cross-encoder (ms-marco-MiniLM-L-6-v2)...**")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

SAMPLE_DOCUMENT = """
Under Principle 11 of the FCA Handbook, all authorized financial firms must deal with their regulators in an open and cooperative manner. Regulated firms are legally obligated to disclose any operational disruption that could compromise market integrity or consumer protection.

In the case of severe IT downtime, critical cybersecurity breaches, or core database corruption, the designated compliance officer must submit an official incident report to the FCA within 2 hours of discovery. Unjustified failure to notify within this window exposes the firm to formal enforcement action, including fines under Section 206 of the Financial Services and Markets Act 2000.

Furthermore, all forensic system logs, network traffic dumps, and incident response communication channels must be retained securely for an audit window of at least seven years.
"""

# 1. Sentence Boundary Chunking
def sentence_aware_chunker(text: str, max_words: int=50, sentence_overlap: int=1) -> list[str]:
    # Split text cleanly on sentence boundaries while keeping sentences intact
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]
    if not sentences:
        return []

    chunks = []
    i = 0
    while i < len(sentences):
        current_chunk = []
        current_word_count = 0
        for j in range(i, len(sentences)):
            sent = sentences[j]
            sent_word_count = len(sent.split())

            if current_word_count + sent_word_count > max_words and current_chunk:
                break

            current_chunk.append(sent)
            current_word_count += sent_word_count

        chunks.append(" ".join(current_chunk))

        # Advance the index, sliding backward by the sentence overlap
        stride = max(1, len(current_chunk) - sentence_overlap)
        i += stride

        # If the last sentence was included, finish
        if i + sentence_overlap >= len(sentences) and j == len(sentences) - 1:
            break

    return chunks

def reingest_clean_data():
    """Wipes table and inserts sentence-boundary chunks."""
    print("*** Re-ingesting document with sentence-aware boundaries...***")
    chunks = sentence_aware_chunker(SAMPLE_DOCUMENT, max_words=45, sentence_overlap=1)

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE document_chunks;")
            for idx, chunk in enumerate(chunks):
                emb = bi_encoder.encode(chunk).tolist()
                cur.execute("""
                INSERT INTO document_chunks(document_name, chunk_index, content, embedding)
                VALUES (%s, %s, %s, %s);
                """, ("fca.handbook.txt", idx, chunk, emb))
            conn.commit()
        print(f"Re-ingested {len(chunks)} high-coherence chunks.")

# --- 2. Hybrid Retrieval Components ---
def simple_tokeniser(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())

def fetch_all_db_chunks():
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("""
            SELECT id, chunk_index, content
            FROM document_chunks
            ORDER BY chunk_index ASC;
            """)
            return cur.fetchall()

def dense_retreival(query: str, top_k: int):
    query_vector = bi_encoder.encode(query).tolist()
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("""
            SELECT id, chunk_index, content, (embedding <=> %s::vector) AS cosine_distance
            FROM document_chunks
            ORDER BY cosine_distance ASC
            LIMIT %s
            """, (query_vector, top_k))
            return cur.fetchall()

def sparse_retreival(query: str, corpus: list, top_k: int):
    tokenised_corpus = [simple_tokeniser(row[2]) for row in corpus]
    bm25 = BM25Okapi(tokenised_corpus)
    scores = bm25.get_scores(simple_tokeniser(query))
    ranked = sorted(zip(corpus, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]

def reciprocal_rank_fusion(dense_hits: list, sparse_hits: list, k: int = 60) -> list[str]:
    rrf_map={}
    for rank, (doc_id, chunk_idx, content, _) in enumerate(dense_hits, start=1):
        if doc_id not in rrf_map:
            rrf_map[doc_id] = {"chunk_index": chunk_idx, "content": content, "rrf_score":0.0}
        rrf_map[doc_id]["rrf_score"] += 1.0 / (rank + k)

    for rank, ((doc_id, chunk_idx, content), score) in enumerate(sparse_hits, start=1):
        if score <= 0.0:
            continue
        if doc_id not in rrf_map:
            rrf_map[doc_id] = {"chunk_index": chunk_idx, "content": content, "rrf_score":0.0}
        rrf_map[doc_id]["rrf_score"] += 1.0 / (rank + k)

    return sorted(rrf_map.values(), key=lambda x: x["rrf_score"], reverse= True)

# --- 3. Stage-2 Cross-Encoder Re-Ranking ---

def rerank_candidates(query: str, candidates: list, top_k: int = 2) -> list:
    if not candidates:
        return []

    pairs = [[query, c["content"]] for c in candidates]
    scores = cross_encoder.predict(pairs)

    for c, score in zip(candidates, scores):
        c["rerank_score"] = float(score)

    return sorted(candidates, key=lambda x: x["rerank_score"], reverse= True)[:top_k]

# --- 4. End-to-End Orchestrator ---
def run_retrieval_pipeline(query: str) -> dict:
    corpus = fetch_all_db_chunks()
    dense_hits = dense_retreival(query, top_k = 4)
    sparse_hits = sparse_retreival(query, corpus, top_k = 4)
    fused = reciprocal_rank_fusion(dense_hits, sparse_hits, k = 60)
    top_ranked = rerank_candidates(query, fused, top_k = 2)
    return top_ranked

if __name__ == "__main__":
    reingest_clean_data()

    q1 = "How long do we have to alert authorities after a system crash?"
    print(f"QUERY: '{q1}'")
    results = run_retrieval_pipeline(q1)

    for idx, r in enumerate(results, start=1):
        print(f"\n[Rank #{idx}] chunk #{r['chunk_index']} | Cross Encoder logit: {r['rerank_score']:.4f}")
        print(f"Content: #{r['content']}")