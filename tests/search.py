import psycopg
from sentence_transformers import SentenceTransformer

DATABASE_URL = "postgresql://ai_user:ai_password@127.0.0.1:5432/rag_engine"

def semantic_search(query: str, top_k: int):

    model = SentenceTransformer("all-MiniLM-L6-v2")

    query_vector = model.encode(query).tolist()

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    chunk_index,
                    content,
                    (embedding <=> %s::vector) AS cosine_distance
                FROM document_chunks
                ORDER BY cosine_distance ASC
                LIMIT %s;
            """, (query_vector, top_k))

            results = cur.fetchall()
            return results

if __name__ == "__main__":
    test_query = "What does Section 206 state?"

    print(f"\n[?] User Query: '{test_query}'\n")

    matches = semantic_search(query= test_query, top_k=2)

    for idx, (chunk_idx, content, dist) in enumerate(matches):
        similarity_score = 1 - dist
        print(f"Result #{idx + 1} (Chunk #{chunk_idx})")
        print(f"Cosine Distance:   {dist:.4f} (Lower is better)")
        print(f"Cosine Similarity: {similarity_score:.4f} (Higher is better)")
        print(f"Content:\n\"{content}\"\n")