import psycopg
import re
from rank_bm25 import BM25Okapi

DATABASE_URL = "postgresql://ai_user:ai_password@127.0.0.1:5432/rag_engine"

def simple_tokeniser(text: str) -> list[str]:
    """
    Converts text to lowercase and extracts alphanumeric tokens.
    Removes periods, commas, and special symbols cleanly.
    """
    return re.findall(r'\w+', text.lower())

def run_bm25_test(query: str):
    # 1. Fetch all chunks currently in our PostgreSQL database
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT chunk_index, content
                FROM document_chunks
                ORDER BY chunk_index ASC
            """)
            rows = cur.fetchall()

    if not rows:
        return "[-] No chunks found in database! Run ingest.py first."

    chunk_indices, chunk_content = zip(*rows)

    # 2. Tokenize our corpus (BM25 expects a list of list of tokens: [["word1", "word2"], ...])
    tokenised_corpus = [simple_tokeniser(doc) for doc in chunk_content]
    
    # 3. Initialize the BM25Okapi model with the corpus
    bm25 = BM25Okapi(tokenised_corpus)

    # 4. Tokenize the incoming user query
    tokenised_query = simple_tokeniser(query)
    print(f"\n[?] Query: '{query}'")
    print(f"[*] Tokenized Query: {tokenised_query}\n")

    # 5. Compute raw BM25 relevance scores for all documents
    doc_scores = bm25.get_scores(tokenised_query)

    # 6. Pair chunks with their scores and sort descending (highest score first)
    ranked_results = sorted(
        zip(chunk_indices, chunk_content, doc_scores),
        key=lambda x: x[2],
        reverse=True
    )

    print("--- BM25 Scoring Table ---")
    for rank, (idx, content, score) in enumerate(ranked_results, start=1):
        print(f"Rank #{rank} | Chunk #{idx} | BM25 Score: {score:.4f}")
        print(f"Content: \"{content[:120]}...\"\n")

if __name__ == "__main__":
    run_bm25_test("What does Section 206 state?")