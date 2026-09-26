import psycopg

DATABASE_URL = "postgresql://ai_user:ai_password@127.0.0.1:5432/rag_engine"

def init_database():
    with psycopg.connect(DATABASE_URL) as conn:

        with conn.cursor() as cur:

            print("[1/4] Enabling vector extension...")
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            print("[2/4] Dropping old table if present...")
            cur.execute("DROP TABLE IF EXISTS document_chunks;")

            print("[3/4] Creating document_chunks table...")
            cur.execute("""
                CREATE TABLE document_chunks(
                    id serial PRIMARY KEY,
                    document_name TEXT NOT NULL,
                    chunk_index INT NOT NULL,
                    content TEXT NOT NULL,
                    embedding vector(384),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            """)

            print("[4/4] Building HNSW graph index on embedding column...")
            cur.execute("""
                CREATE INDEX IF NOT EXISTS document_chunks_embedding_hnsw_idx 
                ON document_chunks
                USING hnsw(embedding vector_cosine_ops)
                WITH (m=16, ef_construction=64)
            """)

            conn.commit()
            print("\n[SUCCESS] Database schema and vector index initialized!")

if __name__ == "__main__":
    init_database()