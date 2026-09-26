import psycopg
from typing import List
from sentence_transformers import SentenceTransformer

DATABASE_URL = "postgresql://ai_user:ai_password@localhost:5432/rag_engine"

def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:

    if overlap >= chunk_size:
        raise ValueError("Overlap must be strictly smaller than chunk_size, otherwise stride is <= 0 (infinite loop)!")

    words = text.split()

    if not words:
        return []

    stride = chunk_size - overlap

    chunks = []

    for start_idx in range(0, len(words), stride):
        end_idx = chunk_size + start_idx

        chunk_words = words[start_idx:end_idx]

        chunk_text = " ".join(chunk_words)

        chunks.append(chunk_text)

        if end_idx >= len(words):
            break

    return chunks

DOC_NAME = "fca_incident_reporting_handbook.txt"

SAMPLE_TEXT = """
Under Principle 11 of the FCA Handbook, all authorized financial firms must deal with their regulators 
in an open and cooperative manner. Regulated firms are legally obligated to disclose any operational disruption 
that could compromise market integrity or consumer protection.

In the case of severe IT downtime, critical cybersecurity breaches, or core database corruption, 
the designated compliance officer must submit an official incident report to the FCA within 2 hours of discovery. 
Unjustified failure to notify within this window exposes the firm to formal enforcement action, including fines 
under Section 206 of the Financial Services and Markets Act 2000.

Furthermore, all forensic system logs, network traffic dumps, and incident response communication channels 
must be retained securely for an audit window of at least seven years.
"""

def run_pipeline():
    model = SentenceTransformer("all-MiniLM-L6-v2")

    chunks = chunk_text(SAMPLE_TEXT, chunk_size=35, overlap = 10)
    print(f"      -> Generated {len(chunks)} chunks with overlap.")

    with psycopg.connect(DATABASE_URL) as conn:

        with conn.cursor() as cur:

            for idx, chunk in enumerate(chunks):
                embedding = model.encode(chunk).tolist()

                cur.execute("""INSERT INTO document_chunks (document_name, chunk_index, content, embedding) 
                VALUES (%s, %s, %s, %s);
                """, (DOC_NAME, idx, chunk, embedding))

                print(f"      -> Inserted Chunk #{idx} ({len(chunk.split())} words)")

        conn.commit()

    print("\nSUCCESS")

if __name__ == "__main__":
    run_pipeline()

