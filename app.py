import os
import psycopg
import re
from typing import List, Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# 1. Initialize FastAPI Application
app = FastAPI(
    title="Hybrid RAG Engine (pgvector + BM25 + Cross-Encoder)",
    description="Two-stage hybrid retrieval engine with sentence-boundary chunking, RRF fusion, and Cross-Encoder re-ranking.",
    version="1.0.0"
)

ai_client = genai.Client()

# 2. Load Models into Memory Once at Startup
print("[*] Loading bi-encoder (all-MiniLM-L6-v2)...")
bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")

print("[*] Loading cross-encoder (ms-marco-MiniLM-L-6-v2)...")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# 3. Pydantic Request & Response Schemas

# what LLM must return to the user
class GroundedAnswer(BaseModel):
    synthenic_answer: str = Field(
        description="A concise, professional answer derived strictly from the provided context."
    ),
    direct_quote: str = Field(
        description="The exact verbatim quote from the text that proves the answer."
    ),
    regulatory_risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = Field(
        description="Risk level associated with non-compliance based on context."
    )

class SearchRequest(BaseModel):
    query: str = Field(
        ..., 
        min_length=3, 
        json_schema_extra={"example": "How long do we have to alert authorities after a system crash?"}
    )
class SearchResponse(BaseModel):
    query: str
    grounded_answer: GroundedAnswer
    cited_chunk_index: int
    rerank_score: float

# 4. Retrieval Helper Functions
def simple_tokenizer(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())

def fetch_all_db_chunks():
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, chunk_index, content FROM document_chunks ORDER BY chunk_index ASC;")
            return cur.fetchall()

def dense_retrieval(query: str, top_k: int = 4):
    query_vector = bi_encoder.encode(query).tolist()
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, chunk_index, content, (embedding <=> %s::vector) AS distance
                FROM document_chunks
                ORDER BY distance ASC
                LIMIT %s;
            """, (query_vector, top_k))
            return cur.fetchall()

def sparse_retrieval(query: str, corpus: list, top_k: int = 4):
    tokenized_corpus = [simple_tokenizer(row[2]) for row in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(simple_tokenizer(query))
    ranked = sorted(zip(corpus, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]

def rrf_fusion(dense_hits: list, sparse_hits: list, k: int = 60) -> list:
    rrf_scores = {}
    for rank, (doc_id, c_idx, content, _) in enumerate(dense_hits, start=1):
        rrf_scores[doc_id] = {"chunk_index": c_idx, "content": content, "score": 1.0 / (k + rank)}

    for rank, ((doc_id, c_idx, content), score) in enumerate(sparse_hits, start=1):
        if score <= 0:
            continue
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = {"chunk_index": c_idx, "content": content, "score": 0.0}
        rrf_scores[doc_id]["score"] += 1.0 / (k + rank)

    return sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)

def rerank(query: str, candidates: list, top_n: int = 1) -> list:
    if not candidates:
        return []
    pairs = [[query, c["content"]] for c in candidates]
    scores = cross_encoder.predict(pairs)
    for c, score in zip(candidates, scores):
        c["rerank_score"] = float(score)
    return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_n]

# 5. LLM Generation Layer (Gemini + Pydantic Structured Output)
def generate_grounded_response(query: str, context: str) -> GroundedAnswer:
    prompt = f"""
You are a regulatory compliance engine. Answer the user query using ONLY the provided source context.
Do not infer, assume, or extrapolate facts outside this text.

[SOURCE CONTEXT]:
{context}

[USER QUERY]:
{query}
"""
    try:
        response = ai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=GroundedAnswer,
                temperature=0.0,  # Zero temperature for deterministic grounding
            ),
        )
        # Parse output into our Pydantic model
        return GroundedAnswer.model_validate_json(response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {str(e)}")

# 6. Production Endpoint
@app.post("/api/v1/query", response_model=SearchResponse)
def query_rag(request: SearchRequest):
    corpus = fetch_all_db_chunks()
    if not corpus:
        raise HTTPException(status_code=500, detail="Database contains no indexed documents.")

    # Stage 1: Coarse Hybrid Retrieval
    dense_hits = dense_retrieval(request.query, top_k=4)
    sparse_hits = sparse_retrieval(request.query, corpus, top_k=4)
    fused_candidates = rrf_fusion(dense_hits, sparse_hits, k=60)

    # Stage 2: Precision Cross-Encoder Re-Ranking
    top_matches = rerank(request.query, fused_candidates, top_n=1)
    if not top_matches:
        raise HTTPException(status_code=404, detail="No relevant context found.")

    best_match = top_matches[0]
    
    # Stage 3: Grounded Synthesis with Gemini
    llm_answer = generate_grounded_response(
        query=request.query,
        context=best_match["content"]
    )

    return SearchResponse(
        query=request.query,
        grounded_answer=llm_answer,
        cited_chunk_index=best_match["chunk_index"],
        rerank_score=round(best_match["rerank_score"], 4),
    )