# from typing import List

# def sliding_window_chunker(text: str, chunk_size: int, overlap: int) -> List[str]:

#     if overlap >= chunk_size:
#         raise ValueError("Overlap must be strictly smaller than chunk_size, otherwise stride is <= 0 (infinite loop)!")

#     words = text.split()

#     if not words:
#         return []

#     chunks = []

#     stride = chunk_size - overlap

#     for start_idx in range(0, len(words), stride):
#         end_idx = start_idx + chunk_size

#         chunk_words = words[start_idx:end_idx]

#         chunk_text = " ".join(chunk_words)
#         chunks.append(chunk_text)

#         if end_idx >= len(words):
#             break

#     return chunks

# if __name__ == "__main__":
#     sample_text = (
#         "Under Principle 11, firms must deal with their regulators in an open and cooperative way, "
#         "and must disclose to the FCA appropriately anything relating to the firm of which the "
#         "FCA would reasonably expect notice."
#     )

#     result_chunks = sliding_window_chunker(sample_text, chunk_size=8, overlap=2)

#     print(f"Total words in text: {len(sample_text.split())}")
#     print(f"Total chunks created: {len(result_chunks)}\n")

#     for idx, chunk in enumerate(result_chunks):
#         print(f"--- Chunk #{idx} ---")
#         print(f"'{chunk}'\n")

# Updated Text Chunker to sentence-based

import re

def sentence_aware_chunker(text: str, max_words: int=50, sentence_overlap: int=1) -> list[str]:
    # Split text cleanly on sentence boundaries while keeping sentences intact
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]
    if not sentences:
        return []

    #[s1, s2, s3, s4, s5, s6]
    chunks = []
    i = 0
    while i < len(sentences):
        current_chunk = []
        current_word_count = 0
        for j in range(i, len(sentences)):
            sent = sentences[j]
            sent_word_count = len(sent)

            if current_word_count + sent_word_count > max_words and current_chunk:
                break

            current_chunk.append(sent)
            current_word_count += sent_word_count

        chunks.append(" ".join(current_chunk))

        stride = max(1, len(current_chunk) - sentence_overlap)
        i += stride

        if i + sentence_overlap >= len(sentences) and j == len(sentences):
            break

    return chunks
