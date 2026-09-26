from sentence_transformers import SentenceTransformer
import numpy as np

print("[*] Loading all-MiniLM-L6-v2...")
model = SentenceTransformer("all-MiniLM-L6-v2")

s1 = "The company reported a major financial loss in Q3."
s2 = "Revenue dropped significantly in the third quarter."
s3 = "A recipe for chocolate chip cookies using brown butter."

v1 = model.encode(s1)
v2 = model.encode(s2)
v3 = model.encode(s3)

print(f"\n[+] Type of output: {type(v1)}")
print(f"[+] Dimensionality: {v1.shape} (Must be exactly 384)")
print(f"[+] First 5 numbers of v1: {v1[:5]}\n")

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

sim_1_2 = cosine_sim(v1, v2)
sim_1_3 = cosine_sim(v1, v3)

print(f"Similarity between (Loss in Q3) and (Revenue dropped in Q3): {sim_1_2:.4f}")
print(f"Similarity between (Loss in Q3) and (Chocolate cookies):       {sim_1_3:.4f}")