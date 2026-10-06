import os
import pickle
import faiss

from sentence_transformers import SentenceTransformer, CrossEncoder


# =========================================
# PATHS
# =========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VECTORSTORE_DIR = os.path.join(
    BASE_DIR,
    "vectorstore"
)

CHUNKS_PATH = os.path.join(
    VECTORSTORE_DIR,
    "chunks.pkl"
)

INDEX_PATH = os.path.join(
    VECTORSTORE_DIR,
    "vectors.index"
)


# =========================================
# MODELS
# =========================================

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# =========================================
# LOAD DATA
# =========================================

with open(CHUNKS_PATH, "rb") as f:
    chunks = pickle.load(f)

index = faiss.read_index(INDEX_PATH)


print("================================")
print("NAIVE RAG")
print("================================")

print("Chunks loaded:", len(chunks))
print("Vectors loaded:", index.ntotal)


# =========================================
# LOAD MODELS
# =========================================

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("Embedding model loaded.")


print("\nLoading reranker model...")

reranker = CrossEncoder(
    RERANKER_MODEL
)

print("Reranker model loaded.")


# =========================================
# ASK QUESTION
# =========================================

question = input("\nAsk your question: ")


# =========================================
# QUERY EMBEDDING
# =========================================

print("\n================================")
print("QUERY EMBEDDING")
print("================================")

query_vector = embedding_model.encode(
    [question],
    convert_to_numpy=True
)

print("Query vector shape:", query_vector.shape)


# =========================================
# RETRIEVAL
# =========================================

print("\n================================")
print("RETRIEVING CHUNKS")
print("================================")

TOP_K = min(5, len(chunks))

distances, indices = index.search(
    query_vector.astype("float32"),
    TOP_K
)

retrieved_chunks = []

for rank, idx in enumerate(indices[0]):

    chunk = chunks[idx]

    retrieved_chunks.append(chunk)

    print(f"\n--- Retrieved {rank + 1} ---")
    print("FAISS distance:", distances[0][rank])
    print("\nChunk:")
    print(chunk)


# =========================================
# RERANKING
# =========================================

print("\n================================")
print("RERANKING")
print("================================")

pairs = [
    [question, chunk]
    for chunk in retrieved_chunks
]

scores = reranker.predict(pairs)

reranked_results = list(
    zip(scores, retrieved_chunks)
)

reranked_results.sort(
    key=lambda x: x[0],
    reverse=True
)


# =========================================
# TOP RERANKED RESULTS
# =========================================

print("\n================================")
print("TOP RERANKED CHUNKS")
print("================================")

TOP_N = min(3, len(reranked_results))

for i in range(TOP_N):

    score, chunk = reranked_results[i]

    print(f"\n--- Context {i + 1} ---")
    print("Reranker score:", score)
    print("\nChunk:")
    print(chunk)


# =========================================
# FINAL ANSWER
# =========================================

print("\n================================")
print("RAG ANSWER")
print("================================")

best_score, best_context = reranked_results[0]

print(best_context)


# =========================================
# SOURCE
# =========================================

print("\n================================")
print("SOURCE USED")
print("================================")

print("Reranker score:", best_score)

print(best_context)