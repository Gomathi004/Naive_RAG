import os
import pickle
import faiss
import ollama

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

OLLAMA_MODEL = "llama3.2:latest"


# =========================================
# LOAD VECTORSTORE
# =========================================

print("========================================")
print("NAIVE RAG + OLLAMA")
print("========================================")

with open(CHUNKS_PATH, "rb") as f:
    chunks = pickle.load(f)

index = faiss.read_index(INDEX_PATH)

print("Chunks loaded:", len(chunks))
print("Vectors loaded:", index.ntotal)


# =========================================
# LOAD EMBEDDING MODEL
# =========================================

print("\n========================================")
print("LOADING EMBEDDING MODEL")
print("========================================")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("Embedding model:", EMBEDDING_MODEL)


# =========================================
# LOAD RERANKER
# =========================================

print("\n========================================")
print("LOADING RERANKER MODEL")
print("========================================")

reranker = CrossEncoder(
    RERANKER_MODEL
)

print("Reranker model:", RERANKER_MODEL)


# =========================================
# ASK QUESTION
# =========================================

question = input("\nAsk your question: ")


# =========================================
# QUERY EMBEDDING
# =========================================

print("\n========================================")
print("QUERY EMBEDDING")
print("========================================")

query_vector = embedding_model.encode(
    [question],
    convert_to_numpy=True
)

print("Query vector shape:", query_vector.shape)


# =========================================
# RETRIEVAL
# =========================================

print("\n========================================")
print("RETRIEVING CHUNKS")
print("========================================")

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
    print("Chunk:")
    print(chunk)


# =========================================
# RERANKING
# =========================================

print("\n========================================")
print("RERANKING")
print("========================================")

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
# TOP RERANKED CHUNKS
# =========================================

print("\n========================================")
print("TOP RERANKED CHUNKS")
print("========================================")

TOP_N = min(3, len(reranked_results))

for i, (score, chunk) in enumerate(
    reranked_results[:TOP_N],
    start=1
):

    print(f"\n--- Context {i} ---")
    print("Reranker score:", score)
    print("Chunk:")
    print(chunk)


# =========================================
# PREPARE CONTEXT
# =========================================

context = "\n\n".join(
    chunk
    for score, chunk in reranked_results[:TOP_N]
)


# =========================================
# GENERATE WITH OLLAMA
# =========================================

print("\n========================================")
print("GENERATING ANSWER WITH OLLAMA")
print("========================================")

prompt = f"""
You are a RAG assistant.

Answer the user's question using ONLY the information
provided in the context below.

Do not use outside knowledge.

If the answer cannot be found in the context, say:
"I could not find the answer in the provided document."

Give a clear and concise answer.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""


response = ollama.chat(
    model=OLLAMA_MODEL,
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

answer = response["message"]["content"]


# =========================================
# FINAL ANSWER
# =========================================

print("\n========================================")
print("RAG ANSWER")
print("========================================")

print(answer)


# =========================================
# SOURCES
# =========================================

print("\n========================================")
print("SOURCES USED")
print("========================================")

for i, (score, chunk) in enumerate(
    reranked_results[:TOP_N],
    start=1
):

    print(f"\n--- Source {i} ---")
    print("Reranker score:", score)
    print(chunk)