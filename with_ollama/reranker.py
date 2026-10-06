import os
import pickle
import faiss

from sentence_transformers import SentenceTransformer, CrossEncoder


# ============================================================
# 1. PROJECT PATH
# ============================================================

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


# ============================================================
# 2. MODEL CONFIGURATION
# ============================================================

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ============================================================
# 3. CHECK FILES
# ============================================================

print("\n========================================")
print("NAIVE RAG - RERANKING")
print("========================================")

print("\nProject directory:")
print(BASE_DIR)

print("\nVectorstore directory:")
print(VECTORSTORE_DIR)

print("\nChunks file:")
print(CHUNKS_PATH)

print("\nIndex file:")
print(INDEX_PATH)


if not os.path.exists(CHUNKS_PATH):

    print("\nERROR: chunks.pkl does not exist!")

    exit()


if not os.path.exists(INDEX_PATH):

    print("\nERROR: vectors.index does not exist!")

    exit()


# ============================================================
# 4. LOAD CHUNKS
# ============================================================

print("\n========================================")
print("LOADING CHUNKS")
print("========================================")

with open(CHUNKS_PATH, "rb") as file:

    chunks = pickle.load(file)


print("Chunks loaded:", len(chunks))


# ============================================================
# 5. LOAD FAISS INDEX
# ============================================================

print("\n========================================")
print("LOADING FAISS INDEX")
print("========================================")

index = faiss.read_index(
    INDEX_PATH
)

print("Vectors loaded:", index.ntotal)


# ============================================================
# 6. VERIFY
# ============================================================

print("\n========================================")
print("VERIFYING VECTOR DATABASE")
print("========================================")


if len(chunks) != index.ntotal:

    print("\nERROR!")

    print("Chunks:", len(chunks))
    print("Vectors:", index.ntotal)

    print("\nChunks and vectors do not match.")

    exit()


print("Vector database is consistent.")


# ============================================================
# 7. SHOW FIRST CHUNK
# ============================================================

print("\n========================================")
print("FIRST CHUNK")
print("========================================")

print(chunks[0][:1000])


# ============================================================
# 8. LOAD EMBEDDING MODEL
# ============================================================

print("\n========================================")
print("LOADING EMBEDDING MODEL")
print("========================================")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("Embedding model:", EMBEDDING_MODEL)


# ============================================================
# 9. LOAD RERANKER
# ============================================================

print("\n========================================")
print("LOADING RERANKER MODEL")
print("========================================")

reranker = CrossEncoder(
    RERANKER_MODEL
)

print("Reranker model:", RERANKER_MODEL)


# ============================================================
# 10. USER QUESTION
# ============================================================

question = input(
    "\nAsk your question: "
)


# ============================================================
# 11. QUERY EMBEDDING
# ============================================================

print("\n========================================")
print("QUERY EMBEDDING")
print("========================================")

query_vector = embedding_model.encode(
    [question],
    convert_to_numpy=True
)

print("Query vector shape:")
print(query_vector.shape)


# ============================================================
# 12. RETRIEVAL
# ============================================================

print("\n========================================")
print("RETRIEVING CHUNKS")
print("========================================")

TOP_K = min(
    5,
    len(chunks)
)


distances, indices = index.search(
    query_vector,
    TOP_K
)


retrieved_chunks = []


for rank, (distance, index_number) in enumerate(
    zip(distances[0], indices[0]),
    start=1
):

    if index_number < 0:
        continue

    chunk = chunks[index_number]

    retrieved_chunks.append(chunk)

    print(
        f"\n--- Retrieved {rank} ---"
    )

    print(
        "FAISS distance:",
        distance
    )

    print("\nChunk:")

    print(chunk)


# ============================================================
# 13. PREPARE RERANKING PAIRS
# ============================================================

print("\n========================================")
print("PREPARING RERANKING")
print("========================================")

pairs = []

for chunk in retrieved_chunks:

    pairs.append(
        [
            question,
            chunk
        ]
    )


print(
    "Pairs sent to reranker:",
    len(pairs)
)


# ============================================================
# 14. RERANK
# ============================================================

scores = reranker.predict(
    pairs
)


reranked_results = list(
    zip(
        retrieved_chunks,
        scores
    )
)


# Highest score first

reranked_results.sort(
    key=lambda item: item[1],
    reverse=True
)


# ============================================================
# 15. DISPLAY RERANKED RESULTS
# ============================================================

print("\n========================================")
print("TOP RERANKED CHUNKS")
print("========================================")


TOP_N = min(
    3,
    len(reranked_results)
)


for rank, (chunk, score) in enumerate(
    reranked_results[:TOP_N],
    start=1
):

    print(
        f"\n--- Context {rank} ---"
    )

    print(
        "Reranker score:",
        score
    )

    print("\nChunk:")

    print(chunk)


# ============================================================
# 16. DONE
# ============================================================

print("\n========================================")
print("RERANKING COMPLETED")
print("========================================")