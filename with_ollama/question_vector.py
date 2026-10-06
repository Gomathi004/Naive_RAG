import pickle
import faiss
from sentence_transformers import SentenceTransformer


# ==========================================
# 1. LOAD STORED DATA
# ==========================================

print("\nLoading vector database...")

with open(
    "vectorstore/chunks.pkl",
    "rb"
) as file:
    chunks = pickle.load(file)


index = faiss.read_index(
    "vectorstore/vectors.index"
)

print("Chunks loaded:", len(chunks))
print("Vectors loaded:", index.ntotal)


# ==========================================
# 2. LOAD EMBEDDING MODEL
# ==========================================

print("\nLoading embedding model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ==========================================
# 3. QUERY
# ==========================================

query = input(
    "\nEnter your question: "
)


# ==========================================
# 4. QUERY PROCESSING
# Query → Embedding
# ==========================================

print("\nConverting question into vector...")

query_vector = model.encode(
    [query],
    convert_to_numpy=True
)

query_vector = query_vector.astype(
    "float32"
)


print(
    "Query vector shape:",
    query_vector.shape
)


# ==========================================
# 5. RETRIEVE
# ==========================================

top_k = min(5, len(chunks))

distances, indices = index.search(
    query_vector,
    top_k
)


# ==========================================
# 6. DISPLAY RESULTS
# ==========================================

print("\n================================")
print("RETRIEVED CHUNKS")
print("================================")

for rank, (idx, distance) in enumerate(
    zip(indices[0], distances[0]),
    start=1
):

    print(f"\n--- Result {rank} ---")

    print("FAISS distance:", distance)

    print("\nChunk:")

    print(chunks[idx])