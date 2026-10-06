import os
import re
import pickle
import pymupdf
import faiss
from sentence_transformers import SentenceTransformer


# =========================================
# PATHS
# =========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PDF_PATH = os.path.join(
    BASE_DIR,
    "data",
    "knowledge.pdf"
)

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
# 1. INGEST PDF
# =========================================

print("\n[1] INGESTING PDF...")

doc = pymupdf.open(PDF_PATH)

full_text = ""

for page in doc:
    full_text += page.get_text() + "\n"

doc.close()

full_text = re.sub(r"\s+", " ", full_text).strip()

print("PDF text extracted successfully.")
print("Total characters:", len(full_text))


# =========================================
# 2. CREATE SECTION-BASED CHUNKS
# =========================================

print("\n[2] CREATING CHUNKS...")

# Find numbered sections such as:
# 1. Ingestion
# 2. Chunking
# 3. Embeddings

matches = list(
    re.finditer(
        r"\b\d+\.\s+[A-Za-z][A-Za-z ]*",
        full_text
    )
)

chunks = []

# Content before the first numbered section
if matches:
    intro = full_text[:matches[0].start()].strip()

    if intro:
        chunks.append(intro)

# Create one chunk for each numbered section
for i, match in enumerate(matches):

    start = match.start()

    if i + 1 < len(matches):
        end = matches[i + 1].start()
    else:
        end = len(full_text)

    chunk = full_text[start:end].strip()

    if chunk:
        chunks.append(chunk)


print("Total chunks:", len(chunks))


# =========================================
# CHUNK PREVIEW
# =========================================

print("\n========== CHUNK PREVIEW ==========")

for i, chunk in enumerate(chunks):

    print(f"\n--- Chunk {i + 1} ---")
    print(chunk[:500])


# =========================================
# 3. CREATE EMBEDDINGS
# =========================================

print("\n[3] CREATING EMBEDDINGS...")

MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)

embeddings = model.encode(
    chunks,
    convert_to_numpy=True
)

print("Embeddings created.")
print("Embedding shape:", embeddings.shape)


# =========================================
# 4. STORE VECTORS
# =========================================

print("\n[4] STORING VECTORS...")

os.makedirs(VECTORSTORE_DIR, exist_ok=True)

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings.astype("float32"))

with open(CHUNKS_PATH, "wb") as f:
    pickle.dump(chunks, f)

faiss.write_index(index, INDEX_PATH)


# =========================================
# COMPLETE
# =========================================

print("\n================================")
print("NAIVE RAG INGESTION COMPLETED")
print("================================")

print("Chunks:", len(chunks))
print("Vector dimension:", dimension)

print("\nSaved files:")
print("→ vectorstore/chunks.pkl")
print("→ vectorstore/vectors.index")