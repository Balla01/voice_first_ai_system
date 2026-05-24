from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, HnswConfigDiff

# ── STEP 1: Load embedding model ──
model = SentenceTransformer("Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True)
model[0].auto_model.config.unpad_inputs = False  # fixes corrupt position_ids bug

# ── STEP 2: Read data.txt ──
with open(r"C:\projects\audio_transition\main\data.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# ── STEP 3: Chunk text ──
chunks = [chunk.strip() for chunk in raw_text.split("\n\n") if chunk.strip()]
print(f"Total chunks: {len(chunks)}")

# ── STEP 4: Generate embeddings ──
embeddings = model.encode(chunks, normalize_embeddings=True, show_progress_bar=True).tolist()

# ── STEP 5: Connect to Qdrant (local folder, no Docker) ──
client = QdrantClient(path=r"C:\projects\audio_transition\main\embed")

# ── STEP 6: Create collection ──
client.delete_collection("insurance_docs")
client.create_collection(
    collection_name="insurance_docs",
    vectors_config=VectorParams(
        size=1024,
        distance=Distance.COSINE
    ),
    hnsw_config=HnswConfigDiff(
        m=16,
        ef_construct=100
    )
)

# ── STEP 7: Insert into Qdrant ──
client.upsert(
    collection_name="insurance_docs",
    points=[
        PointStruct(
            id=i,
            vector=emb,
            payload={"text": chunk}
        )
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
    ]
)

print(f"✅ Inserted {len(chunks)} chunks into Qdrant")
client.close()
#To retrieve later (same local path):
"""
client = QdrantClient(path=r"C:\projects\audio_transition\main\embed")

query_vector = model.encode(["customer needs life insurance"], normalize_embeddings=True).tolist()[0]

results = client.query_points(
    collection_name="insurance_docs",
    query=query_vector,
    limit=3
)

for r in results.points:
    print(f"Score: {r.score:.3f} | {r.payload['text']}")
"""