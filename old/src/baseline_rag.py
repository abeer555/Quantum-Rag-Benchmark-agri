import json
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer

# Paths
DATA_FILE = Path("data/cleaned.jsonl")
COLLECTION_NAME = "agri_rag"

# 1. Load chunks
def load_chunks(file_path):
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records

# 2. Init embeddings + Qdrant
def init_pipeline():
    print("🔄 Loading embedding model...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")  # fast + ~384-dim
    
    print("🔄 Connecting to Qdrant (in-memory)...")
    client = QdrantClient(":memory:")  # runs in RAM (use "localhost" if you run a Qdrant server)
    
    return embedder, client

# 3. Index corpus
def index_chunks(records, embedder, client):
    dim = embedder.get_sentence_embedding_dimension()
    
    # Create collection
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    
    # Build embeddings + insert
    points = []
    for i, r in enumerate(records):
        vec = embedder.encode(r["text"]).tolist()
        points.append(PointStruct(id=i, vector=vec, payload=r))
    
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"✅ Indexed {len(records)} chunks into {COLLECTION_NAME}")

# 4. Search
def search(query, embedder, client, top_k=5):
    qvec = embedder.encode(query).tolist()
    hits = client.search(collection_name=COLLECTION_NAME, query_vector=qvec, limit=top_k)
    
    print(f"\n🔍 Query: {query}\n")
    for h in hits:
        print(f"Score: {h.score:.3f} | ID: {h.payload['id']}")
        print(h.payload["text"][:300] + "...\n")

# --- Main flow ---
if __name__ == "__main__":
    records = load_chunks(DATA_FILE)
    embedder, client = init_pipeline()
    index_chunks(records, embedder, client)
    
    # Example queries
    queries = [
        "global wheat production trends",
        "soil management in rice farming",
        "impact of climate change on maize yield"
    ]
    
    for q in queries:
        search(q, embedder, client)
