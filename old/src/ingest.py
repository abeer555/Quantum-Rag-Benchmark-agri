import json
import time
import csv
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Paths
DATA_FILE = Path("data/cleaned.jsonl")
COLLECTION_NAME = "agri_rag"
LOG_FILE = Path("data/query_logs.csv")

# Init embedding + DB
def init_pipeline():
    print("🔄 Loading embedding model...")
    embedder = SentenceTransformer("intfloat/multilingual-e5-base")  # strong embeddings

    print("🔄 Connecting to Qdrant (in-memory)...")
    client = QdrantClient(":memory:")

    print("🔄 Loading local text-generation model...")
    generator = pipeline("text2text-generation", model="google/flan-t5-base")

    return embedder, client, generator

# Load chunks
def load_chunks(file_path):
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records

# Index corpus
def index_chunks(records, embedder, client):
    dim = embedder.get_sentence_embedding_dimension()

    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    points = []
    for i, r in enumerate(records):
        vec = embedder.encode(r["text"]).tolist()
        points.append(PointStruct(id=i, vector=vec, payload=r))

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"✅ Indexed {len(records)} chunks into {COLLECTION_NAME}")

# Search + timing
def search(query, embedder, client, top_k=5):
    t0 = time.time()
    qvec = embedder.encode(query).tolist()
    t1 = time.time()
    hits = client.search(collection_name=COLLECTION_NAME, query_vector=qvec, limit=top_k)
    t2 = time.time()

    embedding_time = t1 - t0
    retrieval_time = t2 - t1
    total_time = t2 - t0

    contexts = [h.payload["text"] for h in hits]
    return contexts, embedding_time, retrieval_time, total_time

# Generate answer (local HuggingFace model)
def generate_answer(query, contexts, generator):
    context = "\n\n".join(contexts)
    prompt = f"Question: {query}\n\nContext:\n{context}\n\nAnswer clearly:"
    result = generator(prompt, max_length=256, do_sample=False)
    return result[0]["generated_text"]

# Log results
def log_results(query, answer, embedding_time, retrieval_time, total_time):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    header = ["query", "embedding_time", "retrieval_time", "total_time", "answer"]

    new_file = not LOG_FILE.exists()
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(header)
        writer.writerow([query, f"{embedding_time:.4f}", f"{retrieval_time:.4f}",
                         f"{total_time:.4f}", answer[:200]])

if __name__ == "__main__":
    records = load_chunks(DATA_FILE)
    embedder, client, generator = init_pipeline()
    index_chunks(records, embedder, client)

    print("\n🤖 Ask me something about agriculture (type 'exit' to quit)\n")
    while True:
        query = input("❓ Your question: ")
        if query.lower() in ["exit", "quit"]:
            break

        contexts, et, rt, tt = search(query, embedder, client, top_k=5)
        answer = generate_answer(query, contexts, generator)

        print("\n--- Answer ---")
        print(answer)
        print(f"\n⏱ Embedding: {et:.4f}s | Retrieval: {rt:.4f}s | Total: {tt:.4f}s\n")

        log_results(query, answer, et, rt, tt)
