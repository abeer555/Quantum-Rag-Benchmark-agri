import time
import torch
import json
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

# Quantum imports
import pennylane as qml
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Handle missing dependencies gracefully
try:
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct, VectorParams, Distance
    DEPS_AVAILABLE = True
except ImportError:
    print("Warning: Some dependencies not available. Please install requirements.txt")
    DEPS_AVAILABLE = False

try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    print("Warning: PennyLane not available. Quantum features disabled.")
    PENNYLANE_AVAILABLE = False

try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    print("Warning: scikit-learn not available. Using numpy for similarity.")
    SKLEARN_AVAILABLE = False
    
    def cosine_similarity(a, b):
        """Fallback cosine similarity implementation."""
        a, b = np.array(a), np.array(b)
        return np.dot(a, b.T) / (np.linalg.norm(a) * np.linalg.norm(b))

# ================= CONFIG =================
COLLECTION_NAME = "agri_rag"
N_QUBITS = 4
USE_IBM = False
IBM_BACKEND = "ibmq_qasm_simulator"

# Embedding model
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Qdrant client
client = QdrantClient(":memory:")

# Generator model (Flan-T5)
device = "cpu"
gen_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
gen_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to(device)

# ============ Quantum Setup ==============
if USE_IBM:
    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService()
    dev = qml.device("qiskit.ibmq", wires=N_QUBITS, backend=IBM_BACKEND)
else:
    dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev)
def quantum_kernel(x1, x2):
    qml.templates.AngleEmbedding(x1, wires=range(N_QUBITS))
    qml.adjoint(qml.templates.AngleEmbedding)(x2, wires=range(N_QUBITS))
    return qml.probs(wires=range(N_QUBITS))

def quantum_similarity(v1, v2):
    v1 = v1[:N_QUBITS]
    v2 = v2[:N_QUBITS]
    probs = quantum_kernel(v1, v2)
    return probs[0]

# ============ INDEXING ==============
def build_index():
    print("📥 Indexing corpus into Qdrant...")
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    with open("data/cleaned.jsonl", "r", encoding="utf-8") as f:
        docs = [json.loads(line) for line in f]

    points = []
    for i, d in enumerate(docs):
        vec = embedder.encode(d["text"]).tolist()
        points.append(PointStruct(id=i, vector=vec, payload={"text": d["text"]}))

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"✅ Indexed {len(points)} documents into {COLLECTION_NAME}")

build_index()

# ============ CLASSICAL RAG ==============
def classical_rag(query, embedder, client, top_k=5):
    qvec = embedder.encode(query).tolist()

    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=qvec,
        limit=top_k
    )

    contexts = [hit.payload["text"] for hit in results]
    answer = generate_answer(query, contexts)
    return answer, contexts

# ============ QUANTUM RAG ==============

def quantum_rerank(query, contexts, top_k=5):
    """Rerank contexts using quantum similarity."""
    qvec = embedder.encode(query)
    
    scored_contexts = []
    for context in contexts:
        cvec = embedder.encode(context)
        q_score = quantum_similarity(qvec, cvec)
        c_score = cosine_similarity([qvec], [cvec])[0][0]
        
        # Hybrid score combining quantum and classical
        final_score = 0.5 * c_score + 0.5 * q_score
        scored_contexts.append((final_score, context))
    
    # Sort by score and return top_k
    scored_contexts.sort(reverse=True, key=lambda x: x[0])
    return [context for _, context in scored_contexts[:top_k]]

def quantum_rag(query, embedder, client, top_k=5):
    qvec = embedder.encode(query).tolist()

    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=qvec,
        limit=top_k * 2
    )

    # rerank using quantum similarity
    contexts = [hit.payload["text"] for hit in results]
    reranked = quantum_rerank(query, contexts, top_k=top_k)
    answer = generate_answer(query, reranked)
    return answer, reranked

# ============ GENERATOR ==============
def generate_answer(query, contexts):
    context_text = " ".join(contexts)[:600]
    prompt = (
        f"Question: {query}\n"
        f"Context: {context_text}\n"
        f"Instruction: Provide a clear, concise answer in 40–50 words.\n"
        f"Answer:"
    )
    inputs = gen_tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    outputs = gen_model.generate(**inputs, max_new_tokens=100)
    return gen_tokenizer.decode(outputs[0], skip_special_tokens=True)

# ============ MAIN LOOP ==============
def main():
    """Main function to run the RAG comparison."""
    if not DEPS_AVAILABLE:
        print("Error: Required dependencies not available. Please run:")
        print("pip install -r requirements.txt")
        return
    
    print("🤖 Classical vs Quantum RAG")
    print("Type 'exit' to quit\n")

    while True:
        query = input("❓ Your question: ")
        if query.lower() == "exit":
            break

        try:
            t0 = time.time()
            c_ans, _ = classical_rag(query, embedder, client)
            t1 = time.time()
            
            if PENNYLANE_AVAILABLE:
                q_ans, _ = quantum_rag(query, embedder, client)
                t2 = time.time()
            else:
                q_ans = "Quantum features not available"
                t2 = t1

            print("\n--- Classical RAG ---")
            print(f"Answer: {c_ans}")
            print(f"⏱ Time: {t1-t0:.2f}s")

            print("\n--- Quantum RAG ---")
            print(f"Answer: {q_ans}")
            print(f"⏱ Time: {t2-t1:.2f}s\n")
            
        except Exception as e:
            print(f"Error processing query: {e}")
            continue

if __name__ == "__main__":
    if DEPS_AVAILABLE:
        try:
            build_index()
        except Exception as e:
            print(f"Error building index: {e}")
            print("Make sure you have the data/cleaned.jsonl file")
    
    main()
