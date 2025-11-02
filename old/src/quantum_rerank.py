import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sklearn.metrics.pairwise import cosine_similarity

# Quantum imports
import pennylane as qml
from qiskit_ibm_runtime import QiskitRuntimeService

# ============ CONFIG ==============
COLLECTION_NAME = "agri_qrag"
N_QUBITS = 4  # Keep circuits small for demo
USE_IBM = False  # flip to True if you want real IBM backend
IBM_BACKEND = "ibmq_qasm_simulator"  # or e.g. "ibm_perth"

# ============ EMBEDDING MODEL ==============
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ============ QDRANT CLIENT ==============
client = QdrantClient(":memory:")

# ============ QUANTUM DEVICE ==============
if USE_IBM:
    service = QiskitRuntimeService()  # requires saved IBM account
    dev = qml.device("qiskit.ibmq", wires=N_QUBITS, backend=IBM_BACKEND)
else:
    dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev)
def quantum_kernel(x1, x2):
    """Quantum kernel similarity between two vectors"""
    qml.templates.AngleEmbedding(x1, wires=range(N_QUBITS))
    qml.adjoint(qml.templates.AngleEmbedding)(x2, wires=range(N_QUBITS))
    return qml.probs(wires=range(N_QUBITS))

def quantum_similarity(v1, v2):
    """Compute similarity using quantum kernel (fidelity)"""
    v1 = v1[:N_QUBITS]
    v2 = v2[:N_QUBITS]
    probs = quantum_kernel(v1, v2)
    return probs[0]  # probability of measuring |0000>

# ============ PIPELINE ==============
def rerank_with_quantum(query, docs, top_k=5):
    """Rerank retrieved docs using quantum similarity"""
    qvec = embedder.encode([query])[0]

    scores = []
    for doc in docs:
        dvec = embedder.encode([doc["text"]])[0]
        qscore = quantum_similarity(qvec, dvec)
        cscore = cosine_similarity([qvec], [dvec])[0][0]
        final = 0.5 * cscore + 0.5 * qscore  # hybrid score
        scores.append((final, doc["text"][:200]))

    scores.sort(reverse=True, key=lambda x: x[0])
    return scores[:top_k]

# Example run
if __name__ == "__main__":
    # Dummy mini-corpus
    corpus = [
        {"text": "Rice grows best in warm, wet environments with plenty of water."},
        {"text": "Wheat is usually planted in winter or spring depending on the region."},
        {"text": "Maize needs lots of sunlight and moderate rainfall to grow well."},
    ]

    query = "When is the best time to grow rice?"
    reranked = rerank_with_quantum(query, corpus)

    print("\n🔮 Quantum Reranked Results:")
    for score, text in reranked:
        print(f"Score: {score:.4f} | {text}...")
