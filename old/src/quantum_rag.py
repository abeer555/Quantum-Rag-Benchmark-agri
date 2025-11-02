#!/usr/bin/env python3
"""
Quantum-Enhanced Agricultural RAG System

This script implements a RAG (Retrieval-Augmented Generation) system for agricultural
information using quantum embeddings for enhanced semantic similarity computation.
"""

import json
import time
import csv
import os
import numpy as np
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple

# Environment file loading
try:
    from dotenv import load_dotenv
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False

# Gemini API integration
try:
    from google import genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False
    print("⚠️  Gemini API not available. Install with: pip install google-generativeai")

# Quantum embeddings
try:
    import sys
    sys.path.append('src')  # Add src to path for relative imports
    from quantum_embeddings.pennylane_embeddings import (
        AngleEmbedding, AmplitudeEmbedding, IQPEmbedding,
        quantum_similarity_pennylane
    )
    from quantum_embeddings.qiskit_embeddings import (
        ZFeatureMap, ZZFeatureMap, PauliFeatureMap,
        quantum_similarity_qiskit
    )
    HAS_QUANTUM = True
except ImportError as e:
    HAS_QUANTUM = False
    print(f"⚠️  Quantum embeddings not available: {e}")
    print("   Install with: pip install pennylane qiskit")

# Paths
DATA_FILE = Path("data/cleaned.jsonl")
COLLECTION_NAME = "quantum_agri_rag"
LOG_FILE = Path("data/quantum_query_logs.csv")

# -------------------- QUANTUM EMBEDDING WRAPPER --------------------
class QuantumEmbeddingEnhancer:
    """Enhances classical embeddings with quantum feature maps."""
    
    def __init__(self, embedding_type: str = "angle", n_qubits: int = 8):
        self.embedding_type = embedding_type
        self.n_qubits = n_qubits
        
        if not HAS_QUANTUM:
            raise ImportError("Quantum libraries not available")
        
        # Initialize quantum embedding based on type
        if embedding_type == "angle":
            self.quantum_embedder = AngleEmbedding(n_qubits=n_qubits)
        elif embedding_type == "amplitude":
            self.quantum_embedder = AmplitudeEmbedding(n_qubits=n_qubits)
        elif embedding_type == "iqp":
            self.quantum_embedder = IQPEmbedding(n_qubits=n_qubits, depth=2)
        elif embedding_type == "z_feature":
            self.quantum_embedder = ZFeatureMap(feature_dimension=n_qubits)
        elif embedding_type == "zz_feature":
            self.quantum_embedder = ZZFeatureMap(feature_dimension=n_qubits)
        elif embedding_type == "pauli":
            self.quantum_embedder = PauliFeatureMap(feature_dimension=n_qubits)
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")
    
    def enhance_embedding(self, classical_embedding: np.ndarray) -> np.ndarray:
        """Enhance classical embedding with quantum features."""
        # Reduce dimensionality to fit quantum circuit
        if len(classical_embedding) > self.n_qubits:
            # Use PCA-like reduction or simple truncation
            reduced_embedding = classical_embedding[:self.n_qubits]
        else:
            # Pad with zeros if needed
            reduced_embedding = np.pad(
                classical_embedding, 
                (0, max(0, self.n_qubits - len(classical_embedding)))
            )
        
        try:
            # Get quantum state
            quantum_state = self.quantum_embedder.encode(reduced_embedding)
            
            # Extract features from quantum state
            if hasattr(quantum_state, 'numpy'):
                quantum_features = quantum_state.numpy()
            else:
                quantum_features = np.array(quantum_state)
            
            # Combine classical and quantum features
            if len(quantum_features.shape) > 1:
                quantum_features = quantum_features.flatten()
            
            # Create hybrid embedding: classical + quantum features
            hybrid_embedding = np.concatenate([
                classical_embedding,
                quantum_features.real,  # Use real part for practical purposes
                quantum_features.imag[:len(quantum_features.real)]  # Add some imaginary part info
            ])
            
            return hybrid_embedding
            
        except Exception as e:
            print(f"⚠️  Quantum enhancement failed: {e}")
            return classical_embedding  # Fallback to classical embedding
    
    def quantum_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute quantum-enhanced similarity between embeddings."""
        try:
            # Reduce to quantum circuit size
            q_emb1 = emb1[:self.n_qubits] if len(emb1) > self.n_qubits else emb1
            q_emb2 = emb2[:self.n_qubits] if len(emb2) > self.n_qubits else emb2
            
            # Use quantum similarity function
            if "pennylane" in str(type(self.quantum_embedder)).lower():
                return quantum_similarity_pennylane(q_emb1, q_emb2, self.n_qubits)
            else:
                return quantum_similarity_qiskit(q_emb1, q_emb2, self.quantum_embedder)
        except Exception as e:
            print(f"⚠️  Quantum similarity failed: {e}")
            # Fallback to cosine similarity
            return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

# -------------------- INIT --------------------
def init_quantum_pipeline(quantum_type: str = "angle", n_qubits: int = 8):
    """Initialize the quantum-enhanced RAG pipeline."""
    print("🔄 Loading embedding model (MiniLM, fast on CPU)...")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("🔄 Connecting to Qdrant (in-memory)...")
    client = QdrantClient(":memory:")

    print("🔄 Loading local text-generation model...")
    generator = pipeline(
        "text2text-generation", 
        model="google/flan-t5-large",
        device=-1  # Force CPU usage
    )
    
    # Initialize quantum enhancer
    quantum_enhancer = None
    if HAS_QUANTUM:
        try:
            print(f"🔄 Initializing quantum embeddings ({quantum_type}, {n_qubits} qubits)...")
            quantum_enhancer = QuantumEmbeddingEnhancer(
                embedding_type=quantum_type, 
                n_qubits=n_qubits
            )
            print("✅ Quantum embeddings initialized!")
        except Exception as e:
            print(f"⚠️  Quantum embedding initialization failed: {e}")
            print("   Falling back to classical embeddings only")
    else:
        print("⚠️  Quantum libraries not available, using classical embeddings only")
    
    # Initialize Gemini API
    gemini_config = init_gemini()

    return embedder, client, generator, gemini_config, quantum_enhancer

def init_gemini():
    """Initialize Gemini API if available and configured."""
    if not HAS_GEMINI:
        print("⚠️  Gemini API not available")
        return None
    
    # Load .env file if available
    if HAS_DOTENV:
        env_file = Path(".env")
        if env_file.exists():
            load_dotenv(env_file)
            print("✅ Loaded environment variables from .env file")
        else:
            load_dotenv()  # Try to load from default locations
    
    # Check for API key in environment variable
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("⚠️  GEMINI_API_KEY not found")
        print("   Options:")
        print("   1. Create .env file with: GEMINI_API_KEY=your-api-key")
        print("   2. Set environment variable: export GEMINI_API_KEY='your-api-key'")
        return None
    
    # Set the API key for Google AI client (it expects GOOGLE_API_KEY)
    os.environ['GOOGLE_API_KEY'] = api_key

    try:
        # Use the exact format from documentation
        client = genai.Client()
        
        # Try different model names, prioritizing pro models first
        model_names = [
            'gemini-2.5-pro',        # Latest pro model
            'gemini-2.5-flash',      # Experimental flash model
            'gemini-1.5-pro',        # Good performance, prioritized
            'gemini-1.5-flash',      # Fast and usually available
            'gemini-2.0-flash-exp'   # Experimental flash model
        ]
        
        for model_name in model_names:
            try:
                # Test the model with a simple query
                prompt = "Test"
                test_response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                if test_response and test_response.text:
                    print(f"✅ Gemini API ready! Using model: {model_name}")
                    return {
                        'client': client,
                        'model': model_name,
                        'available': True
                    }
            except Exception as e:
                error_msg = str(e)
                if "503" in error_msg or "overloaded" in error_msg.lower():
                    print(f"⚠️  Model {model_name} is overloaded, trying next...")
                elif "404" in error_msg or "not found" in error_msg.lower():
                    print(f"⚠️  Model {model_name} not available, trying next...")
                else:
                    print(f"⚠️  Model {model_name} failed: {error_msg[:50]}...")
                continue
        
        print("❌ No working Gemini models found")
        return None
        
    except Exception as e:
        print(f"❌ Gemini API initialization failed: {e}")
        return None

# -------------------- DATA LOADING --------------------
def load_and_index_data(embedder, client, quantum_enhancer=None):
    """Load data and create quantum-enhanced embeddings."""
    if not DATA_FILE.exists():
        print(f"❌ Data file not found: {DATA_FILE}")
        print("Run merge_and_chunk.py first to create the data file")
        return 0

    # Create collection with appropriate vector size
    base_vector_size = 384  # MiniLM embedding size
    if quantum_enhancer:
        # Estimate enhanced vector size (classical + quantum features)
        sample_embedding = np.random.random(base_vector_size)
        enhanced_embedding = quantum_enhancer.enhance_embedding(sample_embedding)
        vector_size = len(enhanced_embedding)
        print(f"📊 Using quantum-enhanced vectors (size: {vector_size})")
    else:
        vector_size = base_vector_size
        print(f"📊 Using classical vectors (size: {vector_size})")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )

    # Load and embed documents
    documents = []
    embeddings = []
    
    print("🔄 Loading and embedding documents...")
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(tqdm(f, desc="Processing chunks")):
            data = json.loads(line.strip())
            text = data.get('text', '') or data.get('content', '')  # Handle both field names
            if not text.strip():
                continue
            
            # Create classical embedding
            classical_embedding = embedder.encode(text)
            
            # Enhance with quantum features if available
            if quantum_enhancer:
                try:
                    enhanced_embedding = quantum_enhancer.enhance_embedding(classical_embedding)
                except Exception as e:
                    print(f"⚠️  Quantum enhancement failed for chunk {line_num}: {e}")
                    enhanced_embedding = classical_embedding
            else:
                enhanced_embedding = classical_embedding
            
            documents.append(data)
            embeddings.append(enhanced_embedding)

    # Index documents
    print("🔄 Indexing documents in vector database...")
    points = []
    for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
        points.append(PointStruct(
            id=i,
            vector=embedding.tolist(),
            payload=doc
        ))

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"✅ Indexed {len(points)} chunks into {COLLECTION_NAME}")
    return len(points)

# -------------------- RETRIEVAL --------------------
def quantum_retrieve(query: str, embedder, client, quantum_enhancer=None, limit: int = 3):
    """Retrieve relevant documents using quantum-enhanced similarity."""
    # Create query embedding
    query_embedding = embedder.encode(query)
    
    # Enhance with quantum features if available
    if quantum_enhancer:
        try:
            query_embedding = quantum_enhancer.enhance_embedding(query_embedding)
        except Exception as e:
            print(f"⚠️  Quantum enhancement failed for query: {e}")
    
    # Search in vector database
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding.tolist(),
        limit=limit,
        with_payload=True
    )
    
    return results

# -------------------- GENERATION --------------------
def generate_answer(query: str, context_chunks: List[str], generator, gemini_config=None) -> str:
    """Generate answer using retrieved context and available models."""
    context = "\n\n".join(context_chunks)
    
    # Try Gemini first if available
    if gemini_config and gemini_config.get('available'):
        try:
            prompt = f"""Based on the following agricultural information, please provide a comprehensive and accurate answer to the question.

Context:
{context}

Question: {query}

Please provide a detailed, helpful answer based on the agricultural information provided. If the context doesn't contain enough information to fully answer the question, please indicate that and provide what information you can."""

            response = gemini_config['client'].models.generate_content(
                model=gemini_config['model'],
                contents=prompt
            )
            
            if response and response.text:
                return response.text.strip()
                
        except Exception as e:
            print(f"⚠️  Gemini generation failed: {e}")
            print("   Falling back to local model...")
    
    # Fallback to local model
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    
    try:
        response = generator(
            prompt,
            max_length=200,
            min_length=30,
            do_sample=True,
            temperature=0.7,
            pad_token_id=generator.tokenizer.eos_token_id
        )
        return response[0]['generated_text'].replace(prompt, "").strip()
    except Exception as e:
        print(f"⚠️  Local generation failed: {e}")
        return "I apologize, but I'm having trouble generating a response right now."

# -------------------- MAIN LOOP --------------------
def main():
    """Main interactive loop."""
    print("🚀 Initializing Quantum Agricultural RAG System...")
    
    # Ask user for quantum embedding preferences
    print("\n🔬 Quantum Embedding Options:")
    print("1. angle - Angle embedding (simple, fast)")
    print("2. amplitude - Amplitude embedding (dense representation)")
    # print("3. iqp - IQP embedding (good expressivity)")
    # print("4. z_feature - Z feature map (Qiskit)")
    # print("5. zz_feature - ZZ feature map (Qiskit, entangled)")
    # print("6. pauli - Pauli feature map (Qiskit, flexible)")
    # print("7. classical - Classical embeddings only")
    
    choice = input("\nSelect quantum embedding type (1-2, default=1): ").strip()
    quantum_types = {
        '1': 'angle', '2': 'amplitude'
    }
    quantum_type = quantum_types.get(choice, 'angle')
    
    n_qubits = 8
    if quantum_type != 'classical':
        try:
            n_qubits = int(input(f"Number of qubits (default=8): ").strip() or "8")
            n_qubits = max(4, min(n_qubits, 16))  # Limit to reasonable range
        except:
            n_qubits = 8
    
    # Initialize pipeline
    if quantum_type == 'classical':
        embedder, client, generator, gemini_config, quantum_enhancer = (
            *init_quantum_pipeline("angle", 8)[:-1], None
        )
    else:
        embedder, client, generator, gemini_config, quantum_enhancer = init_quantum_pipeline(
            quantum_type, n_qubits
        )
    
    # Load and index data
    num_chunks = load_and_index_data(embedder, client, quantum_enhancer)
    if num_chunks == 0:
        return
    
    # Initialize query log
    if not LOG_FILE.parent.exists():
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    if not LOG_FILE.exists():
        with open(LOG_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'query', 'context_chunks', 'answer', 'quantum_type', 'n_qubits'])
    
    print(f"\n🤖 Quantum Agricultural RAG System Ready!")
    print(f"📊 Loaded {num_chunks} chunks with {quantum_type} embeddings")
    if gemini_config and gemini_config.get('available'):
        print(f"🧠 Using Gemini API ({gemini_config['model']}) + local fallback")
    else:
        print("⚠️  Using local models only (set GEMINI_API_KEY for better answers)")
    print("\nType 'exit' to quit, 'help' for sample questions")
    
    while True:
        query = input("\n🌾 Ask about agriculture: ").strip()
        
        if query.lower() in ['exit', 'quit', 'q']:
            print("👋 Goodbye!")
            break
        
        if query.lower() == 'help':
            print("\n📝 Sample questions:")
            print("- What are the best practices for organic farming?")
            print("- How can I improve soil fertility?")
            print("- What causes plant diseases and how to prevent them?")
            print("- How does climate change affect crop yields?")
            continue
        
        if not query:
            continue
        
        print(f"\n🔍 Searching with {quantum_type} embeddings...")
        start_time = time.time()
        
        # Retrieve relevant chunks
        results = quantum_retrieve(query, embedder, client, quantum_enhancer, limit=3)
        
        # Extract context
        context_chunks = []
        for result in results:
            chunk_text = result.payload.get('text', '') or result.payload.get('content', '')
            source = result.payload.get('source', result.payload.get('id', 'Unknown'))
            score = result.score
            print(f"📄 Source: {source} (similarity: {score:.3f})")
            context_chunks.append(chunk_text)
        
        if not context_chunks:
            print("❌ No relevant information found")
            continue
        
        print(f"\n🤖 Generating answer...")
        
        # Generate answer
        answer = generate_answer(query, context_chunks, generator, gemini_config)
        
        retrieval_time = time.time() - start_time
        print(f"\n📝 Answer (in {retrieval_time:.2f}s):")
        print("-" * 50)
        print(answer)
        print("-" * 50)
        
        # Log the interaction
        try:
            with open(LOG_FILE, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    time.strftime('%Y-%m-%d %H:%M:%S'),
                    query,
                    json.dumps(context_chunks),
                    answer,
                    quantum_type,
                    n_qubits if quantum_enhancer else 0
                ])
        except Exception as e:
            print(f"⚠️  Logging failed: {e}")

if __name__ == "__main__":
    main()