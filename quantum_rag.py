#!/usr/bin/env python3
"""
Quantum-Enhanced Agricultural RAG System
Works with web crawler TXT data from agricultural_data_complete/
"""

import json
import time
import csv
import os
import re
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

# Quantum embeddings
try:
    import sys
    sys.path.append('old/src')
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

# Paths
DATA_DIR = Path("agricultural_data_complete/txt")
COLLECTION_NAME = "quantum_agri_crawler_rag"
LOG_FILE = Path("quantum_query_logs.csv")

# -------------------- TEXT PROCESSING --------------------
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict[str, Any]]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk_text = ' '.join(chunk_words)
        
        if len(chunk_text.strip()) > 50:  # Skip very small chunks
            chunks.append({
                'text': chunk_text,
                'start_idx': i,
                'end_idx': i + len(chunk_words)
            })
    
    return chunks

def load_txt_files(data_dir: Path) -> List[Dict[str, Any]]:
    """Load all TXT files from web crawler output."""
    all_chunks = []
    
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return all_chunks
    
    txt_files = list(data_dir.glob("*.txt"))
    
    if not txt_files:
        print(f"❌ No TXT files found in {data_dir}")
        return all_chunks
    
    print(f"📂 Loading {len(txt_files)} TXT files...")
    
    for txt_file in tqdm(txt_files, desc="Processing files"):
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract source info from header
            source_name = txt_file.stem
            url_match = re.search(r'URL:\s*(.+?)\\n', content)
            source_url = url_match.group(1) if url_match else "Unknown"
            
            # Remove header
            content = re.sub(r'^={80}.*?={80}', '', content, flags=re.DOTALL)
            
            # Chunk the text
            chunks = chunk_text(content.strip())
            
            # Add metadata to each chunk
            for idx, chunk in enumerate(chunks):
                chunk['source'] = source_name
                chunk['source_url'] = source_url
                chunk['chunk_id'] = f"{source_name}_{idx}"
                chunk['file'] = str(txt_file)
                all_chunks.append(chunk)
                
        except Exception as e:
            print(f"⚠️  Error loading {txt_file.name}: {e}")
            continue
    
    print(f"✅ Loaded {len(all_chunks)} chunks from {len(txt_files)} files")
    return all_chunks

# -------------------- QUANTUM EMBEDDING WRAPPER --------------------
class QuantumEmbeddingEnhancer:
    """Enhances classical embeddings with quantum feature maps."""
    
    def __init__(self, embedding_type: str = "angle", n_qubits: int = 8):
        self.embedding_type = embedding_type
        self.n_qubits = n_qubits
        
        if not HAS_QUANTUM:
            raise ImportError("Quantum libraries not available")
        
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
        if len(classical_embedding) > self.n_qubits:
            reduced_embedding = classical_embedding[:self.n_qubits]
        else:
            reduced_embedding = np.pad(
                classical_embedding, 
                (0, max(0, self.n_qubits - len(classical_embedding)))
            )
        
        try:
            quantum_state = self.quantum_embedder.encode(reduced_embedding)
            
            if hasattr(quantum_state, 'numpy'):
                quantum_features = quantum_state.numpy()
            else:
                quantum_features = np.array(quantum_state)
            
            if len(quantum_features.shape) > 1:
                quantum_features = quantum_features.flatten()
            
            hybrid_embedding = np.concatenate([
                classical_embedding,
                quantum_features.real,
                quantum_features.imag[:len(quantum_features.real)]
            ])
            
            return hybrid_embedding
            
        except Exception as e:
            print(f"⚠️  Quantum enhancement failed: {e}")
            return classical_embedding

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
        device=-1
    )
    
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
    
    gemini_config = init_gemini()

    return embedder, client, generator, gemini_config, quantum_enhancer

def init_gemini():
    """Initialize Gemini API if available and configured."""
    if not HAS_GEMINI:
        return None
    
    if HAS_DOTENV:
        env_file = Path(".env")
        if env_file.exists():
            load_dotenv(env_file)
    
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        return None
    
    os.environ['GOOGLE_API_KEY'] = api_key

    try:
        client = genai.Client()
        model_names = [
            'gemini-2.0-flash-exp',
            'gemini-1.5-flash',
            'gemini-1.5-pro',
        ]
        
        for model_name in model_names:
            try:
                test_response = client.models.generate_content(
                    model=model_name,
                    contents="Test"
                )
                if test_response and test_response.text:
                    print(f"✅ Gemini API ready! Using model: {model_name}")
                    return {
                        'client': client,
                        'model': model_name,
                        'available': True
                    }
            except:
                continue
        
        return None
        
    except Exception as e:
        return None

# -------------------- DATA LOADING --------------------
def load_and_index_data(embedder, client, quantum_enhancer=None):
    """Load web crawler data and create quantum-enhanced embeddings."""
    chunks = load_txt_files(DATA_DIR)
    
    if not chunks:
        print("❌ No data chunks loaded")
        return 0

    base_vector_size = 384
    if quantum_enhancer:
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

    print("🔄 Creating embeddings...")
    points = []
    
    for i, chunk in enumerate(tqdm(chunks, desc="Embedding chunks")):
        classical_embedding = embedder.encode(chunk['text'])
        
        if quantum_enhancer:
            try:
                enhanced_embedding = quantum_enhancer.enhance_embedding(classical_embedding)
            except Exception as e:
                enhanced_embedding = classical_embedding
        else:
            enhanced_embedding = classical_embedding
        
        points.append(PointStruct(
            id=i,
            vector=enhanced_embedding.tolist(),
            payload=chunk
        ))

    print("🔄 Indexing documents in vector database...")
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"✅ Indexed {len(points)} chunks into {COLLECTION_NAME}")
    return len(points)

# -------------------- RETRIEVAL --------------------
def quantum_retrieve(query: str, embedder, client, quantum_enhancer=None, limit: int = 3):
    """Retrieve relevant documents using quantum-enhanced similarity."""
    query_embedding = embedder.encode(query)
    
    if quantum_enhancer:
        try:
            query_embedding = quantum_enhancer.enhance_embedding(query_embedding)
        except Exception as e:
            print(f"⚠️  Quantum enhancement failed for query: {e}")
    
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
    context = "\\n\\n".join(context_chunks)
    
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
    
    prompt = f"Context: {context}\\n\\nQuestion: {query}\\n\\nAnswer:"
    
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
        return "I apologize, but I'm having trouble generating a response right now."

# -------------------- MAIN LOOP --------------------
def main():
    """Main interactive loop."""
    print("🚀 Initializing Quantum Agricultural RAG System (Web Crawler Data)...")
    
    print("\\n🔬 Quantum Embedding Options:")
    print("1. angle - Angle embedding (simple, fast)")
    print("2. amplitude - Amplitude embedding (dense representation)")
    print("3. classical - Classical embeddings only")
    
    choice = input("\\nSelect quantum embedding type (1-3, default=1): ").strip()
    quantum_types = {
        '1': 'angle', '2': 'amplitude', '3': 'classical'
    }
    quantum_type = quantum_types.get(choice, 'angle')
    
    n_qubits = 8
    if quantum_type != 'classical':
        try:
            n_qubits = int(input(f"Number of qubits (default=8): ").strip() or "8")
            n_qubits = max(4, min(n_qubits, 16))
        except:
            n_qubits = 8
    
    if quantum_type == 'classical':
        embedder, client, generator, gemini_config, quantum_enhancer = (
            *init_quantum_pipeline("angle", 8)[:-1], None
        )
    else:
        embedder, client, generator, gemini_config, quantum_enhancer = init_quantum_pipeline(
            quantum_type, n_qubits
        )
    
    num_chunks = load_and_index_data(embedder, client, quantum_enhancer)
    if num_chunks == 0:
        return
    
    if not LOG_FILE.parent.exists():
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    if not LOG_FILE.exists():
        with open(LOG_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'query', 'context_chunks', 'answer', 'quantum_type', 'n_qubits', 'retrieval_time'])
    
    print(f"\\n🤖 Quantum Agricultural RAG System Ready!")
    print(f"📊 Loaded {num_chunks} chunks with {quantum_type} embeddings")
    if gemini_config and gemini_config.get('available'):
        print(f"🧠 Using Gemini API ({gemini_config['model']}) + local fallback")
    else:
        print("⚠️  Using local models only")
    print("\\nType 'exit' to quit, 'help' for sample questions")
    
    while True:
        query = input("\\n🌾 Ask about agriculture: ").strip()
        
        if query.lower() in ['exit', 'quit', 'q']:
            print("👋 Goodbye!")
            break
        
        if query.lower() == 'help':
            print("\\n📝 Sample questions:")
            print("- What are the best practices for organic farming?")
            print("- How can I improve soil fertility?")
            print("- What are the main crops produced globally?")
            print("- How does climate change affect agriculture?")
            continue
        
        if not query:
            continue
        
        print(f"\\n🔍 Searching with {quantum_type} embeddings...")
        start_time = time.time()
        
        results = quantum_retrieve(query, embedder, client, quantum_enhancer, limit=3)
        
        context_chunks = []
        for result in results:
            chunk_text = result.payload.get('text', '')
            source = result.payload.get('source', 'Unknown')
            score = result.score
            print(f"📄 Source: {source} (similarity: {score:.3f})")
            context_chunks.append(chunk_text)
        
        if not context_chunks:
            print("❌ No relevant information found")
            continue
        
        print(f"\\n🤖 Generating answer...")
        
        answer = generate_answer(query, context_chunks, generator, gemini_config)
        
        retrieval_time = time.time() - start_time
        print(f"\\n📝 Answer (in {retrieval_time:.2f}s):")
        print("-" * 50)
        print(answer)
        print("-" * 50)
        
        try:
            with open(LOG_FILE, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    time.strftime('%Y-%m-%d %H:%M:%S'),
                    query,
                    json.dumps(context_chunks),
                    answer,
                    quantum_type,
                    n_qubits if quantum_enhancer else 0,
                    retrieval_time
                ])
        except Exception as e:
            print(f"⚠️  Logging failed: {e}")

if __name__ == "__main__":
    main()
