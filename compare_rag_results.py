#!/usr/bin/env python3
"""
RAG Comparison Script
Compares Quantum RAG vs Classical RAG on agricultural web crawler data
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
from typing import List, Dict, Any, Tuple
from datetime import datetime

# Try to import quantum modules
try:
    import sys
    sys.path.append('old/src')
    from quantum_embeddings.pennylane_embeddings import AngleEmbedding
    HAS_QUANTUM = True
except ImportError:
    HAS_QUANTUM = False

# Data paths
DATA_DIR = Path("agricultural_data_complete/txt")
RESULTS_FILE = Path("rag_comparison_results.json")
RESULTS_CSV = Path("rag_comparison_results.csv")

# Test queries for evaluation
TEST_QUERIES = [
    "What are the best practices for organic farming?",
    "How can I improve soil fertility?",
    "What are the main crops produced globally?",
    "How does climate change affect agriculture?",
    "What is sustainable agriculture?",
    "How does irrigation impact crop yields?",
    "What are the benefits of crop rotation?",
    "How can farmers adapt to water scarcity?",
    "What role does livestock play in agriculture?",
    "How can precision agriculture improve farming efficiency?"
]

# -------------------- TEXT PROCESSING --------------------
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict[str, Any]]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk_text = ' '.join(chunk_words)
        
        if len(chunk_text.strip()) > 50:
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
            
            source_name = txt_file.stem
            content = re.sub(r'^={80}.*?={80}', '', content, flags=re.DOTALL)
            
            chunks = chunk_text(content.strip())
            
            for idx, chunk in enumerate(chunks):
                chunk['source'] = source_name
                chunk['chunk_id'] = f"{source_name}_{idx}"
                all_chunks.append(chunk)
                
        except Exception as e:
            continue
    
    return all_chunks

# -------------------- QUANTUM ENHANCER --------------------
class QuantumEmbeddingEnhancer:
    """Enhances classical embeddings with quantum feature maps."""
    
    def __init__(self, n_qubits: int = 8):
        self.n_qubits = n_qubits
        if not HAS_QUANTUM:
            raise ImportError("Quantum libraries not available")
        self.quantum_embedder = AngleEmbedding(n_qubits=n_qubits)
    
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
            return classical_embedding

# -------------------- RAG SYSTEMS --------------------
class ClassicalRAG:
    """Classical RAG system."""
    
    def __init__(self):
        print("🔄 Initializing Classical RAG...")
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.client = QdrantClient(":memory:")
        self.collection_name = "classical_compare_rag"
        self.indexed = False
    
    def index_data(self, chunks: List[Dict[str, Any]]):
        """Index data with classical embeddings."""
        vector_size = 384
        
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        
        points = []
        for i, chunk in enumerate(tqdm(chunks, desc="Classical indexing")):
            embedding = self.embedder.encode(chunk['text'])
            points.append(PointStruct(
                id=i,
                vector=embedding.tolist(),
                payload=chunk
            ))
        
        self.client.upsert(collection_name=self.collection_name, points=points)
        self.indexed = True
        print(f"✅ Classical RAG indexed {len(points)} chunks")
    
    def retrieve(self, query: str, limit: int = 5) -> Tuple[List, float]:
        """Retrieve relevant documents."""
        if not self.indexed:
            return [], 0.0
        
        start_time = time.time()
        query_embedding = self.embedder.encode(query)
        
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=limit,
            with_payload=True
        )
        
        retrieval_time = time.time() - start_time
        return results, retrieval_time


class QuantumRAG:
    """Quantum-enhanced RAG system."""
    
    def __init__(self, n_qubits: int = 8):
        print("🔄 Initializing Quantum RAG...")
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.client = QdrantClient(":memory:")
        self.collection_name = "quantum_compare_rag"
        self.indexed = False
        
        if HAS_QUANTUM:
            self.quantum_enhancer = QuantumEmbeddingEnhancer(n_qubits=n_qubits)
            print(f"✅ Quantum enhancer initialized ({n_qubits} qubits)")
        else:
            self.quantum_enhancer = None
            print("⚠️  Quantum not available, using classical")
    
    def index_data(self, chunks: List[Dict[str, Any]]):
        """Index data with quantum-enhanced embeddings."""
        if self.quantum_enhancer:
            sample_embedding = np.random.random(384)
            enhanced_embedding = self.quantum_enhancer.enhance_embedding(sample_embedding)
            vector_size = len(enhanced_embedding)
        else:
            vector_size = 384
        
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        
        points = []
        for i, chunk in enumerate(tqdm(chunks, desc="Quantum indexing")):
            classical_embedding = self.embedder.encode(chunk['text'])
            
            if self.quantum_enhancer:
                try:
                    embedding = self.quantum_enhancer.enhance_embedding(classical_embedding)
                except:
                    embedding = classical_embedding
            else:
                embedding = classical_embedding
            
            points.append(PointStruct(
                id=i,
                vector=embedding.tolist(),
                payload=chunk
            ))
        
        self.client.upsert(collection_name=self.collection_name, points=points)
        self.indexed = True
        print(f"✅ Quantum RAG indexed {len(points)} chunks")
    
    def retrieve(self, query: str, limit: int = 5) -> Tuple[List, float]:
        """Retrieve relevant documents with quantum enhancement."""
        if not self.indexed:
            return [], 0.0
        
        start_time = time.time()
        query_embedding = self.embedder.encode(query)
        
        if self.quantum_enhancer:
            try:
                query_embedding = self.quantum_enhancer.enhance_embedding(query_embedding)
            except:
                pass
        
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=limit,
            with_payload=True
        )
        
        retrieval_time = time.time() - start_time
        return results, retrieval_time

# -------------------- EVALUATION METRICS --------------------
def calculate_retrieval_overlap(classical_results, quantum_results, top_k: int = 3) -> float:
    """Calculate overlap in top-k retrieved documents."""
    classical_ids = set([r.id for r in classical_results[:top_k]])
    quantum_ids = set([r.id for r in quantum_results[:top_k]])
    
    if not classical_ids:
        return 0.0
    
    overlap = len(classical_ids.intersection(quantum_ids))
    return overlap / top_k

def calculate_avg_similarity_score(results, top_k: int = 3) -> float:
    """Calculate average similarity score."""
    if not results:
        return 0.0
    return np.mean([r.score for r in results[:top_k]])

def calculate_diversity(results, top_k: int = 3) -> float:
    """Calculate diversity of sources in top-k results."""
    if not results:
        return 0.0
    sources = [r.payload.get('source', 'unknown') for r in results[:top_k]]
    unique_sources = len(set(sources))
    return unique_sources / top_k

# -------------------- COMPARISON RUNNER --------------------
def run_comparison(test_queries: List[str] = TEST_QUERIES, top_k: int = 5):
    """Run comprehensive comparison between Classical and Quantum RAG."""
    
    print("="*80)
    print("RAG SYSTEMS COMPARISON: Quantum vs Classical")
    print("="*80)
    
    # Load data
    chunks = load_txt_files(DATA_DIR)
    if not chunks:
        print("❌ No data to compare")
        return
    
    print(f"\\n📊 Loaded {len(chunks)} chunks")
    
    # Initialize systems
    classical_rag = ClassicalRAG()
    
    if HAS_QUANTUM:
        quantum_rag = QuantumRAG(n_qubits=8)
    else:
        print("\\n⚠️  Quantum libraries not available. Comparison will be limited.")
        quantum_rag = None
    
    # Index data
    print("\\n📚 Indexing data...")
    classical_rag.index_data(chunks)
    if quantum_rag:
        quantum_rag.index_data(chunks)
    
    # Run comparisons
    results = []
    
    print(f"\\n🔍 Running {len(test_queries)} test queries...")
    print("="*80)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\\nQuery {i}/{len(test_queries)}: {query}")
        
        # Classical retrieval
        classical_results, classical_time = classical_rag.retrieve(query, limit=top_k)
        classical_avg_score = calculate_avg_similarity_score(classical_results, top_k)
        classical_diversity = calculate_diversity(classical_results, top_k)
        
        print(f"  📘 Classical: {classical_time:.4f}s | Avg Score: {classical_avg_score:.4f} | Diversity: {classical_diversity:.2f}")
        
        # Quantum retrieval
        if quantum_rag:
            quantum_results, quantum_time = quantum_rag.retrieve(query, limit=top_k)
            quantum_avg_score = calculate_avg_similarity_score(quantum_results, top_k)
            quantum_diversity = calculate_diversity(quantum_results, top_k)
            overlap = calculate_retrieval_overlap(classical_results, quantum_results, top_k)
            
            speedup = classical_time / quantum_time if quantum_time > 0 else 1.0
            score_improvement = ((quantum_avg_score - classical_avg_score) / classical_avg_score * 100) if classical_avg_score > 0 else 0.0
            
            print(f"  ⚛️  Quantum:   {quantum_time:.4f}s | Avg Score: {quantum_avg_score:.4f} | Diversity: {quantum_diversity:.2f}")
            print(f"  📊 Overlap: {overlap:.2f} | Speed Ratio: {speedup:.2f}x | Score Diff: {score_improvement:+.2f}%")
        else:
            quantum_results, quantum_time, quantum_avg_score = [], 0.0, 0.0
            quantum_diversity, overlap, speedup, score_improvement = 0.0, 0.0, 0.0, 0.0
        
        # Store results
        result = {
            'query': query,
            'classical': {
                'retrieval_time': classical_time,
                'avg_similarity': classical_avg_score,
                'diversity': classical_diversity,
                'top_sources': [r.payload.get('source', 'unknown') for r in classical_results[:3]]
            },
            'quantum': {
                'retrieval_time': quantum_time,
                'avg_similarity': quantum_avg_score,
                'diversity': quantum_diversity,
                'top_sources': [r.payload.get('source', 'unknown') for r in quantum_results[:3]] if quantum_results else []
            },
            'comparison': {
                'overlap': overlap,
                'speedup': speedup,
                'score_improvement_pct': score_improvement
            }
        }
        results.append(result)
    
    # Calculate aggregate statistics
    print("\\n" + "="*80)
    print("📊 AGGREGATE RESULTS")
    print("="*80)
    
    classical_avg_time = np.mean([r['classical']['retrieval_time'] for r in results])
    classical_avg_similarity = np.mean([r['classical']['avg_similarity'] for r in results])
    classical_avg_diversity = np.mean([r['classical']['diversity'] for r in results])
    
    print(f"\\n📘 Classical RAG:")
    print(f"   Avg Retrieval Time: {classical_avg_time:.4f}s")
    print(f"   Avg Similarity Score: {classical_avg_similarity:.4f}")
    print(f"   Avg Source Diversity: {classical_avg_diversity:.4f}")
    
    if quantum_rag:
        quantum_avg_time = np.mean([r['quantum']['retrieval_time'] for r in results])
        quantum_avg_similarity = np.mean([r['quantum']['avg_similarity'] for r in results])
        quantum_avg_diversity = np.mean([r['quantum']['diversity'] for r in results])
        avg_overlap = np.mean([r['comparison']['overlap'] for r in results])
        avg_speedup = np.mean([r['comparison']['speedup'] for r in results])
        avg_score_improvement = np.mean([r['comparison']['score_improvement_pct'] for r in results])
        
        print(f"\\n⚛️  Quantum RAG:")
        print(f"   Avg Retrieval Time: {quantum_avg_time:.4f}s")
        print(f"   Avg Similarity Score: {quantum_avg_similarity:.4f}")
        print(f"   Avg Source Diversity: {quantum_avg_diversity:.4f}")
        
        print(f"\\n🔬 Comparison:")
        print(f"   Avg Overlap: {avg_overlap:.4f}")
        print(f"   Avg Speed Ratio: {avg_speedup:.2f}x")
        print(f"   Avg Score Improvement: {avg_score_improvement:+.2f}%")
        
        # Determine winner
        print(f"\\n🏆 Results:")
        if quantum_avg_similarity > classical_avg_similarity:
            print(f"   ✅ Quantum RAG has better retrieval quality ({quantum_avg_similarity:.4f} vs {classical_avg_similarity:.4f})")
        else:
            print(f"   ✅ Classical RAG has better retrieval quality ({classical_avg_similarity:.4f} vs {quantum_avg_similarity:.4f})")
        
        if quantum_avg_time < classical_avg_time:
            print(f"   ✅ Quantum RAG is faster ({quantum_avg_time:.4f}s vs {classical_avg_time:.4f}s)")
        else:
            print(f"   ✅ Classical RAG is faster ({classical_avg_time:.4f}s vs {quantum_avg_time:.4f}s)")
    
    # Save results
    print(f"\\n💾 Saving results...")
    
    # Save JSON
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'num_queries': len(test_queries),
            'num_chunks': len(chunks),
            'top_k': top_k,
            'has_quantum': HAS_QUANTUM,
            'queries': results,
            'aggregates': {
                'classical': {
                    'avg_retrieval_time': classical_avg_time,
                    'avg_similarity': classical_avg_similarity,
                    'avg_diversity': classical_avg_diversity
                },
                'quantum': {
                    'avg_retrieval_time': quantum_avg_time if quantum_rag else 0,
                    'avg_similarity': quantum_avg_similarity if quantum_rag else 0,
                    'avg_diversity': quantum_avg_diversity if quantum_rag else 0
                } if quantum_rag else None,
                'comparison': {
                    'avg_overlap': avg_overlap if quantum_rag else 0,
                    'avg_speedup': avg_speedup if quantum_rag else 0,
                    'avg_score_improvement_pct': avg_score_improvement if quantum_rag else 0
                } if quantum_rag else None
            }
        }, f, indent=2)
    
    # Save CSV
    with open(RESULTS_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'query', 
            'classical_time', 'classical_similarity', 'classical_diversity',
            'quantum_time', 'quantum_similarity', 'quantum_diversity',
            'overlap', 'speedup', 'score_improvement_pct'
        ])
        
        for r in results:
            writer.writerow([
                r['query'],
                r['classical']['retrieval_time'],
                r['classical']['avg_similarity'],
                r['classical']['diversity'],
                r['quantum']['retrieval_time'],
                r['quantum']['avg_similarity'],
                r['quantum']['diversity'],
                r['comparison']['overlap'],
                r['comparison']['speedup'],
                r['comparison']['score_improvement_pct']
            ])
    
    print(f"✅ Results saved to:")
    print(f"   - {RESULTS_FILE}")
    print(f"   - {RESULTS_CSV}")
    print("\\n" + "="*80)
    print("✅ COMPARISON COMPLETE!")
    print("="*80)

# -------------------- MAIN --------------------
def main():
    """Main entry point."""
    print("\\n🚀 Starting RAG Comparison...")
    
    # Check if data exists
    if not DATA_DIR.exists() or not list(DATA_DIR.glob("*.txt")):
        print(f"\\n❌ No data found in {DATA_DIR}")
        print("Please run the web crawler first:")
        print("   python web_crawler.py")
        return
    
    # Run comparison
    run_comparison(test_queries=TEST_QUERIES, top_k=5)

if __name__ == "__main__":
    main()
