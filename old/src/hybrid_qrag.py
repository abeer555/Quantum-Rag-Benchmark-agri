"""
Enhanced Hybrid Quantum RAG Pipeline

This module provides a comprehensive quantum-enhanced RAG system with
multiple embedding strategies, benchmarking, and evaluation capabilities.
"""

import time
import json
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path

# Core dependencies
try:
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct, VectorParams, Distance
    CORE_DEPS_AVAILABLE = True
except ImportError:
    CORE_DEPS_AVAILABLE = False
    print("Warning: Core dependencies not available. Install requirements.txt")

# Quantum dependencies
try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False

try:
    from qiskit import QuantumCircuit
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

# Local imports
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if PENNYLANE_AVAILABLE or QISKIT_AVAILABLE:
    from quantum_embeddings import (
        quantum_similarity_pennylane, create_feature_map, compare_embeddings
    )

from evaluation import (
    evaluate_retrieval_performance, evaluate_answer_quality, 
    benchmark_rag_pipeline, HumanEvaluationFramework
)

from benchmarks import QRAGBenchmark, BenchmarkConfig, create_sample_agricultural_benchmark


class HybridQuantumRAG:
    """
    Enhanced RAG system with quantum embeddings and comprehensive evaluation.
    """
    
    def __init__(
        self,
        collection_name: str = "agri_rag_enhanced",
        quantum_method: str = "angle",
        quantum_framework: str = "pennylane",
        n_qubits: int = 4,
        classical_weight: float = 0.5,
        quantum_weight: float = 0.5,
        use_quantum: bool = True,
        generator_model: str = "google/flan-t5-base"
    ):
        """
        Initialize the Hybrid Quantum RAG system.
        
        Args:
            collection_name: Name for the vector database collection
            quantum_method: Quantum embedding method ("angle", "amplitude", "iqp", etc.)
            quantum_framework: Framework to use ("pennylane" or "qiskit")
            n_qubits: Number of qubits for quantum circuits
            classical_weight: Weight for classical similarity
            quantum_weight: Weight for quantum similarity
            use_quantum: Whether to use quantum enhancements
            generator_model: Name of the text generation model
        """
        if not CORE_DEPS_AVAILABLE:
            raise ImportError("Core dependencies not available. Please install requirements.txt")
        
        self.collection_name = collection_name
        self.quantum_method = quantum_method
        self.quantum_framework = quantum_framework
        self.n_qubits = n_qubits
        self.classical_weight = classical_weight
        self.quantum_weight = quantum_weight
        self.use_quantum = use_quantum
        
        # Initialize classical components
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.client = QdrantClient(":memory:")
        
        # Initialize generator
        self.device = "cpu"
        self.gen_tokenizer = AutoTokenizer.from_pretrained(generator_model)
        self.gen_model = AutoModelForSeq2SeqLM.from_pretrained(generator_model).to(self.device)
        
        # Quantum setup
        if self.use_quantum:
            if not (PENNYLANE_AVAILABLE or QISKIT_AVAILABLE):
                print("Warning: Quantum frameworks not available. Falling back to classical mode.")
                self.use_quantum = False
            else:
                self.quantum_embedder = create_feature_map(
                    quantum_method, n_qubits, quantum_framework
                )
        
        # Performance tracking
        self.performance_stats = {
            "total_queries": 0,
            "avg_retrieval_time": 0.0,
            "avg_generation_time": 0.0,
            "avg_quantum_time": 0.0
        }
        
        print(f"Initialized Hybrid Quantum RAG")
        print(f"  Quantum enabled: {self.use_quantum}")
        print(f"  Method: {self.quantum_method}")
        print(f"  Framework: {self.quantum_framework}")
        print(f"  Qubits: {self.n_qubits}")
    
    def build_index(self, documents: List[str], batch_size: int = 100):
        """
        Build the vector index from documents.
        
        Args:
            documents: List of documents to index
            batch_size: Batch size for processing
        """
        print(f"Building index with {len(documents)} documents...")
        
        # Create collection
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        
        # Process documents in batches
        points = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            # Generate embeddings
            embeddings = self.embedder.encode(batch)
            
            # Create points
            for j, (doc, embedding) in enumerate(zip(batch, embeddings)):
                point_id = i + j
                points.append(PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload={"text": doc, "doc_id": point_id}
                ))
        
        # Upload to vector database
        self.client.upsert(collection_name=self.collection_name, points=points)
        print(f"✅ Indexed {len(points)} documents")
    
    def retrieve_documents(
        self, 
        query: str, 
        top_k: int = 10,
        rerank_with_quantum: bool = True
    ) -> Tuple[List[str], List[float]]:
        """
        Retrieve and optionally rerank documents using quantum similarity.
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve
            rerank_with_quantum: Whether to use quantum reranking
            
        Returns:
            Tuple of (documents, scores)
        """
        start_time = time.time()
        
        # Classical retrieval
        query_embedding = self.embedder.encode(query)
        
        # Retrieve more documents than needed for reranking
        retrieve_k = top_k * 2 if rerank_with_quantum and self.use_quantum else top_k
        
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=retrieve_k
        )
        
        documents = [hit.payload["text"] for hit in results]
        classical_scores = [hit.score for hit in results]
        
        # Quantum reranking
        if rerank_with_quantum and self.use_quantum and len(documents) > 0:
            quantum_start = time.time()
            
            try:
                # Compute quantum similarities
                quantum_scores = []
                for doc in documents:
                    doc_embedding = self.embedder.encode(doc)
                    
                    if self.quantum_framework == "pennylane":
                        q_sim = quantum_similarity_pennylane(
                            query_embedding, doc_embedding, 
                            self.quantum_method, self.n_qubits
                        )
                    else:
                        # Fallback to classical for now
                        q_sim = np.dot(query_embedding, doc_embedding) / (
                            np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                        )
                    
                    quantum_scores.append(q_sim)
                
                # Combine classical and quantum scores
                combined_scores = []
                for c_score, q_score in zip(classical_scores, quantum_scores):
                    combined = (self.classical_weight * c_score + 
                              self.quantum_weight * q_score)
                    combined_scores.append(combined)
                
                # Rerank by combined scores
                doc_score_pairs = list(zip(documents, combined_scores))
                doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
                
                documents = [doc for doc, _ in doc_score_pairs[:top_k]]
                final_scores = [score for _, score in doc_score_pairs[:top_k]]
                
                quantum_time = time.time() - quantum_start
                self.performance_stats["avg_quantum_time"] = (
                    self.performance_stats["avg_quantum_time"] * self.performance_stats["total_queries"] +
                    quantum_time
                ) / (self.performance_stats["total_queries"] + 1)
                
            except Exception as e:
                print(f"Quantum reranking failed: {e}")
                # Fall back to classical results
                documents = documents[:top_k]
                final_scores = classical_scores[:top_k]
        else:
            documents = documents[:top_k]
            final_scores = classical_scores[:top_k]
        
        retrieval_time = time.time() - start_time
        self.performance_stats["avg_retrieval_time"] = (
            self.performance_stats["avg_retrieval_time"] * self.performance_stats["total_queries"] +
            retrieval_time
        ) / (self.performance_stats["total_queries"] + 1)
        
        return documents, final_scores
    
    def generate_answer(
        self, 
        query: str, 
        contexts: List[str],
        max_context_length: int = 600,
        max_new_tokens: int = 100
    ) -> str:
        """
        Generate answer using retrieved contexts.
        
        Args:
            query: User query
            contexts: Retrieved context documents
            max_context_length: Maximum length of context to use
            max_new_tokens: Maximum new tokens to generate
            
        Returns:
            Generated answer
        """
        start_time = time.time()
        
        # Prepare context
        context_text = " ".join(contexts)[:max_context_length]
        
        # Create prompt
        prompt = (
            f"Question: {query}\n"
            f"Context: {context_text}\n"
            f"Instruction: Provide a clear, concise answer based on the context. "
            f"Focus on agricultural information and be specific.\n"
            f"Answer:"
        )
        
        # Generate answer
        inputs = self.gen_tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        ).to(self.device)
        
        outputs = self.gen_model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=self.gen_tokenizer.eos_token_id
        )
        
        answer = self.gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the answer part
        if "Answer:" in answer:
            answer = answer.split("Answer:")[-1].strip()
        
        generation_time = time.time() - start_time
        self.performance_stats["avg_generation_time"] = (
            self.performance_stats["avg_generation_time"] * self.performance_stats["total_queries"] +
            generation_time
        ) / (self.performance_stats["total_queries"] + 1)
        
        return answer
    
    def query(
        self, 
        query: str, 
        top_k: int = 5,
        return_contexts: bool = False,
        use_quantum_rerank: bool = None
    ) -> str:
        """
        Process a query through the full RAG pipeline.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            return_contexts: Whether to return contexts along with answer
            use_quantum_rerank: Override quantum reranking setting
            
        Returns:
            Generated answer (and contexts if requested)
        """
        if use_quantum_rerank is None:
            use_quantum_rerank = self.use_quantum
        
        # Retrieve documents
        documents, scores = self.retrieve_documents(
            query, top_k, use_quantum_rerank
        )
        
        # Generate answer
        answer = self.generate_answer(query, documents)
        
        # Update stats
        self.performance_stats["total_queries"] += 1
        
        if return_contexts:
            return answer, documents
        else:
            return answer
    
    def benchmark_performance(
        self, 
        test_queries: List[str],
        methods_to_compare: List[str] = ["classical", "quantum", "hybrid"]
    ) -> Dict[str, Any]:
        """
        Benchmark different retrieval methods.
        
        Args:
            test_queries: Queries to test with
            methods_to_compare: Methods to compare
            
        Returns:
            Benchmark results
        """
        print("Running performance benchmark...")
        
        results = {}
        
        for method in methods_to_compare:
            print(f"Testing {method} method...")
            
            # Configure method
            if method == "classical":
                use_quantum = False
                weights = (1.0, 0.0)
            elif method == "quantum":
                use_quantum = True
                weights = (0.0, 1.0)
            else:  # hybrid
                use_quantum = True
                weights = (0.5, 0.5)
            
            # Store original settings
            orig_use_quantum = self.use_quantum
            orig_weights = (self.classical_weight, self.quantum_weight)
            
            # Apply test settings
            self.use_quantum = use_quantum
            self.classical_weight, self.quantum_weight = weights
            
            # Measure performance
            start_time = time.time()
            answers = []
            
            for query in test_queries:
                try:
                    answer = self.query(query)
                    answers.append(answer)
                except Exception as e:
                    print(f"Error with query '{query}': {e}")
                    answers.append("")
            
            end_time = time.time()
            
            # Record results
            results[method] = {
                "total_time": end_time - start_time,
                "avg_time_per_query": (end_time - start_time) / len(test_queries),
                "successful_queries": len([a for a in answers if a]),
                "answers": answers
            }
            
            # Restore original settings
            self.use_quantum = orig_use_quantum
            self.classical_weight, self.quantum_weight = orig_weights
        
        return results
    
    def comprehensive_evaluation(
        self,
        test_queries: List[str],
        reference_answers: List[str],
        relevant_docs: List[List[str]] = None,
        include_human_eval: bool = False,
        output_dir: str = "evaluation_results"
    ) -> Dict[str, Any]:
        """
        Run comprehensive evaluation of the RAG system.
        
        Args:
            test_queries: Test queries
            reference_answers: Reference answers for comparison
            relevant_docs: Relevant documents for each query (for retrieval eval)
            include_human_eval: Whether to set up human evaluation
            output_dir: Directory to save results
            
        Returns:
            Comprehensive evaluation results
        """
        print("Running comprehensive evaluation...")
        
        # Generate answers and collect contexts
        generated_answers = []
        retrieved_contexts = []
        
        for query in test_queries:
            answer, contexts = self.query(query, return_contexts=True)
            generated_answers.append(answer)
            retrieved_contexts.append(contexts)
        
        # Evaluate answer quality
        answer_metrics = evaluate_answer_quality(
            queries=test_queries,
            contexts=retrieved_contexts,
            generated_answers=generated_answers,
            reference_answers=reference_answers
        )
        
        # Evaluate retrieval (if relevant docs provided)
        retrieval_metrics = {}
        if relevant_docs:
            # Convert contexts to doc IDs for evaluation
            retrieved_doc_ids = []
            for contexts in retrieved_contexts:
                doc_ids = [f"doc_{hash(ctx[:50])}" for ctx in contexts]
                retrieved_doc_ids.append(doc_ids)
            
            retrieval_metrics = evaluate_retrieval_performance(
                queries=test_queries,
                retrieved_docs=retrieved_doc_ids,
                relevant_docs=relevant_docs
            )
        
        # Performance benchmarking
        performance_results = self.benchmark_performance(test_queries[:5])  # Sample for speed
        
        # Human evaluation setup
        human_eval_info = {}
        if include_human_eval:
            eval_framework = HumanEvaluationFramework()
            eval_framework.conduct_full_evaluation(
                queries=test_queries,
                contexts=retrieved_contexts,
                system_answers={"HybridQuantumRAG": generated_answers},
                output_dir=output_dir
            )
            human_eval_info = {
                "evaluation_forms_generated": True,
                "output_directory": output_dir
            }
        
        # Compile results
        evaluation_results = {
            "system_info": {
                "quantum_enabled": self.use_quantum,
                "quantum_method": self.quantum_method,
                "quantum_framework": self.quantum_framework,
                "n_qubits": self.n_qubits,
                "weights": (self.classical_weight, self.quantum_weight)
            },
            "answer_quality_metrics": answer_metrics,
            "retrieval_metrics": retrieval_metrics,
            "performance_results": performance_results,
            "performance_stats": self.performance_stats,
            "human_evaluation": human_eval_info,
            "generated_answers": generated_answers,
            "retrieved_contexts": retrieved_contexts
        }
        
        # Save results
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        results_file = Path(output_dir) / "comprehensive_evaluation.json"
        with open(results_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        
        print(f"Evaluation results saved to {results_file}")
        
        return evaluation_results
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get current performance statistics."""
        return self.performance_stats.copy()
    
    def compare_quantum_methods(
        self, 
        test_queries: List[str],
        methods: List[str] = ["angle", "amplitude", "iqp"]
    ) -> Dict[str, Any]:
        """
        Compare different quantum embedding methods.
        
        Args:
            test_queries: Queries to test
            methods: Quantum methods to compare
            
        Returns:
            Comparison results
        """
        if not self.use_quantum:
            print("Quantum not enabled. Cannot compare quantum methods.")
            return {}
        
        print("Comparing quantum embedding methods...")
        
        original_method = self.quantum_method
        results = {}
        
        for method in methods:
            print(f"Testing {method} embedding...")
            
            # Switch method
            self.quantum_method = method
            
            # Measure performance
            start_time = time.time()
            answers = []
            
            for query in test_queries[:5]:  # Sample for speed
                try:
                    answer = self.query(query)
                    answers.append(answer)
                except Exception as e:
                    print(f"Error with {method} method: {e}")
                    answers.append("")
            
            end_time = time.time()
            
            results[method] = {
                "total_time": end_time - start_time,
                "avg_time": (end_time - start_time) / len(test_queries[:5]),
                "successful_queries": len([a for a in answers if a]),
                "answers": answers
            }
        
        # Restore original method
        self.quantum_method = original_method
        
        return results


def create_sample_rag_system() -> HybridQuantumRAG:
    """
    Create a sample quantum RAG system with agricultural data.
    
    Returns:
        Configured quantum RAG system
    """
    print("Creating sample Quantum RAG system...")
    
    # Sample agricultural documents
    agricultural_docs = [
        "Rice cultivation requires flooded fields and warm temperatures between 20-30°C for optimal growth.",
        "Wheat should be planted in fall for winter varieties or spring for spring varieties, depending on climate.",
        "Corn requires well-drained soil with pH between 6.0-6.8 and adequate nitrogen fertilization.",
        "Tomatoes need consistent watering and temperatures between 18-24°C to prevent blossom end rot.",
        "Crop rotation helps prevent soil depletion and reduces pest and disease pressure in agricultural systems.",
        "Integrated pest management combines biological, cultural, and chemical methods for sustainable farming.",
        "Drip irrigation systems can reduce water usage by 30-50% compared to traditional flood irrigation.",
        "Cover crops like clover and rye grass improve soil health by adding organic matter and nitrogen.",
        "Precision agriculture uses GPS and sensors to optimize input application and increase efficiency.",
        "Organic farming prohibits synthetic pesticides and fertilizers, relying on natural methods instead."
    ]
    
    # Initialize system
    rag_system = HybridQuantumRAG(
        collection_name="sample_agri_rag",
        quantum_method="angle",
        use_quantum=True,
        n_qubits=4
    )
    
    # Build index
    rag_system.build_index(agricultural_docs)
    
    print("Sample system ready!")
    return rag_system


def main():
    """Main function to demonstrate the quantum RAG system."""
    print("🌾 Hybrid Quantum RAG System Demo")
    print("=" * 50)
    
    # Create sample system
    rag = create_sample_rag_system()
    
    # Sample queries
    test_queries = [
        "When should I plant rice?",
        "How do I manage water for crops?",
        "What is crop rotation?",
        "How can I reduce pesticide use?",
        "What temperature does corn need?"
    ]
    
    print("\n🔍 Testing queries...")
    for query in test_queries:
        print(f"\nQ: {query}")
        answer = rag.query(query)
        print(f"A: {answer}")
    
    print("\n📊 Performance benchmark...")
    benchmark_results = rag.benchmark_performance(test_queries)
    
    for method, results in benchmark_results.items():
        print(f"\n{method.upper()} Method:")
        print(f"  Average time: {results['avg_time_per_query']:.3f}s")
        print(f"  Success rate: {results['successful_queries']}/{len(test_queries)}")
    
    print("\n✅ Demo completed!")


if __name__ == "__main__":
    main()