# Quantum RAG Quick Start Guide

## 🚀 Installation & Setup

### Option 1: Run Demo Immediately (No Dependencies)

```bash
# The demo works without any external dependencies!
python demo_quantum_rag.py
```

### Option 2: Full Installation for Production Use

```bash
# Install all dependencies for full functionality
pip install -r requirements.txt
```

### Option 3: Minimal Installation

```bash
# Core dependencies only
pip install sentence-transformers transformers qdrant-client numpy scikit-learn
```

## 🔥 Quick Demo

```python
from src.hybrid_qrag import HybridQuantumRAG, create_sample_rag_system

# Create a sample system (works out of the box)
rag = create_sample_rag_system()

# Ask questions
answer = rag.query("When should I plant rice?")
print(f"Answer: {answer}")

# Compare different methods
results = rag.benchmark_performance([
    "When should I plant rice?",
    "How do I manage water for crops?",
    "What are organic farming practices?"
])
```

## 📊 Run Comprehensive Evaluation

```python
from src.benchmarks import QRAGBenchmark, create_sample_agricultural_benchmark

# Create benchmark configuration
config = create_sample_agricultural_benchmark()

# Initialize benchmark
benchmark = QRAGBenchmark(config)

# Run evaluation (replace with your RAG function)
def my_rag_function(query):
    answer = "Your generated answer"
    contexts = ["Retrieved context 1", "Retrieved context 2"]
    return answer, contexts

results = benchmark.run_full_benchmark(my_rag_function, "MySystem")
```

## 🔮 Compare Quantum Methods

```python
from src.quantum_embeddings import compare_embeddings

# Compare different quantum embedding approaches
embeddings = [/* your text embeddings */]
methods = ["classical", "angle", "amplitude", "iqp"]

comparison = compare_embeddings(
    embeddings,
    methods=methods,
    n_qubits=4
)

for method, stats in comparison.items():
    print(f"{method}: {stats['mean_similarity']:.3f}")
```

## 👥 Human Evaluation

```python
from src.evaluation import HumanEvaluationFramework

# Setup human evaluation
evaluator = HumanEvaluationFramework()

queries = ["Your test queries"]
contexts = [["context 1", "context 2"]]  # Retrieved contexts for each query
system_answers = {
    "Classical": ["classical answer 1", ...],
    "Quantum": ["quantum answer 1", ...]
}

# Generate evaluation forms
evaluator.conduct_full_evaluation(
    queries=queries,
    contexts=contexts,
    system_answers=system_answers,
    output_dir="human_eval_forms"
)
# Open the generated HTML files in a browser to conduct evaluation
```

## 📈 Performance Benchmarking

```python
from src.evaluation import benchmark_rag_pipeline

def your_rag_pipeline(query):
    # Your implementation here
    return "Generated answer"

# Benchmark performance
results = benchmark_rag_pipeline(
    your_rag_pipeline,
    test_queries=["query1", "query2", "query3"],
    system_name="MyRAG",
    duration=300  # 5 minutes
)

print(f"Average latency: {results['latency_metrics']['mean_latency']:.3f}s")
print(f"Throughput: {results['throughput_metrics']['throughput_ops_per_sec']:.1f} ops/sec")
```

## 🌾 Agricultural Domain Evaluation

```python
from src.evaluation import AgricultureQAEvaluator

evaluator = AgricultureQAEvaluator()

queries = ["When to plant corn?", "How to prevent crop diseases?"]
generated_answers = ["Plant corn in spring...", "Use integrated pest management..."]
reference_answers = ["Corn should be planted...", "Disease prevention involves..."]

metrics = evaluator.evaluate_agricultural_answers(
    queries, generated_answers, reference_answers
)

print(f"Domain relevance: {metrics['generated_domain_relevance']:.3f}")
print(f"Specificity: {metrics['generated_specificity']:.3f}")
```

## 🔧 Configuration Options

```python
# Initialize with custom settings
rag = HybridQuantumRAG(
    quantum_method="iqp",           # "angle", "amplitude", "iqp", "data_reuploading"
    quantum_framework="pennylane",  # "pennylane" or "qiskit"
    n_qubits=6,                     # Number of qubits (2-10 recommended)
    classical_weight=0.7,           # Weight for classical similarity (0.0-1.0)
    quantum_weight=0.3,             # Weight for quantum similarity (0.0-1.0)
    use_quantum=True                # Enable/disable quantum features
)
```

## 📁 File Structure

```
agri_qrag/
├── src/
│   ├── quantum_embeddings/     # Quantum embedding implementations
│   ├── evaluation/             # Evaluation metrics and tools
│   ├── benchmarks/             # Benchmarking framework
│   ├── hybrid_qrag.py          # Main quantum RAG system
│   └── demo_quantum_rag.py     # Working demo (run this!)
├── requirements.txt            # Dependencies
├── QUANTUM_RAG_TODO.md        # Project roadmap (completed ✅)
└── IMPLEMENTATION_SUMMARY.md  # Detailed implementation overview
```

## 🎯 Key Features

### ✅ Quantum Embeddings

- **5 Methods**: Angle, Amplitude, IQP, Data-reuploading, Trainable
- **2 Frameworks**: PennyLane and Qiskit support
- **Hybrid Scoring**: Configurable classical/quantum weights

### ✅ Comprehensive Evaluation

- **Retrieval Metrics**: MAP, NDCG, Precision@K, Recall@K, MRR
- **Answer Quality**: BLEU, ROUGE, BERTScore, Semantic Similarity
- **Performance**: Latency, Memory, Throughput profiling
- **Domain-Specific**: Agricultural relevance and specificity

### ✅ Human Evaluation

- **HTML Forms**: Auto-generated evaluation interfaces
- **Multiple Types**: Relevance, accuracy, comparative preference
- **Result Analysis**: Automated processing of evaluation results

### ✅ Benchmarking

- **Cross-Method**: Compare classical vs quantum vs hybrid
- **Performance vs Quality**: Analyze trade-offs
- **Agricultural Focus**: Crop and farming-specific metrics

## 🚨 Troubleshooting

### Missing Dependencies

```python
# The system gracefully handles missing dependencies
# Core demo works without any external libraries
# Full features require: pip install -r requirements.txt
```

### Quantum Libraries Not Available

```python
# System automatically falls back to classical mode
# Quantum features are optional enhancements
# Demo script works without PennyLane/Qiskit
```

### Memory Issues

```python
# Reduce n_qubits for lower memory usage
# Use batch processing for large document sets
# Enable caching for repeated queries
```

## 🏆 Success!

The Quantum RAG system is fully implemented and ready to use. Start with `demo_quantum_rag.py` to see it in action, then integrate the components into your own agricultural RAG system!

**Happy quantum computing! 🌾🔮✨**
