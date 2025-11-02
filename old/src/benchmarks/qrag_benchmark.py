"""
Comprehensive QRAG Benchmark Suite

This module provides the main benchmarking framework for evaluating
and comparing classical vs quantum RAG systems.
"""

import time
import json
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass, asdict
from pathlib import Path

# Import evaluation modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.retrieval_metrics import evaluate_retrieval_performance
from evaluation.answer_quality import evaluate_answer_quality, AgricultureQAEvaluator
from evaluation.performance_metrics import benchmark_rag_pipeline, compute_efficiency_metrics
from evaluation.human_eval import HumanEvaluationFramework


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    test_queries: List[str]
    reference_answers: List[str]
    relevant_docs: List[List[str]]
    corpus: List[str]
    
    # Evaluation settings
    include_human_eval: bool = False
    include_bert_score: bool = True
    k_values: List[int] = None
    
    # Performance settings
    benchmark_duration: float = 300.0  # 5 minutes
    warmup_queries: int = 3
    
    # Output settings
    output_dir: str = "benchmark_results"
    save_detailed_results: bool = True
    
    def __post_init__(self):
        if self.k_values is None:
            self.k_values = [1, 3, 5, 10]


@dataclass
class BenchmarkResults:
    """Container for benchmark results."""
    system_name: str
    timestamp: float
    
    # Performance metrics
    retrieval_metrics: Dict[str, float]
    answer_quality_metrics: Dict[str, float]
    performance_metrics: Dict[str, float]
    
    # Additional metrics
    agricultural_metrics: Optional[Dict[str, float]] = None
    human_evaluation_results: Optional[Dict[str, Any]] = None
    
    # Raw data
    generated_answers: List[str] = None
    retrieved_contexts: List[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkResults':
        """Create from dictionary."""
        return cls(**data)


class QRAGBenchmark:
    """
    Main benchmark class for evaluating RAG systems.
    """
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.agriculture_evaluator = AgricultureQAEvaluator()
        self.human_eval_framework = HumanEvaluationFramework()
        
        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def run_full_benchmark(
        self, 
        rag_system: Callable[[str], Tuple[str, List[str]]],
        system_name: str
    ) -> BenchmarkResults:
        """
        Run comprehensive benchmark on a RAG system.
        
        Args:
            rag_system: Function that takes a query and returns (answer, contexts)
            system_name: Name of the system being benchmarked
            
        Returns:
            Benchmark results
        """
        print(f"Starting comprehensive benchmark for {system_name}...")
        
        # Generate answers and collect contexts
        print("Generating answers...")
        generated_answers = []
        retrieved_contexts = []
        
        for query in self.config.test_queries:
            try:
                answer, contexts = rag_system(query)
                generated_answers.append(answer)
                retrieved_contexts.append(contexts)
            except Exception as e:
                print(f"Error processing query '{query}': {e}")
                generated_answers.append("")
                retrieved_contexts.append([])
        
        # Run retrieval evaluation
        print("Evaluating retrieval performance...")
        retrieval_metrics = self._evaluate_retrieval(retrieved_contexts)
        
        # Run answer quality evaluation
        print("Evaluating answer quality...")
        answer_quality_metrics = self._evaluate_answer_quality(
            generated_answers, retrieved_contexts
        )
        
        # Run performance benchmarking
        print("Benchmarking performance...")
        performance_metrics = self._benchmark_performance(rag_system)
        
        # Agricultural domain evaluation
        print("Evaluating agricultural domain specifics...")
        agricultural_metrics = self._evaluate_agricultural_domain(generated_answers)
        
        # Human evaluation (if requested)
        human_evaluation_results = None
        if self.config.include_human_eval:
            print("Setting up human evaluation...")
            human_evaluation_results = self._setup_human_evaluation(
                generated_answers, retrieved_contexts, system_name
            )
        
        # Compile results
        results = BenchmarkResults(
            system_name=system_name,
            timestamp=time.time(),
            retrieval_metrics=retrieval_metrics,
            answer_quality_metrics=answer_quality_metrics,
            performance_metrics=performance_metrics,
            agricultural_metrics=agricultural_metrics,
            human_evaluation_results=human_evaluation_results,
            generated_answers=generated_answers,
            retrieved_contexts=retrieved_contexts
        )
        
        # Save results
        if self.config.save_detailed_results:
            self._save_results(results)
        
        print(f"Benchmark completed for {system_name}")
        return results
    
    def _evaluate_retrieval(
        self, 
        retrieved_contexts: List[List[str]]
    ) -> Dict[str, float]:
        """Evaluate retrieval performance."""
        # Convert contexts to document IDs for evaluation
        retrieved_docs = []
        for contexts in retrieved_contexts:
            # Simple approach: use first few words as document ID
            doc_ids = [ctx[:50] for ctx in contexts]
            retrieved_docs.append(doc_ids)
        
        return evaluate_retrieval_performance(
            queries=self.config.test_queries,
            retrieved_docs=retrieved_docs,
            relevant_docs=self.config.relevant_docs,
            k_values=self.config.k_values,
            corpus=self.config.corpus
        )
    
    def _evaluate_answer_quality(
        self,
        generated_answers: List[str],
        retrieved_contexts: List[List[str]]
    ) -> Dict[str, float]:
        """Evaluate answer quality."""
        return evaluate_answer_quality(
            queries=self.config.test_queries,
            contexts=retrieved_contexts,
            generated_answers=generated_answers,
            reference_answers=self.config.reference_answers,
            include_bert_score=self.config.include_bert_score
        )
    
    def _benchmark_performance(
        self,
        rag_system: Callable[[str], Tuple[str, List[str]]]
    ) -> Dict[str, float]:
        """Benchmark system performance."""
        # Wrapper function for performance testing
        def pipeline_func(query: str) -> str:
            answer, _ = rag_system(query)
            return answer
        
        results = benchmark_rag_pipeline(
            pipeline_func,
            self.config.test_queries,
            duration=self.config.benchmark_duration
        )
        
        # Extract relevant metrics
        performance_metrics = {}
        if "latency_metrics" in results:
            performance_metrics.update(results["latency_metrics"])
        if "memory_metrics" in results:
            performance_metrics.update(results["memory_metrics"])
        if "throughput_metrics" in results:
            performance_metrics.update(results["throughput_metrics"])
        
        return performance_metrics
    
    def _evaluate_agricultural_domain(
        self,
        generated_answers: List[str]
    ) -> Dict[str, float]:
        """Evaluate agricultural domain-specific metrics."""
        return self.agriculture_evaluator.evaluate_agricultural_answers(
            queries=self.config.test_queries,
            generated_answers=generated_answers,
            reference_answers=self.config.reference_answers
        )
    
    def _setup_human_evaluation(
        self,
        generated_answers: List[str],
        retrieved_contexts: List[List[str]],
        system_name: str
    ) -> Dict[str, str]:
        """Setup human evaluation forms."""
        human_eval_dir = Path(self.config.output_dir) / "human_evaluation" / system_name
        human_eval_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate evaluation forms
        self.human_eval_framework.conduct_full_evaluation(
            queries=self.config.test_queries,
            contexts=retrieved_contexts,
            system_answers={system_name: generated_answers},
            output_dir=str(human_eval_dir)
        )
        
        return {
            "evaluation_dir": str(human_eval_dir),
            "forms_generated": True,
            "instructions": "Open HTML files in browser and complete evaluations"
        }
    
    def _save_results(self, results: BenchmarkResults):
        """Save benchmark results to files."""
        timestamp_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(results.timestamp))
        
        # Save main results
        results_file = Path(self.config.output_dir) / f"{results.system_name}_{timestamp_str}.json"
        with open(results_file, 'w') as f:
            json.dump(results.to_dict(), f, indent=2, default=str)
        
        # Save summary report
        report_file = Path(self.config.output_dir) / f"{results.system_name}_{timestamp_str}_report.txt"
        with open(report_file, 'w') as f:
            f.write(self._generate_summary_report(results))
        
        print(f"Results saved to {results_file}")
        print(f"Report saved to {report_file}")
    
    def _generate_summary_report(self, results: BenchmarkResults) -> str:
        """Generate a human-readable summary report."""
        report = f"""
{'='*60}
Benchmark Report: {results.system_name}
{'='*60}
Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(results.timestamp))}

RETRIEVAL METRICS
{'-'*20}
"""
        
        for metric, value in results.retrieval_metrics.items():
            report += f"{metric}: {value:.4f}\n"
        
        report += f"""
ANSWER QUALITY METRICS
{'-'*22}
"""
        
        for metric, value in results.answer_quality_metrics.items():
            report += f"{metric}: {value:.4f}\n"
        
        report += f"""
PERFORMANCE METRICS
{'-'*19}
"""
        
        for metric, value in results.performance_metrics.items():
            report += f"{metric}: {value:.4f}\n"
        
        if results.agricultural_metrics:
            report += f"""
AGRICULTURAL DOMAIN METRICS
{'-'*27}
"""
            for metric, value in results.agricultural_metrics.items():
                report += f"{metric}: {value:.4f}\n"
        
        return report


def create_benchmark_config(
    queries_file: str,
    answers_file: str,
    relevance_file: str,
    corpus_file: str,
    **kwargs
) -> BenchmarkConfig:
    """
    Create benchmark config from files.
    
    Args:
        queries_file: Path to queries file (one per line)
        answers_file: Path to reference answers file (one per line)
        relevance_file: Path to relevance judgments file (JSON)
        corpus_file: Path to corpus file (one document per line)
        **kwargs: Additional config parameters
        
    Returns:
        Benchmark configuration
    """
    # Load queries
    with open(queries_file, 'r', encoding='utf-8') as f:
        test_queries = [line.strip() for line in f if line.strip()]
    
    # Load reference answers
    with open(answers_file, 'r', encoding='utf-8') as f:
        reference_answers = [line.strip() for line in f if line.strip()]
    
    # Load corpus
    with open(corpus_file, 'r', encoding='utf-8') as f:
        corpus = [line.strip() for line in f if line.strip()]
    
    # Load relevance judgments
    with open(relevance_file, 'r', encoding='utf-8') as f:
        relevance_data = json.load(f)
    
    # Extract relevant documents for each query
    relevant_docs = []
    for query in test_queries:
        if query in relevance_data:
            relevant = [doc for doc, score in relevance_data[query].items() if score > 0]
            relevant_docs.append(relevant)
        else:
            relevant_docs.append([])
    
    return BenchmarkConfig(
        test_queries=test_queries,
        reference_answers=reference_answers,
        relevant_docs=relevant_docs,
        corpus=corpus,
        **kwargs
    )


def run_quantum_vs_classical_benchmark(
    classical_rag_func: Callable[[str], Tuple[str, List[str]]],
    quantum_rag_func: Callable[[str], Tuple[str, List[str]]],
    config: BenchmarkConfig
) -> Tuple[BenchmarkResults, BenchmarkResults]:
    """
    Run comparative benchmark between classical and quantum RAG.
    
    Args:
        classical_rag_func: Classical RAG system function
        quantum_rag_func: Quantum RAG system function
        config: Benchmark configuration
        
    Returns:
        Tuple of (classical_results, quantum_results)
    """
    benchmark = QRAGBenchmark(config)
    
    print("Running classical RAG benchmark...")
    classical_results = benchmark.run_full_benchmark(classical_rag_func, "Classical_RAG")
    
    print("Running quantum RAG benchmark...")
    quantum_results = benchmark.run_full_benchmark(quantum_rag_func, "Quantum_RAG")
    
    # Generate comparative analysis
    print("Generating comparative analysis...")
    comparison = compute_efficiency_metrics(
        classical_results.performance_metrics,
        quantum_results.performance_metrics
    )
    
    # Save comparison
    comparison_file = Path(config.output_dir) / "classical_vs_quantum_comparison.json"
    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=2, default=str)
    
    print(f"Comparative analysis saved to {comparison_file}")
    
    return classical_results, quantum_results


def create_sample_agricultural_benchmark() -> BenchmarkConfig:
    """
    Create a sample agricultural benchmark configuration.
    
    Returns:
        Sample benchmark configuration
    """
    sample_queries = [
        "What is the best time to plant rice?",
        "How do you prevent wheat diseases?",
        "What are sustainable farming practices?",
        "How does crop rotation benefit soil health?",
        "What fertilizers are best for corn growth?",
        "How do you manage irrigation for tomatoes?",
        "What are the effects of climate change on agriculture?",
        "How do you control pests organically?",
        "What are the benefits of companion planting?",
        "How do you improve soil fertility naturally?"
    ]
    
    sample_answers = [
        "The best time to plant rice is during the wet season when temperatures are warm and water is abundant.",
        "Wheat diseases can be prevented through crop rotation, resistant varieties, and proper field sanitation.",
        "Sustainable farming practices include organic methods, crop rotation, and reduced chemical inputs.",
        "Crop rotation benefits soil health by preventing nutrient depletion and breaking disease cycles.",
        "The best fertilizers for corn include nitrogen-rich fertilizers applied at planting and during growth.",
        "Tomato irrigation should be consistent and deep, avoiding water on leaves to prevent disease.",
        "Climate change affects agriculture through altered rainfall patterns and increased temperatures.",
        "Organic pest control uses beneficial insects, companion planting, and natural repellents.",
        "Companion planting benefits include natural pest control and improved soil nutrients.",
        "Soil fertility can be improved through composting, cover crops, and organic matter addition."
    ]
    
    # Simple relevance judgments (in practice, this would be more comprehensive)
    relevance_judgments = {}
    for i, query in enumerate(sample_queries):
        relevance_judgments[query] = {
            f"doc_{i}_relevant": 1.0,
            f"doc_{i}_somewhat": 0.5,
            f"doc_{i}_irrelevant": 0.0
        }
    
    sample_corpus = [
        "Rice cultivation requires warm temperatures and adequate water supply throughout the growing season.",
        "Wheat is susceptible to various diseases that can be managed through integrated pest management.",
        "Sustainable agriculture focuses on long-term productivity while protecting the environment.",
        "Crop rotation is a fundamental practice that maintains soil health and reduces pest pressure.",
        "Corn requires significant nitrogen inputs for optimal growth and yield production.",
        "Proper irrigation management is crucial for tomato production and disease prevention.",
        "Climate change poses significant challenges to global agricultural production systems.",
        "Integrated pest management combines biological, cultural, and chemical control methods.",
        "Companion planting utilizes beneficial plant interactions to improve crop performance.",
        "Soil health is fundamental to sustainable agriculture and long-term productivity."
    ]
    
    # Create temporary files for the sample data
    import tempfile
    temp_dir = tempfile.mkdtemp()
    
    queries_file = Path(temp_dir) / "sample_queries.txt"
    with open(queries_file, 'w') as f:
        f.write('\n'.join(sample_queries))
    
    answers_file = Path(temp_dir) / "sample_answers.txt"
    with open(answers_file, 'w') as f:
        f.write('\n'.join(sample_answers))
    
    relevance_file = Path(temp_dir) / "sample_relevance.json"
    with open(relevance_file, 'w') as f:
        json.dump(relevance_judgments, f, indent=2)
    
    corpus_file = Path(temp_dir) / "sample_corpus.txt"
    with open(corpus_file, 'w') as f:
        f.write('\n'.join(sample_corpus))
    
    return create_benchmark_config(
        str(queries_file),
        str(answers_file),
        str(relevance_file),
        str(corpus_file),
        output_dir="sample_benchmark_results"
    )