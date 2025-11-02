"""
Evaluation Module for Quantum RAG

This module provides comprehensive evaluation metrics for comparing
classical and quantum RAG systems.
"""

from .retrieval_metrics import (
    compute_precision_at_k,
    compute_recall_at_k,
    compute_map_score,
    compute_ndcg_score,
    compute_mrr_score,
    evaluate_retrieval_performance
)

from .answer_quality import (
    compute_bleu_score,
    compute_rouge_scores,
    compute_bert_score,
    evaluate_answer_quality,
    AgricultureQAEvaluator
)

from .performance_metrics import (
    LatencyProfiler,
    MemoryProfiler,
    QuantumCircuitProfiler,
    compute_efficiency_metrics
)

from .human_eval import (
    HumanEvaluationFramework,
    RelevanceScorer,
    FactualAccuracyEvaluator,
    ComparativePreferenceTest
)

__all__ = [
    'compute_precision_at_k',
    'compute_recall_at_k', 
    'compute_map_score',
    'compute_ndcg_score',
    'compute_mrr_score',
    'evaluate_retrieval_performance',
    'compute_bleu_score',
    'compute_rouge_scores',
    'compute_bert_score',
    'evaluate_answer_quality',
    'AgricultureQAEvaluator',
    'LatencyProfiler',
    'MemoryProfiler',
    'QuantumCircuitProfiler',
    'compute_efficiency_metrics',
    'HumanEvaluationFramework',
    'RelevanceScorer',
    'FactualAccuracyEvaluator', 
    'ComparativePreferenceTest'
]