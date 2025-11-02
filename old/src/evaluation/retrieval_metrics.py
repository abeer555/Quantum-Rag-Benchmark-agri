"""
Retrieval Metrics for RAG Evaluation

This module provides metrics for evaluating retrieval performance
in RAG systems.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict


def compute_precision_at_k(
    retrieved_docs: List[List[str]], 
    relevant_docs: List[List[str]], 
    k: int
) -> float:
    """
    Compute Precision@K for retrieved documents.
    
    Args:
        retrieved_docs: List of retrieved document lists for each query
        relevant_docs: List of relevant document lists for each query
        k: Number of top documents to consider
        
    Returns:
        Average Precision@K across all queries
    """
    precisions = []
    
    for retrieved, relevant in zip(retrieved_docs, relevant_docs):
        if not relevant:  # No relevant documents for this query
            continue
            
        # Take top k retrieved documents
        top_k = retrieved[:k]
        relevant_set = set(relevant)
        
        # Count relevant documents in top k
        relevant_in_k = sum(1 for doc in top_k if doc in relevant_set)
        
        # Precision@K = relevant_in_k / k
        precision = relevant_in_k / k
        precisions.append(precision)
    
    return np.mean(precisions) if precisions else 0.0


def compute_recall_at_k(
    retrieved_docs: List[List[str]], 
    relevant_docs: List[List[str]], 
    k: int
) -> float:
    """
    Compute Recall@K for retrieved documents.
    
    Args:
        retrieved_docs: List of retrieved document lists for each query
        relevant_docs: List of relevant document lists for each query
        k: Number of top documents to consider
        
    Returns:
        Average Recall@K across all queries
    """
    recalls = []
    
    for retrieved, relevant in zip(retrieved_docs, relevant_docs):
        if not relevant:  # No relevant documents for this query
            continue
            
        # Take top k retrieved documents
        top_k = retrieved[:k]
        relevant_set = set(relevant)
        
        # Count relevant documents in top k
        relevant_in_k = sum(1 for doc in top_k if doc in relevant_set)
        
        # Recall@K = relevant_in_k / total_relevant
        recall = relevant_in_k / len(relevant_set)
        recalls.append(recall)
    
    return np.mean(recalls) if recalls else 0.0


def compute_map_score(
    retrieved_docs: List[List[str]], 
    relevant_docs: List[List[str]]
) -> float:
    """
    Compute Mean Average Precision (MAP).
    
    Args:
        retrieved_docs: List of retrieved document lists for each query
        relevant_docs: List of relevant document lists for each query
        
    Returns:
        MAP score
    """
    average_precisions = []
    
    for retrieved, relevant in zip(retrieved_docs, relevant_docs):
        if not relevant:  # No relevant documents for this query
            continue
            
        relevant_set = set(relevant)
        num_relevant = 0
        precision_sum = 0
        
        for i, doc in enumerate(retrieved):
            if doc in relevant_set:
                num_relevant += 1
                precision_at_i = num_relevant / (i + 1)
                precision_sum += precision_at_i
        
        if num_relevant > 0:
            average_precision = precision_sum / len(relevant_set)
            average_precisions.append(average_precision)
    
    return np.mean(average_precisions) if average_precisions else 0.0


def compute_ndcg_score(
    retrieved_docs: List[List[str]], 
    relevance_scores: List[Dict[str, float]], 
    k: Optional[int] = None
) -> float:
    """
    Compute Normalized Discounted Cumulative Gain (NDCG).
    
    Args:
        retrieved_docs: List of retrieved document lists for each query
        relevance_scores: List of relevance score dictionaries for each query
        k: Number of top documents to consider (None for all)
        
    Returns:
        Average NDCG@K across all queries
    """
    ndcg_scores = []
    
    for retrieved, relevance in zip(retrieved_docs, relevance_scores):
        if not relevance:  # No relevance scores for this query
            continue
        
        # Limit to top k if specified
        if k is not None:
            retrieved = retrieved[:k]
        
        # Compute DCG
        dcg = 0
        for i, doc in enumerate(retrieved):
            rel_score = relevance.get(doc, 0)
            dcg += rel_score / np.log2(i + 2)  # i+2 because log2(1) = 0
        
        # Compute IDCG (Ideal DCG)
        ideal_scores = sorted(relevance.values(), reverse=True)
        if k is not None:
            ideal_scores = ideal_scores[:k]
            
        idcg = 0
        for i, score in enumerate(ideal_scores):
            idcg += score / np.log2(i + 2)
        
        # Compute NDCG
        if idcg > 0:
            ndcg = dcg / idcg
            ndcg_scores.append(ndcg)
    
    return np.mean(ndcg_scores) if ndcg_scores else 0.0


def compute_mrr_score(
    retrieved_docs: List[List[str]], 
    relevant_docs: List[List[str]]
) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).
    
    Args:
        retrieved_docs: List of retrieved document lists for each query
        relevant_docs: List of relevant document lists for each query
        
    Returns:
        MRR score
    """
    reciprocal_ranks = []
    
    for retrieved, relevant in zip(retrieved_docs, relevant_docs):
        if not relevant:  # No relevant documents for this query
            continue
            
        relevant_set = set(relevant)
        
        # Find rank of first relevant document
        for i, doc in enumerate(retrieved):
            if doc in relevant_set:
                reciprocal_rank = 1.0 / (i + 1)
                reciprocal_ranks.append(reciprocal_rank)
                break
        else:
            # No relevant document found
            reciprocal_ranks.append(0.0)
    
    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0


def compute_coverage(
    retrieved_docs: List[List[str]], 
    corpus: List[str]
) -> float:
    """
    Compute coverage - fraction of corpus covered by retrievals.
    
    Args:
        retrieved_docs: List of retrieved document lists
        corpus: Complete document corpus
        
    Returns:
        Coverage score
    """
    all_retrieved = set()
    for retrieved in retrieved_docs:
        all_retrieved.update(retrieved)
    
    corpus_set = set(corpus)
    coverage = len(all_retrieved.intersection(corpus_set)) / len(corpus_set)
    
    return coverage


def compute_diversity(retrieved_docs: List[List[str]], k: int = 10) -> float:
    """
    Compute diversity of retrieved documents.
    
    Args:
        retrieved_docs: List of retrieved document lists
        k: Number of top documents to consider
        
    Returns:
        Average diversity score
    """
    diversity_scores = []
    
    for retrieved in retrieved_docs:
        top_k = retrieved[:k]
        
        if len(top_k) <= 1:
            diversity_scores.append(0.0)
            continue
        
        # Simple diversity metric: number of unique documents / k
        unique_docs = len(set(top_k))
        diversity = unique_docs / k
        diversity_scores.append(diversity)
    
    return np.mean(diversity_scores) if diversity_scores else 0.0


def evaluate_retrieval_performance(
    queries: List[str],
    retrieved_docs: List[List[str]],
    relevant_docs: List[List[str]],
    relevance_scores: Optional[List[Dict[str, float]]] = None,
    k_values: List[int] = [1, 3, 5, 10],
    corpus: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Comprehensive retrieval evaluation.
    
    Args:
        queries: List of query strings
        retrieved_docs: List of retrieved document lists for each query
        relevant_docs: List of relevant document lists for each query
        relevance_scores: List of relevance score dictionaries (optional)
        k_values: List of k values for Precision@K and Recall@K
        corpus: Complete document corpus (for coverage calculation)
        
    Returns:
        Dictionary of evaluation metrics
    """
    results = {}
    
    # Precision@K and Recall@K for different k values
    for k in k_values:
        results[f"precision_at_{k}"] = compute_precision_at_k(
            retrieved_docs, relevant_docs, k
        )
        results[f"recall_at_{k}"] = compute_recall_at_k(
            retrieved_docs, relevant_docs, k
        )
    
    # MAP
    results["map"] = compute_map_score(retrieved_docs, relevant_docs)
    
    # MRR
    results["mrr"] = compute_mrr_score(retrieved_docs, relevant_docs)
    
    # NDCG (if relevance scores provided)
    if relevance_scores:
        for k in k_values:
            results[f"ndcg_at_{k}"] = compute_ndcg_score(
                retrieved_docs, relevance_scores, k
            )
    
    # Coverage (if corpus provided)
    if corpus:
        results["coverage"] = compute_coverage(retrieved_docs, corpus)
    
    # Diversity
    for k in k_values:
        results[f"diversity_at_{k}"] = compute_diversity(retrieved_docs, k)
    
    return results


def compare_retrieval_systems(
    system_results: Dict[str, Dict[str, float]]
) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple retrieval systems.
    
    Args:
        system_results: Dictionary mapping system names to their evaluation results
        
    Returns:
        Comparison summary with rankings and improvements
    """
    comparison = {}
    
    # Get all metrics
    all_metrics = set()
    for results in system_results.values():
        all_metrics.update(results.keys())
    
    # For each metric, rank systems and compute improvements
    for metric in all_metrics:
        metric_scores = {}
        for system, results in system_results.items():
            if metric in results:
                metric_scores[system] = results[metric]
        
        if not metric_scores:
            continue
        
        # Sort systems by metric (higher is better for most metrics)
        sorted_systems = sorted(
            metric_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Compute relative improvements
        baseline_score = sorted_systems[-1][1]  # Worst score as baseline
        best_score = sorted_systems[0][1]
        
        improvements = {}
        for system, score in metric_scores.items():
            if baseline_score > 0:
                improvement = (score - baseline_score) / baseline_score * 100
            else:
                improvement = 0.0
            improvements[system] = improvement
        
        comparison[metric] = {
            "rankings": [system for system, _ in sorted_systems],
            "scores": metric_scores,
            "improvements": improvements,
            "best_system": sorted_systems[0][0],
            "best_score": best_score,
            "worst_system": sorted_systems[-1][0],
            "worst_score": baseline_score
        }
    
    return comparison


def create_retrieval_benchmark_dataset(
    corpus_file: str,
    queries_file: str,
    relevance_file: str
) -> Tuple[List[str], List[str], List[List[str]], List[Dict[str, float]]]:
    """
    Create a benchmark dataset for retrieval evaluation.
    
    Args:
        corpus_file: Path to corpus file (one document per line)
        queries_file: Path to queries file (one query per line)
        relevance_file: Path to relevance judgments file (JSON format)
        
    Returns:
        Tuple of (corpus, queries, relevant_docs, relevance_scores)
    """
    import json
    
    # Load corpus
    with open(corpus_file, 'r', encoding='utf-8') as f:
        corpus = [line.strip() for line in f if line.strip()]
    
    # Load queries
    with open(queries_file, 'r', encoding='utf-8') as f:
        queries = [line.strip() for line in f if line.strip()]
    
    # Load relevance judgments
    with open(relevance_file, 'r', encoding='utf-8') as f:
        relevance_data = json.load(f)
    
    relevant_docs = []
    relevance_scores = []
    
    for query in queries:
        if query in relevance_data:
            query_relevance = relevance_data[query]
            
            # Extract relevant documents (score > 0)
            relevant = [doc for doc, score in query_relevance.items() if score > 0]
            relevant_docs.append(relevant)
            
            # Extract relevance scores
            relevance_scores.append(query_relevance)
        else:
            relevant_docs.append([])
            relevance_scores.append({})
    
    return corpus, queries, relevant_docs, relevance_scores