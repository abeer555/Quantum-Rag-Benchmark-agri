"""
Answer Quality Evaluation Metrics

This module provides metrics for evaluating the quality of generated answers
in RAG systems, including BLEU, ROUGE, BERTScore, and domain-specific metrics.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import re
from collections import Counter


def compute_bleu_score(
    references: List[str], 
    candidates: List[str], 
    max_n: int = 4
) -> Dict[str, float]:
    """
    Compute BLEU scores for generated answers.
    
    Args:
        references: List of reference answers
        candidates: List of candidate answers
        max_n: Maximum n-gram order
        
    Returns:
        Dictionary with BLEU scores
    """
    try:
        import nltk
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
    except ImportError:
        print("NLTK not available. Install with: pip install nltk")
        return {}
    
    def tokenize(text):
        """Simple tokenization."""
        return re.findall(r'\w+', text.lower())
    
    smoothing = SmoothingFunction()
    bleu_scores = {f"bleu_{i}": [] for i in range(1, max_n + 1)}
    
    for ref, cand in zip(references, candidates):
        ref_tokens = tokenize(ref)
        cand_tokens = tokenize(cand)
        
        for n in range(1, max_n + 1):
            # Compute BLEU-n score
            weights = [1.0/n] * n + [0.0] * (4 - n)
            score = sentence_bleu(
                [ref_tokens], 
                cand_tokens, 
                weights=weights,
                smoothing_function=smoothing.method1
            )
            bleu_scores[f"bleu_{n}"].append(score)
    
    # Average scores
    avg_scores = {}
    for key, scores in bleu_scores.items():
        avg_scores[key] = np.mean(scores) if scores else 0.0
    
    return avg_scores


def compute_rouge_scores(
    references: List[str], 
    candidates: List[str]
) -> Dict[str, float]:
    """
    Compute ROUGE scores for generated answers.
    
    Args:
        references: List of reference answers
        candidates: List of candidate answers
        
    Returns:
        Dictionary with ROUGE scores
    """
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        print("rouge-score not available. Install with: pip install rouge-score")
        return {}
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge_scores = {
        'rouge1_f': [],
        'rouge1_p': [],
        'rouge1_r': [],
        'rouge2_f': [],
        'rouge2_p': [],
        'rouge2_r': [],
        'rougeL_f': [],
        'rougeL_p': [],
        'rougeL_r': []
    }
    
    for ref, cand in zip(references, candidates):
        scores = scorer.score(ref, cand)
        
        for rouge_type in ['rouge1', 'rouge2', 'rougeL']:
            rouge_scores[f'{rouge_type}_f'].append(scores[rouge_type].fmeasure)
            rouge_scores[f'{rouge_type}_p'].append(scores[rouge_type].precision)
            rouge_scores[f'{rouge_type}_r'].append(scores[rouge_type].recall)
    
    # Average scores
    avg_scores = {}
    for key, scores in rouge_scores.items():
        avg_scores[key] = np.mean(scores) if scores else 0.0
    
    return avg_scores


def compute_bert_score(
    references: List[str], 
    candidates: List[str],
    model_type: str = "bert-base-uncased"
) -> Dict[str, float]:
    """
    Compute BERTScore for semantic similarity.
    
    Args:
        references: List of reference answers
        candidates: List of candidate answers
        model_type: BERT model to use
        
    Returns:
        Dictionary with BERTScore metrics
    """
    try:
        from bert_score import score
    except ImportError:
        print("bert-score not available. Install with: pip install bert-score")
        return {}
    
    P, R, F1 = score(candidates, references, model_type=model_type, verbose=False)
    
    return {
        "bert_precision": float(P.mean()),
        "bert_recall": float(R.mean()),
        "bert_f1": float(F1.mean())
    }


def compute_semantic_similarity(
    references: List[str], 
    candidates: List[str],
    model_name: str = "all-MiniLM-L6-v2"
) -> float:
    """
    Compute semantic similarity using sentence transformers.
    
    Args:
        references: List of reference answers
        candidates: List of candidate answers
        model_name: Sentence transformer model name
        
    Returns:
        Average cosine similarity
    """
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        print("Required packages not available")
        return 0.0
    
    model = SentenceTransformer(model_name)
    
    ref_embeddings = model.encode(references)
    cand_embeddings = model.encode(candidates)
    
    similarities = []
    for ref_emb, cand_emb in zip(ref_embeddings, cand_embeddings):
        sim = cosine_similarity([ref_emb], [cand_emb])[0][0]
        similarities.append(sim)
    
    return np.mean(similarities)


def compute_factual_consistency(
    contexts: List[str],
    answers: List[str]
) -> List[float]:
    """
    Compute factual consistency scores (simplified implementation).
    
    Args:
        contexts: List of context passages
        answers: List of generated answers
        
    Returns:
        List of consistency scores
    """
    consistency_scores = []
    
    for context, answer in zip(contexts, answers):
        # Simple keyword overlap-based consistency
        context_words = set(re.findall(r'\w+', context.lower()))
        answer_words = set(re.findall(r'\w+', answer.lower()))
        
        if not answer_words:
            consistency_scores.append(0.0)
            continue
        
        # Jaccard similarity as a proxy for factual consistency
        intersection = len(context_words.intersection(answer_words))
        union = len(context_words.union(answer_words))
        
        consistency = intersection / union if union > 0 else 0.0
        consistency_scores.append(consistency)
    
    return consistency_scores


def evaluate_answer_quality(
    queries: List[str],
    contexts: List[List[str]],
    generated_answers: List[str],
    reference_answers: List[str],
    include_bert_score: bool = True
) -> Dict[str, float]:
    """
    Comprehensive answer quality evaluation.
    
    Args:
        queries: List of queries
        contexts: List of context lists for each query
        generated_answers: List of generated answers
        reference_answers: List of reference answers
        include_bert_score: Whether to compute BERTScore (slower)
        
    Returns:
        Dictionary of evaluation metrics
    """
    results = {}
    
    # BLEU scores
    print("Computing BLEU scores...")
    bleu_results = compute_bleu_score(reference_answers, generated_answers)
    results.update(bleu_results)
    
    # ROUGE scores
    print("Computing ROUGE scores...")
    rouge_results = compute_rouge_scores(reference_answers, generated_answers)
    results.update(rouge_results)
    
    # BERTScore (optional, slower)
    if include_bert_score:
        print("Computing BERTScore...")
        bert_results = compute_bert_score(reference_answers, generated_answers)
        results.update(bert_results)
    
    # Semantic similarity
    print("Computing semantic similarity...")
    semantic_sim = compute_semantic_similarity(reference_answers, generated_answers)
    results["semantic_similarity"] = semantic_sim
    
    # Factual consistency
    print("Computing factual consistency...")
    context_texts = [" ".join(ctx) for ctx in contexts]
    consistency_scores = compute_factual_consistency(context_texts, generated_answers)
    results["factual_consistency"] = np.mean(consistency_scores)
    
    # Answer length statistics
    gen_lengths = [len(ans.split()) for ans in generated_answers]
    ref_lengths = [len(ans.split()) for ans in reference_answers]
    
    results["avg_generated_length"] = np.mean(gen_lengths)
    results["avg_reference_length"] = np.mean(ref_lengths)
    results["length_ratio"] = np.mean(gen_lengths) / np.mean(ref_lengths) if np.mean(ref_lengths) > 0 else 0.0
    
    return results


class AgricultureQAEvaluator:
    """
    Domain-specific evaluator for agricultural Q&A.
    """
    
    def __init__(self):
        self.agricultural_keywords = {
            'crops': ['wheat', 'rice', 'corn', 'maize', 'barley', 'soy', 'soybean', 'potato', 'tomato'],
            'farming': ['plant', 'grow', 'harvest', 'seed', 'soil', 'fertilizer', 'irrigation', 'cultivation'],
            'livestock': ['cattle', 'cow', 'pig', 'chicken', 'sheep', 'goat', 'livestock', 'dairy'],
            'techniques': ['rotation', 'pesticide', 'organic', 'sustainable', 'greenhouse', 'hydroponics'],
            'seasons': ['spring', 'summer', 'fall', 'autumn', 'winter', 'season', 'seasonal'],
            'weather': ['rain', 'drought', 'temperature', 'climate', 'weather', 'precipitation']
        }
    
    def compute_domain_relevance(self, answers: List[str]) -> List[float]:
        """
        Compute domain relevance scores for answers.
        
        Args:
            answers: List of generated answers
            
        Returns:
            List of relevance scores
        """
        relevance_scores = []
        
        for answer in answers:
            answer_words = set(re.findall(r'\w+', answer.lower()))
            
            # Count agricultural keywords
            total_keywords = 0
            matched_keywords = 0
            
            for category, keywords in self.agricultural_keywords.items():
                for keyword in keywords:
                    total_keywords += 1
                    if keyword in answer_words:
                        matched_keywords += 1
            
            # Relevance as proportion of matched keywords
            relevance = matched_keywords / total_keywords if total_keywords > 0 else 0.0
            relevance_scores.append(relevance)
        
        return relevance_scores
    
    def compute_specificity(self, answers: List[str]) -> List[float]:
        """
        Compute specificity scores (how specific/detailed the answers are).
        
        Args:
            answers: List of generated answers
            
        Returns:
            List of specificity scores
        """
        specificity_scores = []
        
        for answer in answers:
            # Simple specificity based on presence of numbers, units, specific terms
            answer_lower = answer.lower()
            
            specificity = 0.0
            
            # Numbers indicate specific quantities
            numbers = re.findall(r'\d+', answer)
            specificity += len(numbers) * 0.1
            
            # Units indicate specific measurements
            units = ['kg', 'lb', 'pound', 'ton', 'acre', 'hectare', 'inch', 'cm', 'meter', 'feet', 'gallon', 'liter']
            for unit in units:
                if unit in answer_lower:
                    specificity += 0.2
            
            # Specific agricultural terms
            specific_terms = ['variety', 'cultivar', 'strain', 'breed', 'species', 'phylum']
            for term in specific_terms:
                if term in answer_lower:
                    specificity += 0.1
            
            # Cap at 1.0
            specificity = min(specificity, 1.0)
            specificity_scores.append(specificity)
        
        return specificity_scores
    
    def evaluate_agricultural_answers(
        self,
        queries: List[str],
        generated_answers: List[str],
        reference_answers: List[str]
    ) -> Dict[str, float]:
        """
        Comprehensive agricultural domain evaluation.
        
        Args:
            queries: List of queries
            generated_answers: List of generated answers
            reference_answers: List of reference answers
            
        Returns:
            Dictionary of agricultural-specific metrics
        """
        results = {}
        
        # Domain relevance
        gen_relevance = self.compute_domain_relevance(generated_answers)
        ref_relevance = self.compute_domain_relevance(reference_answers)
        
        results["generated_domain_relevance"] = np.mean(gen_relevance)
        results["reference_domain_relevance"] = np.mean(ref_relevance)
        results["relevance_preservation"] = np.mean(gen_relevance) / np.mean(ref_relevance) if np.mean(ref_relevance) > 0 else 0.0
        
        # Specificity
        gen_specificity = self.compute_specificity(generated_answers)
        ref_specificity = self.compute_specificity(reference_answers)
        
        results["generated_specificity"] = np.mean(gen_specificity)
        results["reference_specificity"] = np.mean(ref_specificity)
        results["specificity_preservation"] = np.mean(gen_specificity) / np.mean(ref_specificity) if np.mean(ref_specificity) > 0 else 0.0
        
        return results


def create_answer_quality_report(
    evaluation_results: Dict[str, float],
    system_name: str = "System"
) -> str:
    """
    Create a formatted report for answer quality evaluation.
    
    Args:
        evaluation_results: Dictionary of evaluation metrics
        system_name: Name of the system being evaluated
        
    Returns:
        Formatted evaluation report
    """
    report = f"\n{'='*50}\n"
    report += f"Answer Quality Evaluation Report: {system_name}\n"
    report += f"{'='*50}\n\n"
    
    # Group metrics by category
    categories = {
        "BLEU Scores": [k for k in evaluation_results.keys() if k.startswith('bleu')],
        "ROUGE Scores": [k for k in evaluation_results.keys() if k.startswith('rouge')],
        "BERT Scores": [k for k in evaluation_results.keys() if k.startswith('bert')],
        "Semantic Metrics": ['semantic_similarity', 'factual_consistency'],
        "Length Metrics": ['avg_generated_length', 'avg_reference_length', 'length_ratio'],
        "Domain Metrics": [k for k in evaluation_results.keys() if 'domain' in k or 'specificity' in k]
    }
    
    for category, metrics in categories.items():
        if not metrics:
            continue
            
        report += f"{category}:\n"
        report += "-" * len(category) + "\n"
        
        for metric in metrics:
            if metric in evaluation_results:
                value = evaluation_results[metric]
                report += f"  {metric}: {value:.4f}\n"
        
        report += "\n"
    
    return report


def compare_answer_quality(
    system_results: Dict[str, Dict[str, float]]
) -> Dict[str, Any]:
    """
    Compare answer quality across multiple systems.
    
    Args:
        system_results: Dictionary mapping system names to their evaluation results
        
    Returns:
        Comparison summary
    """
    comparison = {}
    
    # Get all metrics
    all_metrics = set()
    for results in system_results.values():
        all_metrics.update(results.keys())
    
    # For each metric, rank systems
    for metric in all_metrics:
        metric_scores = {}
        for system, results in system_results.items():
            if metric in results:
                metric_scores[system] = results[metric]
        
        if not metric_scores:
            continue
        
        # Sort systems by metric (higher is better)
        sorted_systems = sorted(
            metric_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        comparison[metric] = {
            "best_system": sorted_systems[0][0],
            "best_score": sorted_systems[0][1],
            "worst_system": sorted_systems[-1][0],
            "worst_score": sorted_systems[-1][1],
            "all_scores": metric_scores,
            "rankings": [system for system, _ in sorted_systems]
        }
    
    return comparison