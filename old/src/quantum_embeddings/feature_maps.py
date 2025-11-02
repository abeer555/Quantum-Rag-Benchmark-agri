"""
Feature Map Utilities and Comparisons

This module provides utilities for creating, comparing, and evaluating
different quantum feature maps.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from sklearn.metrics.pairwise import cosine_similarity

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


def create_feature_map(
    method: str,
    n_qubits: int = 4,
    framework: str = "pennylane",
    **kwargs
) -> Any:
    """
    Create a quantum feature map using specified method and framework.
    
    Args:
        method: Feature map method ("angle", "amplitude", "iqp", "zz", etc.)
        n_qubits: Number of qubits
        framework: "pennylane" or "qiskit"
        **kwargs: Additional parameters for the feature map
        
    Returns:
        Feature map object
    """
    if framework == "pennylane":
        if not PENNYLANE_AVAILABLE:
            raise ImportError("PennyLane not available")
            
        from .pennylane_embeddings import (
            AngleEmbedding, AmplitudeEmbedding, IQPEmbedding, DataReuploadingEmbedding
        )
        
        if method == "angle":
            return AngleEmbedding(n_qubits, **kwargs)
        elif method == "amplitude":
            return AmplitudeEmbedding(n_qubits, **kwargs)
        elif method == "iqp":
            return IQPEmbedding(n_qubits, **kwargs)
        elif method == "data_reuploading":
            return DataReuploadingEmbedding(n_qubits, **kwargs)
        else:
            return AngleEmbedding(n_qubits, **kwargs)
            
    elif framework == "qiskit":
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit not available")
            
        from .qiskit_embeddings import ZFeatureMap, ZZFeatureMap, PauliFeatureMap
        
        if method == "z":
            return ZFeatureMap(n_qubits, **kwargs)
        elif method == "zz":
            return ZZFeatureMap(n_qubits, **kwargs)
        elif method == "pauli":
            return PauliFeatureMap(n_qubits, **kwargs)
        else:
            return ZZFeatureMap(n_qubits, **kwargs)
    else:
        raise ValueError(f"Unknown framework: {framework}")


def quantum_kernel_matrix(
    embeddings: List[np.ndarray],
    method: str = "angle",
    framework: str = "pennylane",
    n_qubits: int = 4,
    **kwargs
) -> np.ndarray:
    """
    Compute quantum kernel matrix for a set of embeddings.
    
    Args:
        embeddings: List of embedding vectors
        method: Quantum method to use
        framework: "pennylane" or "qiskit"
        n_qubits: Number of qubits
        **kwargs: Additional parameters
        
    Returns:
        Kernel matrix
    """
    n = len(embeddings)
    kernel_matrix = np.zeros((n, n))
    
    if framework == "pennylane":
        from .pennylane_embeddings import quantum_similarity_pennylane
        
        for i in range(n):
            for j in range(i, n):
                similarity = quantum_similarity_pennylane(
                    embeddings[i], embeddings[j], method, n_qubits
                )
                kernel_matrix[i, j] = similarity
                kernel_matrix[j, i] = similarity
                
    elif framework == "qiskit":
        from .qiskit_embeddings import quantum_kernel_matrix_qiskit
        kernel_matrix = quantum_kernel_matrix_qiskit(
            embeddings, method, n_qubits, **kwargs
        )
    else:
        raise ValueError(f"Unknown framework: {framework}")
    
    return kernel_matrix


def compare_embeddings(
    text_embeddings: List[np.ndarray],
    methods: List[str] = ["classical", "angle", "zz"],
    n_qubits: int = 4,
    sample_size: Optional[int] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compare different embedding methods on similarity computation.
    
    Args:
        text_embeddings: List of classical text embeddings
        methods: List of methods to compare
        n_qubits: Number of qubits for quantum methods
        sample_size: Limit sample size for faster comparison
        
    Returns:
        Dictionary with comparison results
    """
    if sample_size and len(text_embeddings) > sample_size:
        # Random sample for faster computation
        indices = np.random.choice(len(text_embeddings), sample_size, replace=False)
        embeddings = [text_embeddings[i] for i in indices]
    else:
        embeddings = text_embeddings
    
    results = {}
    
    for method in methods:
        print(f"Computing similarities using {method}...")
        
        if method == "classical":
            # Classical cosine similarity
            similarities = cosine_similarity(embeddings)
            
        elif method in ["angle", "amplitude", "iqp", "data_reuploading"]:
            # PennyLane methods
            if not PENNYLANE_AVAILABLE:
                print(f"Skipping {method}: PennyLane not available")
                continue
                
            similarities = quantum_kernel_matrix(
                embeddings, method, "pennylane", n_qubits
            )
            
        elif method in ["z", "zz", "pauli"]:
            # Qiskit methods
            if not QISKIT_AVAILABLE:
                print(f"Skipping {method}: Qiskit not available")
                continue
                
            similarities = quantum_kernel_matrix(
                embeddings, method, "qiskit", n_qubits
            )
        else:
            print(f"Unknown method: {method}")
            continue
        
        # Compute statistics
        # Exclude diagonal (self-similarity)
        upper_triangle = similarities[np.triu_indices_from(similarities, k=1)]
        
        results[method] = {
            "mean_similarity": float(np.mean(upper_triangle)),
            "std_similarity": float(np.std(upper_triangle)),
            "min_similarity": float(np.min(upper_triangle)),
            "max_similarity": float(np.max(upper_triangle)),
            "median_similarity": float(np.median(upper_triangle))
        }
    
    return results


def evaluate_embedding_quality(
    embeddings: List[np.ndarray],
    labels: List[str],
    method: str = "angle",
    framework: str = "pennylane",
    n_qubits: int = 4
) -> Dict[str, float]:
    """
    Evaluate the quality of embeddings using clustering metrics.
    
    Args:
        embeddings: List of embedding vectors
        labels: True labels for the embeddings
        method: Embedding method
        framework: Framework to use
        n_qubits: Number of qubits
        
    Returns:
        Dictionary with evaluation metrics
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    
    # Get quantum similarities
    if method == "classical":
        similarities = cosine_similarity(embeddings)
    else:
        similarities = quantum_kernel_matrix(
            embeddings, method, framework, n_qubits
        )
    
    # Convert similarities to distances for clustering
    distances = 1 - similarities
    
    # Perform clustering
    n_clusters = len(set(labels))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    
    # Use similarities as features (each row of similarity matrix)
    cluster_labels = kmeans.fit_predict(similarities)
    
    # Compute metrics
    ari = adjusted_rand_score(labels, cluster_labels)
    silhouette = silhouette_score(similarities, cluster_labels, metric='precomputed')
    
    return {
        "adjusted_rand_index": ari,
        "silhouette_score": silhouette,
        "num_clusters_found": len(set(cluster_labels)),
        "num_true_clusters": n_clusters
    }


def benchmark_similarity_computation(
    embeddings: List[np.ndarray],
    methods: List[str] = ["classical", "angle", "zz"],
    n_qubits: int = 4,
    n_trials: int = 3
) -> Dict[str, Dict[str, float]]:
    """
    Benchmark the computational time for different similarity methods.
    
    Args:
        embeddings: List of embedding vectors
        methods: Methods to benchmark
        n_qubits: Number of qubits
        n_trials: Number of trials for timing
        
    Returns:
        Timing results for each method
    """
    import time
    
    results = {}
    
    # Use small sample for timing
    sample_embeddings = embeddings[:10] if len(embeddings) > 10 else embeddings
    
    for method in methods:
        times = []
        
        for trial in range(n_trials):
            start_time = time.time()
            
            if method == "classical":
                _ = cosine_similarity(sample_embeddings)
            elif method in ["angle", "amplitude", "iqp", "data_reuploading"]:
                if not PENNYLANE_AVAILABLE:
                    continue
                _ = quantum_kernel_matrix(
                    sample_embeddings, method, "pennylane", n_qubits
                )
            elif method in ["z", "zz", "pauli"]:
                if not QISKIT_AVAILABLE:
                    continue
                _ = quantum_kernel_matrix(
                    sample_embeddings, method, "qiskit", n_qubits
                )
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        if times:
            results[method] = {
                "mean_time": np.mean(times),
                "std_time": np.std(times),
                "min_time": np.min(times),
                "max_time": np.max(times)
            }
    
    return results


def visualize_similarity_matrices(
    embeddings: List[np.ndarray],
    methods: List[str] = ["classical", "angle", "zz"],
    n_qubits: int = 4,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize similarity matrices for different methods.
    
    Args:
        embeddings: List of embedding vectors
        methods: Methods to visualize
        n_qubits: Number of qubits
        save_path: Path to save the plot
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("Matplotlib/Seaborn not available for visualization")
        return
    
    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 4))
    
    if n_methods == 1:
        axes = [axes]
    
    for i, method in enumerate(methods):
        if method == "classical":
            similarities = cosine_similarity(embeddings)
        else:
            try:
                framework = "pennylane" if method in ["angle", "amplitude", "iqp"] else "qiskit"
                similarities = quantum_kernel_matrix(
                    embeddings, method, framework, n_qubits
                )
            except Exception as e:
                print(f"Error computing {method}: {e}")
                continue
        
        sns.heatmap(
            similarities, 
            annot=False, 
            cmap="viridis", 
            ax=axes[i],
            cbar_kws={'label': 'Similarity'}
        )
        axes[i].set_title(f"{method.capitalize()} Similarity")
        axes[i].set_xlabel("Documents")
        axes[i].set_ylabel("Documents")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()