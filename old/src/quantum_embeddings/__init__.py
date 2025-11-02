"""
Quantum Embeddings Module

This module provides various quantum embedding strategies for encoding
classical text vectors into quantum states.
"""

from .pennylane_embeddings import (
    AngleEmbedding,
    AmplitudeEmbedding, 
    IQPEmbedding,
    DataReuploadingEmbedding,
    quantum_similarity_pennylane
)

from .qiskit_embeddings import (
    ZFeatureMap,
    ZZFeatureMap,
    PauliFeatureMap,
    quantum_similarity_qiskit
)

from .feature_maps import (
    create_feature_map,
    quantum_kernel_matrix,
    compare_embeddings
)

from .trainable_embeddings import (
    VariationalQuantumEmbedding,
    QuantumAutoencoder
)

__all__ = [
    'AngleEmbedding',
    'AmplitudeEmbedding', 
    'IQPEmbedding',
    'DataReuploadingEmbedding',
    'quantum_similarity_pennylane',
    'ZFeatureMap',
    'ZZFeatureMap', 
    'PauliFeatureMap',
    'quantum_similarity_qiskit',
    'create_feature_map',
    'quantum_kernel_matrix',
    'compare_embeddings',
    'VariationalQuantumEmbedding',
    'QuantumAutoencoder'
]