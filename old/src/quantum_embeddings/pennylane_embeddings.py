"""
PennyLane-based Quantum Embeddings

Various quantum feature map implementations using PennyLane.
"""

import pennylane as qml
import numpy as np
from typing import List, Tuple, Optional, Union
import torch


class AngleEmbedding:
    """Angle embedding using rotation gates."""
    
    def __init__(self, n_qubits: int, normalize: bool = True):
        self.n_qubits = n_qubits
        self.normalize = normalize
        self.device = qml.device("default.qubit", wires=n_qubits)
        
    def encode(self, features: np.ndarray) -> qml.QNode:
        """Encode features using angle embedding."""
        if self.normalize:
            features = features / (np.linalg.norm(features) + 1e-8)
        
        # Pad or truncate features to match n_qubits
        if len(features) < self.n_qubits:
            features = np.pad(features, (0, self.n_qubits - len(features)))
        else:
            features = features[:self.n_qubits]
            
        @qml.qnode(self.device)
        def circuit():
            qml.templates.AngleEmbedding(features, wires=range(self.n_qubits))
            return qml.state()
            
        return circuit()


class AmplitudeEmbedding:
    """Amplitude embedding for dense feature representation."""
    
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.n_features = 2 ** n_qubits
        self.device = qml.device("default.qubit", wires=n_qubits)
        
    def encode(self, features: np.ndarray) -> qml.QNode:
        """Encode features using amplitude embedding."""
        # Normalize and pad/truncate features
        features = features / (np.linalg.norm(features) + 1e-8)
        
        if len(features) < self.n_features:
            features = np.pad(features, (0, self.n_features - len(features)))
        else:
            features = features[:self.n_features]
            
        @qml.qnode(self.device)
        def circuit():
            qml.templates.AmplitudeEmbedding(features, wires=range(self.n_qubits), normalize=True)
            return qml.state()
            
        return circuit()


class IQPEmbedding:
    """Instantaneous Quantum Polynomial (IQP) embedding."""
    
    def __init__(self, n_qubits: int, depth: int = 1):
        self.n_qubits = n_qubits
        self.depth = depth
        self.device = qml.device("default.qubit", wires=n_qubits)
        
    def encode(self, features: np.ndarray) -> qml.QNode:
        """Encode features using IQP circuits."""
        # Pad or truncate features
        if len(features) < self.n_qubits:
            features = np.pad(features, (0, self.n_qubits - len(features)))
        else:
            features = features[:self.n_qubits]
            
        @qml.qnode(self.device)
        def circuit():
            # Initial Hadamard layer
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
                
            # IQP layers
            for d in range(self.depth):
                # Z rotations
                for i in range(self.n_qubits):
                    qml.RZ(features[i] * np.pi, wires=i)
                    
                # ZZ interactions
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                    qml.RZ(features[i] * features[i + 1] * np.pi, wires=i + 1)
                    qml.CNOT(wires=[i, i + 1])
                    
            return qml.state()
            
        return circuit()


class DataReuploadingEmbedding:
    """Data re-uploading embedding with multiple layers."""
    
    def __init__(self, n_qubits: int, n_layers: int = 2):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.device = qml.device("default.qubit", wires=n_qubits)
        
    def encode(self, features: np.ndarray) -> qml.QNode:
        """Encode features using data re-uploading."""
        # Pad or truncate features
        if len(features) < self.n_qubits:
            features = np.pad(features, (0, self.n_qubits - len(features)))
        else:
            features = features[:self.n_qubits]
            
        @qml.qnode(self.device)
        def circuit():
            for layer in range(self.n_layers):
                # Data encoding
                for i in range(self.n_qubits):
                    qml.RY(features[i] * np.pi, wires=i)
                    
                # Entangling layer
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                    
                # Additional rotations
                for i in range(self.n_qubits):
                    qml.RZ(features[i] * np.pi / 2, wires=i)
                    
            return qml.state()
            
        return circuit()


def quantum_similarity_pennylane(
    embedding1: np.ndarray, 
    embedding2: np.ndarray,
    method: str = "angle",
    n_qubits: int = 4
) -> float:
    """
    Compute quantum similarity between two embeddings using PennyLane.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector  
        method: Embedding method ("angle", "amplitude", "iqp", "data_reuploading")
        n_qubits: Number of qubits to use
        
    Returns:
        Similarity score between 0 and 1
    """
    device = qml.device("default.qubit", wires=n_qubits)
    
    # Prepare features
    if len(embedding1) < n_qubits:
        embedding1 = np.pad(embedding1, (0, n_qubits - len(embedding1)))
    else:
        embedding1 = embedding1[:n_qubits]
        
    if len(embedding2) < n_qubits:
        embedding2 = np.pad(embedding2, (0, n_qubits - len(embedding2)))
    else:
        embedding2 = embedding2[:n_qubits]
    
    @qml.qnode(device)
    def quantum_kernel(x1, x2):
        """Quantum kernel for similarity."""
        if method == "angle":
            qml.templates.AngleEmbedding(x1, wires=range(n_qubits))
            qml.adjoint(qml.templates.AngleEmbedding)(x2, wires=range(n_qubits))
        elif method == "amplitude":
            # For amplitude embedding, we need 2^n_qubits features
            n_features = 2 ** n_qubits
            x1_amp = x1 / (np.linalg.norm(x1) + 1e-8)
            x2_amp = x2 / (np.linalg.norm(x2) + 1e-8)
            
            if len(x1_amp) < n_features:
                x1_amp = np.pad(x1_amp, (0, n_features - len(x1_amp)))
            else:
                x1_amp = x1_amp[:n_features]
                
            if len(x2_amp) < n_features:
                x2_amp = np.pad(x2_amp, (0, n_features - len(x2_amp)))
            else:
                x2_amp = x2_amp[:n_features]
                
            qml.templates.AmplitudeEmbedding(x1_amp, wires=range(n_qubits), normalize=True)
            qml.adjoint(qml.templates.AmplitudeEmbedding)(x2_amp, wires=range(n_qubits), normalize=True)
        else:
            # Default to angle embedding
            qml.templates.AngleEmbedding(x1, wires=range(n_qubits))
            qml.adjoint(qml.templates.AngleEmbedding)(x2, wires=range(n_qubits))
            
        return qml.probs(wires=range(n_qubits))
    
    # Compute quantum kernel (fidelity)
    probs = quantum_kernel(embedding1, embedding2)
    return float(probs[0])  # Probability of measuring |0...0⟩


def batch_quantum_similarity(
    embeddings1: List[np.ndarray],
    embeddings2: List[np.ndarray],
    method: str = "angle",
    n_qubits: int = 4
) -> np.ndarray:
    """
    Compute quantum similarity for batches of embeddings.
    
    Args:
        embeddings1: First batch of embeddings
        embeddings2: Second batch of embeddings
        method: Embedding method
        n_qubits: Number of qubits
        
    Returns:
        Similarity matrix
    """
    n1, n2 = len(embeddings1), len(embeddings2)
    similarities = np.zeros((n1, n2))
    
    for i in range(n1):
        for j in range(n2):
            similarities[i, j] = quantum_similarity_pennylane(
                embeddings1[i], embeddings2[j], method, n_qubits
            )
            
    return similarities


def get_quantum_features(
    text_embeddings: List[np.ndarray],
    method: str = "angle",
    n_qubits: int = 4
) -> List[np.ndarray]:
    """
    Convert classical text embeddings to quantum features.
    
    Args:
        text_embeddings: List of classical embeddings
        method: Quantum embedding method
        n_qubits: Number of qubits
        
    Returns:
        List of quantum state vectors
    """
    quantum_features = []
    
    for embedding in text_embeddings:
        if method == "angle":
            encoder = AngleEmbedding(n_qubits)
        elif method == "amplitude":
            encoder = AmplitudeEmbedding(n_qubits)
        elif method == "iqp":
            encoder = IQPEmbedding(n_qubits)
        elif method == "data_reuploading":
            encoder = DataReuploadingEmbedding(n_qubits)
        else:
            encoder = AngleEmbedding(n_qubits)
            
        quantum_state = encoder.encode(embedding)
        quantum_features.append(quantum_state)
        
    return quantum_features