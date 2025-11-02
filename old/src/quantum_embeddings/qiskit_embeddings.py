"""
Qiskit-based Quantum Embeddings

Various quantum feature map implementations using Qiskit.
"""

try:
    from qiskit import QuantumCircuit, Aer, execute
    from qiskit.circuit.library import ZFeatureMap as QiskitZFeatureMap
    from qiskit.circuit.library import ZZFeatureMap as QiskitZZFeatureMap
    from qiskit.circuit.library import PauliFeatureMap as QiskitPauliFeatureMap
    from qiskit.quantum_info import state_fidelity, Statevector
    from qiskit_machine_learning.kernels import QuantumKernel
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

import numpy as np
from typing import List, Tuple, Optional, Union


class ZFeatureMap:
    """Z feature map implementation."""
    
    def __init__(self, n_qubits: int, reps: int = 1):
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit not available. Install with: pip install qiskit")
            
        self.n_qubits = n_qubits
        self.reps = reps
        self.feature_map = QiskitZFeatureMap(n_qubits, reps=reps)
        self.backend = Aer.get_backend('statevector_simulator')
        
    def encode(self, features: np.ndarray) -> np.ndarray:
        """Encode features using Z feature map."""
        # Pad or truncate features
        if len(features) < self.n_qubits:
            features = np.pad(features, (0, self.n_qubits - len(features)))
        else:
            features = features[:self.n_qubits]
            
        # Create circuit with feature map
        circuit = self.feature_map.bind_parameters(features)
        
        # Execute and get statevector
        job = execute(circuit, self.backend)
        result = job.result()
        statevector = result.get_statevector()
        
        return np.array(statevector)


class ZZFeatureMap:
    """ZZ feature map with two-qubit interactions."""
    
    def __init__(self, n_qubits: int, reps: int = 1):
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit not available. Install with: pip install qiskit")
            
        self.n_qubits = n_qubits
        self.reps = reps
        self.feature_map = QiskitZZFeatureMap(n_qubits, reps=reps)
        self.backend = Aer.get_backend('statevector_simulator')
        
    def encode(self, features: np.ndarray) -> np.ndarray:
        """Encode features using ZZ feature map."""
        # Pad or truncate features
        if len(features) < self.n_qubits:
            features = np.pad(features, (0, self.n_qubits - len(features)))
        else:
            features = features[:self.n_qubits]
            
        # Create circuit with feature map
        circuit = self.feature_map.bind_parameters(features)
        
        # Execute and get statevector
        job = execute(circuit, self.backend)
        result = job.result()
        statevector = result.get_statevector()
        
        return np.array(statevector)


class PauliFeatureMap:
    """Pauli feature map with customizable Pauli strings."""
    
    def __init__(self, n_qubits: int, reps: int = 1, paulis: Optional[List[str]] = None):
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit not available. Install with: pip install qiskit")
            
        self.n_qubits = n_qubits
        self.reps = reps
        
        if paulis is None:
            paulis = ['Z', 'ZZ']
            
        self.feature_map = QiskitPauliFeatureMap(n_qubits, reps=reps, paulis=paulis)
        self.backend = Aer.get_backend('statevector_simulator')
        
    def encode(self, features: np.ndarray) -> np.ndarray:
        """Encode features using Pauli feature map."""
        # Pad or truncate features
        if len(features) < self.n_qubits:
            features = np.pad(features, (0, self.n_qubits - len(features)))
        else:
            features = features[:self.n_qubits]
            
        # Create circuit with feature map
        circuit = self.feature_map.bind_parameters(features)
        
        # Execute and get statevector
        job = execute(circuit, self.backend)
        result = job.result()
        statevector = result.get_statevector()
        
        return np.array(statevector)


def quantum_similarity_qiskit(
    embedding1: np.ndarray,
    embedding2: np.ndarray,
    method: str = "zz",
    n_qubits: int = 4,
    reps: int = 1
) -> float:
    """
    Compute quantum similarity using Qiskit feature maps.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        method: Feature map method ("z", "zz", "pauli")
        n_qubits: Number of qubits
        reps: Number of repetitions in feature map
        
    Returns:
        Similarity score (fidelity) between 0 and 1
    """
    if not QISKIT_AVAILABLE:
        raise ImportError("Qiskit not available. Install with: pip install qiskit")
    
    # Prepare features
    if len(embedding1) < n_qubits:
        embedding1 = np.pad(embedding1, (0, n_qubits - len(embedding1)))
    else:
        embedding1 = embedding1[:n_qubits]
        
    if len(embedding2) < n_qubits:
        embedding2 = np.pad(embedding2, (0, n_qubits - len(embedding2)))
    else:
        embedding2 = embedding2[:n_qubits]
    
    # Create feature map
    if method == "z":
        feature_map = QiskitZFeatureMap(n_qubits, reps=reps)
    elif method == "zz":
        feature_map = QiskitZZFeatureMap(n_qubits, reps=reps)
    elif method == "pauli":
        feature_map = QiskitPauliFeatureMap(n_qubits, reps=reps, paulis=['Z', 'ZZ'])
    else:
        feature_map = QiskitZZFeatureMap(n_qubits, reps=reps)
    
    # Create quantum kernel
    quantum_kernel = QuantumKernel(feature_map=feature_map)
    
    # Compute kernel matrix for the two embeddings
    kernel_matrix = quantum_kernel.evaluate(
        x_vec=embedding1.reshape(1, -1),
        y_vec=embedding2.reshape(1, -1)
    )
    
    return float(kernel_matrix[0, 0])


def create_custom_feature_map(
    n_qubits: int,
    feature_dimension: int,
    entanglement: str = "linear",
    reps: int = 1
) -> QuantumCircuit:
    """
    Create a custom quantum feature map circuit.
    
    Args:
        n_qubits: Number of qubits
        feature_dimension: Dimension of input features
        entanglement: Entanglement pattern ("linear", "circular", "full")
        reps: Number of repetitions
        
    Returns:
        Quantum circuit for feature map
    """
    if not QISKIT_AVAILABLE:
        raise ImportError("Qiskit not available. Install with: pip install qiskit")
    
    from qiskit.circuit import Parameter, ParameterVector
    
    # Create parameter vector for features
    parameters = ParameterVector('x', feature_dimension)
    
    # Create quantum circuit
    qc = QuantumCircuit(n_qubits)
    
    for rep in range(reps):
        # Single-qubit rotations
        for i in range(min(n_qubits, feature_dimension)):
            qc.ry(parameters[i], i)
            
        # Entangling gates
        if entanglement == "linear":
            for i in range(n_qubits - 1):
                qc.cx(i, i + 1)
        elif entanglement == "circular":
            for i in range(n_qubits - 1):
                qc.cx(i, i + 1)
            qc.cx(n_qubits - 1, 0)  # Close the circle
        elif entanglement == "full":
            for i in range(n_qubits):
                for j in range(i + 1, n_qubits):
                    qc.cx(i, j)
                    
        # Additional single-qubit rotations
        for i in range(min(n_qubits, feature_dimension)):
            qc.rz(parameters[i % feature_dimension], i)
    
    return qc


def quantum_kernel_matrix_qiskit(
    embeddings: List[np.ndarray],
    method: str = "zz",
    n_qubits: int = 4,
    reps: int = 1
) -> np.ndarray:
    """
    Compute quantum kernel matrix for a set of embeddings using Qiskit.
    
    Args:
        embeddings: List of embedding vectors
        method: Feature map method
        n_qubits: Number of qubits
        reps: Number of repetitions
        
    Returns:
        Kernel matrix
    """
    if not QISKIT_AVAILABLE:
        raise ImportError("Qiskit not available. Install with: pip install qiskit")
    
    # Prepare embeddings matrix
    embeddings_matrix = []
    for emb in embeddings:
        if len(emb) < n_qubits:
            emb = np.pad(emb, (0, n_qubits - len(emb)))
        else:
            emb = emb[:n_qubits]
        embeddings_matrix.append(emb)
    
    embeddings_matrix = np.array(embeddings_matrix)
    
    # Create feature map
    if method == "z":
        feature_map = QiskitZFeatureMap(n_qubits, reps=reps)
    elif method == "zz":
        feature_map = QiskitZZFeatureMap(n_qubits, reps=reps)
    elif method == "pauli":
        feature_map = QiskitPauliFeatureMap(n_qubits, reps=reps, paulis=['Z', 'ZZ'])
    else:
        feature_map = QiskitZZFeatureMap(n_qubits, reps=reps)
    
    # Create quantum kernel
    quantum_kernel = QuantumKernel(feature_map=feature_map)
    
    # Compute kernel matrix
    kernel_matrix = quantum_kernel.evaluate(x_vec=embeddings_matrix)
    
    return kernel_matrix


def get_qiskit_quantum_features(
    text_embeddings: List[np.ndarray],
    method: str = "zz",
    n_qubits: int = 4,
    reps: int = 1
) -> List[np.ndarray]:
    """
    Convert classical text embeddings to quantum features using Qiskit.
    
    Args:
        text_embeddings: List of classical embeddings
        method: Quantum embedding method
        n_qubits: Number of qubits
        reps: Number of repetitions
        
    Returns:
        List of quantum state vectors
    """
    if not QISKIT_AVAILABLE:
        raise ImportError("Qiskit not available. Install with: pip install qiskit")
    
    quantum_features = []
    
    for embedding in text_embeddings:
        if method == "z":
            encoder = ZFeatureMap(n_qubits, reps)
        elif method == "zz":
            encoder = ZZFeatureMap(n_qubits, reps)
        elif method == "pauli":
            encoder = PauliFeatureMap(n_qubits, reps)
        else:
            encoder = ZZFeatureMap(n_qubits, reps)
            
        quantum_state = encoder.encode(embedding)
        quantum_features.append(quantum_state)
        
    return quantum_features