"""
Trainable Quantum Embeddings

Variational quantum circuits for learning optimal embeddings.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any

try:
    import pennylane as qml
    from pennylane import numpy as pnp
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class VariationalQuantumEmbedding:
    """
    Variational quantum embedding with trainable parameters.
    """
    
    def __init__(
        self, 
        n_qubits: int, 
        n_layers: int = 2,
        learning_rate: float = 0.01,
        optimization_steps: int = 100
    ):
        if not PENNYLANE_AVAILABLE:
            raise ImportError("PennyLane not available")
            
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.optimization_steps = optimization_steps
        
        # Initialize device
        self.device = qml.device("default.qubit", wires=n_qubits)
        
        # Initialize parameters
        self.params = self._initialize_parameters()
        
        # Create quantum node
        self.qnode = qml.QNode(self._quantum_circuit, self.device)
        
        # Optimizer
        self.optimizer = qml.AdamOptimizer(stepsize=learning_rate)
        
    def _initialize_parameters(self) -> np.ndarray:
        """Initialize variational parameters."""
        # Parameters for data encoding + variational layers
        n_data_params = self.n_qubits  # For data encoding
        n_var_params = self.n_layers * self.n_qubits * 3  # 3 rotations per qubit per layer
        
        return np.random.normal(0, 0.1, n_data_params + n_var_params)
    
    def _quantum_circuit(self, params: np.ndarray, x: np.ndarray) -> List[float]:
        """Variational quantum circuit."""
        # Data encoding
        data_params = params[:self.n_qubits]
        var_params = params[self.n_qubits:].reshape(self.n_layers, self.n_qubits, 3)
        
        # Encode input data
        for i in range(min(len(x), self.n_qubits)):
            qml.RY(x[i] * data_params[i], wires=i)
        
        # Variational layers
        for layer in range(self.n_layers):
            # Parameterized rotations
            for i in range(self.n_qubits):
                qml.RX(var_params[layer, i, 0], wires=i)
                qml.RY(var_params[layer, i, 1], wires=i)
                qml.RZ(var_params[layer, i, 2], wires=i)
            
            # Entangling gates
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        
        # Return expectation values
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
    
    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode input using trained variational circuit."""
        # Pad or truncate input
        if len(x) < self.n_qubits:
            x = np.pad(x, (0, self.n_qubits - len(x)))
        else:
            x = x[:self.n_qubits]
            
        return np.array(self.qnode(self.params, x))
    
    def train(
        self, 
        training_data: List[np.ndarray],
        target_similarities: Optional[np.ndarray] = None
    ) -> List[float]:
        """
        Train the variational quantum embedding.
        
        Args:
            training_data: List of input vectors
            target_similarities: Target similarity matrix (optional)
            
        Returns:
            List of cost values during training
        """
        costs = []
        
        def cost_function(params):
            """Cost function for training."""
            if target_similarities is not None:
                # Supervised training with target similarities
                embeddings = []
                for x in training_data:
                    emb = self.qnode(params, x)
                    embeddings.append(emb)
                
                # Compute pairwise similarities
                similarities = []
                for i in range(len(embeddings)):
                    for j in range(i + 1, len(embeddings)):
                        sim = np.dot(embeddings[i], embeddings[j])
                        similarities.append(sim)
                
                similarities = np.array(similarities)
                target_flat = target_similarities[np.triu_indices_from(target_similarities, k=1)]
                
                # Mean squared error
                return np.mean((similarities - target_flat) ** 2)
            else:
                # Unsupervised training - maximize variance
                embeddings = []
                for x in training_data:
                    emb = self.qnode(params, x)
                    embeddings.append(emb)
                
                embeddings = np.array(embeddings)
                
                # Maximize variance (minimize negative variance)
                variance = np.var(embeddings, axis=0).sum()
                return -variance
        
        # Training loop
        for step in range(self.optimization_steps):
            self.params, cost = self.optimizer.step_and_cost(cost_function, self.params)
            costs.append(float(cost))
            
            if step % 20 == 0:
                print(f"Step {step}, Cost: {cost:.6f}")
        
        return costs
    
    def get_parameters(self) -> np.ndarray:
        """Get current parameters."""
        return self.params.copy()
    
    def set_parameters(self, params: np.ndarray) -> None:
        """Set parameters."""
        self.params = params.copy()


class QuantumAutoencoder:
    """
    Quantum autoencoder for dimensionality reduction.
    """
    
    def __init__(
        self, 
        n_qubits: int,
        encoding_qubits: int,
        n_layers: int = 2,
        learning_rate: float = 0.01,
        optimization_steps: int = 100
    ):
        if not PENNYLANE_AVAILABLE:
            raise ImportError("PennyLane not available")
        
        if encoding_qubits >= n_qubits:
            raise ValueError("Encoding qubits must be less than total qubits")
            
        self.n_qubits = n_qubits
        self.encoding_qubits = encoding_qubits
        self.trash_qubits = n_qubits - encoding_qubits
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.optimization_steps = optimization_steps
        
        # Initialize device
        self.device = qml.device("default.qubit", wires=n_qubits)
        
        # Initialize parameters
        self.encoder_params = np.random.normal(0, 0.1, n_layers * n_qubits * 3)
        self.decoder_params = np.random.normal(0, 0.1, n_layers * n_qubits * 3)
        
        # Create quantum nodes
        self.encoder_qnode = qml.QNode(self._encoder_circuit, self.device)
        self.decoder_qnode = qml.QNode(self._decoder_circuit, self.device)
        self.full_qnode = qml.QNode(self._full_circuit, self.device)
        
        # Optimizer
        self.optimizer = qml.AdamOptimizer(stepsize=learning_rate)
    
    def _encoder_circuit(self, params: np.ndarray, x: np.ndarray) -> List[float]:
        """Encoder circuit."""
        params = params.reshape(self.n_layers, self.n_qubits, 3)
        
        # Data encoding
        for i in range(min(len(x), self.n_qubits)):
            qml.RY(x[i], wires=i)
        
        # Encoding layers
        for layer in range(self.n_layers):
            for i in range(self.n_qubits):
                qml.RX(params[layer, i, 0], wires=i)
                qml.RY(params[layer, i, 1], wires=i)
                qml.RZ(params[layer, i, 2], wires=i)
            
            # Entangling gates
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        
        # Return encoding qubits
        return [qml.expval(qml.PauliZ(i)) for i in range(self.encoding_qubits)]
    
    def _decoder_circuit(self, encoder_params: np.ndarray, decoder_params: np.ndarray, x: np.ndarray) -> List[float]:
        """Full autoencoder circuit."""
        # Encoder
        encoder_params = encoder_params.reshape(self.n_layers, self.n_qubits, 3)
        
        # Data encoding
        for i in range(min(len(x), self.n_qubits)):
            qml.RY(x[i], wires=i)
        
        # Encoding layers
        for layer in range(self.n_layers):
            for i in range(self.n_qubits):
                qml.RX(encoder_params[layer, i, 0], wires=i)
                qml.RY(encoder_params[layer, i, 1], wires=i)
                qml.RZ(encoder_params[layer, i, 2], wires=i)
            
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        
        # Decoder
        decoder_params = decoder_params.reshape(self.n_layers, self.n_qubits, 3)
        
        for layer in range(self.n_layers):
            for i in range(self.n_qubits - 1, -1, -1):
                qml.CNOT(wires=[i - 1 if i > 0 else self.n_qubits - 1, i])
            
            for i in range(self.n_qubits):
                qml.RZ(-decoder_params[layer, i, 2], wires=i)
                qml.RY(-decoder_params[layer, i, 1], wires=i)
                qml.RX(-decoder_params[layer, i, 0], wires=i)
        
        # Return all qubits
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
    
    def _full_circuit(self, encoder_params: np.ndarray, decoder_params: np.ndarray, x: np.ndarray) -> List[float]:
        """Full autoencoder circuit."""
        return self._decoder_circuit(encoder_params, decoder_params, x)
    
    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode input to compressed representation."""
        if len(x) < self.n_qubits:
            x = np.pad(x, (0, self.n_qubits - len(x)))
        else:
            x = x[:self.n_qubits]
            
        return np.array(self.encoder_qnode(self.encoder_params, x))
    
    def decode(self, encoded: np.ndarray) -> np.ndarray:
        """Decode compressed representation."""
        # This is simplified - in practice, you'd need to prepare the state
        # corresponding to the encoded representation
        return encoded  # Placeholder
    
    def train(self, training_data: List[np.ndarray]) -> List[float]:
        """Train the quantum autoencoder."""
        costs = []
        
        def cost_function(params):
            """Reconstruction cost."""
            encoder_params, decoder_params = params
            
            total_cost = 0
            for x in training_data:
                if len(x) < self.n_qubits:
                    x_padded = np.pad(x, (0, self.n_qubits - len(x)))
                else:
                    x_padded = x[:self.n_qubits]
                
                # Get reconstruction
                reconstructed = self.full_qnode(encoder_params, decoder_params, x_padded)
                
                # Compute reconstruction error
                error = np.sum((x_padded - reconstructed) ** 2)
                total_cost += error
            
            return total_cost / len(training_data)
        
        # Combined parameters
        combined_params = [self.encoder_params, self.decoder_params]
        
        # Training loop
        for step in range(self.optimization_steps):
            combined_params, cost = self.optimizer.step_and_cost(cost_function, combined_params)
            self.encoder_params, self.decoder_params = combined_params
            costs.append(float(cost))
            
            if step % 20 == 0:
                print(f"Step {step}, Reconstruction Cost: {cost:.6f}")
        
        return costs


def train_quantum_embedding_supervised(
    embeddings: List[np.ndarray],
    labels: List[str],
    n_qubits: int = 4,
    n_layers: int = 2,
    learning_rate: float = 0.01,
    optimization_steps: int = 100
) -> VariationalQuantumEmbedding:
    """
    Train a variational quantum embedding in a supervised manner.
    
    Args:
        embeddings: Input embeddings
        labels: Corresponding labels
        n_qubits: Number of qubits
        n_layers: Number of layers
        learning_rate: Learning rate
        optimization_steps: Number of optimization steps
        
    Returns:
        Trained quantum embedding
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Create target similarity matrix based on labels
    n = len(embeddings)
    target_similarities = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if labels[i] == labels[j]:
                target_similarities[i, j] = 1.0
            else:
                target_similarities[i, j] = 0.0
    
    # Initialize and train quantum embedding
    qemb = VariationalQuantumEmbedding(
        n_qubits=n_qubits,
        n_layers=n_layers,
        learning_rate=learning_rate,
        optimization_steps=optimization_steps
    )
    
    costs = qemb.train(embeddings, target_similarities)
    
    print(f"Training completed. Final cost: {costs[-1]:.6f}")
    
    return qemb


def compare_trainable_vs_fixed_embeddings(
    embeddings: List[np.ndarray],
    labels: List[str],
    n_qubits: int = 4
) -> Dict[str, Dict[str, float]]:
    """
    Compare trainable vs fixed quantum embeddings.
    
    Args:
        embeddings: Input embeddings
        labels: Corresponding labels
        n_qubits: Number of qubits
        
    Returns:
        Comparison results
    """
    from sklearn.metrics import adjusted_rand_score
    from sklearn.cluster import KMeans
    
    results = {}
    
    # Fixed angle embedding
    from .pennylane_embeddings import quantum_similarity_pennylane
    
    print("Testing fixed angle embedding...")
    fixed_similarities = []
    for i in range(len(embeddings)):
        row = []
        for j in range(len(embeddings)):
            sim = quantum_similarity_pennylane(
                embeddings[i], embeddings[j], "angle", n_qubits
            )
            row.append(sim)
        fixed_similarities.append(row)
    
    fixed_similarities = np.array(fixed_similarities)
    
    # Clustering with fixed embeddings
    kmeans_fixed = KMeans(n_clusters=len(set(labels)), random_state=42)
    fixed_clusters = kmeans_fixed.fit_predict(fixed_similarities)
    fixed_ari = adjusted_rand_score(labels, fixed_clusters)
    
    results["fixed_angle"] = {
        "adjusted_rand_index": fixed_ari,
        "mean_similarity": np.mean(fixed_similarities),
        "std_similarity": np.std(fixed_similarities)
    }
    
    # Trainable embedding
    print("Training variational quantum embedding...")
    qemb = train_quantum_embedding_supervised(
        embeddings, labels, n_qubits, optimization_steps=50
    )
    
    # Get trainable embeddings
    trainable_embeddings = []
    for emb in embeddings:
        quantum_emb = qemb.encode(emb)
        trainable_embeddings.append(quantum_emb)
    
    trainable_embeddings = np.array(trainable_embeddings)
    
    # Clustering with trainable embeddings
    kmeans_trainable = KMeans(n_clusters=len(set(labels)), random_state=42)
    trainable_clusters = kmeans_trainable.fit_predict(trainable_embeddings)
    trainable_ari = adjusted_rand_score(labels, trainable_clusters)
    
    results["trainable"] = {
        "adjusted_rand_index": trainable_ari,
        "mean_embedding": np.mean(trainable_embeddings),
        "std_embedding": np.std(trainable_embeddings)
    }
    
    return results