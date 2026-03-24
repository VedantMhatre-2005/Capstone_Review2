import torch
import torch.nn as nn
import pennylane as qml

# The system strictly matches the 4 physical junctions in the open path
N_QUBITS = 4

class TrafficQuantumLayer(nn.Module):
    """
    Quantum Open Path Layer leveraging PennyLane VQC.
    
    This layer wraps a Variational Quantum Circuit (VQC) inside
    a PyTorch nn.Module for integration with the classical GNN.
    
    CRITICAL TOPOLOGY:
    The entanglement strategy physically mirrors the classical open path line network.
    """
    def __init__(self, n_layers=2):
        super(TrafficQuantumLayer, self).__init__()
        self.n_layers = n_layers
        
        # 1. Define the Quantum Device
        # Using default.qubit simulator
        self.dev = qml.device("default.qubit", wires=N_QUBITS)
        
        # 2. Define the Quantum Circuit QNode
        @qml.qnode(self.dev, interface="torch")
        def quantum_circuit(inputs, weights):
            # inputs shape: (N_QUBITS,)
            # weights shape: (n_layers, N_QUBITS)
            
            # --- Angle Embedding ---
            # Loads classical node features as rotation angles on the qubits
            # Handles both unbatched (4,) and batched (batch_size, 4) tensors automatically
            qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation='Y')
                
            # --- Variational & Entanglement Layers ---
            for layer in range(self.n_layers):
                
                # 1. Parameterized Rotations (Trainable parameters)
                for i in range(N_QUBITS):
                    qml.RY(weights[layer, i], wires=i)
                
                # 2. Linear CNOT Entanglement (CRITICAL TOPOLOGY REQUIREMENT)
                # Node N controls Node N+1. 
                # OMITTING [3, 0] to strictly enforce the Open Path Topology.
                for i in range(N_QUBITS - 1):
                    qml.CNOT(wires=[i, i + 1])
                    
            # --- Measurement ---
            # Return Pauli-Z expectation value for each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]
            
        # 3. Wrap QNode in a PyTorch Layer
        weight_shapes = {"weights": (self.n_layers, N_QUBITS)}
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

    def forward(self, x):
        """
        Forward pass for the Quantum Layer.
        
        Args:
            x (torch.Tensor): Input feature tensor of shape (batch_size, N_QUBITS)
            
        Returns:
            torch.Tensor: Evaluated expectation values of shape (batch_size, N_QUBITS)
        """
        return self.qlayer(x)
