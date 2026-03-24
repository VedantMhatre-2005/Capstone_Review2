# Quantum Open Path Network — Team Integration Guide

**To: Vedant, Sanjay, & Aditya**  
**From: Yash**  

This document outlines the strict **Open Path** topology requirements for integrating the Hybrid GCN-Quantum System to correctly model the 4-node traffic intersection structure as directed by our faculty advisor. We are moving away from the Ring Network to a linear sequence graph.

---

## 1. The Quantum Topology Explanation

To accurately model the physical flow of traffic where intersections form a sequential line with no loops, the quantum layer incorporates a **Linear CNOT Entanglement Sequence**.

In our Variational Quantum Circuit (VQC) defined in `quantum_path_layer.py`:
- Each of the `4` physical junctions is assigned to one of the `4` qubits.
- Entanglement connects the qubits sequentially: **Qubit $N$ controls Qubit $N+1$**.
- To perfectly mirror the physical open path structure, there is **NO loop back** from Qubit 3 to Qubit 0. 
- We use the exact following operations for $q \in \{0, 1, 2\}$:
  ```python
  qml.CNOT(wires=[i, i + 1]) # where i ranges from 0 to 2
  ```
This mathematical structure guarantees that the entanglement successfully maps the layout of the physical corridor without creating cyclical overlaps.

---

## 2. Instructions for Vedant & Sanjay: Classical `edge_index`

For the classical GNN side, the simulated topology **must explicitly match** this same 4-node sequential path. 

You must update the `edge_index` tensor generated in your `generate_graph()` function to the following structure:
```python
# The exactly mirrored PyTorch tensor for a 4-node Open Path Network
import torch

# Directed graph: 0 -> 1 -> 2 -> 3
edge_index = torch.tensor([
    [0, 1, 2],  # Source nodes
    [1, 2, 3]   # Target nodes
], dtype=torch.long)
```
*(Note: If you are building an undirected graph, append the reversed edges: `[1, 2, 3]` as source and `[0, 1, 2]` as target.)*

**CRITICAL:** Ensure that the connection `[3, 0]` is strictly **omitted**. **DO NOT** use the old fully-connected mesh or the old completely connected ring `edge_index`.

---

## 3. Code Snippet for GNN Integration

Here is the exact 3-step required process to integrate the `TrafficQuantumLayer` directly into your existing PyTorch PyG pipeline.

```python
import torch
import torch.nn as nn

# Step 1: Import the newly built module from my quantum pipeline
from quantum_path_layer import TrafficQuantumLayer

class HybridTrafficGNN(nn.Module):
    def __init__(self):
        super(HybridTrafficGNN, self).__init__()
        
        # ... [Your Classical GNN Layers Here] ...
        
        # Step 2: Initialize the Quantum Path Layer in your constructor
        # (This automatically constructs the 4-qubit parameterized open path circuit)
        self.quantum_layer = TrafficQuantumLayer()
        
    def forward(self, x, edge_index):
        # ... [Your Classical GNN Forward Pass Here] ...
        
        # Assume `h_classical` is the batched output feature tensor from the GNN 
        # that has been reduced/projected to shape (batch_size, 4).
        
        # Step 3: Pass classical features through the Quantum Path Layer
        h_quantum = self.quantum_layer(h_classical)
        
        return h_quantum
```

If you encounter any size mismatch errors from `TorchLayer`, please assure your bottleneck layer projects completely down to `out_features=4` before feeding it to `TrafficQuantumLayer`.
