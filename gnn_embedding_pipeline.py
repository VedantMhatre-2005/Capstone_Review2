"""
PyTorch GNN Pipeline for Node Embedding Generation
Follows strict message-passing framework for traffic graph analysis
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from itertools import combinations
import os

# ============================================================================
# SETUP: Random Seeds & Hyperparameters
# ============================================================================

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Graph parameters
N_NODES = 10
EMBEDDING_DIM = 8
HIDDEN_DIM = 16
EPOCHS = 100
LEARNING_RATE = 0.01
BATCH_SIZE = N_NODES  # Full batch

device = torch.device('cpu')


# ============================================================================
# 1. SYNTHETIC GRAPH CREATION
# ============================================================================

def generate_graph(n_nodes=N_NODES):
    """
    Create a directed mesh topology (fully connected, no self-loops).
    
    Returns:
        edge_index: torch.Tensor of shape (2, E) containing edge indices
    """
    edges = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:  # No self-loops
                edges.append([i, j])
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index


# ============================================================================
# 2. NODE FEATURES GENERATION
# ============================================================================

def generate_features(edge_index, n_nodes=N_NODES):
    """
    Generate node features X ∈ R^(N × 6).
    
    Features:
    - phi_flow: traffic flow (200-2000)
    - phi_signal: signal time (0-120)
    - phi_type: node type {0,1,2}
    - phi_x: x-coordinate [0,1]
    - phi_y: y-coordinate [0,1]
    - phi_deg: node degree
    
    Returns:
        X: torch.Tensor of shape (N, 6)
    """
    X = np.zeros((n_nodes, 6))
    
    # phi_flow: 200-2000
    X[:, 0] = np.random.uniform(200, 2000, n_nodes)
    
    # phi_signal: 0-120
    X[:, 1] = np.random.uniform(0, 120, n_nodes)
    
    # phi_type: {0, 1, 2}
    X[:, 2] = np.random.choice([0, 1, 2], n_nodes)
    
    # phi_x, phi_y: [0, 1]
    X[:, 3] = np.random.uniform(0, 1, n_nodes)
    X[:, 4] = np.random.uniform(0, 1, n_nodes)
    
    # phi_deg: compute degree from edge_index
    degree = np.zeros(n_nodes)
    edge_np = edge_index.numpy()
    for src, tgt in edge_np.T:
        degree[tgt] += 1  # Count incoming edges
    X[:, 5] = degree
    
    return torch.tensor(X, dtype=torch.float32)


# ============================================================================
# 3. EDGE FEATURES GENERATION
# ============================================================================

def generate_edge_features(edge_index):
    """
    Generate edge features edge_attr ∈ R^(E × 5).
    
    Features:
    - epsilon_cap: capacity (100-1000)
    - epsilon_speed: speed (20-80)
    - epsilon_lanes: lanes {1,2,3,4}
    - epsilon_len: length (50-500)
    - epsilon_type: type {0,1}
    
    Returns:
        edge_attr: torch.Tensor of shape (E, 5)
    """
    n_edges = edge_index.size(1)
    edge_attr = np.zeros((n_edges, 5))
    
    # epsilon_cap: 100-1000
    edge_attr[:, 0] = np.random.uniform(100, 1000, n_edges)
    
    # epsilon_speed: 20-80
    edge_attr[:, 1] = np.random.uniform(20, 80, n_edges)
    
    # epsilon_lanes: {1, 2, 3, 4}
    edge_attr[:, 2] = np.random.choice([1, 2, 3, 4], n_edges)
    
    # epsilon_len: 50-500
    edge_attr[:, 3] = np.random.uniform(50, 500, n_edges)
    
    # epsilon_type: {0, 1}
    edge_attr[:, 4] = np.random.choice([0, 1], n_edges)
    
    return torch.tensor(edge_attr, dtype=torch.float32)


# ============================================================================
# 4. FEATURE NORMALIZATION (Z-SCORE)
# ============================================================================

def normalize_features(X):
    """
    Apply Z-score normalization column-wise.
    
    X_norm = (X - mean) / std
    
    Args:
        X: torch.Tensor of shape (N, d)
    
    Returns:
        X_norm: normalized features
        mean: column means
        std: column standard deviations
    """
    X_np = X.numpy()
    mean = np.mean(X_np, axis=0, keepdims=True)
    std = np.std(X_np, axis=0, keepdims=True)
    
    # Avoid division by zero
    std[std == 0] = 1.0
    
    X_norm = (X_np - mean) / std
    
    return torch.tensor(X_norm, dtype=torch.float32), mean, std


# ============================================================================
# 5. TARGET LABELS GENERATION
# ============================================================================

def generate_labels(X, n_nodes=N_NODES):
    """
    Create synthetic labels y ∈ R^(N × 1) representing traffic congestion.
    
    y = 0.6 * flow + 0.4 * noise
    
    Args:
        X: original features (before normalization)
    
    Returns:
        y: torch.Tensor of shape (N, 1)
    """
    flow = X[:, 0].numpy()  # phi_flow
    noise = np.random.normal(0, 100, n_nodes)
    
    y = 0.6 * flow + 0.4 * noise
    
    # Normalize to [0, 1] for regression
    y_min = y.min()
    y_max = y.max()
    y_norm = (y - y_min) / (y_max - y_min)
    
    return torch.tensor(y_norm.reshape(-1, 1), dtype=torch.float32), y


# ============================================================================
# 6. CUSTOM GNN MODEL (MESSAGE PASSING FRAMEWORK)
# ============================================================================

class CustomGNN(nn.Module):
    """
    Custom Graph Neural Network following strict message-passing equations.
    
    Equations:
    (a) Node embedding: h_i^(0) = ReLU(W_v x_i + b_v)
    (b) Message function: m_{j→i} = MLP(h_j || e_ji)
    (c) Aggregation: a_i = sum of incoming messages
    (d) Residual update: h_i^(1) = h_i^(0) + a_i
    (e) Prediction head: y_hat = Linear(h_i^(1))
    """
    
    def __init__(self, input_dim, embedding_dim, hidden_dim, edge_dim):
        """
        Args:
            input_dim: dimension of input features
            embedding_dim: dimension of node embeddings
            hidden_dim: hidden dimension for MLPs
            edge_dim: dimension of edge features
        """
        super(CustomGNN, self).__init__()
        
        # (a) Node embedding layer: h_i^(0) = ReLU(W_v x_i + b_v)
        self.node_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # (b) Message function: m_{j→i} = MLP(h_j || e_ji)
        # Concatenation of embedding and edge features
        message_input_dim = embedding_dim + edge_dim
        self.message_mlp = nn.Sequential(
            nn.Linear(message_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # (e) Prediction head: y_hat = Linear(h_i^(1))
        self.prediction_head = nn.Linear(embedding_dim, 1)
        
        self.embedding_dim = embedding_dim
    
    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass following message-passing framework.
        
        Args:
            x: node features (N, input_dim)
            edge_index: edge indices (2, E)
            edge_attr: edge features (E, edge_dim)
        
        Returns:
            out: predictions (N, 1)
            h1: node embeddings (N, embedding_dim)
        """
        # (a) Initial node embedding
        h0 = self.node_embedding(x)  # (N, embedding_dim)
        
        # (b) & (c) Message passing and aggregation
        src, tgt = edge_index[0], edge_index[1]
        
        # Concatenate embeddings with edge features
        h_src = h0[src]  # (E, embedding_dim)
        edge_with_features = torch.cat([h_src, edge_attr], dim=1)  # (E, embedding_dim + edge_dim)
        
        # Message function
        messages = self.message_mlp(edge_with_features)  # (E, embedding_dim)
        
        # Aggregation: sum incoming messages per node
        aggregated = torch.zeros_like(h0)
        aggregated.scatter_add_(0, tgt.unsqueeze(1).expand(-1, self.embedding_dim), messages)
        
        # (d) Residual update: h_i^(1) = h_i^(0) + a_i
        h1 = h0 + aggregated  # (N, embedding_dim)
        
        # (e) Prediction head
        out = self.prediction_head(h1)  # (N, 1)
        
        return out, h1


# ============================================================================
# 7. TRAINING LOOP
# ============================================================================

def train_model(model, x, edge_index, edge_attr, y, epochs=EPOCHS, lr=LEARNING_RATE):
    """
    Train the GNN model using MSE loss and Adam optimizer.
    
    Args:
        model: CustomGNN model
        x: node features
        edge_index: edge indices
        edge_attr: edge features
        y: target labels
        epochs: number of training epochs
        lr: learning rate
    
    Returns:
        losses: list of loss values per epoch
        embeddings: final node embeddings
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    losses = []
    
    model.to(device)
    x = x.to(device)
    edge_index = edge_index.to(device)
    edge_attr = edge_attr.to(device)
    y = y.to(device)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        y_hat, embeddings = model(x, edge_index, edge_attr)
        
        # Compute loss
        loss = criterion(y_hat, y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:3d}/{epochs} | Loss: {loss.item():.6f}")
    
    # Extract final embeddings
    with torch.no_grad():
        _, embeddings = model(x, edge_index, edge_attr)
    
    return losses, embeddings


# ============================================================================
# 8. SAVE EMBEDDINGS
# ============================================================================

def save_embeddings(embeddings, X_norm, y, output_dir='./outputs'):
    """
    Save embeddings and features to files.
    
    Args:
        embeddings: node embeddings (N, d)
        X_norm: normalized features (N, 6)
        y: labels (N, 1)
        output_dir: output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to numpy
    embeddings_np = embeddings.detach().cpu().numpy()
    X_norm_np = X_norm.numpy()
    y_np = y.numpy()
    
    # Save as .npz
    np.savez(
        os.path.join(output_dir, 'embeddings.npz'),
        embeddings=embeddings_np,
        features=X_norm_np,
        labels=y_np
    )
    
    # Save embeddings as .csv
    np.savetxt(
        os.path.join(output_dir, 'embeddings.csv'),
        embeddings_np,
        delimiter=',',
        header=','.join([f'dim_{i}' for i in range(embeddings_np.shape[1])]),
        comments=''
    )
    
    # Save normalized features as .csv
    np.savetxt(
        os.path.join(output_dir, 'features_normalized.csv'),
        X_norm_np,
        delimiter=',',
        header=','.join(['flow', 'signal', 'type', 'x', 'y', 'degree']),
        comments=''
    )
    
    # Save labels as .csv
    np.savetxt(
        os.path.join(output_dir, 'labels.csv'),
        y_np,
        delimiter=',',
        header='congestion',
        comments=''
    )
    
    # Save training dataset (features + labels combined)
    training_data = np.hstack([X_norm_np, y_np])
    header = ','.join(['flow', 'signal', 'type', 'x', 'y', 'degree', 'congestion_label'])
    np.savetxt(
        os.path.join(output_dir, 'training_dataset.csv'),
        training_data,
        delimiter=',',
        header=header,
        comments=''
    )
    
    print(f"\n✓ Outputs saved to '{output_dir}/'")
    print(f"  - embeddings.npz")
    print(f"  - embeddings.csv")
    print(f"  - features_normalized.csv")
    print(f"  - labels.csv")
    print(f"  - training_dataset.csv")


# ============================================================================
# 9. MAIN PIPELINE
# ============================================================================

def main():
    """Execute the complete GNN embedding pipeline."""
    
    print("=" * 80)
    print("PyTorch GNN Pipeline for Node Embedding Generation")
    print("=" * 80)
    
    # Step 1-5: Generate multiple graphs, features, labels (100 iterations)
    print("\n[1/5] Generating 100 iterations of graphs, features, and labels...")
    
    all_X_norm = []
    all_y = []
    all_edge_indices = []
    all_edge_attrs = []
    
    for t in range(100):
        # Generate graph
        edge_index = generate_graph(N_NODES)
        all_edge_indices.append(edge_index)
        
        # Generate node features
        X = generate_features(edge_index, N_NODES)
        
        # Generate edge features
        edge_attr = generate_edge_features(edge_index)
        all_edge_attrs.append(edge_attr)
        
        # Normalize features
        X_norm, mean, std = normalize_features(X)
        all_X_norm.append(X_norm)
        
        # Generate labels
        y, y_raw = generate_labels(X, N_NODES)
        all_y.append(y)
        
        if (t + 1) % 20 == 0:
            print(f"  ✓ Completed {t + 1}/100 iterations")
    
    # Concatenate all data
    X_norm = torch.cat(all_X_norm, dim=0)
    y = torch.cat(all_y, dim=0)
    
    print(f"  ✓ Total dataset generated: {X_norm.shape[0]} samples")
    print(f"  ✓ Node features shape: {X_norm.shape}")
    print(f"  ✓ Labels shape: {y.shape}")
    print(f"  ✓ Label range: [{y.min():.4f}, {y.max():.4f}]")
    
    # Step 2: Create a single representative graph for model training
    # (use the first graph as the structure, though edges vary per iteration)
    print("\n[2/5] Creating representative graph for training...")
    edge_index = all_edge_indices[0]  # Use first graph
    edge_attr = all_edge_attrs[0]     # Use first edge attributes
    print(f"  ✓ Graph created: {N_NODES} nodes, {edge_index.size(1)} edges")
    print(f"  ✓ Edge attributes shape: {edge_attr.shape}")
    
    # Step 3: Create model
    print("\n[3/5] Building custom GNN model...")
    input_dim = X_norm.shape[1]
    edge_dim = edge_attr.shape[1]
    model = CustomGNN(
        input_dim=input_dim,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        edge_dim=edge_dim
    )
    print(f"  ✓ Model architecture:")
    print(f"    - Input dimension: {input_dim}")
    print(f"    - Embedding dimension: {EMBEDDING_DIM}")
    print(f"    - Hidden dimension: {HIDDEN_DIM}")
    print(f"    - Edge dimension: {edge_dim}")
    
    # Step 4: Train model
    print(f"\n[4/5] Training GNN for {EPOCHS} epochs...")
    losses, embeddings = train_model(
        model, X_norm, edge_index, edge_attr, y,
        epochs=EPOCHS, lr=LEARNING_RATE
    )
    print(f"  ✓ Training complete!")
    print(f"  ✓ Final loss: {losses[-1]:.6f}")
    
    # Step 5: Save outputs
    print("\n[5/5] Saving dataset and outputs...")
    save_embeddings(embeddings, X_norm, y)
    
    # Summary
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"\nSummary:")
    print(f"  Dataset iterations: 100")
    print(f"  Graph:        {N_NODES} nodes, {edge_index.size(1)} edges (fully connected)")
    print(f"  Total samples: {X_norm.shape[0]} (100 iterations × {N_NODES} nodes)")
    print(f"  Features:     X ∈ R^({X_norm.shape[0]} × {X_norm.shape[1]})")
    print(f"  Edge attrs:   E ∈ R^({edge_attr.shape[0]} × {edge_attr.shape[1]})")
    print(f"  Embeddings:   H ∈ R^({embeddings.shape[0]} × {embeddings.shape[1]})")
    print(f"  Labels:       y ∈ R^({y.shape[0]} × {y.shape[1]})")
    print(f"  Epochs:       {EPOCHS}")
    print(f"  Final Loss:   {losses[-1]:.6f}")
    print("\nEmbeddings are ready for dimensionality reduction and quantum processing!")
    print("=" * 80)
    
    return {
        'edge_index': edge_index,
        'X': X,
        'X_norm': X_norm,
        'edge_attr': edge_attr,
        'y': y,
        'embeddings': embeddings,
        'losses': losses,
        'model': model
    }


if __name__ == '__main__':
    results = main()
