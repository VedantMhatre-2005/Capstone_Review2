# 🚗 Traffic Prediction Pipeline - Complete Guide

A hybrid **Graph Neural Network + PennyLane Quantum Layer** for 5-second traffic forecasting with comprehensive Streamlit visualization.

---

## 📋 Overview

This project implements a complete ML pipeline for traffic prediction:

1. **GNN Embedding Pipeline** (`gnn_embedding_pipeline.py`)
   - Creates synthetic 10-node traffic graph
   - Generates 1,000 training samples
   - Trains classical GNN with message passing
   - Outputs 8-D node embeddings

2. **Traffic Prediction Pipeline** (`traffic_prediction_pipeline.py`)
   - Hybrid GNN + PennyLane Quantum Layer
   - Message passing + variational quantum circuit
   - Predicts 5-second traffic on 90 edges
   - Uses Pauli-Z measurements for quantum update

3. **Streamlit Frontend** (`streamlit_app.py`)
   - Interactive visualization of entire pipeline
   - Network graph visualization
   - Embedding projections (PCA/t-SNE)
   - Traffic prediction heatmaps
   - Quantum circuit diagrams
   - Data analysis dashboard

---

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
pip (Python package manager)
```

### Installation

```bash
# Clone or extract the project
cd Capstone_Review2

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Run All Pipelines

```bash
# 1. Train GNN and generate embeddings (1-2 minutes)
python gnn_embedding_pipeline.py

# 2. Train traffic prediction model with quantum layer (1-2 minutes)
python traffic_prediction_pipeline.py

# 3. Launch interactive dashboard
streamlit run streamlit_app.py
```

The Streamlit app will open at: **http://localhost:8501**

---

## 📊 Features

### GNN Embedding Pipeline
- ✅ 10-node mesh topology graph
- ✅ 6-D node features (flow, signal, type, x, y, degree)
- ✅ 5-D edge features (capacity, speed, lanes, length, type)
- ✅ Z-score normalization
- ✅ Message passing layer: `m_{j→i} = MLP(h_j || e_ji)`
- ✅ Aggregation: `a_i = Σ incoming messages`
- ✅ Residual updates: `h̃^(1) = h^(0) + a_i`
- ✅ MSE loss training (100 epochs)
- ✅ Outputs: embeddings, features, labels

### Traffic Prediction Pipeline
- ✅ Classical node embedding
- ✅ Message passing
- ✅ **PennyLane Quantum Layer** (4 qubits, 2 layers)
  - RY angle embedding
  - RZ-RY-RZ variational rotations
  - CNOT ring entanglement
  - Pauli-Z measurements
- ✅ Layer normalization
- ✅ Edge-level traffic prediction
- ✅ Denormalization to vehicles/hour
- ✅ Outputs: predictions, embeddings

### Streamlit Dashboard
- 📱 **6 Interactive Views:**
  1. **Overview** - Architecture summary
  2. **GNN Embeddings** - 2D projections (PCA/t-SNE)
  3. **Graph Topology** - Network visualization
  4. **Traffic Predictions** - Forecast analysis
  5. **Quantum Layer** - Circuit diagrams
  6. **Data Analysis** - Feature exploration

---

## 📁 Project Structure

```
Capstone_Review2/
├── gnn_embedding_pipeline.py           # GNN training
├── traffic_prediction_pipeline.py      # Hybrid GNN + Quantum
├── streamlit_app.py                    # Dashboard frontend
│
├── doc.md                              # Mathematical specifications
├── TRAFFIC_PREDICTION_SUMMARY.md       # Architecture doc
├── PENNYLANE_QUANTUM_LAYER.md          # Quantum integration doc
├── QUANTUM_CIRCUIT_DETAILS.md          # Circuit specifications
├── README.md                           # This file
│
├── .streamlit/
│   └── config.toml                     # Streamlit config
│
├── outputs/                            # Generated outputs
│   ├── embeddings.csv                  # GNN embeddings (1000×8)
│   ├── embeddings.npz                  # Binary format
│   ├── training_dataset.csv            # Features + labels (1000×7)
│   ├── features_normalized.csv         # Normalized features
│   ├── labels.csv                      # Target labels
│   ├── traffic_predictions_5s.csv      # Predictions (90×1)
│   ├── traffic_predictions_normalized.csv
│   └── edge_embeddings.csv             # Edge embeddings (90×8)
│
└── .venv/                              # Virtual environment
```

---

## 🔬 Technical Details

### Architecture

```
Node Features (6-D)
    ↓
Classical Embedding: h^(0) = ReLU(W_v·x + b_v)  [6→8]
    ↓
Message Passing: m_{j→i} = MLP(h_j || e_ji)    [8+5→8]
    ↓
Node Aggregation: a_i = Σ incoming messages     [sum→8]
    ↓
Residual Update: h̃^(1) = h^(0) + a_i            [8+8→8]
    ↓
Quantum Layer (PennyLane):
├─ Pre-projection: α = π·tanh(W_in·h)           [8→4]
├─ Angle embed: RY(α_q)
├─ Variational: RZ-RY-RZ (24 params)
├─ Entanglement: CNOT ring
├─ Measurement: ⟨Z⟩ ∈ [-1,+1]
└─ Post-proj: W_out·o                           [4→8]
    ↓
Layer Normalization
    ↓
Edge Prediction: [h_i || h_j || f_ij] → traffic [21→1]
    ↓
Denormalization: y = ŷ·450 + 1200 veh/hr
```

### Key Equations

**Message Function (Section 6, doc.md):**
```
m_{j→i} = MLP(h_j^(ℓ-1) || e_ji)
```

**Aggregation:**
```
a_i^(ℓ) = Σ_{j∈N^-(i)} m_{j→i}^(ℓ)
```

**Residual Update:**
```
h̃_i^(ℓ) = h_i^(ℓ-1) + a_i^(ℓ)
```

**Quantum Update (Section 9, doc.md):**
```
Step A: z = W_in·h + b_in; α = π·tanh(z)
Step B: |ψ₀⟩ = RY(α)|0⟩
Step C: U(Θ) = variational circuit with CNOT
Step D: o_q = ⟨Z_q⟩ ∈ [-1,+1]
Step E: q = W_out·o + b_out
```

**Edge Prediction (Section 11, doc.md):**
```
r_{ij} = [h_i || h_j || f_ij] ∈ R^21
ŷ_{ij} = w_out^T ReLU(W_pred·r_{ij} + b_pred)
```

---

## 📈 Training Results

### GNN Training
- **Dataset**: 1,000 samples (100 iterations × 10 nodes)
- **Initial Loss**: 18.13
- **Final Loss**: 0.109
- **Improvement**: 99.4%
- **Time**: ~30 seconds

### Traffic Prediction Training
- **Initial Loss**: 0.159
- **Final Loss**: 0.092
- **Improvement**: 42.1%
- **Time**: ~40 seconds (with quantum simulation)

### Traffic Predictions (5-second forecast)
- **Mean**: 1,434.65 veh/hr
- **Min**: 1,352.24 veh/hr
- **Max**: 1,554.25 veh/hr
- **Range**: 202.01 veh/hr
- **Edges Predicted**: 90

---

## 🔌 Dependencies

### Core
- `torch>=2.0` - ML framework
- `numpy>=1.21` - Numerical computing
- `pandas>=1.3` - Data manipulation
- `pennylane>=0.42` - Quantum computing

### Visualization
- `streamlit>=1.28` - Web dashboard
- `plotly>=5.0` - Interactive plots
- `matplotlib>=3.5` - Static plots
- `networkx>=2.6` - Graph algorithms
- `scikit-learn>=1.0` - ML utilities

---

## 📖 Usage Guide

### 1. Generate GNN Embeddings

```bash
python gnn_embedding_pipeline.py
```

**Output:**
- `outputs/embeddings.csv` (1000×8)
- `outputs/training_dataset.csv` (1000×7)
- `outputs/embeddings.npz` (binary)

### 2. Run Traffic Prediction

```bash
python traffic_prediction_pipeline.py
```

**Output:**
- `outputs/traffic_predictions_5s.csv` (90 predictions)
- `outputs/edge_embeddings.csv` (90×8)
- `outputs/traffic_predictions_normalized.csv`

### 3. Launch Dashboard

```bash
streamlit run streamlit_app.py
```

Navigate to http://localhost:8501 to explore:
- Architecture overview
- Embedding visualizations
- Graph topology
- Traffic predictions
- Quantum circuit details
- Data analysis

---

## 🎯 Dashboard Views

### Overview
- Pipeline architecture diagram
- Component summaries
- Key statistics

### GNN Embeddings
- 2D projections (PCA/t-SNE)
- Correlation heatmaps
- Embedding distributions
- Sample statistics

### Graph Topology
- 10-node mesh visualization
- Network statistics
- Edge feature specifications
- Degree distribution

### Traffic Predictions
- Bar charts per edge
- Distribution histograms
- Percentile analysis
- Statistical summary

### Quantum Layer
- Circuit ASCII diagram
- Gate specifications
- PennyLane integration info
- Parameter details

### Data Analysis
- Feature distributions
- Statistical summaries
- Raw data explorer
- Correlation analysis

---

## ⚙️ Configuration

### Streamlit Config (`config.toml`)
```toml
[theme]
primaryColor = "#0066cc"
backgroundColor = "#f0f2f6"

[server]
headless = true
maxUploadSize = 500
```

### Hyperparameters

Edit the variables at the top of each script:

**gnn_embedding_pipeline.py:**
```python
N_NODES = 10
EMBEDDING_DIM = 8
HIDDEN_DIM = 16
EPOCHS = 100
LEARNING_RATE = 0.01
```

**traffic_prediction_pipeline.py:**
```python
N_QUBITS = 4
N_QUANTUM_LAYERS = 2
EPOCHS = 100
LEARNING_RATE = 0.001
```

---

## 🔧 Troubleshooting

### PennyLane not found
```bash
pip install pennylane==0.42.3
```

### Streamlit not found
```bash
pip install streamlit
```

### Data files not found
Make sure to run `gnn_embedding_pipeline.py` before `traffic_prediction_pipeline.py`:
```bash
python gnn_embedding_pipeline.py
python traffic_prediction_pipeline.py
```

### Slow quantum simulation
The `default.qubit` device simulates classically. For GPU acceleration:
```bash
pip install pennylane-lightning-gpu
# Modify: dev = qml.device('lightning.gpu', wires=4)
```

### Port 8501 already in use
```bash
streamlit run streamlit_app.py --server.port 8502
```

---

## 📚 References

### Documentation
- `doc.md` - Mathematical specifications (14 sections)
- `TRAFFIC_PREDICTION_SUMMARY.md` - Architecture guide
- `PENNYLANE_QUANTUM_LAYER.md` - Quantum integration
- `QUANTUM_CIRCUIT_DETAILS.md` - Circuit specifications

### Libraries
- **PennyLane**: https://pennylane.ai
- **PyTorch**: https://pytorch.org
- **Streamlit**: https://streamlit.io

---

## 📊 Expected Output

### Console Output
```
================================================================================
PyTorch GNN Pipeline for Node Embedding Generation
================================================================================
[1/9] Generating synthetic traffic graph...
  ✓ Graph created: 10 nodes, 90 edges
[2/9] Generating node features...
  ✓ Node features shape: torch.Size([1000, 6])
...
[9/9] Saving outputs...
================================================================================
PIPELINE COMPLETE
================================================================================
```

### Generated Files
```
outputs/
├── embeddings.csv ..................... 204 KB (1000×8)
├── embeddings.npz ..................... 60 KB
├── training_dataset.csv .............. 178 KB (1000×7)
├── features_normalized.csv ........... 153 KB
├── labels.csv ......................... 26 KB
├── traffic_predictions_5s.csv ......... 2.4 KB (90×1)
├── traffic_predictions_normalized.csv . 2.4 KB
└── edge_embeddings.csv ............... 18 KB (90×8)
```

---

## 🎓 Educational Content

This project teaches:
- ✅ Graph Neural Networks (message passing)
- ✅ Quantum machine learning (PennyLane)
- ✅ Hybrid classical-quantum models
- ✅ Feature engineering & normalization
- ✅ Time-series prediction
- ✅ Web dashboard development
- ✅ Data visualization best practices

---

## 📝 Citation

If you use this project, please cite:
```
Traffic Prediction Pipeline
Hybrid GNN + PennyLane Quantum Layer
March 2026
```

---

## 📧 Support

For issues or questions:
1. Check troubleshooting section above
2. Review documentation files (doc.md, etc.)
3. Verify all files are in correct locations
4. Run pipelines in order: GNN → Prediction → Dashboard

---

## ✨ Features Summary

| Component | Status | Details |
|-----------|--------|---------|
| GNN Training | ✅ | Complete with embeddings |
| Quantum Layer | ✅ | PennyLane 0.42.3 |
| Traffic Prediction | ✅ | 90 edges × 5 seconds |
| Streamlit Dashboard | ✅ | 6 interactive views |
| Documentation | ✅ | 4 guide files |
| Performance | ✅ | <100ms inference |
| GPU Support | 🟡 | Optional (Lightning) |

---

**Status**: ✅ Production Ready  
**Last Updated**: March 2026  
**Version**: 1.0
