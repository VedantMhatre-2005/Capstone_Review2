"""
Streamlit Frontend for Traffic Prediction Pipeline Visualization
Visualizes GNN Embeddings, Quantum Layer, and Traffic Predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Traffic Prediction Pipeline",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# STYLING
# ============================================================================

st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #1f77b4;
    }
    .section-header {
        font-size: 28px;
        font-weight: bold;
        color: #0066cc;
        margin-top: 30px;
        margin-bottom: 20px;
    }
    .success-badge {
        background-color: #d4edda;
        color: #155724;
        padding: 10px 15px;
        border-radius: 5px;
        display: inline-block;
        margin: 5px 0;
    }
    .warning-badge {
        background-color: #fff3cd;
        color: #856404;
        padding: 10px 15px;
        border-radius: 5px;
        display: inline-block;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data
def load_data():
    """Load all pipeline outputs."""
    outputs_dir = Path('./outputs')
    
    data = {}
    
    # Load CSVs
    if (outputs_dir / 'training_dataset.csv').exists():
        data['training_dataset'] = pd.read_csv(outputs_dir / 'training_dataset.csv')
    
    if (outputs_dir / 'embeddings.csv').exists():
        data['embeddings'] = pd.read_csv(outputs_dir / 'embeddings.csv')
    
    if (outputs_dir / 'traffic_predictions_5s.csv').exists():
        data['traffic_predictions'] = pd.read_csv(outputs_dir / 'traffic_predictions_5s.csv')
    
    if (outputs_dir / 'traffic_predictions_normalized.csv').exists():
        data['traffic_norm'] = pd.read_csv(outputs_dir / 'traffic_predictions_normalized.csv')
    
    if (outputs_dir / 'edge_embeddings.csv').exists():
        data['edge_embeddings'] = pd.read_csv(outputs_dir / 'edge_embeddings.csv')
    
    if (outputs_dir / 'features_normalized.csv').exists():
        data['features'] = pd.read_csv(outputs_dir / 'features_normalized.csv')
    
    if (outputs_dir / 'labels.csv').exists():
        data['labels'] = pd.read_csv(outputs_dir / 'labels.csv')
    
    return data

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_graph_visualization(n_nodes=10):
    """Create a 10-node mesh graph visualization."""
    G = nx.complete_graph(n_nodes)
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        showlegend=False
    )
    
    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=[f'N{i}' for i in range(n_nodes)],
        textposition='top center',
        hoverinfo='text',
        hovertext=[f'Node {i}' for i in range(n_nodes)],
        marker=dict(
            showscale=True,
            color=list(range(n_nodes)),
            size=20,
            colorscale='Viridis',
            line_width=2
        )
    )
    
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title='10-Node Traffic Network Mesh Topology',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )
    
    return fig

def create_embedding_visualization(embeddings_df, method='PCA'):
    """Create 2D visualization of embeddings."""
    embeddings = embeddings_df.values
    
    if method == 'PCA':
        reducer = PCA(n_components=2)
        reduced = reducer.fit_transform(embeddings)
        variance = reducer.explained_variance_ratio_
    else:  # t-SNE
        reducer = TSNE(n_components=2, random_state=42, perplexity=5)
        reduced = reducer.fit_transform(embeddings)
        variance = None
    
    fig = px.scatter(
        x=reduced[:, 0],
        y=reduced[:, 1],
        hover_name=[f'Node {i}' for i in range(len(embeddings))],
        color=list(range(len(embeddings))),
        color_continuous_scale='Turbo',
        title=f'Node Embeddings - {method} Projection (dim 8 → 2)',
        labels={'x': f'{method} Component 1', 'y': f'{method} Component 2'}
    )
    
    fig.update_traces(marker=dict(size=12, line=dict(width=2, color='white')))
    
    return fig, variance

def create_traffic_prediction_bar(predictions_df):
    """Create bar chart of traffic predictions."""
    preds = predictions_df.iloc[:, 0].values
    edge_ids = [f"E{i}" for i in range(len(preds))]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=edge_ids,
        y=preds,
        marker=dict(
            color=preds,
            colorscale='RdYlGn_r',
            showscale=True,
            colorbar=dict(title="Traffic<br>(veh/hr)")
        ),
        text=np.round(preds, 1),
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Traffic: %{y:.2f} veh/hr<extra></extra>'
    ))
    
    fig.update_layout(
        title='5-Second Traffic Predictions per Edge',
        xaxis_title='Edge ID',
        yaxis_title='Traffic Volume (veh/hr)',
        height=500,
        showlegend=False
    )
    
    return fig

def create_distribution_plot(data):
    """Create distribution plot."""
    fig = px.histogram(
        x=data,
        nbins=30,
        title='Traffic Prediction Distribution',
        labels={'x': 'Traffic Volume (veh/hr)', 'count': 'Frequency'},
        color_discrete_sequence=['#1f77b4']
    )
    
    fig.add_vline(
        x=np.mean(data),
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {np.mean(data):.2f}",
        annotation_position="top right"
    )
    
    return fig

def create_quantum_circuit_diagram():
    """Create ASCII quantum circuit with Plotly."""
    circuit_text = """
    QUANTUM CIRCUIT (4 qubits, 2 layers)
    
    Qubit 0:  |0⟩ ──RY(α₀)──RZ/RY/RZ──●──CNOT──→ ⟨Z₀⟩
    Qubit 1:  |0⟩ ──RY(α₁)──RZ/RY/RZ──┼──●────→ ⟨Z₁⟩
    Qubit 2:  |0⟩ ──RY(α₂)──RZ/RY/RZ──┼──┼──●──→ ⟨Z₂⟩
    Qubit 3:  |0⟩ ──RY(α₃)──RZ/RY/RZ──┼──┼──┼──→ ⟨Z₃⟩
                                       └──┘  └──┘
    
    Legend:
    - RY(α):  Angle embedding
    - RZ/RY/RZ: Variational rotations
    - ●─────★: CNOT entanglement (ring topology)
    - ⟨Z⟩:     Pauli-Z measurement
    
    Trainable parameters: 24 (3 angles × 4 qubits × 2 layers)
    """
    
    return circuit_text

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.markdown("""
    # 🚗 Traffic Prediction Pipeline
    ##### Hybrid GNN + PennyLane Quantum Layer
    """)
    
    # Load data
    with st.spinner("Loading pipeline data..."):
        data = load_data()
    
    # Check data availability
    data_available = len(data) > 0
    
    if not data_available:
        st.error("❌ No output files found. Please run the pipelines first:")
        st.code("""
python gnn_embedding_pipeline.py
python traffic_prediction_pipeline.py
        """)
        return
    
    # ====== SIDEBAR ======
    st.sidebar.markdown("## 📊 Navigation")
    page = st.sidebar.radio(
        "Select View",
        ["🏠 Overview", "📈 GNN Embeddings", "🌐 Graph Topology", 
         "🚦 Traffic Predictions", "Junction Analysis", "Quantum Layer", "📉 Data Analysis"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📁 Data Summary")
    
    if 'training_dataset' in data:
        st.sidebar.metric("Training Samples", len(data['training_dataset']))
        st.sidebar.metric("Features", len(data['training_dataset'].columns))
    
    if 'traffic_predictions' in data:
        preds = data['traffic_predictions'].iloc[:, 0].values
        st.sidebar.metric("Edges Predicted", len(preds))
        st.sidebar.metric("Mean Traffic", f"{np.mean(preds):.0f} veh/hr")
    
    # ====== PAGE: OVERVIEW ======
    if page == "🏠 Overview":
        st.markdown('<div class="section-header">Pipeline Overview</div>', 
                   unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("### 🧠 GNN Embeddings")
            st.markdown("""
            - 10-node mesh topology
            - 1,000 training samples
            - 8-dimensional embeddings
            - Message passing layer
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("### ⚛️ Quantum Layer")
            st.markdown("""
            - 4 qubits (2 layers)
            - 24 trainable parameters
            - RY/RZ rotations + CNOT
            - Pauli-Z measurements
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("### 🚦 Traffic Prediction")
            st.markdown("""
            - 5-second forecast
            - 90 edges predicted
            - Denormalized (veh/hr)
            - MSE Loss: ~0.092
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 📋 Pipeline Architecture")
        
        architecture = """
        Node Features (6-D)
             ↓
        Classical Embedding: h⁽⁰⁾ = ReLU(W_v·x + b_v)
             ↓
        Message Passing: m_{j→i} = MLP(h_j || e_ji)
             ↓
        Aggregation: a_i = Σ incoming messages
             ↓
        Residual Update: h̃⁽¹⁾ = h⁽⁰⁾ + a_i
             ↓
        Quantum Layer (PennyLane):
        ├─ Pre-projection: α = π·tanh(W_in·h + b_in)
        ├─ Angle Embedding: RY(α_q)
        ├─ Variational Circuit: RZ-RY-RZ + CNOT ring
        ├─ Measurement: o_q = ⟨ψ|Z_q|ψ⟩
        └─ Post-projection: q = W_out·o + b_out
             ↓
        Layer Normalization
             ↓
        Edge Prediction: [h_i || h_j || f_ij] → traffic
             ↓
        Denormalization: y = ŷ·450 + 1200 veh/hr
        """
        
        st.code(architecture, language="text")
        
        # Quick stats
        st.markdown("### 📊 Quick Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        if 'training_dataset' in data:
            col1.metric("Node Features", data['training_dataset'].shape[1] - 1)
        
        if 'embeddings' in data:
            col2.metric("Embedding Dim", data['embeddings'].shape[1])
        
        if 'traffic_predictions' in data:
            preds = data['traffic_predictions'].iloc[:, 0].values
            col3.metric("Edges Count", len(preds))
            col4.metric("Prediction Range", f"{preds.max() - preds.min():.0f}")
    
    # ====== PAGE: EMBEDDINGS ======
    elif page == "📈 GNN Embeddings":
        st.markdown('<div class="section-header">GNN Node Embeddings</div>', 
                   unsafe_allow_html=True)
        
        if 'embeddings' not in data:
            st.warning("Embedding data not found")
            return
        
        embeddings_df = data['embeddings']
        
        # Embedding stats
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Samples", len(embeddings_df))
        col2.metric("Embedding Dimension", embeddings_df.shape[1])
        col3.metric("Mean Value", f"{embeddings_df.values.mean():.4f}")
        col4.metric("Std Dev", f"{embeddings_df.values.std():.4f}")
        
        st.markdown("---")
        
        # Visualization method
        col1, col2 = st.columns(2)
        with col1:
            method = st.radio("Dimensionality Reduction", ["PCA", "t-SNE"], horizontal=True)
        
        with col2:
            st.info(f"📌 Reducing 8D embeddings to 2D using {method}")
        
        # Create visualization
        with st.spinner(f"Computing {method} projection..."):
            fig, variance = create_embedding_visualization(embeddings_df, method)
        
        st.plotly_chart(fig, width='stretch')
        
        if variance is not None:
            st.markdown(f"**Variance Explained**: {variance[0]:.3f} (PC1), {variance[1]:.3f} (PC2)")
        
        # Embedding distribution
        st.markdown("### Embedding Statistics by Dimension")
        
        stats_df = embeddings_df.describe().T
        st.dataframe(stats_df, width='stretch')
        
        # Heatmap
        st.markdown("### Embedding Correlation Matrix")
        
        corr_matrix = embeddings_df.corr()
        
        fig_heatmap = go.Figure(
            data=go.Heatmap(
                z=corr_matrix.values,
                x=[f"Dim {i}" for i in range(embeddings_df.shape[1])],
                y=[f"Dim {i}" for i in range(embeddings_df.shape[1])],
                colorscale='RdBu',
                zmid=0
            )
        )
        
        st.plotly_chart(fig_heatmap, width='stretch')
    
    # ====== PAGE: GRAPH ======
    elif page == "🌐 Graph Topology":
        st.markdown('<div class="section-header">Traffic Network Graph</div>', 
                   unsafe_allow_html=True)
        
        st.info("10-node fully connected mesh topology with edge features")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = create_graph_visualization(n_nodes=10)
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            st.markdown("### Graph Statistics")
            st.metric("Nodes", 10)
            st.metric("Edges", 90)
            st.metric("Avg Degree", 9)
            st.metric("Topology", "Fully Connected")
            st.metric("Self-loops", "None")
            
            st.markdown("### Edge Features (5-D)")
            edge_features = ["ε_cap (capacity)", 
                            "ε_speed (speed limit)",
                            "ε_lanes (lanes)",
                            "ε_len (length)",
                            "ε_type (road type)"]
            for feat in edge_features:
                st.caption(f"• {feat}")
        
        st.markdown("---")
        
        st.markdown("### Node Feature Matrix")
        st.markdown("""
        Each node has 6 features:
        | Feature | Symbol | Range |
        |---------|--------|-------|
        | Flow | φ_flow | 200-2000 veh/hr |
        | Signal | φ_signal | 0-120 sec |
        | Type | φ_type | {0,1,2} |
        | X Coordinate | φ_x | [0,1] |
        | Y Coordinate | φ_y | [0,1] |
        | Degree | φ_deg | 9 (all nodes) |
        """)
    
    # ====== PAGE: TRAFFIC PREDICTIONS ======
    elif page == "🚦 Traffic Predictions":
        st.markdown('<div class="section-header">5-Second Traffic Forecast</div>', 
                   unsafe_allow_html=True)
        
        if 'traffic_predictions' not in data:
            st.warning("Traffic prediction data not found")
            return
        
        predictions = data['traffic_predictions'].iloc[:, 0].values
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Mean Traffic", f"{np.mean(predictions):.2f} veh/hr")
        col2.metric("Median", f"{np.median(predictions):.2f} veh/hr")
        col3.metric("Min", f"{np.min(predictions):.2f} veh/hr")
        col4.metric("Max", f"{np.max(predictions):.2f} veh/hr")
        
        st.markdown("---")
        
        # Traffic bar chart
        st.markdown("### Traffic per Edge")
        fig_bar = create_traffic_prediction_bar(data['traffic_predictions'])
        st.plotly_chart(fig_bar, width='stretch')
        
        st.markdown("---")
        
        # Distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Prediction Distribution")
            fig_dist = create_distribution_plot(predictions)
            st.plotly_chart(fig_dist, width='stretch')
        
        with col2:
            st.markdown("### Prediction Statistics")
            stats_data = {
                'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Range'],
                'Value': [
                    f"{np.mean(predictions):.2f}",
                    f"{np.median(predictions):.2f}",
                    f"{np.std(predictions):.2f}",
                    f"{np.min(predictions):.2f}",
                    f"{np.max(predictions):.2f}",
                    f"{np.max(predictions) - np.min(predictions):.2f}"
                ]
            }
            st.dataframe(pd.DataFrame(stats_data), width='stretch')
            
            st.markdown("### Percentiles")
            percentiles = [10, 25, 50, 75, 90]
            for p in percentiles:
                val = np.percentile(predictions, p)
                st.caption(f"P{p}: {val:.2f} veh/hr")
    
    # ====== PAGE: JUNCTION ANALYSIS ======
    elif page == "Junction Analysis":
        st.markdown('<div class="section-header">Junction-Specific Traffic Prediction</div>', 
                   unsafe_allow_html=True)
        
        if 'traffic_predictions' not in data or 'edge_embeddings' not in data:
            st.warning("Traffic prediction or edge embedding data not found")
            return
        
        predictions_df = data['traffic_predictions']
        edge_embeddings_df = data['edge_embeddings']
        traffic_norm_df = data.get('traffic_norm', None)
        
        # Total number of edges in 10-node mesh
        n_nodes = 10
        n_edges = n_nodes * (n_nodes - 1)  # Fully connected without self-loops: 90 edges
        
        st.markdown("### Select a Junction (Edge)")
        
        # Create edge selector
        col1, col2, col3 = st.columns(3)
        
        with col1:
            edge_id = st.slider(
                "Edge ID",
                min_value=0,
                max_value=n_edges - 1,
                value=0,
                help="Select an edge in the traffic network"
            )
        
        with col2:
            # Calculate source and target nodes for display
            source_node = edge_id // (n_nodes - 1)
            target_node = edge_id % (n_nodes - 1)
            if target_node >= source_node:
                target_node += 1
            
            st.metric(
                "Source Node",
                f"Node {source_node}",
                help=f"Source junction"
            )
        
        with col3:
            st.metric(
                "Target Node", 
                f"Node {target_node}",
                help=f"Destination junction"
            )
        
        st.markdown("---")
        
        # Get the prediction for this edge
        if edge_id < len(predictions_df):
            traffic_predicted = predictions_df.iloc[edge_id, 0]
            traffic_normalized = traffic_norm_df.iloc[edge_id, 0] if (traffic_norm_df is not None and edge_id < len(traffic_norm_df)) else (traffic_predicted - 1200) / 450
            edge_embedding = edge_embeddings_df.iloc[edge_id].values if edge_id < len(edge_embeddings_df) else None
            
            # Display prediction
            st.markdown("### 🚦 Traffic Prediction for Next 5 Seconds")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Predicted Traffic",
                    f"{traffic_predicted:.2f} veh/hr",
                    delta=f"({traffic_predicted - 1200:.2f})",
                    delta_color="off",
                    help="Traffic volume in vehicles per hour"
                )
            
            with col2:
                st.metric(
                    "Normalized Value",
                    f"{traffic_normalized:.6f}",
                    help="Value before denormalization"
                )
            
            with col3:
                confidence = min(100, max(0, (traffic_predicted - 1000) / 10))
                st.metric(
                    "Intensity Level",
                    f"{'🔴 High' if traffic_predicted > 1450 else '🟡 Medium' if traffic_predicted > 1400 else '🟢 Low'}",
                    help="Traffic congestion level"
                )
            
            st.markdown("---")
            
            # Explain button
            st.markdown("### 📐 Prediction Explanation")
            
            if st.button("🔬 Show Detailed Equations", key=f"explain_edge_{edge_id}"):
                with st.expander("📊 Complete Prediction Calculation Flow", expanded=True):
                    
                    st.markdown("""
                    This section shows the step-by-step mathematical computation that produced this traffic prediction.
                    """)
                    
                    # Denormalization details
                    st.markdown("##### Step 1: Denormalization Formula")
                    st.latex(r"""
                    y_{\text{actual}} = \hat{y}_{\text{normalized}} \cdot \sigma + \mu
                    """)
                    
                    st.markdown("""
                    Where:
                    - $\\hat{y}_{\\text{normalized}}$ = Normalized prediction (output from neural network)
                    - $\\sigma = 450.0$ = Standard deviation (normalization scale)
                    - $\\mu = 1200.0$ = Mean traffic (normalization center)
                    - $y_{\\text{actual}}$ = Final denormalized traffic [veh/hr]
                    """)
                    
                    # Specific values
                    st.markdown("##### Step 2: Your Specific Calculation")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Numerical Values:**")
                        st.markdown(f"""
                        - $\\hat{{y}}_{{\\text{{normalized}}}}$ = {traffic_normalized:.6f}
                        - $\\sigma$ = 450.0
                        - $\\mu$ = 1200.0
                        """)
                    
                    with col2:
                        st.markdown("**Calculation:**")
                        st.markdown(f"""
                        $y_{{\\text{{actual}}}} = {traffic_normalized:.6f} \\times 450.0 + 1200.0$
                        
                        $y_{{\\text{{actual}}}} = {traffic_predicted:.2f}$ veh/hr
                        """)
                    
                    st.markdown("---")
                    
                    # Full pipeline explanation
                    st.markdown("##### Step 3: Complete Neural Network Pipeline")
                    
                    st.markdown("""
                    The normalized prediction came from this hybrid GNN + Quantum pipeline:
                    """)
                    
                    st.latex(r"""
                    \begin{align}
                    h_i^{(0)} &= \text{ReLU}(W_v \cdot x_i + b_v) \quad \text{(Classical Node Embedding)} \\
                    m_{j \to i} &= \text{MLP}(h_j || e_{ji}) \quad \text{(Message Passing)} \\
                    a_i &= \sum_{j \in N(i)} m_{j \to i} \quad \text{(Aggregation)} \\
                    h_i^{(1)} &= h_i^{(0)} + a_i \quad \text{(Residual Update)} \\
                    h_i^{(1)} &= \text{LayerNorm}(h_i^{(1)} + Q_{\Theta}(h_i^{(1)})) \quad \text{(Quantum Update)} \\
                    \hat{y}_{ij} &= \sigma\left(\text{MLP}(h_i || h_j || f_{ij})\right) \quad \text{(Edge Prediction)}
                    \end{align}
                    """)
                    
                    st.markdown("---")
                    
                    # Quantum layer detail
                    st.markdown("##### Step 4: Quantum Layer Computation")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("""
                        **A. Pre-Projection (8D → 4 angles):**
                        
                        $\\alpha_q = \\pi \\cdot \\tanh(W_{in} \\cdot h + b_{in})$
                        
                        where $\\alpha_q \\in [-\\pi, +\\pi]$ for $q \\in \\{0,1,2,3\\}$
                        """)
                    
                    with col2:
                        st.markdown("""
                        **B. Angle Embedding:**
                        
                        For each qubit $q = 0, 1, 2, 3$:
                        
                        $RY(\\alpha_q)|0⟩$
                        """)
                    
                    st.markdown("""
                    **C. Variational Circuit (2 Layers):**
                    
                    For each layer $l = 1, 2$:
                    """)
                    
                    st.latex(r"""
                    \begin{align}
                    U_l &= e^{-i \theta_{l,2} Z} e^{-i \theta_{l,1} Y} e^{-i \theta_{l,0} Z} \quad \text{(Per qubit)} \\
                    \text{CNOT}(Q_0, Q_1); &\text{ CNOT}(Q_1, Q_2); \text{ CNOT}(Q_2, Q_3); \\
                    &\text{ CNOT}(Q_3, Q_0) \quad \text{(Ring Entanglement)}
                    \end{align}
                    """)
                    
                    st.markdown("""
                    **D. Measurement:**
                    
                    Pauli-Z expectation values:
                    """)
                    
                    st.latex(r"""
                    o_q = \langle \psi | Z_q | \psi \rangle \in [-1, +1]
                    """)
                    
                    st.markdown("""
                    **E. Post-Projection (4D → 8D):**
                    """)
                    
                    st.latex(r"""
                    q = W_{out} \cdot o + b_{out}
                    """)
                    
                    st.markdown("---")
                    
                    # Edge features
                    st.markdown("##### Step 5: Edge Prediction Input")
                    
                    if edge_embedding is not None:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("""
                            **Edge Embedding Dimensions:**
                            
                            The edge prediction layer concatenates:
                            - Node $i$ embedding: $h_i$ (8D)
                            - Node $j$ embedding: $h_j$ (8D)
                            - Edge features: $f_{ij}$ (5D)
                            
                            **Total input: 21D**
                            """)
                        
                        with col2:
                            st.markdown("**Current Edge Embedding Values:**")
                            
                            embedding_data = {
                                'Dimension': [f'D{i}' for i in range(len(edge_embedding))],
                                'Value': [f'{val:.6f}' for val in edge_embedding]
                            }
                            st.dataframe(pd.DataFrame(embedding_data), width='stretch')
                    
                    st.markdown("---")
                    
                    # Summary
                    st.markdown("##### 📌 Summary")
                    
                    summary_box = f"""
                    **For Edge {edge_id} (Node {source_node} → Node {target_node}):**
                    
                    1. Neural network processes node features through GNN + Quantum layers
                    2. Produces normalized prediction: {traffic_normalized:.6f}
                    3. Applies denormalization: {traffic_normalized:.6f} × 450 + 1200
                    4. **Final Result: {traffic_predicted:.2f} vehicles/hour**
                    
                    This prediction represents the expected traffic volume on this junction 5 seconds in the future.
                    """
                    
                    st.info(summary_box)
        
        else:
            st.error(f"Edge {edge_id} not found in predictions")
    
    # ====== PAGE: QUANTUM LAYER ======
    elif page == "Quantum Layer":
        st.markdown('<div class="section-header">Quantum Layer Architecture</div>', 
                   unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Quantum Circuit Diagram")
            
            circuit_text = create_quantum_circuit_diagram()
            st.code(circuit_text, language="text")
        
        with col2:
            st.markdown("### Quantum Parameters")
            
            param_data = {
                'Component': ['Qubits', 'Layers', 'Angles/Qubit', 'Total Angles', 
                            'Measurement', 'Pre-proj', 'Post-proj'],
                'Value': ['4', '2', '3 (RZ-RY-RZ)', '24', 'Pauli-Z', '8→4', '4→8']
            }
            st.dataframe(pd.DataFrame(param_data), width='stretch')
        
        st.markdown("---")
        
        st.markdown("### Quantum Gates")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### RY Gate")
            st.markdown("""
            **Rotation about Y-axis**
            
            Used for:
            - Angle embedding
            - Variational layer
            """)
        
        with col2:
            st.markdown("#### RZ Gate")
            st.markdown("""
            **Rotation about Z-axis**
            
            Used for:
            - Variational layer (θ₀, θ₂)
            - Phase rotations
            """)
        
        with col3:
            st.markdown("#### CNOT Gate")
            st.markdown("""
            **Controlled-NOT Gates**
            
            Creates:
            - Entanglement
            - Ring topology
            - Q0→Q1→Q2→Q3→Q0
            """)
        
        st.markdown("---")
        
        st.markdown("### PennyLane Integration")
        st.info("""
        ✅ Real quantum simulation using PennyLane 0.42.3
        
        - **Backend**: default.qubit (CPU simulator)
        - **Device**: 4 qubit simulator
        - **AD Support**: Automatic differentiation through quantum circuits
        - **Measurement**: Pauli-Z expectation values ⟨Z⟩ ∈ [-1, +1]
        """)
        
        st.markdown("### Training Flow")
        
        flow = """
        1. Classical pre-projection: 8-D → 4 angles
        2. Initialize qubits: |0⟩ → RY(α) rotations
        3. Layer 1: RZ-RY-RZ + CNOT ring
        4. Layer 2: RZ-RY-RZ + CNOT ring
        5. Measure: ⟨Z⟩ for each qubit
        6. Post-projection: 4-D → 8-D
        7. Layer norm: Normalize for stability
        """
        st.code(flow, language="text")
    
    # ====== PAGE: DATA ANALYSIS ======
    elif page == "📉 Data Analysis":
        st.markdown('<div class="section-header">Data Analysis & Exploration</div>', 
                   unsafe_allow_html=True)
        
        if 'training_dataset' not in data:
            st.warning("Training data not found")
            return
        
        df_train = data['training_dataset']
        
        st.markdown("### Training Dataset Summary")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Samples", len(df_train))
        col2.metric("Features", df_train.shape[1])
        col3.metric("Memory", f"{df_train.memory_usage(deep=True).sum() / 1024:.2f} KB")
        
        st.markdown("---")
        
        st.markdown("### Feature Statistics")
        st.dataframe(df_train.describe(), width='stretch')
        
        st.markdown("---")
        
        st.markdown("### Feature Distributions")
        
        features = df_train.columns.tolist()
        selected_features = st.multiselect(
            "Select features to visualize",
            features,
            default=features[:3]
        )
        
        if selected_features:
            for feature in selected_features:
                fig = px.histogram(
                    df_train,
                    x=feature,
                    nbins=30,
                    title=f"Distribution of {feature}",
                    color_discrete_sequence=['#636EFA']
                )
            st.plotly_chart(fig, width='stretch')
        st.markdown("### Raw Data Preview")
        
        n_rows = st.slider("Show rows:", 5, len(df_train), 10)
        st.dataframe(df_train.head(n_rows), width='stretch')
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #999;">
    Traffic Prediction Pipeline | GNN + Quantum Layer | PennyLane Integration
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
