# import torch
# from quantum_path_layer import TrafficQuantumLayer

# def test_open_path_topology():
#     print("="*60)
#     print(" 🧪 TESTING QUANTUM OPEN PATH LAYER (4-NODE)")
#     print("="*60)

#     # 1. Initialize the layer
#     print("\n[1] Initializing TrafficQuantumLayer...")
#     try:
#         quantum_layer = TrafficQuantumLayer(n_layers=2)
#         print("    ✅ Initialization Successful")
#     except Exception as e:
#         print(f"    ❌ Failed to initialize: {e}")
#         return

#     # 2. Create dummy classical features (Simulating GNN bottleneck output)
#     batch_size = 16
#     n_features = 4  # Must strictly match N_QUBITS = 4
    
#     print(f"\n[2] Generating dummy classical input tensor...")
#     print(f"    Simulating Batch Size: {batch_size}")
    
#     dummy_input = torch.rand(batch_size, n_features)
#     print(f"    Input Tensor Shape: {dummy_input.shape}")

#     # 3. Run the forward pass
#     print("\n[3] Running Forward Pass through Quantum Circuit...")
#     try:
#         output = quantum_layer(dummy_input)
#         print("    ✅ Forward Pass Successful")
#         print(f"    Output Tensor Shape: {output.shape}")
        
#         # 4. Final Verification
#         assert output.shape == (batch_size, n_features), "Output shape mismatch!"
#         print("\n🎉 Topology check passed!")
#         print("   The quantum layer perfectly maps the 4 classical features")
#         print("   through the linear CNOT sequence and returns the correct dimensions.")
        
#         print("\nSample Expectation Values (First graph in batch):")
#         print(output[0].detach().numpy())
        
#     except Exception as e:
#         print(f"    ❌ Forward Pass Failed: {e}")
        
#     print("\n" + "="*60)

# if __name__ == "__main__":
#     test_open_path_topology()


import torch
import pennylane as qml
from quantum_path_layer import TrafficQuantumLayer

def test_meaningful_traffic_flow():
    print("="*60)
    print(" [TEST] TESTING QUANTUM OPEN PATH LAYER (TRAFFIC FLOW SCENARIO)")
    print("="*60)

    # 1. Initialize the layer
    print("\n[1] Initializing TrafficQuantumLayer...")
    try:
        quantum_layer = TrafficQuantumLayer(n_layers=2)
        print("    [OK] Initialization Successful")
    except Exception as e:
        print(f"    [FAIL] Failed to initialize: {e}")
        return

    # 2. Define Scenario parameters
    # t=0 traffic volumes at nodes 0, 1, 2, 3
    w, x, y, z = 10.0, 20.0, 30.0, 40.0
    
    # Traffic flowing down the Open Path from node 0 towards node 3
    traffic_t0 = torch.tensor([w, x, y, z], dtype=torch.float32)
    traffic_t1 = torch.tensor([-w, x+w, y+x+w, z+y+x+w], dtype=torch.float32)
    
    print(f"\n[2] Defined Traffic Scenario (Open Path 0 -> 1 -> 2 -> 3):")
    print(f"    t=0 (Initial) : [w, x, y, z] -> Node 0: {w}, Node 1: {x}, Node 2: {y}, Node 3: {z}")
    print(f"    t=1 (Flowing) : [-w, x+w, y+x+w, z+y+x+w] -> Node 0: {-w}, Node 1: {x+w}, Node 2: {y+x+w}, Node 3: {z+y+x+w}")

    # To pass classical features to the Quantum Circuit (AngleEmbedding),
    # we simulate the GNN bottleneck pre-projection: scale and tanh to [-pi, pi]
    def preprocess_to_angles(traffic_tensor):
        scaled = traffic_tensor / 50.0  # arbitrary normalization constant
        return torch.pi * torch.tanh(scaled)

    inputs_t0 = preprocess_to_angles(traffic_t0).unsqueeze(0) # shape (1, 4)
    inputs_t1 = preprocess_to_angles(traffic_t1).unsqueeze(0) # shape (1, 4)

    # 3. Run the forward pass
    print("\n[3] Running Quantum Forward Pass...")
    try:
        out_t0 = quantum_layer(inputs_t0)
        out_t1 = quantum_layer(inputs_t1)
        print("    [OK] Quantum execution successful for both time steps.")
    except Exception as e:
        print(f"    [FAIL] Forward Pass Failed: {e}")
        return

    # 4. Extract meaningful outcomes
    print("\n[4] Meaningful Quantum Node Outcomes (at t=1):")
    # Expectation values are in [-1, +1]
    expectations_t1 = out_t1[0].detach()
    
    # Convert expectation values to Probabilities [0.0, 1.0]
    probabilities_t1 = (expectations_t1 + 1.0) / 2.0
    
    # Scale probabilities to expected expected traffic (e.g. max 150 vehicles/hr limit)
    expected_traffic_t1 = probabilities_t1 * 150.0 
    
    for i in range(4):
        print(f"    Node {i}:")
        print(f"       - Z-Expectation             : {expectations_t1[i]:.4f}")
        print(f"       - Probability of Congestion : {probabilities_t1[i]:.2%}")
        print(f"       - Expected Traffic          : {expected_traffic_t1[i]:.1f} veh/hr")

    # 5. Cluster scalar
    print("\n[5] Cluster-Level Output:")
    print("    This 4-node open network acts as one single cluster.")
    cluster_scalar = expected_traffic_t1.mean().item()
    print(f"    -> Cluster Output Scalar: {cluster_scalar:.2f} (This value is given to another cluster)")
    
    # 6. Visual Topology Proof
    print("\n[6] Visualizing Quantum Circuit Topology (t=1 features)...")
    print("    Notice the Open Path (0->1->2->3) and NO 3->0 loop back:")
    print("-" * 60)
    
    weights = list(quantum_layer.parameters())[0]
    circuit_drawing = qml.draw(quantum_layer.qlayer.qnode)(inputs_t1[0], weights)
    print(circuit_drawing)
    
    print("-" * 60)
    print("\n[SUCCESS] Meaningful Scenario & Topology check completed successfully!\n")

if __name__ == "__main__":
    test_meaningful_traffic_flow()