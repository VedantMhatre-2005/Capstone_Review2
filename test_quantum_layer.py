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
import pandas as pd
import numpy as np
import pennylane as qml
from quantum_path_layer import TrafficQuantumLayer
import os

def test_real_time_classical_integration():
    print("="*60)
    print(" [TEST] INTEGRATING QUANTUM LAYER WITH CLASSICAL GCN STREAM")
    print("="*60)

    # 1. Initialize the layer
    print("\n[1] Initializing TrafficQuantumLayer...")
    try:
        quantum_layer = TrafficQuantumLayer(n_layers=2)
        print("    [OK] Initialization Successful")
    except Exception as e:
        print(f"    [FAIL] Failed to initialize: {e}")
        return

    # 2. Ingest real-time compressed classical inputs (Frequency Domain FFT/DCT)
    csv_path = os.path.join("outputs", "quantum_input.csv")
    if not os.path.exists(csv_path):
        print(f"    [FAIL] Cannot find {csv_path}. Make sure classical pipeline has been run.")
        return

    print(f"\n[2] Ingesting Classical Spatio-Temporal Embeddings...")
    print(f"    Reading from: {csv_path}")
    
    df_inputs = pd.read_csv(csv_path, header=None)
    classical_tensor = torch.tensor(df_inputs.values, dtype=torch.float32)
    total_steps = classical_tensor.shape[0]
    
    print(f"    [OK] Loaded {total_steps} real-time spatio-temporal frames.")
    
    # 3. Simulate real-time continuous traffic flow processing
    print("\n[3] Simulating Live Quantum Flow Processing (First 5 frames)...")
    
    expected_traffic_per_node = []
    cluster_scalars = []
    
    # The quantum_input.csv values are pre-projected/compressed, so no need for further tanh
    # We feed them straight into the AngleEmbedding
    
    for t in range(min(5, total_steps)):
        # Grab a single classical time-frame
        frame_input = classical_tensor[t].unsqueeze(0) # shape (1, 4)
        
        # Execute Quantum pass
        quantum_out = quantum_layer(frame_input)
        z_expectations = quantum_out[0].detach()
        
        # Map Z out to probability space [0, 1]
        probabilities = (z_expectations + 1.0) / 2.0
        
        # Compute meaningful macro outcomes
        expected_traffic = probabilities * 2000.0 # Realistic constraint (veh/hr scale factor)
        cluster_scalar = expected_traffic.mean().item()
        
        print(f"    [Frame T={t}] Classical Input (Compressed): [{frame_input[0,0]:.2f}, {frame_input[0,1]:.2f}, {frame_input[0,2]:.2f}, {frame_input[0,3]:.2f}]")
        print(f"                -> Quantum Node Predicts:   [Q0: {expected_traffic[0]:.1f}, Q1: {expected_traffic[1]:.1f}, Q2: {expected_traffic[2]:.1f}, Q3: {expected_traffic[3]:.1f}] veh/hr")
        print(f"                -> Cluster Scalar Out:      {cluster_scalar:.2f} veh/hr")

    # 4. Batch Process entire simulation for MARL pipeline
    print(f"\n[4] Batch processing all {total_steps} frames for MARL algorithm...")
    try:
        # Pytorch automatic batch processing over Pennylane
        all_quantum_out = quantum_layer(classical_tensor) # shape (total_steps, 4)
        z_out_batch = all_quantum_out.detach()
        
        prob_batch = (z_out_batch + 1.0) / 2.0
        expected_traffic_batch = prob_batch * 2000.0
        
        print("    [OK] Full dataset mapped through VQC Entanglement block.")
        
        # 5. Output enhanced embeddings integration
        out_csv = os.path.join("outputs", "quantum_enhanced_embeddings.csv")
        
        # Convert expected traffic into DataFrame and save for MARL agent
        df_out = pd.DataFrame(expected_traffic_batch.numpy(), columns=["Q0_Traffic", "Q1_Traffic", "Q2_Traffic", "Q3_Traffic"])
        
        # Also compute and save the Cluster Scalar column
        df_out["Cluster_Scalar"] = df_out.mean(axis=1)
        
        df_out.to_csv(out_csv, index=False)
        print(f"\n[5] Quantum Enhanced Integration Complete.")
        print(f"    [OK] Saved resulting outputs & Cluster Scalars to: {out_csv}")
        print(f"    These realistic embeddings are now ready to be consumed by the MARL Layer!")
        
    except Exception as e:
        print(f"    [FAIL] Batch processing failed: {e}")
        return

    print("\n[SUCCESS] Pipeline Integration check completed successfully!\n")

if __name__ == "__main__":
    test_real_time_classical_integration()