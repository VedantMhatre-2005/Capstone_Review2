# LOAD THE EMBEDDINGS
import numpy as np

# Load .npz file
data = np.load("outputs/embeddings.npz")

# Check available keys (IMPORTANT for safety)
print("Keys in file:", data.files)

# Replace 'H' with correct key if needed
H = data[data.files[0]]   # auto-pick first array

print("Original Embeddings Shape:", H.shape)

# NORMALISE (Z-score)
H_norm = (H - np.mean(H)) / (np.std(H) + 1e-8)

# REDUCE DIMENSION (for quantum)
Z = H_norm[:, :4]   # 4 qubits

# SCALE FOR QUANTUM (angle encoding)
Z = np.tanh(Z) * np.pi

# FINAL SHAPE CHECK
print("Quantum Input Shape:", Z.shape)

# OPTIONAL: Save quantum-ready data
np.savez("outputs/quantum_input.npz", Z=Z)

# OPTIONAL: Save CSV for inspection
np.savetxt("outputs/quantum_input.csv", Z, delimiter=",")

print("Quantum-ready embeddings saved successfully!")