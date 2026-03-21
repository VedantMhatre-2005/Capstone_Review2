# ================================
# QUANTUM LAYER USING PENNYLANE
# ================================

import numpy as np
import pennylane as qml

# -------------------------------
# 1. LOAD QUANTUM INPUT
# -------------------------------
data = np.load("outputs/quantum_input.npz")
Z = data["Z"]   # shape: (N, 4)

print("Loaded Quantum Input Shape:", Z.shape)

# -------------------------------
# 2. DEFINE QUANTUM DEVICE
# -------------------------------
n_qubits = 4

dev = qml.device("default.qubit", wires=n_qubits)

# -------------------------------
# 3. DEFINE QUANTUM CIRCUIT
# -------------------------------
@qml.qnode(dev)
def quantum_circuit(x, weights):
    # x: input features (length = n_qubits)

    # --- Angle Encoding ---
    for i in range(n_qubits):
        qml.RY(x[i], wires=i)

    # --- Entanglement ---
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])

    # --- Variational Layer ---
    for i in range(n_qubits):
        qml.RY(weights[i], wires=i)

    # --- Output ---
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


# -------------------------------
# 4. INITIALIZE PARAMETERS
# -------------------------------
np.random.seed(42)
weights = np.random.randn(n_qubits)

print("Initial Quantum Weights:", weights)

# -------------------------------
# 5. RUN QUANTUM LAYER
# -------------------------------
quantum_outputs = []

for i, z in enumerate(Z):
    output = quantum_circuit(z, weights)
    quantum_outputs.append(output)

quantum_outputs = np.array(quantum_outputs)

print("Quantum Output Shape:", quantum_outputs.shape)

# -------------------------------
# 6. SAVE OUTPUTS
# -------------------------------
np.savez("outputs/quantum_output.npz", Q=quantum_outputs)
np.savetxt("outputs/quantum_output.csv", quantum_outputs, delimiter=",")

print("Quantum outputs saved successfully!")

# -------------------------------
# 7. SAMPLE OUTPUT
# -------------------------------
print("\nSample Input (Z[0]):", Z[0])
print("Quantum Output (Q[0]):", quantum_outputs[0])