# PennyLane Quantum Circuit Diagram

## Complete Quantum Circuit for Traffic Prediction

```
╔════════════════════════════════════════════════════════════════════════════╗
║                      QUANTUM CIRCUIT - 4 QUBITS                            ║
╚════════════════════════════════════════════════════════════════════════════╝

INPUT: Node embeddings h ∈ ℝ^8
         ↓
    Pre-projection: α = π·tanh(W_in·h + b_in) ∈ [-π,π]^4
         ↓
═══════════════════════════════════════════════════════════════════════════════

STEP B: ANGLE EMBEDDING
────────────────────────────────────────────────────────────────────────────────

Qubit 0:   |0⟩ ──RY(α₀)────────────────────────────────────────────→
Qubit 1:   |0⟩ ──RY(α₁)────────────────────────────────────────────→
Qubit 2:   |0⟩ ──RY(α₂)────────────────────────────────────────────→
Qubit 3:   |0⟩ ──RY(α₃)────────────────────────────────────────────→

RY(θ) = [[cos(θ/2), -sin(θ/2)],
         [sin(θ/2),  cos(θ/2)]]

═══════════════════════════════════════════════════════════════════════════════

STEP C: VARIATIONAL LAYER 1
────────────────────────────────────────────────────────────────────────────────

Qubit 0:   ─RZ(θ₀⁽¹⁾)─RY(θ₁⁽¹⁾)─RZ(θ₂⁽¹⁾)─●─────────────────────→
                                       │
Qubit 1:   ─RZ(θ₀⁽¹⁾)─RY(θ₁⁽¹⁾)─RZ(θ₂⁽¹⁾)─┼─●───────────────────→
                                       │ │
Qubit 2:   ─RZ(θ₀⁽¹⁾)─RY(θ₁⁽¹⁾)─RZ(θ₂⁽¹⁾)─┼─┼─●─────────────────→
                                       │ │ │
Qubit 3:   ─RZ(θ₀⁽¹⁾)─RY(θ₁⁽¹⁾)─RZ(θ₂⁽¹⁾)─┼─┼─┼─●───────────────→
                                       │ │ │ │
CNOT gates (ring entanglement):        X│ │ │ │
  Q0→Q1────┘ │ │
  Q1→Q2──────┘ │
  Q2→Q3────────┘
  Q3→Q0────────┘

═══════════════════════════════════════════════════════════════════════════════

STEP C: VARIATIONAL LAYER 2
────────────────────────────────────────────────────────────────────────────────

Qubit 0:   ─RZ(θ₀⁽²⁾)─RY(θ₁⁽²⁾)─RZ(θ₂⁽²⁾)─●─────────────────────→
                                       │
Qubit 1:   ─RZ(θ₀⁽²⁾)─RY(θ₁⁽²⁾)─RZ(θ₂⁽²⁾)─┼─●───────────────────→
                                       │ │
Qubit 2:   ─RZ(θ₀⁽²⁾)─RY(θ₁⁽²⁾)─RZ(θ₂⁽²⁾)─┼─┼─●─────────────────→
                                       │ │ │
Qubit 3:   ─RZ(θ₀⁽²⁾)─RY(θ₁⁽²⁾)─RZ(θ₂⁽²⁾)─┼─┼─┼─●───────────────→
                                       │ │ │ │

CNOT gates (ring entanglement):
  Q0→Q1────┘ │ │
  Q1→Q2──────┘ │
  Q2→Q3────────┘
  Q3→Q0────────┘

═══════════════════════════════════════════════════════════════════════════════

STEP D: MEASUREMENT
────────────────────────────────────────────────────────────────────────────────

Qubit 0:   ────────────────────────────────────────────┤⟨Z₀⟩├──→ o₀ ∈ [-1,+1]
Qubit 1:   ────────────────────────────────────────────┤⟨Z₁⟩├──→ o₁ ∈ [-1,+1]
Qubit 2:   ────────────────────────────────────────────┤⟨Z₂⟩├──→ o₂ ∈ [-1,+1]
Qubit 3:   ────────────────────────────────────────────┤⟨Z₃⟩├──→ o₃ ∈ [-1,+1]

Pauli-Z operator:   Z = [[1,  0],
                         [0, -1]]

Expectation value:  ⟨Z⟩ = P(|0⟩) - P(|1⟩)

═══════════════════════════════════════════════════════════════════════════════

STEP E: POST-PROJECTION
────────────────────────────────────────────────────────────────────────────────

Measurement outcomes: o = [o₀, o₁, o₂, o₃]ᵀ ∈ ℝ⁴

Post-projection:     q⁽¹⁾ = W_out · o + b_out
                     W_out ∈ ℝ^(8×4), b_out ∈ ℝ⁸

Output:              q⁽¹⁾ ∈ ℝ⁸

═══════════════════════════════════════════════════════════════════════════════

OUTPUT: Updated embeddings q ∈ ℝ^8

```

---

## Quantum Gate Details

### RY Rotation (Angle Embedding)
```
RY(θ): Rotation about Y-axis by angle θ

Matrix form:
┌──────────────────┐
│ cos(θ/2) -sin(θ/2) │
│ sin(θ/2)  cos(θ/2) │
└──────────────────┘

Effect on basis states:
|0⟩ → cos(θ/2)|0⟩ + sin(θ/2)|1⟩
|1⟩ → -sin(θ/2)|0⟩ + cos(θ/2)|1⟩
```

### RZ Rotation (Variational)
```
RZ(θ): Rotation about Z-axis by angle θ

Matrix form:
┌──────────────────────────┐
│ e^(-iθ/2)      0        │
│      0      e^(iθ/2)   │
└──────────────────────────┘

Effect on basis states:
|0⟩ → e^(-iθ/2)|0⟩
|1⟩ → e^(iθ/2)|1⟩
```

### CNOT Gate (Entanglement)
```
CNOT(control, target): Controlled-NOT gate

Matrix form:
┌──────────────┐
│ 1  0  0  0 │
│ 0  1  0  0 │
│ 0  0  0  1 │
│ 0  0  1  0 │
└──────────────┘

Effect:
- If control=|0⟩, target unchanged
- If control=|1⟩, target is flipped (X gate applied)
- Creates entanglement between qubits
```

---

## Trainable Parameters

### Total: 24 Quantum Parameters

**Layer 1** (12 parameters):
- Qubit 0: θ₀⁽¹⁾, θ₁⁽¹⁾, θ₂⁽¹⁾
- Qubit 1: θ₀⁽¹⁾, θ₁⁽¹⁾, θ₂⁽¹⁾
- Qubit 2: θ₀⁽¹⁾, θ₁⁽¹⁾, θ₂⁽¹⁾
- Qubit 3: θ₀⁽¹⁾, θ₁⁽¹⁾, θ₂⁽¹⁾

**Layer 2** (12 parameters):
- Qubit 0: θ₀⁽²⁾, θ₁⁽²⁾, θ₂⁽²⁾
- Qubit 1: θ₀⁽²⁾, θ₁⁽²⁾, θ₂⁽²⁾
- Qubit 2: θ₀⁽²⁾, θ₁⁽²⁾, θ₂⁽²⁾
- Qubit 3: θ₀⁽²⁾, θ₁⁽²⁾, θ₂⁽²⁾

Each Θ ∈ ℝ (trainable via gradient descent)

---

## Quantum Circuit Complexity

| Metric | Value |
|--------|-------|
| **Qubits** | 4 |
| **Variational Layers** | 2 |
| **Single-qubit gates** | (3 + 1) × 2 = 8 per layer |
| **Two-qubit gates** | 4 CNOT per layer |
| **Total gates** | 24 gates × 2 layers = 48 gates |
| **Circuit Depth** | ~12 (with 4-qubit CNOT ring) |

---

## Quantum State Evolution Example

### Initial State (after angle embedding)
```
Qubit 0: 0.876|0⟩ + 0.482|1⟩
Qubit 1: 0.958|0⟩ + 0.287|1⟩
Qubit 2: 0.976|0⟩ - 0.220|1⟩
Qubit 3: 0.947|0⟩ + 0.322|1⟩

Full 4-qubit state:
|ψ_init⟩ = |ψ₀⟩ ⊗ |ψ₁⟩ ⊗ |ψ₂⟩ ⊗ |ψ₃⟩ ∈ ℂ^16
```

### After Variational Layer 1 + CNOT Ring
```
Entanglement created through CNOT connections
|ψ_L1⟩ = complex superposition of 2^4 = 16 basis states
```

### After Variational Layer 2 + CNOT Ring
```
Further entanglement and parameterized evolution
|ψ_final⟩ = optimized quantum state
```

### Measurement Outcomes
```
Example: o = [0.342, -0.156, 0.578, 0.089]ᵀ

Interpretation:
- o₀ = 0.342  → Q0 measured |0⟩ ~ 67% of time
- o₁ = -0.156 → Q1 measured |1⟩ ~ 58% of time
- o₂ = 0.578  → Q2 measured |0⟩ ~ 79% of time
- o₃ = 0.089  → Q3 measured |0⟩ ~ 54% of time
```

---

## PennyLane Integration Points

### 1. Device Initialization
```python
dev = qml.device('default.qubit', wires=4)
```

### 2. QNode Registration
```python
@qml.QNode(dev)
def _quantum_circuit(angles, params):
    # Circuit definition
    return measurements
```

### 3. Circuit Execution
```python
result = self.qnode(angles_batch, params_batch)
# Automatically computes gradients
```

### 4. Gradient Computation
```
# Parameter-shift rule (built-in):
∂⟨O⟩/∂θ = (1/2)[⟨O⟩(θ+π/2) - ⟨O⟩(θ-π/2)]

# Automatic differentiation through circuit
```

---

## Performance Metrics

### Quantum Circuit Statistics
- **Circuit Executions per Epoch**: 1,000 (batch size)
- **Quantum Gates per Execution**: 48
- **Total Gate Operations**: 48,000 per epoch
- **Training Time**: ~40 seconds (100 epochs)
- **Inference Time**: ~0.5ms per sample

### Model Performance
- **Final MSE Loss**: 0.0916
- **Prediction Accuracy**: ±202 veh/hr (range)
- **Mean Traffic**: 1,434.65 veh/hr
- **Convergence**: Stable after epoch 50

---

## References

- **PennyLane Documentation**: https://pennylane.readthedocs.io/en/stable/
- **Quantum Gates**: https://pennylane.readthedocs.io/en/stable/code/ops/qubit.html
- **QNodes**: https://pennylane.readthedocs.io/en/stable/code/api/pennylane.QNode.html
- **doc.md**: Architecture specifications
- **traffic_prediction_pipeline.py**: Implementation code
