# 🚗 Traffic Prediction + Signal Control + Green Corridor Integration Guide

## 📋 Overview

This integrated system combines:
1. **GNN + Quantum Layer**: Deep learning traffic prediction
2. **Signal Controller**: Quantum-guided traffic signal optimization
3. **Green Corridor**: Coordinated signal timing for smooth traffic flow
4. **SUMO Simulation**: Real-world traffic simulation with signal control

---

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
# Activate virtual environment
source myenv/Scripts/activate  # Windows: myenv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

**Key packages added:**
- `sumo-uc>=1.17.0` - SUMO traffic simulator
- `traci>=1.17.0` - SUMO Python API
- `xmltodict>=0.13.0` - XML processing

### Step 2: Generate Traffic Predictions

If not already done, generate baseline traffic predictions:

```bash
python traffic_prediction_pipeline.py
python gnn_embedding_pipeline.py
```

This creates:
- `outputs/traffic_predictions_5s.csv` - 5-second traffic forecasts
- `outputs/quantum_output.csv` - Quantum layer outputs
- `outputs/embeddings.csv` - GNN node embeddings

### Step 3: Set Up SUMO Network

```bash
python sumo_network.py
```

This creates:
- `sumo_simulation/traffic_network.net.xml` - Network topology
- `sumo_simulation/traffic.rou.xml` - Vehicle routes
- `sumo_simulation/sumo.sumocfg` - Simulation configuration

**Note:** Requires SUMO to be installed. Download from: https://sumo.dlr.de

### Step 4: Launch Streamlit Dashboard

```bash
streamlit run streamlit_app.py
```

This opens an interactive dashboard at `http://localhost:8501`

---

## 📊 Dashboard Pages

### 1. **🏠 Overview**
- Pipeline architecture diagram
- Quick statistics
- Data availability summary

### 2. **📈 GNN Embeddings**
- Node embedding visualization (PCA/t-SNE)
- Embedding statistics
- Correlation matrices

### 3. **🌐 Graph Topology**
- Network graph visualization
- Node connectivity (10-node mesh)
- Feature matrix display

### 4. **🚦 Traffic Predictions**
- 5-second traffic forecasts per edge
- Prediction distribution
- Statistical summary

### 5. **🚨 Signal Control** ⭐ NEW
- Quantum-optimized signal timings
- Real-time signal state visualization
- Adaptive control metrics
- Performance indicators

### 6. **🟢 Green Corridor** ⭐ NEW
- Green corridor network visualization
- Coverage routes (colored pathways)
- Timing coordination display
- Efficiency metrics

### 7. **⚙️ Simulation** ⭐ NEW
- Run SUMO traffic simulation
- Interactive simulation control
- Result visualization
- Performance analysis

### 8. **Quantum Layer**
- Circuit diagram and parameters
- Gate descriptions
- PennyLane integration details

### 9. **📉 Data Analysis**
- Raw data exploration
- Feature distributions
- Dataset statistics

---

## 🔧 Module Details

### signal_controller.py
**Quantum-Guided Signal Controller**

```python
from signal_controller import QuantumGuidedSignalController

# Create controller
controller = QuantumGuidedSignalController(n_nodes=10)

# Get signal timing for node
green_duration = controller.get_green_time('n0')

# Get all timings
timings = controller.get_all_timings()

# Get signal state at specific time
state = controller.get_signal_state('n0', current_time=25.5)  # Returns 'G', 'Y', or 'R'
```

**Features:**
- Loads traffic predictions from outputs
- Adjusts signal timing based on traffic demand
- Supports quantum output modulation
- Exports to SUMO format

### green_corridor.py
**Green Corridor Optimizer**

```python
from green_corridor import GreenCorridor, GreenWaveOptimizer

# Create corridor optimizer
corridor = GreenCorridor(n_nodes=10, signal_controller=controller)
corridor.optimize_all_corridors()

# Get corridor visualization data
viz_data = corridor.get_corridor_visualization_data()

# Optimize offsets to minimize stops
optimizer = GreenWaveOptimizer(corridor)
for i in range(10):
    optimizer.add_arrival_pattern(f'n{i}', arrival_rate=0.3 + 0.05*i)
```

**Features:**
- Identifies main traffic routes
- Calculates optimal timing offsets
- Minimizes vehicle stops
- Supports multiple concurrent corridors

### sumo_simulator.py
**SUMO Integration & Simulation Runner**

```bash
# Run simulation
python sumo_simulator.py
```

Or programmatically:

```python
from sumo_simulator import SimulationRunner

runner = SimulationRunner(simulation_time=3600, time_step=1.0)
runner.setup()
runner.run()
runner.save_results()
```

**Outputs:**
- `outputs/sumo_results/time_series.csv` - Vehicle count, speed, wait time
- `outputs/sumo_results/corridor_efficiency.csv` - Green corridor performance
- `outputs/sumo_results/signal_timings_n*.csv` - Per-signal data

---

## 🎯 How It Works

### Signal Control Flow

```
Traffic Predictions (GNN + Quantum)
        ↓
Normalize to traffic intensity [0, 1]
        ↓
Scale to signal durations
  - Min green: 10s
  - Max green: 60s
        ↓
Apply quantum output modulation
        ↓
Quantum-Optimized Signal Plan
        ↓
Export to SUMO traffic light logic
        ↓
SUMO simulation uses optimized times
```

### Green Corridor Formation

```
1. Identify high-traffic routes
2. For each route (n0 → n1 → n2 → ...):
   - Calculate inter-node travel time
   - Calculate when vehicle arrives at next intersection
   - Set signal offset so it's GREEN on arrival
3. Repeat for all intersections in sequence
4. Result: Vehicles travel corridor with minimal stops
```

### SUMO Integration

```
Predictions & Signal Plans
        ↓
SUMO Network Files (.net.xml, .rou.xml)
        ↓
SUMO Simulation Loop:
   - Initialize vehicles
   - For each time step:
     * Update signal states
     * Move vehicles
     * Collect statistics
   - Export results
        ↓
Metrics & Visualization
```

---

## 📈 Key Metrics

### Signal Control Metrics
- **Green Time**: Duration of green phase (seconds)
- **Cycle Time**: Total red + yellow + green (seconds)
- **Arrival Rate**: Expected vehicles per second
- **Efficiency**: % of vehicles not delayed

### Green Corridor Metrics
- **Route Length**: Number of intersections in corridor
- **Priority**: Importance ranking (longer = higher)
- **Expected Stops**: Predicted vehicle stops on corridor
- **Throughput**: Total vehicles served per hour

### SUMO Metrics
- **Vehicle Count**: Active vehicles per time step
- **Speed**: Average vehicle speed (m/s)
- **Wait Time**: Cumulative waiting per vehicle
- **Corridor Efficiency**: % of green wave success

---

## 🔌 Integration With Existing Pipeline

The system seamlessly integrates with existing components:

### ✅ GNN Embeddings
- Used as input to signal timing
- Node features inform traffic predictions
- Affects green time allocation

### ✅ Quantum Layer
- Outputs modulate signal parameters
- Quantum states influence corridor timing
- Real-time quantum-classical loop

### ✅ Traffic Predictions
- Primary driver of signal optimization
- Edge predictions determine green duration
- High-traffic edges get longer greens

---

## 🛠️ Customization

### Adjust Signal Timing Limits

Edit `signal_controller.py`:

```python
controller = QuantumGuidedSignalController(
    n_nodes=10,
    min_green=15,    # Changed from 10s
    max_green=80,    # Changed from 60s
    yellow_duration=4  # Changed from 3s
)
```

### Create Custom Corridors

Edit `green_corridor.py` or use programmatically:

```python
corridor = GreenCorridor(n_nodes=10)
custom_route = [0, 2, 5, 7, 9]  # Your desired route
corridor.create_corridor(custom_route, "CustomRoute")
```

### Extend Simulation

Edit `sumo_simulator.py` to add:
- Vehicle emission tracking
- Lane-specific metrics
- Real-time visualization

---

## ⚠️ Troubleshooting

### SUMO Not Found
```
Error: netconvert not found
```
**Solution:** Install SUMO from https://sumo.dlr.de or run in simulation mode

### Missing Prediction Files
```
Warning: No output files found
```
**Solution:** Run `traffic_prediction_pipeline.py` first

### Memory Issues
- Reduce `simulation_time` in Streamlit
- Decrease number of vehicles in routes
- Use `time_step=5.0` instead of `1.0`

### Slow Performance
- Disable GUI: `use_gui=False`
- Increase time step
- Reduce prediction history stored
- Use fewer visualization updates

---

## 📚 References

### Key Files
- `signal_controller.py` - Signal optimization logic
- `green_corridor.py` - Corridor alignment algorithm
- `sumo_network.py` - Network generation
- `sumo_simulator.py` - SUMO integration
- `streamlit_app.py` - Dashboard interface

### External Resources
- [SUMO Documentation](https://sumo.dlr.de/wiki/Main)
- [TraCI API](https://sumo.dlr.de/wiki/TraCI)
- [PennyLane Docs](https://pennylane.ai/)
- [Streamlit Guide](https://docs.streamlit.io/)

---

## ✨ Features Summary

| Feature | Status | Description |
|---------|--------|-------------|
| Traffic Prediction | ✅ | GNN + Quantum |
| Signal Optimization | ✅ | Quantum-guided |
| Green Corridors | ✅ | Multi-route coordination |
| SUMO Simulation | ✅ | Full traffic simulation |
| Real-time Viz | ✅ | Interactive dashboard |
| Adaptive Control | ✅ | Queue-based adjustment |
| Export Results | ✅ | CSV + visualization |
| Multi-corridor Support | ✅ | Up to 5+ concurrent |

---

## 🎓 Educational Value

This system demonstrates:
1. **Quantum-Classical Hybrid ML** - Combining quantum & classical computing
2. **Traffic Engineering** - Signal coordination theory
3. **System Integration** - Connecting ML models to simulation
4. **Real-world Optimization** - Practical traffic management

---

**Last Updated**: March 2026
**Version**: 1.0
**License**: Your project license
