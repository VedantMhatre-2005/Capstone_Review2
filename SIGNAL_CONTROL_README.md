# 🚗 Traffic Signal Scheduling & SUMO Integration

## NEW FEATURES ADDED

This implementation adds complete traffic signal control and SUMO simulation integration to your existing GNN + Quantum layer traffic prediction system.

---

## 📦 What's New

### 1. **signal_controller.py** - Quantum-Guided Signal Optimization
Implements real-time traffic signal control using GNN + Quantum predictions:
- Loads traffic predictions from the quantum layer
- Optimizes signal timing based on predicted demand
- Adaptive control that responds to queue lengths
- Exports directly to SUMO format

**Key Classes:**
- `QuantumGuidedSignalController` - Main controller with optimization
- `AdaptiveSignalController` - Real-time queue-based adjustment
- `SignalPhase` - Phase timing definition

**Usage:**
```python
from signal_controller import QuantumGuidedSignalController

controller = QuantumGuidedSignalController(n_nodes=10)
green_time = controller.get_green_time('n0')  # Get green duration for node
state = controller.get_signal_state('n0', 25.5)  # Get signal at time 25.5s
```

---

### 2. **green_corridor.py** - Green Wave Coordination
Coordinates signals along routes to create "green waves" for smooth traffic flow:
- Identifies main traffic corridors
- Calculates optimal timing offsets
- Minimizes vehicle stops along corridors
- Supports multiple concurrent green corridors

**Key Classes:**
- `GreenCorridor` - Corridor definition and optimization
- `GreenWaveOptimizer` - Advanced green wave optimization

**Features:**
- Analyzes 5 main traffic routes automatically
- Calculates expected vehicle stops
- Tracks efficiency metrics

---

### 3. **sumo_network.py** - SUMO Network Generation
Creates traffic network and configuration files for SUMO simulation:
- Generates 10-node fully-connected mesh network
- Creates node and edge definitions
- Defines traffic light logic for each intersection
- Produces vehicle routes and simulation config

**Key Classes:**
- `SUMONetworkGenerator` - Network topology creation
- `SUMOConfigGenerator` - Route and config file generation

**Output Files:**
- `traffic_network.nod.xml` - Node definitions
- `traffic_network.edg.xml` - Edge definitions  
- `traffic_network.tll.xml` - Traffic light logic
- `traffic.rou.xml` - Vehicle routes
- `sumo.sumocfg` - Simulation configuration

---

### 4. **sumo_simulator.py** - SUMO Integration & Simulation
Runs traffic simulation with signal control and data collection:
- Starts SUMO traffic simulator with TraCI interface
- Controls signals using quantum-optimized timings
- Collects traffic metrics in real-time
- Evaluates green corridor performance
- Exports results to CSV files

**Key Classes:**
- `SUMOSimulator` - Main simulation wrapper
- `SimulationRunner` - High-level orchestration

**Outputs:**
```
outputs/sumo_results/
├── time_series.csv           # Vehicle count, speed, wait time
├── corridor_efficiency.csv   # Green corridor performance
├── signal_timings_n*.csv     # Per-intersection data
└── simulation_summary.txt    # Configuration & results
```

---

### 5. **Enhanced streamlit_app.py** - New Dashboard Pages
Added three new dashboard pages and helper functions:

#### New Pages:
- **🚨 Signal Control** - Visualize optimized signal timing
- **🟢 Green Corridor** - View coordinated signal paths
- **⚙️ Simulation** - Run and analyze SUMO simulation

#### New Helper Functions:
- `create_signal_timing_chart()` - Bar chart of green/yellow times
- `create_signal_state_visualization()` - Real-time signal states
- `create_green_corridor_visualization()` - Network with colored corridors
- `create_corridor_efficiency_chart()` - Performance metrics
- `create_signal_controller()` - Cached controller instance
- `create_green_corridor()` - Cached corridor instance

---

## 🚀 Quick Start

### 1. Install Updated Requirements
```bash
pip install -r requirements.txt
```

New packages:
- `sumo-uc>=1.17.0` - SUMO simulator
- `traci>=1.17.0` - Python API for SUMO
- `xmltodict>=0.13.0` - XML processing

### 2. Generate Test Data (if not already done)
```bash
python traffic_prediction_pipeline.py
python gnn_embedding_pipeline.py
```

### 3. Set Up SUMO Network
```bash
python sumo_network.py
```

Creates `sumo_simulation/` directory with network files.

**Note:** Requires SUMO installed. Download from: https://sumo.dlr.de

### 4. Run Tests
```bash
python test_integration.py
```

Validates all components and dependencies.

### 5. Launch Dashboard
```bash
streamlit run streamlit_app.py
```

Or use convenience scripts:
- **Linux/Mac**: `bash run.sh`
- **Windows**: `run.bat`

---

## 📊 Dashboard Pages Overview

### 🚨 Signal Control Page
**What it shows:**
- Quantum-optimized signal timing for all 10 intersections
- Green/yellow/red durations
- Real-time signal state changes over time
- Performance metrics (throughput, arrival rates)

**Visual Elements:**
- Stacked bar chart of green vs red time
- Timeline heatmap showing signal states
- Summary statistics and metrics

**Use Cases:**
- Verify signal timing is appropriate for traffic
- Understand how quantum layer influences signals
- Monitor adaptive adjustments

### 🟢 Green Corridor Page
**What it shows:**
- 5 main green corridors through the network
- Color-coded paths on network graph
- Timing synchronization for each corridor
- Efficiency metrics

**Visual Elements:**
- Network graph with colored corridor paths
- Corridor details table
- Efficiency bar chart
- Timing offset data

**Use Cases:**
- Visualize continuous green wave routes
- Check corridor coverage
- Understand signal coordination

### ⚙️ Simulation Page
**What it shows:**
- Interactive SUMO simulation runner
- Real-time progress tracking
- Traffic metrics (vehicle count, speed, wait time)
- Corridor efficiency during simulation

**Interactive Controls:**
- Simulation duration (60-infinite seconds)
- Time step size (0.1, 1.0, 5.0+ seconds)
- GUI toggle (requires SUMO installation)
- Start button

**Outputs:**
- Time series charts
- Corridor efficiency graph
- Wait time analysis

**Use Cases:**
- Run complete traffic simulations
- Test signal control effectiveness
- Analyze green corridor performance in realistic conditions

---

## 🔄 System Architecture

```
┌─────────────────────────────────────────────────────────┐
│         GNN + Quantum Layer Predictions                 │
│  (traffic_predictions_5s.csv, quantum_output.csv)       │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│     Signal Controller (quantum-guided)                   │
│  - Normalizes predictions to traffic intensity          │
│  - Scales to green times (10-60s)                       │
│  - Applies quantum modulation                           │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│     Green Corridor Optimizer                            │
│  - Identifies main traffic routes                       │
│  - Calculates timing offsets                            │
│  - Minimizes vehicle stops                              │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│     SUMO Network & Simulator                            │
│  - 10-node mesh network                                 │
│  - 90 bidirectional edges                               │
│  - 100 vehicles per simulation                          │
│  - Real-world traffic dynamics                          │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│     Results & Visualization                             │
│  - Streamlit dashboard                                  │
│  - CSV exports                                          │
│  - Performance metrics                                  │
└─────────────────────────────────────────────────────────┘
```

---

## 📈 Signal Optimization Algorithm

### Step 1: Load Predictions
- Read traffic predictions from GNN + Quantum: `traffic_predictions_5s.csv`
- Values represent vehicles per hour (veh/hr) on each edge

### Step 2: Normalize Traffic Intensity
```
intensity = (prediction - min) / (max - min)  ∈ [0, 1]
```

### Step 3: Scale to Signal Durations
```
green_time = min_green + intensity × (max_green - min_green)
green_time ∈ [10, 60] seconds
```

### Step 4: Apply Quantum Modulation
```
quantum_factor = (1 + quantum_output) / 2  ∈ [0, 1]
adjusted_green = min_green + quantum_factor × (max_green - min_green)
```

### Step 5: Adaptive Adjustment
```
if queue_length > threshold:
    green_time += extension (max 10s extra)
if queue_length < 5:
    green_time -= reduction (min 10s floor)
```

---

## 🎯 Green Corridor Algorithm

### Corridor Identification
1. Find node pairs with highest combined traffic
2. Calculate shortest paths using traffic-weighted network
3. Rank by traffic volume

### Offset Calculation
For each node in corridor:
```
offset[i] = (offset[i-1] + cycle_time[i-1] - travel_time) mod cycle
```

Where:
- `offset[i]` = Time when node i's green starts
- `travel_time` = Estimated transit between nodes (distance/speed)
- `cycle` = Green + Yellow duration

### Optimization
Grid search over offset variations to minimize expected stops:
```
stops = vehicles × (red_time / cycle_time)
```

---

## 📊 Key Metrics

### Signal Control Metrics
| Metric | Range | Meaning |
|--------|-------|---------|
| Green Time | 10-60s | Duration of green phase |
| Cycle Time | ~30s | Total green + yellow + red |
| Arrival Rate | 0-1 veh/s | Predicted vehicles per second |
| Efficiency | 0-100% | Fraction of vehicles not delayed |

### Green Corridor Metrics
| Metric | Range | Meaning |
|--------|-------|---------|
| Expected Stops | 0-N | Vehicles expected to stop |
| Throughput | 0-∞ | Vehicles/hour on corridor |
| Offset Variance | 0-∞ | Timing distribution uniformity |
| Priority | 1-N | Corridor importance ranking |

### SUMO Simulation Metrics
| Metric | Unit | Meaning |
|--------|------|---------|
| Vehicle Count | # | Active vehicles |
| Speed | m/s | Average velocity |
| Wait Time | s | Cumulative delay |
| Efficiency | % | Green wave success rate |

---

## ⚙️ Configuration Options

### Signal Controller
Edit `signal_controller.py`:
```python
QuantumGuidedSignalController(
    n_nodes=10,           # Number of intersections
    min_green=10,         # Minimum green time (seconds)
    max_green=60,         # Maximum green time (seconds)
    yellow_duration=3     # Yellow phase duration (seconds)
)
```

### Green Corridor
Edit or create corridors programmatically:
```python
corridor = GreenCorridor(n_nodes=10, signal_controller=controller)
custom_route = [0, 2, 5, 7, 9]
corridor.create_corridor(custom_route, "MainRoute")
```

### SUMO Simulation
Edit `sumo_simulator.py`:
```python
SimulationRunner(
    simulation_time=3600,  # Total simulation seconds
    time_step=1.0          # Seconds per simulation step
)
```

---

## 🧪 Testing

Run the comprehensive test suite:
```bash
python test_integration.py
```

Tests:
1. ✅ Module imports
2. ✅ Data availability
3. ✅ Signal controller functionality
4. ✅ Green corridor optimizer
5. ✅ SUMO network generation
6. ✅ Streamlit integration

---

## 📁 File Structure

```
CapStone-qCNN/
├── signal_controller.py           # Signal optimization (NEW)
├── green_corridor.py              # Green wave coordination (NEW)
├── sumo_network.py                # Network generation (NEW)
├── sumo_simulator.py              # SUMO integration (NEW)
├── test_integration.py            # Test suite (NEW)
├── INTEGRATION_GUIDE.md           # Detailed guide (NEW)
├── streamlit_app.py               # Enhanced dashboard (UPDATED)
├── requirements.txt               # Updated dependencies (UPDATED)
├── run.sh                         # Linux/Mac launcher (NEW)
├── run.bat                        # Windows launcher (NEW)
│
├── traffic_prediction_pipeline.py # Existing
├── gnn_embedding_pipeline.py      # Existing
├── quantum_pipeline.py            # Existing
├── ready_embeddings.py            # Existing
│
├── outputs/                       # Generated data
│   ├── traffic_predictions_5s.csv
│   ├── embeddings.csv
│   ├── quantum_output.csv
│   └── sumo_results/              # NEW simulation outputs
│
└── sumo_simulation/               # NEW SUMO files
    ├── traffic_network.nod.xml
    ├── traffic_network.edg.xml
    ├── traffic_network.tll.xml
    ├── traffic.rou.xml
    └── sumo.sumocfg
```

---

## 🚨 Troubleshooting

### Issue: "netconvert not found"
**Cause:** SUMO not installed  
**Solution:** Download from https://sumo.dlr.de or run in simulation mode

### Issue: "No output files found"
**Cause:** Missing traffic predictions  
**Solution:** Run `python traffic_prediction_pipeline.py` first

### Issue: "ModuleNotFoundError: signal_controller"
**Cause:** Not in correct directory  
**Solution:** Run from project root directory

### Issue: Streamlit won't start
**Cause:** Port 8501 in use  
**Solution:** `streamlit run streamlit_app.py --server.port 8502`

### Issue: Slow performance
**Solution:** 
- Increase time step (e.g., `time_step=5.0`)
- Reduce simulation duration
- Disable GUI (`use_gui=False`)

---

## 📚 References

### Project Files
| File | Purpose |
|------|---------|
| `signal_controller.py` | Signal timing optimization |
| `green_corridor.py` | Green corridor coordination |
| `sumo_network.py` | Network file generation |
| `sumo_simulator.py` | SUMO integration |
| `test_integration.py` | Test suite |
| `INTEGRATION_GUIDE.md` | Comprehensive guide |

### External Resources
- [SUMO Wiki](https://sumo.dlr.de/wiki/Main)
- [TraCI Documentation](https://sumo.dlr.de/docs/TraCI/)
- [PennyLane Docs](https://pennylane.ai/)
- [Streamlit Guide](https://docs.streamlit.io/)

---

## ✅ Verification Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment created
- [ ] Dependencies: `pip install -r requirements.txt`
- [ ] Traffic predictions generated
- [ ] Test suite passes: `python test_integration.py`
- [ ] SUMO network created: `python sumo_network.py`
- [ ] Dashboard launches: `streamlit run streamlit_app.py`
- [ ] Signal control page shows timings
- [ ] Green corridor page shows routes
- [ ] Simulation page is interactive

---

## 🎓 Learning Resources

This implementation demonstrates:
1. **Hybrid Quantum-Classical ML** - Combining quantum & classical
2. **Traffic Engineering** - Real-world signal control theory
3. **System Integration** - ML models → Simulation → Visualization
4. **Software Engineering** - Modular, testable architecture

---

## 📄 License & Attribution

This integration builds upon:
- Your existing GNN + Quantum prediction system
- SUMO traffic simulator (open-source)
- PennyLane quantum library
- Streamlit interactive framework

---

**Version:** 1.0  
**Last Updated:** March 2026  
**Status:** ✅ Production Ready

For questions or issues, refer to `INTEGRATION_GUIDE.md` or `test_integration.py`
