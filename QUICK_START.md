# 🚀 QUICK START REFERENCE

## Installation (1 minute)

```bash
# Activate environment
source myenv/Scripts/activate  # Windows: myenv\Scripts\activate

# Install/update packages
pip install -r requirements.txt
```

---

## Generate Predictions (if needed, ~2-5 minutes)

```bash
# Run traffic prediction pipeline
python traffic_prediction_pipeline.py

# Run GNN embedding pipeline
python gnn_embedding_pipeline.py
```

---

## Setup SUMO (if SUMO installed, ~30 seconds)

```bash
# Generate network files
python sumo_network.py
```

---

## Run Tests (optional, ~1 minute)

```bash
# Verify all systems
python test_integration.py
```

---

## Launch Dashboard (instant)

### Option 1: Direct
```bash
streamlit run streamlit_app.py
```

### Option 2: Automated Script  
```bash
# Linux/Mac
bash run.sh

# Windows
run.bat
```

Then open: **http://localhost:8501**

---

## Dashboard Pages Quick Reference

| Page | Icon | What It Shows | Key Feature |
|------|------|--------------|-------------|
| Overview | 🏠 | System info & architecture | Quick stats |
| GNN Embeddings | 📈 | Node embeddings visualization | PCA/t-SNE |
| Graph Topology | 🌐 | Network structure | 10-node mesh |
| Traffic Predictions | 🚦 | Edge traffic forecasts | 5-second horizon |
| Signal Control | 🚨 | Optimized signal timings | Quantum-guided ⭐ |
| Green Corridor | 🟢 | Green wave routes | Coordinated signals ⭐ |
| Simulation | ⚙️ | SUMO traffic simulation | Full integration ⭐ |
| Quantum Layer | ⚛️ | Circuit details | Gates & parameters |
| Data Analysis | 📉 | Raw data exploration | Feature distributions |

---

## 🚨 Signal Control - How It Works

```
Traffic Prediction (veh/hr)
        ↓
Normalize [0, 1]
        ↓
Scale to green time [10-60s]
        ↓
Apply quantum modulation
        ↓
✅ Optimized Signal Timing
```

**Example:**
- High traffic (900 veh/hr) → 55s green
- Medium traffic (600 veh/hr) → 35s green  
- Low traffic (300 veh/hr) → 15s green

---

## 🟢 Green Corridor - How It Works

```
Route: n0 → n1 → n2 → n3

At each intersection, calculate when vehicle arrives:
- Leave n0 at t=0s (green 25s)
- Arrive n1 at t=5s → Green starts at t=22s ✅
- Arrive n2 at t=10s → Green starts at t=44s ✅
- Arrive n3 at t=15s → Green starts at t=66s ✅

Result: Vehicle travels entire corridor with NO STOPS!
```

---

## ⚙️ Run SUMO Simulation

1. Open dashboard: `streamlit run streamlit_app.py`
2. Go to **⚙️ Simulation** page
3. Set parameters:
   - Duration: 600s (10 minutes) recommended
   - Time step: 1.0s (default)
   - GUI: No (unless SUMO installed locally)
4. Click **▶️ Run Simulation**
5. Watch progress bar
6. View results in dashboard

---

## 📊 Key Metrics

### Signal Timing
```
Green Time: 10-60 seconds (adaptive)
Cycle Time: ~30 seconds average
Yellow: 3 seconds fixed
```

### Green Corridor
```
Routes: 5 main corridors
Coverage: ~80% of intersections
Expected Stops: <2 per route
```

### SUMO Simulation  
```
Vehicles: 100 per simulation
Duration: Configurable (60-3600s)
Nodes: 10 intersections
Edges: 90 bidirectional
```

---

## 🔍 File Locations

| Item | Location |
|------|----------|
| Predictions | `outputs/traffic_predictions_5s.csv` |
| Embeddings | `outputs/embeddings.csv` |
| SUMO Network | `sumo_simulation/traffic_network.*` |
| Sim Results | `outputs/sumo_results/*` |
| Dashboard Config | `streamlit_app.py` |
| Signal Controller | `signal_controller.py` |
| Corridors | `green_corridor.py` |

---

## ⚡ Commands Quick Sheet

```bash
# Setup
pip install -r requirements.txt
python test_integration.py

# Predictions
python traffic_prediction_pipeline.py
python gnn_embedding_pipeline.py

# SUMO
python sumo_network.py
python sumo_simulator.py

# Dashboard
streamlit run streamlit_app.py

# Convenience
bash run.sh              # Linux/Mac
run.bat                  # Windows
```

---

## 🆘 Common Issues

| Problem | Solution |
|---------|----------|
| Port 8501 in use | `streamlit run streamlit_app.py --server.port 8502` |
| Missing predictions | `python traffic_prediction_pipeline.py` |
| SUMO not found | Download from sumo.dlr.de or use simulation mode |
| Slow performance | Increase `time_step`, reduce `simulation_time` |
| Module not found | Verify in project root directory |

---

## 📈 Expected Performance

```
System          Time to Complete      Result
─────────────────────────────────────────────
Setup              1 min              Ready
Predictions        3-5 min            CSV files
SUMO setup         <1 min             Network
Integration tests  ~2 min             ✓/✗ report
Dashboard launch   <5 sec             Browser
Simulation (600s)  2-5 min            Results & charts
```

---

## 🎯 Next Steps

1. ✅ Run `streamlit run streamlit_app.py`
2. ✅ Explore **Signal Control** page
3. ✅ Check **Green Corridor** visualizations
4. ✅ Run a simulation on **⚙️ Simulation** page
5. ✅ Analyze results in **📉 Data Analysis**

---

## 📖 For More Detail

- **Full Guide**: Read `INTEGRATION_GUIDE.md`
- **New Features**: Read `SIGNAL_CONTROL_README.md`
- **Code Docs**: View docstrings in Python files
- **Testing**: Run `python test_integration.py`

---

**Happy Traffic Optimization! 🚗** 💚
