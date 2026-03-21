"""
Signal Controller using GNN + Quantum Layer Predictions
Optimizes signal timing based on predicted traffic flow
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from dataclasses import dataclass


@dataclass
class SignalPhase:
    """Traffic signal phase definition."""
    duration: float  # In seconds
    state: str       # Signal state (G/Y/R for each lane)
    arrival_rate: float  # Predicted vehicles/sec


class QuantumGuidedSignalController:
    """
    GNN + Quantum Layer-based traffic signal controller.
    Optimizes signal timing using traffic predictions.
    """
    
    def __init__(self, n_nodes=10, min_green=10, max_green=60, yellow_duration=3):
        """
        Initialize signal controller.
        
        Args:
            n_nodes: Number of intersections
            min_green: Minimum green time (sec)
            max_green: Maximum green time (sec)
            yellow_duration: Yellow phase duration (sec)
        """
        self.n_nodes = n_nodes
        self.min_green = min_green
        self.max_green = max_green
        self.yellow_duration = yellow_duration
        
        # Load traffic predictions if available
        self.predictions = self._load_predictions()
        
        # Initialize signal timings
        self.signal_timings = self._initialize_signal_timings()
        self.current_phases = {f'n{i}': 0 for i in range(n_nodes)}  # Current phase index
        self.phase_start_time = {f'n{i}': 0 for i in range(n_nodes)}
        
    def _load_predictions(self):
        """Load GNN + Quantum predictions."""
        pred_file = Path('./outputs/traffic_predictions_5s.csv')
        
        if pred_file.exists():
            try:
                df = pd.read_csv(pred_file)
                return df.values.flatten()
            except Exception as e:
                print(f"⚠ Error loading predictions: {e}")
                return np.ones(90) * 1200  # Default traffic volume
        else:
            return np.ones(90) * 1200  # Default: ~1200 veh/hr per edge
    
    def _initialize_signal_timings(self):
        """Initialize signal timing plan based on predictions."""
        timings = {}
        
        # Normalize predictions to get relative traffic intensity
        min_pred = np.min(self.predictions)
        max_pred = np.max(self.predictions)
        
        if max_pred > min_pred:
            intensity = (self.predictions - min_pred) / (max_pred - min_pred)
        else:
            intensity = np.ones_like(self.predictions)
        
        # Each node handles multiple edges
        edges_per_node = len(self.predictions) // self.n_nodes
        
        for node_idx in range(self.n_nodes):
            edge_indices = range(
                node_idx * edges_per_node,
                min((node_idx + 1) * edges_per_node, len(self.predictions))
            )
            
            # Average intensity for this node's edges
            node_intensity = np.mean(intensity[list(edge_indices)])
            
            # Scale green time based on traffic intensity
            green_time = self.min_green + node_intensity * (self.max_green - self.min_green)
            
            timings[f'n{node_idx}'] = SignalPhase(
                duration=green_time,
                state='G',
                arrival_rate=np.mean(self.predictions[list(edge_indices)]) / 3600  # veh/sec
            )
        
        return timings
    
    def get_signal_state(self, node_id, current_time):
        """
        Get signal state for a node at given time.
        
        Args:
            node_id: Node identifier (e.g., 'n0')
            current_time: Current simulation time (seconds)
        
        Returns:
            Signal state ('G', 'Y', or 'R')
        """
        if node_id not in self.signal_timings:
            return 'R'  # Default to red
        
        timing = self.signal_timings[node_id]
        phase_time = current_time - self.phase_start_time[node_id]
        cycle_time = timing.duration + self.yellow_duration
        
        if phase_time < timing.duration:
            return 'G'  # Green
        elif phase_time < timing.duration + self.yellow_duration:
            return 'Y'  # Yellow
        else:
            # Reset for next cycle
            self.phase_start_time[node_id] = current_time
            return 'G'
    
    def update_timings_from_quantum(self, quantum_output):
        """
        Update signal timings based on quantum layer output.
        
        Args:
            quantum_output: Output from quantum circuit (shape: (n_nodes, 4) or similar)
        """
        if isinstance(quantum_output, np.ndarray):
            q_out = quantum_output
        else:
            q_out = np.array(quantum_output)
        
        # Normalize quantum output to reasonable durations
        q_out = np.abs(q_out)  # Ensure positive
        
        for node_idx in range(self.n_nodes):
            node_id = f'n{node_idx}'
            
            # Use quantum output to modulate signal timing
            if node_idx < q_out.shape[0]:
                quantum_factor = (1 + q_out[node_idx]) / 2  # Scale [0, 1]
            else:
                quantum_factor = 0.5
            
            # Adjust green time
            timing = self.signal_timings[node_id]
            adjusted_green = self.min_green + quantum_factor * (self.max_green - self.min_green)
            
            timing.duration = adjusted_green
    
    def get_cycle_time(self, node_id):
        """Get total cycle time for a node."""
        if node_id in self.signal_timings:
            return self.signal_timings[node_id].duration + self.yellow_duration
        return self.max_green + self.yellow_duration
    
    def get_green_time(self, node_id):
        """Get green duration for a node."""
        if node_id in self.signal_timings:
            return self.signal_timings[node_id].duration
        return self.max_green
    
    def get_all_timings(self):
        """Get all signal timings as dictionary."""
        return {
            node_id: {
                'green': timing.duration,
                'yellow': self.yellow_duration,
                'cycle': timing.duration + self.yellow_duration,
                'arrival_rate': timing.arrival_rate
            }
            for node_id, timing in self.signal_timings.items()
        }
    
    def export_to_sumo(self, filepath):
        """Export signal timings to SUMO format."""
        import xml.etree.ElementTree as ET
        from xml.dom import minidom
        
        tllogics = ET.Element('tlLogics')
        
        for node_id, timing in self.signal_timings.items():
            tllogic = ET.SubElement(tllogics, 'tlLogic')
            tllogic.set('id', node_id)
            tllogic.set('type', 'static')
            tllogic.set('programID', 'optimized')
            tllogic.set('offset', '0')
            
            # Green phase
            phase_g = ET.SubElement(tllogic, 'phase')
            phase_g.set('duration', str(int(timing.duration)))
            phase_g.set('state', 'GGrrGGrrGGrr')
            
            # Yellow phase
            phase_y = ET.SubElement(tllogic, 'phase')
            phase_y.set('duration', str(self.yellow_duration))
            phase_y.set('state', 'yyrryyrryyrr')
        
        # Pretty print and save
        rough_str = ET.tostring(tllogics, 'utf-8')
        reparsed = minidom.parseString(rough_str)
        pretty_str = reparsed.toprettyxml(indent="  ")
        pretty_str = '\n'.join([line for line in pretty_str.split('\n') if line.strip()])
        
        with open(filepath, 'w') as f:
            f.write(pretty_str)
        
        print(f"✓ Exported signal timings to: {filepath}")


class AdaptiveSignalController:
    """
    Adaptive controller that adjusts signals in real-time based on
    network state and predictions.
    """
    
    def __init__(self, base_controller):
        self.base_controller = base_controller
        self.queue_lengths = {}
        self.waiting_times = {}
        
    def update_with_traffic_state(self, queue_data):
        """
        Update signal timings based on current traffic state.
        
        Args:
            queue_data: Dictionary of queue lengths per node
        """
        for node_id, queue_length in queue_data.items():
            # Adaptive green time extension
            if queue_length > 20:  # Threshold
                # Extend green time
                current_timing = self.base_controller.signal_timings[node_id]
                extension = min(10, queue_length / 10)
                current_timing.duration = min(
                    current_timing.duration + extension,
                    self.base_controller.max_green
                )
            elif queue_length < 5:
                # Reduce green time to serve other movements
                current_timing = self.base_controller.signal_timings[node_id]
                current_timing.duration = max(
                    current_timing.duration - 2,
                    self.base_controller.min_green
                )


# ============================================================================
# MAIN: Test Signal Controller
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Quantum-Guided Signal Controller Test")
    print("="*60 + "\n")
    
    # Create controller
    controller = QuantumGuidedSignalController(n_nodes=10)
    
    # Display timings
    print("Initial Signal Timings:")
    print("-" * 60)
    timings = controller.get_all_timings()
    
    timing_df = pd.DataFrame(timings).T
    print(timing_df.round(2))
    
    # Test signal state at different times
    print("\n\nSignal States at Different Times (Node n0):")
    print("-" * 60)
    for t in range(0, 100, 10):
        state = controller.get_signal_state('n0', t)
        print(f"Time {t:3d}s: Signal = {state}")
    
    # Export to SUMO
    output_dir = Path('./sumo_simulation')
    output_dir.mkdir(exist_ok=True)
    controller.export_to_sumo(output_dir / 'optimized_signals.xml')
    
    print("\n✓ Signal controller test completed!")
