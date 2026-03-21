"""
SUMO Simulator Integration
Runs traffic simulation and integrates with GNN + Quantum predictions
"""

import numpy as np
import pandas as pd
import os
import time
from pathlib import Path
from datetime import datetime
from signal_controller import QuantumGuidedSignalController, AdaptiveSignalController
from green_corridor import GreenCorridor, GreenWaveOptimizer
import warnings

warnings.filterwarnings('ignore')


class SUMOSimulator:
    """
    Wrapper for SUMO simulation with signal control integration.
    """
    
    def __init__(self, config_file='./sumo_simulation/sumo.sumocfg',
                 use_gui=False, seed=42):
        """
        Initialize SUMO simulator.
        
        Args:
            config_file: Path to SUMO configuration file
            use_gui: Whether to use SUMO GUI
            seed: Random seed
        """
        self.config_file = config_file
        self.use_gui = use_gui
        self.seed = seed
        self.is_running = False
        self.current_time = 0
        
        # Initialize controllers
        self.signal_controller = QuantumGuidedSignalController()
        self.adaptive_controller = AdaptiveSignalController(self.signal_controller)
        self.green_corridor = GreenCorridor(signal_controller=self.signal_controller)
        self.green_corridor.optimize_all_corridors()
        
        # Data collection
        self.simulation_data = {
            'time': [],
            'vehicle_count': [],
            'average_speed': [],
            'total_wait_time': [],
            'node_occupancy': {},
            'signal_timings': {},
            'corridor_efficiency': []
        }
        
        # Initialize node tracking
        for i in range(10):
            node_id = f'n{i}'
            self.simulation_data['node_occupancy'][node_id] = []
            self.simulation_data['signal_timings'][node_id] = []
    
    def start(self):
        """Start SUMO simulation."""
        self._try_start_sumo()
        self.is_running = True
        self.start_time = time.time()
        print("✓ SUMO Simulator started")
    
    def _try_start_sumo(self):
        """Attempt to start SUMO with fallback to simulation mode."""
        try:
            import traci
            
            # Build SUMO command
            if self.use_gui:
                sumo_cmd = ['sumo-gui']
            else:
                sumo_cmd = ['sumo', '-c', str(self.config_file), '--start']
            
            sumo_cmd.extend(['--seed', str(self.seed), '--quit-on-end'])
            
            traci.start(sumo_cmd)
            self.traci = traci
            self.use_traci = True
            print("✓ Connected to SUMO via TraCI")
            
        except ImportError:
            print("⚠ TraCI not available. Running in simulation mode.")
            self.use_traci = False
        except Exception as e:
            print(f"⚠ Could not connect to SUMO: {e}")
            print("  Running in simulation mode...")
            self.use_traci = False
    
    def step(self, dt=1.0):
        """
        Advance simulation by dt seconds.
        
        Args:
            dt: Time step in seconds
        """
        self.current_time += dt
        
        if self.use_traci:
            try:
                self.traci.simulationStep()
                self._collect_traci_data()
            except Exception as e:
                print(f"⚠ TraCI error: {e}")
                self.use_traci = False
        
        # Collect data from our controllers
        self._collect_controller_data()
        self._update_signals()
        
        return self.current_time
    
    def _collect_traci_data(self):
        """Collect data from SUMO via TraCI."""
        try:
            vehicle_count = self.traci.vehicle.getIDCount()
            avg_speed = self._get_average_vehicle_speed()
            total_wait_time = self._get_total_wait_time()
            
            self.simulation_data['vehicle_count'].append(vehicle_count)
            self.simulation_data['average_speed'].append(avg_speed)
            self.simulation_data['total_wait_time'].append(total_wait_time)
            self.simulation_data['time'].append(self.current_time)
            
        except Exception as e:
            print(f"⚠ Error collecting TraCI data: {e}")
    
    def _get_average_vehicle_speed(self):
        """Get average speed of all vehicles."""
        try:
            vehicle_ids = self.traci.vehicle.getIDList()
            if not vehicle_ids:
                return 0
            
            speeds = [self.traci.vehicle.getSpeed(vid) for vid in vehicle_ids]
            return np.mean(speeds)
        except:
            return 0
    
    def _get_total_wait_time(self):
        """Get total accumulated wait time."""
        try:
            vehicle_ids = self.traci.vehicle.getIDList()
            total_wait = 0
            for vid in vehicle_ids:
                wait_time = self.traci.vehicle.getAccumulatedWaitingTime(vid)
                total_wait += wait_time
            return total_wait
        except:
            return 0
    
    def _collect_controller_data(self):
        """Collect data from controllers."""
        # Collect signal timings
        timings = self.signal_controller.get_all_timings()
        
        for node_id, timing in timings.items():
            signal_state = self.signal_controller.get_signal_state(node_id, self.current_time)
            self.simulation_data['signal_timings'][node_id].append({
                'time': self.current_time,
                'state': signal_state,
                'green_time': timing['green'],
                'cycle_time': timing['cycle']
            })
        
        # Evaluate corridor efficiency
        efficiency = self._evaluate_corridor_efficiency()
        self.simulation_data['corridor_efficiency'].append({
            'time': self.current_time,
            'efficiency': efficiency
        })
    
    def _update_signals(self):
        """Update signal timings based on current state."""
        if self.use_traci:
            try:
                # Get current lane data
                lane_ids = self.traci.lane.getIDList()
                queue_data = {}
                
                for lane_id in lane_ids[:10]:  # Approximate node 0-9
                    occupancy = self.traci.lane.getLastStepOccupancy(lane_id)
                    queue_data[f'n{int(lane_id[0])}'] = occupancy
                
                # Update adaptive controller
                self.adaptive_controller.update_with_traffic_state(queue_data)
                
            except Exception as e:
                pass  # Silent fail during simulation mode
    
    def _evaluate_corridor_efficiency(self):
        """Evaluate current corridor efficiency."""
        if not self.green_corridor.corridors:
            return 0
        
        total_efficiency = 0
        
        # Rough estimate based on signal alignment
        for corridor_name, corridor in self.green_corridor.corridors.items():
            # Check how many signals are in sync
            nodes = corridor['route']
            in_phase_count = 0
            
            for node_id in nodes:
                state = self.signal_controller.get_signal_state(node_id, self.current_time)
                if state == 'G':
                    in_phase_count += 1
            
            efficiency = (in_phase_count / len(nodes)) * 100
            total_efficiency += efficiency
        
        avg_efficiency = total_efficiency / len(self.green_corridor.corridors)
        return avg_efficiency
    
    def stop(self):
        """Stop SUMO simulation."""
        if self.use_traci:
            try:
                self.traci.close()
            except:
                pass
        
        self.is_running = False
        print("✓ SUMO Simulator stopped")
    
    def get_simulation_data(self):
        """Get collected simulation data."""
        return self.simulation_data
    
    def export_results(self, output_dir='./outputs/sumo_results'):
        """Export simulation results."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export time series data
        if self.simulation_data['time']:
            ts_data = pd.DataFrame({
                'time': self.simulation_data['time'],
                'vehicle_count': self.simulation_data['vehicle_count'],
                'avg_speed': self.simulation_data['average_speed'],
                'wait_time': self.simulation_data['total_wait_time']
            })
            ts_data.to_csv(output_path / 'time_series.csv', index=False)
        
        # Export corridor efficiency
        if self.simulation_data['corridor_efficiency']:
            corridor_df = pd.DataFrame(self.simulation_data['corridor_efficiency'])
            corridor_df.to_csv(output_path / 'corridor_efficiency.csv', index=False)
        
        # Export signal timings for each node
        for node_id, timings_list in self.simulation_data['signal_timings'].items():
            if timings_list:
                timing_df = pd.DataFrame(timings_list)
                timing_df.to_csv(output_path / f'signal_timings_{node_id}.csv', index=False)
        
        print(f"✓ Results exported to: {output_path}")
        return output_path


class SimulationRunner:
    """
    High-level runner for complete traffic simulation with signal control.
    """
    
    def __init__(self, simulation_time=3600, time_step=1.0):
        """
        Initialize simulation runner.
        
        Args:
            simulation_time: Total simulation time in seconds
            time_step: Simulation time step in seconds
        """
        self.simulation_time = simulation_time
        self.time_step = time_step
        self.simulator = None
        self.results = None
    
    def setup(self):
        """Setup simulation."""
        print("\n" + "="*60)
        print("Traffic Simulation with Quantum-Guided Signal Control")
        print("="*60 + "\n")
        
        self.simulator = SUMOSimulator(use_gui=False)
        self.simulator.start()
    
    def run(self):
        """Run simulation."""
        print(f"Running simulation for {self.simulation_time}s...")
        print("-" * 60)
        
        start = time.time()
        step_count = 0
        
        while self.simulator.current_time < self.simulation_time:
            self.simulator.step(self.time_step)
            step_count += 1
            
            # Progress indicator
            if step_count % 100 == 0:
                progress = (self.simulator.current_time / self.simulation_time) * 100
                elapsed = time.time() - start
                eta = elapsed / (progress + 1) * (100 - progress) if progress < 100 else 0
                
                print(f"Progress: {progress:5.1f}% | Elapsed: {elapsed:6.1f}s | "
                      f"ETA: {eta:6.1f}s | Step: {step_count}")
        
        self.simulator.stop()
        print("-" * 60)
        print(f"✓ Simulation completed in {time.time() - start:.1f}s")
    
    def save_results(self, output_dir='./outputs/sumo_results'):
        """Save simulation results."""
        self.simulator.export_results(output_dir)
        
        # Additional summary
        summary_file = Path(output_dir) / 'simulation_summary.txt'
        with open(summary_file, 'w') as f:
            f.write("Traffic Simulation Summary\n")
            f.write("="*60 + "\n\n")
            f.write(f"Simulation Time: {self.simulation_time}s\n")
            f.write(f"Time Step: {self.time_step}s\n")
            f.write(f"Simulation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Signal Control Configuration:\n")
            f.write("-"*60 + "\n")
            timings = self.simulator.signal_controller.get_all_timings()
            for node_id, timing in timings.items():
                f.write(f"{node_id}: Green={timing['green']:.1f}s, "
                       f"Cycle={timing['cycle']:.1f}s\n")
            
            f.write("\n\nGreen Corridors Created:\n")
            f.write("-"*60 + "\n")
            for name, corridor in self.simulator.green_corridor.corridors.items():
                f.write(f"{name}: {' → '.join(corridor['route'])}\n")
        
        print(f"✓ Summary saved to: {summary_file}")


# ============================================================================
# MAIN: Run Simulation
# ============================================================================

if __name__ == '__main__':
    # Create and run simulation
    runner = SimulationRunner(simulation_time=600, time_step=1.0)  # 10 minute simulation
    
    runner.setup()
    runner.run()
    runner.save_results()
    
    print("\n✓ Traffic simulation completed!")
