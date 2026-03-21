"""
Green Corridor Optimization
Coordinates traffic signals along key routes to minimize stops and delays
"""

import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from typing import List, Tuple, Dict


class GreenCorridor:
    """
    Implements green corridor logic to create synchronized signal timing
    along defined traffic routes.
    """
    
    def __init__(self, n_nodes=10, signal_controller=None):
        """
        Initialize green corridor optimizer.
        
        Args:
            n_nodes: Number of nodes in network
            signal_controller: Base signal controller instance
        """
        self.n_nodes = n_nodes
        self.signal_controller = signal_controller
        self.graph = self._create_network_graph()
        self.corridors = {}
        self.main_routes = self._identify_main_routes()
        
    def _create_network_graph(self):
        """Create network graph for path analysis."""
        G = nx.complete_graph(self.n_nodes)
        
        # Add weights based on traffic predictions
        pred_file = Path('./outputs/traffic_predictions_5s.csv')
        if pred_file.exists():
            try:
                preds = pd.read_csv(pred_file).values.flatten()
                # Normalize to edge weights
                edge_idx = 0
                for i in range(self.n_nodes):
                    for j in range(self.n_nodes):
                        if i != j and edge_idx < len(preds):
                            G[i][j]['weight'] = 1 / (preds[edge_idx] + 1)  # Inverse
                            edge_idx += 1
            except:
                pass
        
        return G
    
    def _identify_main_routes(self):
        """Identify main traffic routes in the network."""
        routes = []
        
        # Find paths with highest traffic (lowest cumulative weight)
        # These represent main corridors
        main_pairs = [
            (0, 9), (1, 8), (2, 7), (3, 6), (4, 5),  # Major diagonal routes
        ]
        
        for start, end in main_pairs:
            try:
                path = nx.shortest_path(self.graph, start, end, weight='weight')
                routes.append(path)
            except nx.NetworkXNoPath:
                # If no path found, use direct
                routes.append([start, end])
        
        return routes
    
    def create_corridor(self, route: List[int], name: str = None):
        """
        Create a green corridor for a specific route.
        
        Args:
            route: List of node indices forming the corridor
            name: Corridor name
        
        Returns:
            Corridor definition with timing offsets
        """
        if name is None:
            name = f"Corridor_{len(self.corridors)}"
        
        route_nodes = [f'n{node_id}' for node_id in route]
        
        # Calculate optimal offsets for smooth progression
        offsets = self._calculate_offsets(route_nodes)
        
        corridor = {
            'name': name,
            'route': route_nodes,
            'offsets': offsets,
            'priority': len(route),  # Longer routes get higher priority
            'expected_speed': 13.89,  # m/s (~50 km/h)
            'timing_sync': self._calculate_sync_timing(route_nodes, offsets)
        }
        
        self.corridors[name] = corridor
        return corridor
    
    def _calculate_offsets(self, route_nodes: List[str]) -> Dict[str, float]:
        """
        Calculate timing offsets to create green waves.
        
        Args:
            route_nodes: List of node IDs in route
        
        Returns:
            Dictionary mapping node_id to offset in seconds
        """
        offsets = {}
        
        if not self.signal_controller:
            # Default: uniform offset
            for i, node_id in enumerate(route_nodes):
                base_green = 25  # Default green time
                offsets[node_id] = i * (base_green + 3) / len(route_nodes)
        else:
            # Use actual signal timings
            for i, node_id in enumerate(route_nodes):
                if i == 0:
                    offsets[node_id] = 0
                else:
                    # Calculate offset based on travel time between nodes
                    prev_node = route_nodes[i - 1]
                    distance_km = 1.0  # Approximate inter-node distance
                    travel_time = (distance_km / 13.89) * 3.6  # Convert to seconds
                    
                    prev_offset = offsets[prev_node]
                    prev_cycle = self.signal_controller.get_cycle_time(prev_node)
                    
                    # Offset to hit green light
                    offsets[node_id] = (prev_offset + prev_cycle - travel_time) % prev_cycle
        
        return offsets
    
    def _calculate_sync_timing(self, route_nodes: List[str], offsets: Dict):
        """Calculate synchronized timing plan."""
        timing = {}
        
        for node_id in route_nodes:
            if self.signal_controller:
                green = self.signal_controller.get_green_time(node_id)
                cycle = self.signal_controller.get_cycle_time(node_id)
            else:
                green = 25
                cycle = 28  # 25 + 3 yellow
            
            timing[node_id] = {
                'offset': offsets.get(node_id, 0),
                'green': green,
                'yellow': 3,
                'cycle': cycle,
                'start_time': offsets.get(node_id, 0)
            }
        
        return timing
    
    def optimize_all_corridors(self):
        """Create and optimize all main corridors."""
        corridors_created = []
        
        for i, route in enumerate(self.main_routes):
            corridor = self.create_corridor(route, f"MainRoute_{i}")
            corridors_created.append(corridor)
            print(f"✓ Created {corridor['name']}: {' → '.join(corridor['route'])}")
        
        return corridors_created
    
    def get_corridor_visualization_data(self):
        """Get data for corridor visualization."""
        viz_data = []
        
        for corridor_name, corridor in self.corridors.items():
            for node_id in corridor['route']:
                timing = corridor['timing_sync'][node_id]
                viz_data.append({
                    'corridor': corridor_name,
                    'node': node_id,
                    'offset': timing['offset'],
                    'green_duration': timing['green'],
                    'cycle_time': timing['cycle'],
                    'start_time': timing['start_time']
                })
        
        return pd.DataFrame(viz_data)
    
    def get_corridor_status_report(self):
        """Generate corridor status report."""
        report = {}
        
        for corridor_name, corridor in self.corridors.items():
            offsets = corridor['offsets']
            nodes = corridor['route']
            
            # Calculate progression quality
            offsets_list = [offsets.get(node, 0) for node in nodes]
            offset_variance = np.var(offsets_list)
            
            report[corridor_name] = {
                'length': len(nodes),
                'nodes': nodes,
                'avg_offset': np.mean(offsets_list),
                'offset_variance': offset_variance,
                'priority': corridor['priority'],
                'expected_speed': corridor['expected_speed']
            }
        
        return pd.DataFrame(report).T


class GreenWaveOptimizer:
    """
    Advanced optimizer for creating optimal green waves considering
    vehicle platoons and arrival patterns.
    """
    
    def __init__(self, corridor):
        """
        Initialize green wave optimizer.
        
        Args:
            corridor: GreenCorridor instance
        """
        self.corridor = corridor
        self.vehicle_arrivals = {}
        self.optimization_metrics = {}
        
    def add_arrival_pattern(self, node_id, arrival_rate, arrival_variance=0.1):
        """
        Add predicted arrival pattern for a node.
        
        Args:
            node_id: Node identifier
            arrival_rate: Vehicles per second
            arrival_variance: Variance in arrivals
        """
        self.vehicle_arrivals[node_id] = {
            'rate': arrival_rate,
            'variance': arrival_variance,
            'volume': arrival_rate * 3600  # Convert to vehicles per hour
        }
    
    def calculate_stops(self, corridor_name):
        """
        Calculate expected number of stops on corridor.
        
        Args:
            corridor_name: Name of corridor to evaluate
        
        Returns:
            Number of expected stops
        """
        if corridor_name not in self.corridor.corridors:
            return 0
        
        corridor = self.corridor.corridors[corridor_name]
        nodes = corridor['route']
        timing = corridor['timing_sync']
        
        total_stops = 0
        
        for node_id in nodes:
            if node_id in self.vehicle_arrivals:
                arrival = self.vehicle_arrivals[node_id]
                node_timing = timing[node_id]
                
                # Probability of hitting red (simplified)
                red_duration = node_timing['cycle'] - node_timing['green']
                red_ratio = red_duration / node_timing['cycle']
                
                expected_stops = arrival['volume'] * red_ratio / 3600
                total_stops += expected_stops
        
        self.optimization_metrics[corridor_name] = {
            'expected_stops': total_stops,
            'throughput': sum(
                self.vehicle_arrivals.get(node, {}).get('volume', 0) 
                for node in nodes
            ),
            'efficiency': (1 - total_stops / max(1, len(nodes) * 100)) * 100
        }
        
        return total_stops
    
    def optimize_offsets(self, corridor_name):
        """
        Optimize timing offsets to minimize stops.
        
        Args:
            corridor_name: Name of corridor to optimize
        
        Returns:
            Optimized offsets
        """
        corridor = self.corridor.corridors[corridor_name]
        nodes = corridor['route']
        
        best_offsets = corridor['offsets'].copy()
        best_stops = float('inf')
        
        # Grid search over offset variations
        for phase_offset in np.linspace(0, 5, 11):
            test_offsets = {
                node: (corridor['offsets'].get(node, 0) + phase_offset) % 28
                for node in nodes
            }
            
            corridor['offsets'] = test_offsets
            stops = self.calculate_stops(corridor_name)
            
            if stops < best_stops:
                best_stops = stops
                best_offsets = test_offsets.copy()
        
        corridor['offsets'] = best_offsets
        return best_offsets
    
    def get_optimization_report(self):
        """Get optimization report for all corridors."""
        return pd.DataFrame(self.optimization_metrics).T


# ============================================================================
# MAIN: Test Green Corridor
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Green Corridor Optimizer Test")
    print("="*60 + "\n")
    
    # Create green corridor
    corridor = GreenCorridor(n_nodes=10)
    
    # Create corridors
    print("Creating Green Corridors:")
    print("-" * 60)
    corridors = corridor.optimize_all_corridors()
    
    # Display corridor data
    print("\n\nCorridor Details:")
    print("-" * 60)
    viz_data = corridor.get_corridor_visualization_data()
    print(viz_data.head(20))
    
    # Get status report
    print("\n\nCorridor Status Report:")
    print("-" * 60)
    report = corridor.get_corridor_status_report()
    print(report)
    
    # Test green wave optimizer
    print("\n\nGreen Wave Optimization:")
    print("-" * 60)
    optimizer = GreenWaveOptimizer(corridor)
    
    # Add arrival patterns
    for i in range(10):
        optimizer.add_arrival_pattern(f'n{i}', arrival_rate=0.3 + 0.05*i)
    
    # Calculate stops for first corridor
    if corridor.corridors:
        first_corridor = list(corridor.corridors.keys())[0]
        stops = optimizer.calculate_stops(first_corridor)
        print(f"Expected stops on {first_corridor}: {stops:.2f}")
        
        # Optimize
        optimizer.optimize_offsets(first_corridor)
        optimized_stops = optimizer.calculate_stops(first_corridor)
        print(f"After optimization: {optimized_stops:.2f}")
    
    print("\n✓ Green corridor test completed!")
