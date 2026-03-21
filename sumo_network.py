"""
SUMO Network Generator for 10-Node Traffic Network
Creates a grid-based traffic network with traffic lights for signal control
"""

import os
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
from xml.dom import minidom


class SUMONetworkGenerator:
    """Generate SUMO network files for traffic simulation."""
    
    def __init__(self, n_nodes=10, grid_size=1000):
        """
        Initialize network generator.
        
        Args:
            n_nodes: Number of nodes in the network
            grid_size: Size of the grid (meters)
        """
        self.n_nodes = n_nodes
        self.grid_size = grid_size
        self.node_positions = self._generate_grid_positions()
        self.output_dir = Path('./sumo_simulation')
        self.output_dir.mkdir(exist_ok=True)
        
    def _generate_grid_positions(self):
        """Generate node positions in a grid layout."""
        # For 10 nodes, create a roughly 3x3 + 1 grid
        side_length = int(np.ceil(np.sqrt(self.n_nodes)))
        spacing = self.grid_size / (side_length + 1)
        
        positions = {}
        idx = 0
        for i in range(side_length):
            for j in range(side_length):
                if idx < self.n_nodes:
                    x = (i + 1) * spacing
                    y = (j + 1) * spacing
                    positions[f'n{idx}'] = (x, y)
                    idx += 1
        
        return positions
    
    def create_node_file(self):
        """Create SUMO node definition XML file."""
        nodes_xml = ET.Element('nodes')
        
        for node_id, (x, y) in self.node_positions.items():
            node = ET.SubElement(nodes_xml, 'node')
            node.set('id', node_id)
            node.set('x', str(x))
            node.set('y', str(y))
            node.set('type', 'traffic_light')  # All nodes have traffic lights
        
        # Write to file
        node_file = self.output_dir / 'traffic_network.nod.xml'
        self._prettify_and_save(nodes_xml, node_file)
        print(f"✓ Created node file: {node_file}")
        return node_file
    
    def create_edge_file(self):
        """Create SUMO edge definition XML file."""
        edges_xml = ET.Element('edges')
        
        n_nodes = self.n_nodes
        edge_count = 0
        
        # Create bidirectional edges between nodes (fully connected)
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                from_node = f'n{i}'
                to_node = f'n{j}'
                
                # Edge i->j
                edge1 = ET.SubElement(edges_xml, 'edge')
                edge1.set('id', f'e{edge_count}_0')
                edge1.set('from', from_node)
                edge1.set('to', to_node)
                edge1.set('numLanes', '2')
                edge1.set('speed', '13.89')  # ~50 km/h
                edge1.set('length', str(self._calculate_distance(from_node, to_node)))
                edge_count += 1
                
                # Edge j->i
                edge2 = ET.SubElement(edges_xml, 'edge')
                edge2.set('id', f'e{edge_count}_0')
                edge2.set('from', to_node)
                edge2.set('to', from_node)
                edge2.set('numLanes', '2')
                edge2.set('speed', '13.89')
                edge2.set('length', str(self._calculate_distance(from_node, to_node)))
                edge_count += 1
        
        # Write to file
        edge_file = self.output_dir / 'traffic_network.edg.xml'
        self._prettify_and_save(edges_xml, edge_file)
        print(f"✓ Created edge file: {edge_file} with {edge_count} edges")
        return edge_file
    
    def create_tllogic_file(self):
        """Create traffic light logic definitions."""
        tllogics_xml = ET.Element('tlLogics')
        
        # Create traffic light for each node
        for node_id in self.node_positions.keys():
            tllogic = ET.SubElement(tllogics_xml, 'tlLogic')
            tllogic.set('id', node_id)
            tllogic.set('type', 'static')
            tllogic.set('programID', 'default')
            tllogic.set('offset', '0')
            
            # Main phase (green)
            phase1 = ET.SubElement(tllogic, 'phase')
            phase1.set('duration', '25')  # 25 seconds green
            phase1.set('state', 'GGGGGGrrrrr')  # Example state for multiple connections
            
            # Transition phase (yellow)
            phase2 = ET.SubElement(tllogic, 'phase')
            phase2.set('duration', '3')   # 3 seconds yellow
            phase2.set('state', 'yyyyyyyyyyy')
        
        # Write to file
        tl_file = self.output_dir / 'traffic_network.tll.xml'
        self._prettify_and_save(tllogics_xml, tl_file)
        print(f"✓ Created traffic light logic file: {tl_file}")
        return tl_file
    
    def create_net_file(self):
        """Create combined network file by running netconvert."""
        import subprocess
        
        nod_file = self.output_dir / 'traffic_network.nod.xml'
        edg_file = self.output_dir / 'traffic_network.edg.xml'
        tll_file = self.output_dir / 'traffic_network.tll.xml'
        net_file = self.output_dir / 'traffic_network.net.xml'
        
        # Create node and edge files first
        self.create_node_file()
        self.create_edge_file()
        self.create_tllogic_file()
        
        # Use netconvert to create the network
        cmd = [
            'netconvert',
            '-n', str(nod_file),
            '-e', str(edg_file),
            '-t', str(tll_file),
            '-o', str(net_file),
            '--geometry.remove', '--geometry.remove.keep-edges.explicit',
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"✓ Created network file: {net_file}")
            return net_file
        except FileNotFoundError:
            print("⚠ netconvert not found. Please install SUMO from")
            print("  https://sumo.dlr.de/wiki/Downloads or use simulator with existing files")
            return None
        except subprocess.CalledProcessError as e:
            print(f"⚠ Error running netconvert: {e.stderr}")
            return None
    
    def _calculate_distance(self, node1, node2):
        """Calculate Euclidean distance between two nodes."""
        x1, y1 = self.node_positions[node1]
        x2, y2 = self.node_positions[node2]
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def _prettify_and_save(self, elem, filepath):
        """Pretty print and save XML element."""
        rough_str = ET.tostring(elem, 'utf-8')
        reparsed = minidom.parseString(rough_str)
        pretty_str = reparsed.toprettyxml(indent="  ")
        
        # Remove extra blank lines
        pretty_str = '\n'.join([line for line in pretty_str.split('\n') if line.strip()])
        
        with open(filepath, 'w') as f:
            f.write(pretty_str)


class SUMOConfigGenerator:
    """Generate SUMO configuration files."""
    
    def __init__(self, net_file, output_dir='./sumo_simulation'):
        self.net_file = net_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def create_routes_file(self, n_vehicles=100, simulation_time=3600):
        """Create vehicle routes file."""
        routes_xml = ET.Element('routes')
        
        # Add vehicle types
        vtype = ET.SubElement(routes_xml, 'vType')
        vtype.set('id', 'car')
        vtype.set('accel', '0.8')
        vtype.set('decel', '4.5')
        vtype.set('sigma', '0.5')
        vtype.set('length', '5.0')
        vtype.set('minGap', '2.5')
        vtype.set('maxSpeed', '13.89')
        
        # Get available edges from network file (simplified)
        # In real scenario, would parse the network file
        edges = self._get_random_edges(n_routes=60)
        
        # Add routes
        for i, (origin, dest) in enumerate(edges):
            route = ET.SubElement(routes_xml, 'route')
            route.set('id', f'route_{i}')
            route.set('edges', f'{origin} {dest}')
        
        # Add vehicles with random departures
        np.random.seed(42)
        for i in range(n_vehicles):
            vehicle = ET.SubElement(routes_xml, 'vehicle')
            vehicle.set('id', f'veh_{i}')
            vehicle.set('type', 'car')
            vehicle.set('route', f'route_{i % len(edges)}')
            vehicle.set('depart', str(np.random.randint(0, simulation_time)))
        
        # Write to file
        routes_file = self.output_dir / 'traffic.rou.xml'
        self._prettify_and_save(routes_xml, routes_file)
        print(f"✓ Created routes file: {routes_file}")
        return routes_file
    
    def create_config_file(self, net_file, routes_file, simulation_time=3600):
        """Create main SUMO configuration file."""
        
        config_xml = ET.Element('configuration')
        
        # Input section
        input_elem = ET.SubElement(config_xml, 'input')
        ET.SubElement(input_elem, 'net-file').set('value', str(net_file))
        ET.SubElement(input_elem, 'route-files').set('value', str(routes_file))
        
        # Time section
        time_elem = ET.SubElement(config_xml, 'time')
        ET.SubElement(time_elem, 'begin').set('value', '0')
        ET.SubElement(time_elem, 'end').set('value', str(simulation_time))
        
        # Output section
        output_elem = ET.SubElement(config_xml, 'output')
        ET.SubElement(output_elem, 'tripinfo-output').set('value', str(self.output_dir / 'tripinfo.xml'))
        ET.SubElement(output_elem, 'lanearea-detectors').set('value', str(self.output_dir / 'detectors.xml'))
        
        # GUI section
        gui_elem = ET.SubElement(config_xml, 'gui_only')
        ET.SubElement(gui_elem, 'gui-settings-file').set('value', str(self.output_dir / 'gui.xml'))
        
        # Write to file
        config_file = self.output_dir / 'sumo.sumocfg'
        self._prettify_and_save(config_xml, config_file)
        print(f"✓ Created config file: {config_file}")
        return config_file
    
    def _get_random_edges(self, n_routes=60):
        """Generate random edges for routes."""
        # Simplified: generate edges based on node count
        n_nodes = 10
        edges = []
        np.random.seed(42)
        
        for _ in range(n_routes):
            i, j = np.random.choice(n_nodes, 2, replace=False)
            origin = f'e{min(i,j)}{max(i,j)}_0'
            dest = f'e{min(i,j)+1}_{(max(i,j)-min(i,j)-1)%10}_0' if i < j else origin
            edges.append((origin, dest))
        
        return edges
    
    def _prettify_and_save(self, elem, filepath):
        """Pretty print and save XML element."""
        rough_str = ET.tostring(elem, 'utf-8')
        reparsed = minidom.parseString(rough_str)
        pretty_str = reparsed.toprettyxml(indent="  ")
        pretty_str = '\n'.join([line for line in pretty_str.split('\n') if line.strip()])
        
        with open(filepath, 'w') as f:
            f.write(pretty_str)


# ============================================================================
# MAIN: Generate Network Files
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("SUMO Network Generator")
    print("="*60 + "\n")
    
    # Generate network
    generator = SUMONetworkGenerator(n_nodes=10, grid_size=1000)
    net_file = generator.create_net_file()
    
    if net_file:
        # Generate configuration
        config_gen = SUMOConfigGenerator(net_file)
        routes_file = config_gen.create_routes_file(n_vehicles=100, simulation_time=3600)
        config_file = config_gen.create_config_file(net_file, routes_file, simulation_time=3600)
        
        print("\n✓ SUMO network files created successfully!")
        print(f"  Network: {net_file}")
        print(f"  Routes: {routes_file}")
        print(f"  Config: {config_file}")
    else:
        print("\n⚠ Network generation incomplete. Install SUMO to enable full functionality.")
        print("  Download from: https://sumo.dlr.de/wiki/Downloads")
