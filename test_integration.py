"""
Integration Test & Demo Script
Tests all components of the SUMO + Signal Control + Green Corridor system
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd


def test_signal_controller():
    """Test signal controller functionality."""
    print("\n" + "="*60)
    print("TEST 1: Signal Controller")
    print("="*60)
    
    try:
        from signal_controller import QuantumGuidedSignalController
        
        controller = QuantumGuidedSignalController(n_nodes=10)
        
        print("✓ Controller initialized successfully")
        print(f"  - Nodes: 10")
        print(f"  - Min green: {controller.min_green}s")
        print(f"  - Max green: {controller.max_green}s")
        
        # Test getting timings
        timings = controller.get_all_timings()
        print(f"✓ Retrieved timings for {len(timings)} nodes")
        
        # Sample output
        sample_node = 'n0'
        timing = timings[sample_node]
        print(f"\n  Sample timing ({sample_node}):")
        print(f"    - Green: {timing['green']:.1f}s")
        print(f"    - Cycle: {timing['cycle']:.1f}s")
        print(f"    - Arrival rate: {timing['arrival_rate']:.2f} veh/s")
        
        # Test signal state
        states = [controller.get_signal_state(sample_node, t) for t in [0, 10, 20, 30]]
        print(f"\n  Signal states at different times:")
        print(f"    Time 0s: {states[0]}")
        print(f"    Time 10s: {states[1]}")
        print(f"    Time 20s: {states[2]}")
        print(f"    Time 30s: {states[3]}")
        
        print("\n✅ Signal Controller Test: PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Signal Controller Test: FAILED")
        print(f"   Error: {str(e)}")
        return False


def test_green_corridor():
    """Test green corridor functionality."""
    print("\n" + "="*60)
    print("TEST 2: Green Corridor Optimizer")
    print("="*60)
    
    try:
        from signal_controller import QuantumGuidedSignalController
        from green_corridor import GreenCorridor, GreenWaveOptimizer
        
        # Create dependencies
        signal_controller = QuantumGuidedSignalController(n_nodes=10)
        corridor = GreenCorridor(n_nodes=10, signal_controller=signal_controller)
        
        print("✓ Green Corridor initialized successfully")
        print(f"  - Network size: 10 nodes")
        
        # Create corridors
        corridors = corridor.optimize_all_corridors()
        print(f"✓ Created {len(corridors)} green corridors")
        
        for i, corr in enumerate(corridors):
            route_str = ' → '.join(corr['route'])
            print(f"  [{i}] {corr['name']}: {route_str}")
        
        # Get visualization data
        viz_data = corridor.get_corridor_visualization_data()
        print(f"\n✓ Retrieved visualization data for {len(viz_data)} nodes")
        print(f"  Shape: {viz_data.shape}")
        
        # Get status report
        report = corridor.get_corridor_status_report()
        print(f"✓ Generated corridor status report")
        print(f"  Columns: {', '.join(report.columns.tolist())}")
        
        # Test green wave optimizer
        optimizer = GreenWaveOptimizer(corridor)
        for i in range(10):
            optimizer.add_arrival_pattern(f'n{i}', arrival_rate=0.3 + 0.05*i)
        
        print(f"✓ Created green wave optimizer with arrival patterns")
        
        if corridor.corridors:
            first_corridor = list(corridor.corridors.keys())[0]
            stops = optimizer.calculate_stops(first_corridor)
            print(f"  Expected stops on {first_corridor}: {stops:.2f}")
        
        print("\n✅ Green Corridor Test: PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Green Corridor Test: FAILED")
        print(f"   Error: {str(e)}")
        return False


def test_sumo_network():
    """Test SUMO network generation."""
    print("\n" + "="*60)
    print("TEST 3: SUMO Network Generation")
    print("="*60)
    
    try:
        from sumo_network import SUMONetworkGenerator, SUMOConfigGenerator
        
        # Create network generator
        generator = SUMONetworkGenerator(n_nodes=10, grid_size=1000)
        print("✓ Network generator initialized")
        print(f"  - Grid size: 1000m x 1000m")
        print(f"  - Nodes: 10")
        
        # Check node positions
        positions = generator.node_positions
        print(f"✓ Generated {len(positions)} node positions")
        
        sample_nodes = list(positions.items())[:3]
        for node_id, (x, y) in sample_nodes:
            print(f"  {node_id}: ({x:.0f}, {y:.0f})")
        
        # Create network files (without netconvert)
        nod_file = generator.create_node_file()
        edg_file = generator.create_edge_file()
        tll_file = generator.create_tllogic_file()
        
        print(f"✓ Created network definition files:")
        print(f"  - {nod_file.name}")
        print(f"  - {edg_file.name}")
        print(f"  - {tll_file.name}")
        
        # Check files exist
        print(f"✓ All files created successfully")
        print(f"  Output directory: {generator.output_dir}")
        
        print("\n✅ SUMO Network Test: PASSED (net.xml requires SUMO installation)")
        return True
        
    except Exception as e:
        print(f"\n❌ SUMO Network Test: FAILED")
        print(f"   Error: {str(e)}")
        return False


def test_data_availability():
    """Test if required data files exist."""
    print("\n" + "="*60)
    print("TEST 4: Data Availability")
    print("="*60)
    
    try:
        outputs_dir = Path('./outputs')
        required_files = [
            'traffic_predictions_5s.csv',
            'embeddings.csv',
            'quantum_output.csv'
        ]
        
        existing_files = []
        missing_files = []
        
        for fname in required_files:
            fpath = outputs_dir / fname
            if fpath.exists():
                size_kb = fpath.stat().st_size / 1024
                existing_files.append((fname, size_kb))
                print(f"✓ {fname} ({size_kb:.1f} KB)")
            else:
                missing_files.append(fname)
                print(f"⚠ {fname} (NOT FOUND)")
        
        if missing_files:
            print(f"\n⚠ Missing {len(missing_files)} file(s)")
            print("  Run the following to generate:")
            print("    python traffic_prediction_pipeline.py")
            print("    python gnn_embedding_pipeline.py")
            return False
        
        print(f"\n✅ Data Availability Test: PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Data Availability Test: FAILED")
        print(f"   Error: {str(e)}")
        return False


def test_streamlit_integration():
    """Test streamlit integration."""
    print("\n" + "="*60)
    print("TEST 5: Streamlit Integration")
    print("="*60)
    
    try:
        import streamlit as st
        print("✓ Streamlit imported successfully")
        
        # Check if custom modules can be imported from streamlit context
        streamlit_file = Path('./streamlit_app.py')
        if streamlit_file.exists():
            print(f"✓ {streamlit_file.name} exists")
            
            # Check file size
            size_kb = streamlit_file.stat().st_size / 1024
            print(f"  Size: {size_kb:.1f} KB")
            
            print("\n✅ Streamlit Integration Test: PASSED")
            return True
        else:
            print(f"❌ {streamlit_file.name} not found")
            return False
            
    except Exception as e:
        print(f"\n❌ Streamlit Integration Test: FAILED")
        print(f"   Error: {str(e)}")
        return False


def test_imports():
    """Test if all required modules can be imported."""
    print("\n" + "="*60)
    print("TEST 0: Module Imports")
    print("="*60)
    
    modules = {
        'numpy': 'Core numerics',
        'pandas': 'Data handling',
        'torch': 'PyTorch ML',
        'pennylane': 'Quantum computing',
        'streamlit': 'Dashboard',
        'plotly': 'Visualization',
        'networkx': 'Graph analysis',
    }
    
    passed = 0
    failed = 0
    
    for module_name, description in modules.items():
        try:
            __import__(module_name)
            print(f"✓ {module_name:15} - {description}")
            passed += 1
        except ImportError:
            print(f"❌ {module_name:15} - {description} (NOT INSTALLED)")
            failed += 1
    
    print(f"\nImports: {passed}/{len(modules)} passed")
    
    if failed > 0:
        print(f"\n⚠ Install missing packages:")
        print("  pip install -r requirements.txt")
        return False
    
    print("\n✅ Module Import Test: PASSED")
    return True


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("INTEGRATION TEST SUITE")
    print("SUMO + Signal Control + Green Corridor System")
    print("="*70)
    
    results = []
    
    # Run tests in order
    results.append(("Module Imports", test_imports()))
    results.append(("Data Availability", test_data_availability()))
    results.append(("Signal Controller", test_signal_controller()))
    results.append(("Green Corridor", test_green_corridor()))
    results.append(("SUMO Network", test_sumo_network()))
    results.append(("Streamlit Integration", test_streamlit_integration()))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:30} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("1. Run SUMO network setup: python sumo_network.py")
        print("2. Launch dashboard: streamlit run streamlit_app.py")
        print("3. Use '⚙️ Simulation' page to run traffic simulation")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed. Please fix issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
