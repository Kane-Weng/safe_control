"""
Example: Running Rerun 3D Visualization

This example demonstrates how to use the Rerun 3D visualization module
with the safe control simulation.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from safe_control.rerun_sim.main3D_rerun import run_rerun_simulation


def example_1_simple_scenario():
    """
    Example 1: Run the simple scenario

    Features:
    - Linear path from (1, 7.5, 0) to (20, 7.5, 0) with mid-waypoint at higher altitude
    - 6 dynamic obstacles with 3D motion
    - CBF-QP controller
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Simple 3D Scenario")
    print("="*70)

    controller = run_rerun_simulation(
        scenario='simple',
        num_steps=500,
        verbose=True
    )

    return controller


def example_2_complex_scenario():
    """
    Example 2: Run the complex scenario

    Features:
    - 5 waypoints with varying altitudes (0-2.5m)
    - 18 dynamic obstacles in 3D grid
    - More challenging collision avoidance
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Complex 3D Scenario")
    print("="*70)

    controller = run_rerun_simulation(
        scenario='complex',
        num_steps=1000,
        verbose=True
    )

    return controller


def example_3_custom_scenario():
    """
    Example 3: Define and run a custom scenario
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Custom 3D Scenario")
    print("="*70)

    from safe_control.rerun_sim.rerun_controller import RerunControllerDyn

    # Define custom waypoints
    waypoints = np.array([
        [2.0, 5.0, 0.0],
        [5.0, 8.0, 1.0],
        [10.0, 5.0, 0.5],
        [15.0, 10.0, 1.5],
    ])

    # Define custom obstacles
    obstacles = np.array([
        [6.0, 6.0, 0.5, 0.4, -0.2, 0.1, 0.0],
        [8.0, 7.0, 1.0, 0.5, -0.3, -0.1, 0.1],
        [12.0, 8.0, 0.8, 0.4, -0.1, 0.2, -0.05],
    ])

    # Robot specification
    robot_spec = {
        'model': 'DoubleIntegrator3D_DPCBF',
        'v_max': 2.5,
        'a_max': 4.0,
        'radius': 0.3,
        'reached_threshold': 0.4,
    }

    # Initial state
    x_init = np.array([
        waypoints[0, 0],
        waypoints[0, 1],
        waypoints[0, 2],
        0.0, 0.0, 0.0
    ])

    # Create controller
    controller = RerunControllerDyn(
        X0=x_init,
        robot_spec=robot_spec,
        dt=0.05,
        controller_type={'pos': 'cbf_qp'},
    )

    # Configure and run
    controller.set_obstacles(obstacles)
    controller.set_waypoints(waypoints)
    controller.run(max_steps=800, verbose=True)

    return controller


def example_4_trajectory_analysis():
    """
    Example 4: Analyze trajectory and statistics after simulation
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Trajectory Analysis")
    print("="*70)

    controller = run_rerun_simulation(
        scenario='simple',
        num_steps=600,
        verbose=False
    )

    # Analyze trajectory
    trajectory = np.array(controller.trajectory)

    print(f"\nTrajectory Statistics:")
    print(f"  Total points: {len(trajectory)}")
    print(f"  Start: {trajectory[0]}")
    print(f"  End: {trajectory[-1]}")

    # Compute path length
    diffs = np.diff(trajectory, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    total_distance = np.sum(distances)

    print(f"  Total distance traveled: {total_distance:.2f}m")
    print(f"  Average velocity: {total_distance / len(trajectory) * 20:.2f}m/s")  # 20 steps/sec

    # Compute altitude profile
    altitudes = trajectory[:, 2]
    print(f"  Min altitude: {altitudes.min():.2f}m")
    print(f"  Max altitude: {altitudes.max():.2f}m")

    return controller


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " Rerun 3D Visualization - Usage Examples ".center(68) + "║")
    print("╚" + "="*68 + "╝")

    # Run examples
    controllers = []

    try:
        controllers.append(example_1_simple_scenario())
    except Exception as e:
        print(f"Example 1 failed: {e}")

    try:
        controllers.append(example_2_complex_scenario())
    except Exception as e:
        print(f"Example 2 failed: {e}")

    try:
        controllers.append(example_3_custom_scenario())
    except Exception as e:
        print(f"Example 3 failed: {e}")

    try:
        controllers.append(example_4_trajectory_analysis())
    except Exception as e:
        print(f"Example 4 failed: {e}")

    print("\n" + "="*70)
    print(f"Examples completed! Generated {len(controllers)} simulations.")
    print("="*70)
    print("\nNext steps:")
    print("  1. Open the Rerun viewer")
    print("  2. Load the recordings from this session")
    print("  3. Scrub through the timeline to inspect collision avoidance")
    print("  4. Observe the 3D DPCBF paraboloid surfaces around obstacles")
    print("\n")


if __name__ == "__main__":
    main()
