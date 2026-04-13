"""
Created on Apr 09
@author: Kane Weng

@description:
This script runs the DoubleIntegrator3D_DPCBF simulation with full 3D Rerun
visualization. It completely bypasses Matplotlib and provides a modern,
interactive 3D visualization experience.

Usage:
    python -m safe_control.rerun_sim.main3D_rerun

To view the results in Rerun:
    1. Create the recording if needed: rr init
    2. Open in viewer: rerun "path/to/recording"
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from safe_control.rerun_sim.rerun_controller import RerunControllerDyn


def create_3d_scenario_simple():
    """Create a simple 3D scenario with obstacles and waypoints."""
    dt = 0.05

    # Define 3D waypoints
    waypoints_3d = np.array([
        [1.0, 7.5, 0.0],
        [10.0, 7.5, 2.0],
        [20.0, 7.5, 0.0],
    ], dtype=np.float64)

    # Define 3D dynamic obstacles (moving in 3D space)
    dynamic_obs = []
    obs_positions_base = [
        [8.0, 9.0, 0.5],
        [10.0, 4.0, 1.0],
        [12.0, 5.0, 0.5],
        [14.0, 9.0, 1.5],
        [16.0, 6.0, 0.5],
        [18.0, 14.0, 1.0],
    ]

    for i, pos in enumerate(obs_positions_base):
        ox, oy, oz = pos
        r = 0.5
        
        # Give every single obstacle a unique velocity vector
        vx = -0.3 - (i * 0.05)           # Varies based on index
        vy = 0.5 if i % 2 == 0 else -0.5 # Alternates left/right
        vz = 0.1 * np.sin(i)             # Adds a unique vertical wobble

        dynamic_obs.append([ox, oy, oz, r, vx, vy, vz])

    known_obs = np.array(dynamic_obs, dtype=np.float64)

    # Robot specification
    robot_spec = {
        'model': 'DoubleIntegrator3D_DPCBF',
        'v_max': 3.0,
        'a_max': 5.0,
        'radius': 0.25,
        'reached_threshold': 0.5,
    }

    # Initial state: [x, y, z, vx, vy, vz]
    x_init = np.array([
        waypoints_3d[0, 0],
        waypoints_3d[0, 1],
        waypoints_3d[0, 2],
        0.0, 0.0, 0.0
    ], dtype=np.float64)

    return {
        'waypoints': waypoints_3d,
        'obstacles': known_obs,
        'robot_spec': robot_spec,
        'x_init': x_init,
        'dt': dt,
    }


def create_3d_scenario_complex():
    """Create a more complex 3D scenario with vertical movement."""
    dt = 0.05

    # 3D waypoints with varying altitudes
    waypoints_3d = np.array([
        [2.0, 5.0, 0.5],
        [8.0, 8.0, 1.5],
        [15.0, 12.0, 2.5],
        [22.0, 8.0, 1.0],
        [25.0, 5.0, 0.0],
    ], dtype=np.float64)

    # Generate a grid of dynamic obstacles in 3D space
    dynamic_obs = []
    for xi in [5, 10, 15, 20]:
        for yi in [4, 8, 12]:
            for zi in [0.5, 1.5, 2.5]:
                r = 0.4
                vx = -0.3
                vy = np.sin(xi / 10.0) * 0.2
                vz = np.sin(yi / 10.0) * 0.1

                dynamic_obs.append([xi, yi, zi, r, vx, vy, vz])

    known_obs = np.array(dynamic_obs, dtype=np.float64)

    robot_spec = {
        'model': 'DoubleIntegrator3D_DPCBF',
        'v_max': 3.5,
        'a_max': 6.0,
        'radius': 0.25,
        'reached_threshold': 0.6,
    }

    x_init = np.array([
        waypoints_3d[0, 0],
        waypoints_3d[0, 1],
        waypoints_3d[0, 2],
        0.0, 0.0, 0.0
    ], dtype=np.float64)

    return {
        'waypoints': waypoints_3d,
        'obstacles': known_obs,
        'robot_spec': robot_spec,
        'x_init': x_init,
        'dt': dt,
    }


def run_rerun_simulation(scenario: str = 'simple',
                         num_steps: int = 1000,
                         controller_type: dict = None,
                         verbose: bool = True):
    """
    Run the Rerun 3D visualization simulation.

    Args:
        scenario: 'simple' or 'complex' scenario selection
        num_steps: Number of simulation steps
        controller_type: Controller configuration (default: CBF-QP)
        verbose: Print progress information
    """
    if controller_type is None:
        controller_type = {'pos': 'cbf_qp'}

    # Create scenario
    if scenario == 'simple':
        scenario_data = create_3d_scenario_simple()
    elif scenario == 'complex':
        scenario_data = create_3d_scenario_complex()
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    # Extract scenario parameters
    waypoints = scenario_data['waypoints']
    obstacles = scenario_data['obstacles']
    robot_spec = scenario_data['robot_spec']
    x_init = scenario_data['x_init']
    dt = scenario_data['dt']

    if verbose:
        print("=" * 70)
        print(f"Rerun 3D Visualization - {scenario.upper()} Scenario")
        print("=" * 70)
        print(f"Robot Spec: {robot_spec}")
        print(f"Initial State: {x_init.T}")
        print(f"Waypoints: {len(waypoints)}")
        print(f"Obstacles: {len(obstacles)}")
        print(f"Simulation Time: {num_steps * dt:.2f}s ({num_steps} steps)")
        print("=" * 70)

    # Create controller
    controller = RerunControllerDyn(
        X0=x_init,
        robot_spec=robot_spec,
        dt=dt,
        controller_type=controller_type,
        enable_rotation=True,
        raise_error=False
    )

    # Set obstacles and waypoints
    controller.set_obstacles(obstacles)
    controller.set_waypoints(waypoints)

    # Run simulation
    controller.run(max_steps=num_steps, verbose=verbose)

    if verbose:
        print("\n" + "=" * 70)
        print("Simulation completed!")
        print(f"Total trajectory length: {len(controller.trajectory)} steps")
        print("Open in Rerun viewer with: rerun <your_recording>")
        print("=" * 70)

    return controller


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run 3D Rerun simulation for safe control algorithms"
    )
    parser.add_argument(
        '--scenario',
        type=str,
        choices=['simple', 'complex'],
        default='simple',
        help='Simulation scenario'
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=1000,
        help='Number of simulation steps'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )

    args = parser.parse_args()

    # Run simulation
    controller = run_rerun_simulation(
        scenario=args.scenario,
        num_steps=args.steps,
        verbose=not args.quiet
    )

    return controller


if __name__ == "__main__":
    controller = main()
