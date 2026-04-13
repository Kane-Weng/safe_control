"""
Created on Apr 09
@author: Kane Weng

@description:
This script acts as a testing suite to validate the 3D Dynamic Paraboloid 
Control Barrier Function (DPCBF) across various edge cases and stress tests.

Usage:
    python evaluate_dpcbf_3d.py --test <test_name> [--analyze] [--steps 1000]

Available Tests:
    - head_on
    - cross_traffic
    - vertical_drop
    - moving_wall
    - asteroid_field
    - all
"""

import numpy as np
import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from safe_control.rerun_sim.rerun_controller import RerunControllerDyn


def get_scenario_config(scenario_name: str) -> dict:
    """Generates specific 3D configurations to test DPCBF behavior."""
    dt = 0.05
    robot_spec = {
        'model': 'DoubleIntegrator3D_DPCBF',
        'v_max': 3.5,
        'a_max': 6.0,
        'radius': 0.25,
        'reached_threshold': 0.5,
    }
    dynamic_obs = []

    if scenario_name == 'head_on':
        # GOAL: Test response to high relative velocity directly on the X-axis.
        waypoints_3d = np.array([[0.0, 0.0, 1.0], [20.0, 0.0, 1.0]], dtype=np.float64)
        dynamic_obs.append([18.0, 0.0, 1.0, 0.6, -2.5, 0.0, 0.0]) # Fast incoming

    elif scenario_name == 'cross_traffic':
        # GOAL: Test lateral dodging and timing (Y-axis crossing).
        waypoints_3d = np.array([[0.0, 0.0, 1.0], [20.0, 0.0, 1.0]], dtype=np.float64)
        dynamic_obs = [
            [6.0, -8.0, 1.0, 0.5, 0.0,  2.0, 0.0],
            [12.0, 8.0, 1.0, 0.5, 0.0, -2.0, 0.0],
            [16.0, -6.0, 1.0, 0.5, 0.0,  1.5, 0.0],
        ]

    elif scenario_name == 'vertical_drop':
        # GOAL: Test pure 3D vertical avoidance (Z-axis). 2D CBFs fail here.
        waypoints_3d = np.array([[0.0, 0.0, 2.0], [15.0, 0.0, 2.0]], dtype=np.float64)
        dynamic_obs = [
            [5.0,  0.0, 10.0, 0.6, 0.0, 0.0, -2.5],
            [10.0, 0.0, 12.0, 0.6, 0.0, 0.0, -3.0],
        ]

    elif scenario_name == 'moving_wall':
        # GOAL: Edge case - local minima vs gap finding.
        waypoints_3d = np.array([[0.0, 0.0, 2.0], [20.0, 0.0, 2.0]], dtype=np.float64)
        wall_x = 15.0
        wall_vx = -1.0 # Wall moves towards robot
        
        # Build the wall in the Y-Z plane
        for y in [-2.5, -1.0, 0.5, 2.0]:
            for z in [0.5, 2.0, 3.5]:
                # Leave a gap at y=0.5, z=2.0
                if y == 0.5 and z == 2.0:
                    continue 
                dynamic_obs.append([wall_x, y, z, 0.6, wall_vx, 0.0, 0.0])

    elif scenario_name == 'asteroid_field':
        # GOAL: Stress test the QP solver with dense, multi-directional paraboloids.
        waypoints_3d = np.array([
            [0.0, 0.0, 1.0], 
            [10.0, 8.0, 4.0], 
            [20.0, 0.0, 1.0]
        ], dtype=np.float64)
        
        # Deterministic pseudo-random generation for repeatable tests
        for i in range(12):
            ox = 5.0 + (i * 1.2)
            oy = np.sin(i) * 5.0
            oz = 1.0 + (i % 4)
            r = 0.3 + (i % 3) * 0.15
            vx = -0.5 + np.cos(i) * 0.5
            vy = np.sin(i * 2) * 0.8
            vz = np.cos(i) * 0.4
            dynamic_obs.append([ox, oy, oz, r, vx, vy, vz])

    else:
        raise ValueError(f"Unknown scenario: {scenario_name}")

    x_init = np.array([
        waypoints_3d[0, 0],
        waypoints_3d[0, 1],
        waypoints_3d[0, 2],
        0.0, 0.0, 0.0
    ], dtype=np.float64)

    return {
        'waypoints': waypoints_3d,
        'obstacles': np.array(dynamic_obs, dtype=np.float64),
        'robot_spec': robot_spec,
        'x_init': x_init,
        'dt': dt,
    }


def analyze_trajectory(controller, scenario_name: str):
    """Computes and prints statistics about the completed flight path."""
    trajectory = np.array(controller.trajectory)
    if len(trajectory) < 2:
        print(f"  [!] Not enough trajectory data to analyze for {scenario_name}.")
        return

    diffs = np.diff(trajectory, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    total_distance = np.sum(distances)
    
    # 1 / dt gives steps per second.
    avg_velocity = total_distance / len(trajectory) * (1.0 / controller.dt)
    altitudes = trajectory[:, 2]

    print(f"\n--- Trajectory Analysis: {scenario_name.upper()} ---")
    print(f"  Total steps:           {len(trajectory)}")
    print(f"  Start Pos:             [{trajectory[0][0]:.2f}, {trajectory[0][1]:.2f}, {trajectory[0][2]:.2f}]")
    print(f"  End Pos:               [{trajectory[-1][0]:.2f}, {trajectory[-1][1]:.2f}, {trajectory[-1][2]:.2f}]")
    print(f"  Total Path Distance:   {total_distance:.2f} m")
    print(f"  Average Velocity:      {avg_velocity:.2f} m/s")
    print(f"  Altitude Range:        {altitudes.min():.2f} m  ->  {altitudes.max():.2f} m")
    print("------------------------------------------\n")


def run_test(scenario_name: str, num_steps: int, verbose: bool, do_analyze: bool):
    """Initializes and runs a single scenario."""
    if verbose:
        print("\n" + "=" * 70)
        print(f" INITIALIZING TEST: {scenario_name.upper()}")
        print("=" * 70)

    scenario_data = get_scenario_config(scenario_name)

    controller = RerunControllerDyn(
        X0=scenario_data['x_init'],
        robot_spec=scenario_data['robot_spec'],
        dt=scenario_data['dt'],
        controller_type={'pos': 'cbf_qp'},
        enable_rotation=True,
        raise_error=False
    )

    controller.set_obstacles(scenario_data['obstacles'])
    controller.set_waypoints(scenario_data['waypoints'])

    # Run the simulation loop
    controller.run(max_steps=num_steps, verbose=verbose)

    if do_analyze:
        analyze_trajectory(controller, scenario_name)

    return controller


def main():
    valid_tests = [
        'head_on', 
        'cross_traffic', 
        'vertical_drop', 
        'moving_wall', 
        'asteroid_field',
        'all'
    ]

    parser = argparse.ArgumentParser(description="Evaluate 3D DPCBF Simulation via Rerun")
    
    parser.add_argument(
        '-t', '--test',
        type=str,
        choices=valid_tests,
        default='head_on',
        help='Which scenario to run (use "all" to run sequentially).'
    )
    parser.add_argument(
        '-s', '--steps',
        type=int,
        default=1000,
        help='Maximum number of simulation steps.'
    )
    parser.add_argument(
        '-a', '--analyze',
        action='store_true',
        help='Print trajectory statistics after the simulation completes.'
    )
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress verbose step-by-step output.'
    )

    args = parser.parse_args()

    # Determine which tests to run
    tests_to_run = valid_tests[:-1] if args.test == 'all' else [args.test]

    print(f"\nStarting DPCBF Evaluation Suite...")
    print(f"Tests queued: {len(tests_to_run)}")

    for test_name in tests_to_run:
        try:
            run_test(
                scenario_name=test_name,
                num_steps=args.steps,
                verbose=not args.quiet,
                do_analyze=args.analyze
            )
        except Exception as e:
            print(f"\n[!] FAILED test '{test_name}': {e}")

    print("\n" + "=" * 70)
    print(" Evaluation suite completed. Open the Rerun viewer to inspect.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()