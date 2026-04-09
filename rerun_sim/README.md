"""
Rerun 3D Visualization Module - Architecture Documentation

This module provides a complete 3D visualization layer for safe control algorithms
using the Rerun SDK (v0.15+). It is designed with strict architectural constraints
to preserve the existing Matplotlib-based 2D/pseudo-3D pipeline.

================================================================================
DIRECTORY STRUCTURE
================================================================================

rerun_sim/
├── __init__.py                 # Package marker
├── rerun_logger.py             # RerunLogger3D - all rr.log() calls
├── rerun_controller.py         # RerunControllerDyn - execution logic
├── main3D_rerun.py            # Entry point (executable)
└── README.md                   # This file

================================================================================
ARCHITECTURAL DESIGN
================================================================================

1. ZERO DESTRUCTIVE CHANGES
   - Original files remain completely untouched:
     * utils/plotting.py (Matplotlib plotting)
     * utils/animation.py (Matplotlib animation)
     * dynamic_env/robot.py (rendering logic)
     * tracking.py (LocalTrackingController base)
   - All new code lives exclusively in rerun_sim/

2. DECOUPLED EXECUTION PATHS
   Existing Path (Matplotlib):
     main3D.py → LocalTrackingControllerDyn → render_plot() → Matplotlib

   New Path (Rerun):
     main3D_rerun.py → RerunControllerDyn → rerun_logger → rr.log()

3. MATHEMATICAL COMPONENT REUSE
   - RerunControllerDyn wraps DoubleIntegrator3D_DPCBF for physics
   - NO additional mathematical implementations
   - Pure visualization/logging layer

================================================================================
KEY COMPONENTS
================================================================================

A. RerunLogger3D (rerun_logger.py)
   ────────────────────────────────
   Purpose: Centralized Rerun logging handler

   Key Methods:
   - set_time_step(step)              → rr.set_time("step", sequence=step)
   - log_robot_state(pos, vel, r)     → rr.Points3D + rr.Arrows3D
   - log_obstacles(obs_array)         → 3D spheres + velocity vectors
   - log_waypoints(wp_array)          → 3D target points
   - log_goal(goal_pos)               → Current goal marker
   - log_trajectory(trajectory)       → Path history as point cloud
   - log_dpcbf_paraboloid()           → 3D collision surface (main feature!)

   Implementation Detail (DPCBF Paraboloid):
   ───────────────────────────────────────
   The paraboloid is defined in relative velocity space:
     h(v_rel) = v_rel_x + λ(v_rel_y² + v_rel_z²) + μ

   Algorithm:
   1. Extract λ and μ from DoubleIntegrator3D_DPCBF.{k_lambda, k_mu}
   2. Generate meshgrid in 2D (y_rel, z_rel) space
   3. Solve for x_rel at boundary where h=0:
        x_rel = -(λ·(y_rel² + z_rel²) + μ)
   4. Assemble 3D point cloud in relative velocity frame
   5. Rotate using yaw/pitch angles relative to obstacle
   6. Transform to world frame
   7. Log as rr.Points3D with semi-transparent gradient coloring

B. RerunControllerDyn (rerun_controller.py)
   ──────────────────────────────────────────
   Purpose: Main simulation loop with Rerun integration

   Inheritance:
   ├── Wraps DoubleIntegrator3D_DPCBF (physics model)
   ├── Manages state machine (idle → track → stop/rotate)
   ├── Handles waypoint tracking and goal updates
   ├── Solves CBF-QP control problem
   └── Logs everything via RerunLogger3D

   State Machine:
     idle → [goal update & track] ↔ rotate/stop

   Key Methods:
   - set_waypoints(wp_array)       → Configure navigation goals
   - set_obstacles(obs_array)      → Update obstacle field
   - step()                        → Execute one time step
   - run(max_steps)                → Run full simulation loop
   - _log_frame()                  → Log current state to Rerun

C. Main Entry Point (main3D_rerun.py)
   ──────────────────────────────────
   Purpose: Executable simulation runner

   Features:
   - Two pre-configured scenarios: 'simple' and 'complex'
   - Argument parsing for customization
   - Scenario generation with proper 3D initialization
   - Verbose progress reporting

   Usage:
     python -m safe_control.rerun_sim.main3D_rerun --scenario simple --steps 1000
     python -m safe_control.rerun_sim.main3D_rerun --scenario complex --steps 2000 --quiet

================================================================================
DATA FORMATS
================================================================================

Robot State (X):
   X = [x, y, z, vx, vy, vz]ᵀ  (6D state)
   ├─ Position: [x, y, z]
   └─ Velocity: [vx, vy, vz]

Obstacle Array (obs):
   Each row: [ox, oy, oz, r, vx, vy, vz]  (7D obstacle)
   ├─ Position: [ox, oy, oz]
   ├─ Radius: r (sphere radius)
   └─ Velocity: [vx, vy, vz]

Waypoints (wp):
   Each row: [x, y, z]  (3D waypoint)

DPCBF Parameters:
   From DoubleIntegrator3D_DPCBF.__init__:
   ├─ k_lambda = 0.1 (default)
   └─ k_mu = 0.5     (default)

   Computed at runtime:
   ├─ ego_dim = (r_obs + r_robot) × s  (combined safety radius)
   ├─ func_lambda = k_lambda × √(d_safe) / (||v_rel|| + ε)
   └─ func_mu = k_mu × √(d_safe)

================================================================================
USAGE EXAMPLES
================================================================================

1. Simple 3D Navigation (with default settings):
   ─────────────────────────────────────────────

   from safe_control.rerun_sim.main3D_rerun import run_rerun_simulation

   controller = run_rerun_simulation(
       scenario='simple',
       num_steps=1000,
       controller_type={'pos': 'cbf_qp'},
       verbose=True
   )

2. Command-line execution:
   ──────────────────────

   python -m safe_control.rerun_sim.main3D_rerun --scenario complex --steps 2000

3. Custom scenario:
   ────────────────

   from safe_control.rerun_sim.rerun_controller import RerunControllerDyn
   import numpy as np

   # Define your scenario
   waypoints = np.array([...])
   obstacles = np.array([...])
   robot_spec = {...}
   x_init = np.array([...])

   # Create and run controller
   controller = RerunControllerDyn(x_init, robot_spec)
   controller.set_waypoints(waypoints)
   controller.set_obstacles(obstacles)
   controller.run(max_steps=5000)

4. Accessing trajectory after simulation:
   ───────────────────────────────────────

   controller = run_rerun_simulation(scenario='simple', num_steps=1000)
   trajectory = controller.trajectory  # List of 3D positions
   print(f"Total path length: {len(trajectory)} points")
   print(f"Final position: {trajectory[-1]}")

================================================================================
RERUN VISUALIZATION FEATURES
================================================================================

Logged Elements:
   ✓ Robot position (3D sphere, color: blue)
   ✓ Robot velocity (3D arrow, color: blue)
   ✓ Obstacles (3D spheres, color: red)
   ✓ Obstacle velocities (3D arrows, color: orange)
   ✓ Waypoints (3D targets, color: green)
   ✓ Current goal (3D marker, color: green)
   ✓ Trajectory history (3D path, color: yellow)
   ✓ DPCBF paraboloid surfaces (3D point cloud, color: cyan/semi-transparent)

Timeline Navigation:
   - Full 3D trajectory playback with timeline scrubbing
   - Step-by-step inspection of collision avoidance
   - Real-time computation of safety margins

Interaction:
   - 3D camera: Rotate, zoom, pan
   - Time slider: Navigate through simulation
   - Toggle visibility by element type
   - Color-coded depth perception on paraboloid surfaces

================================================================================
MODERN RERUN API COMPLIANCE
================================================================================

Version: v0.15+

Using:
   ✓ rr.set_time("step", sequence=step)   [MODERN, not deprecated rr.set_time_sequence()]
   ✓ rr.Points3D() with radii and colors
   ✓ rr.Arrows3D() for vector fields
   ✓ rr.log() for all data recording

================================================================================
INTEGRATION WITH EXISTING CODEBASE
================================================================================

No modification required to:
   ✗ safe_control/utils/plotting.py
   ✗ safe_control/utils/animation.py
   ✗ safe_control/dynamic_env/robot.py
   ✗ safe_control/tracking.py (base LocalTrackingController)

Leverages:
   ✓ safe_control/robots/double_integrator3D.py (physics model)
   ✓ safe_control/dynamic_env/double_integrator3D_dpcbf.py (DPCBF math)
   ✓ safe_control/position_control/cbf_qp.py (control solver)

================================================================================
PERFORMANCE NOTES
================================================================================

- Paraboloid mesh density: 25×25 = 625 points per obstacle per step
- Trajectory subsampled to 500 points for visualization
- Nearest obstacles limited to 3 for paraboloid rendering
- Memory: ~O(N × mesh_density²) where N = number of obstacles

Optimization tips:
   1. Reduce mesh_density in log_dpcbf_paraboloid() to speed up rendering
   2. Limit max_steps to avoid huge trajectory arrays
   3. Use Rerun's built-in compression (automatic)

================================================================================
TROUBLESHOOTING
================================================================================

Issue: "ModuleNotFoundError: No module named 'rerun'"
Solution: pip install rerun-sdk

Issue: "CBF-QP solver not found"
Solution: Ensure safe_control.position_control.cbf_qp is available

Issue: "Paraboloid surface doesn't render"
Solution: Check that nearest obstacles have meaningful velocities

================================================================================
EXTENDING THE MODULE
================================================================================

To add visualization for other robot types:
   1. Create RerunControllerXXX inheriting from appropriate model class
   2. Implement _log_frame() method
   3. Add scenario generation function in main3D_rerun.py
   4. Keep RerunLogger3D unchanged (it's generic)

To modify DPCBF paraboloid visualization:
   1. Edit RerunLogger3D.log_dpcbf_paraboloid()
   2. Adjust mesh_density parameter
   3. Modify color/alpha gradient in _generate_gradient_colors()

================================================================================
"""

print(__doc__)
