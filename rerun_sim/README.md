# rerun_sim — 3D Safe Control Visualization

Real-time 3D visualization for the **DoubleIntegrator3D + DPCBF** pipeline using [Rerun SDK](https://rerun.io/) (v0.31+). Completely bypasses Matplotlib — no changes required to the existing codebase.

## Files

| File | Purpose |
|---|---|
| `rerun_logger.py` | `RerunLogger3D` — all `rr.log()` calls (robot, obstacles, trajectory, DPCBF paraboloids) |
| `rerun_controller.py` | `RerunControllerDyn` — simulation loop, CBF-QP safety filter, state machine |
| `main3D_rerun.py` | Entry point with `simple` and `complex` pre-built scenarios |
| `evaluate_dpcbf_3d.py` | Evaluation suite with 5 targeted test scenarios |

## Quick Start

```bash
# Simple scenario
python -m safe_control.rerun_sim.main3D_rerun --scenario simple --steps 1000

# Complex scenario (multi-altitude waypoints, dense obstacle grid)
python -m safe_control.rerun_sim.main3D_rerun --scenario complex --steps 2000 --quiet
```

## Evaluation Suite

```bash
# Single test
python evaluate_dpcbf_3d.py --test head_on --steps 1000 --analyze

# All tests sequentially
python evaluate_dpcbf_3d.py --test all --analyze
```

Available tests: `head_on`, `cross_traffic`, `vertical_drop`, `moving_wall`, `asteroid_field`

## Programmatic Usage

```python
from safe_control.rerun_sim.main3D_rerun import run_rerun_simulation

controller = run_rerun_simulation(scenario='simple', num_steps=1000)
print(controller.trajectory[-1])  # final position
```

## State Formats

- **Robot state**: `[x, y, z, vx, vy, vz]`
- **Obstacle row**: `[ox, oy, oz, r, vx, vy, vz]`
- **Waypoint row**: `[x, y, z]`

## DPCBF Paraboloid

The safety surface is defined in relative velocity space:

```
h(v_rel) = v_rel_x + λ(v_rel_y² + v_rel_z²) + μ = 0
```

At each step, the paraboloid boundary is solved, rotated into world frame via a line-of-sight rotation matrix, and logged as a semi-transparent point cloud per obstacle.

## Visualized Elements

- Robot position (blue sphere) + velocity (blue arrow)
- Obstacles (red spheres) + velocities (orange arrows)
- Waypoints (green markers)
- Trajectory history (yellow point cloud, last 100 steps, logged every 10 steps)
- DPCBF paraboloid surface per nearby obstacle (color-coded by obstacle ID)

## Dependencies

```bash
pip install rerun-sdk==0.31.2
```

Requires `safe_control.robots.double_integrator3D`, `safe_control.dynamic_env.double_integrator3D_dpcbf`, and `safe_control.position_control.cbf_qp`.
