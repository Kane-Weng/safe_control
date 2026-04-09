"""
Rerun Controller Module

Custom tracking controller that extends DoubleIntegrator3D_DPCBF and integrates
Rerun 3D visualization. Completely bypasses Matplotlib.

This controller:
1. Inherits all mathematical components from DoubleIntegrator3D_DPCBF
2. Uses Rerun for visualization instead of Matplotlib
3. Logs robot state, obstacles, DPCBF surfaces, and trajectories
4. Manages simulation stepping and time progression
"""

import numpy as np
from typing import Optional, List, Tuple
from safe_control.dynamic_env.double_integrator3D_dpcbf import DoubleIntegrator3D_DPCBF
from safe_control.rerun_sim.rerun_logger import RerunLogger3D


class RerunControllerDyn:
    """
    Rerun-based tracking controller for 3D double integrator with DPCBF.

    Manages:
    - Robot dynamics and stepping
    - Goal tracking and state machine
    - Obstacle avoidance
    - 3D visualization via Rerun
    """

    def __init__(self,
                 X0: np.ndarray,
                 robot_spec: dict,
                 dt: float = 0.05,
                 controller_type: Optional[dict] = None,
                 enable_rotation: bool = True,
                 raise_error: bool = False):
        """
        Initialize Rerun controller.

        Args:
            X0: Initial state [x, y, z, vx, vy, vz]
            robot_spec: Robot specification dictionary
            dt: Simulation timestep
            controller_type: Controller selection {'pos': 'cbf_qp', ...}
            enable_rotation: Whether to enable rotational behavior
            raise_error: Raise exception on collision/infeasibility
        """
        self.robot_spec = robot_spec
        self.dt = dt
        self.enable_rotation = enable_rotation
        self.raise_error = raise_error

        # Parse controller type
        if controller_type is None:
            controller_type = {'pos': 'cbf_qp'}
        self.pos_controller_type = controller_type.get('pos', 'cbf_qp')
        self.att_controller_type = controller_type.get('att', 'velocity_tracking_yaw')

        # State machine for tracking
        self.state_machine = 'idle'  # idle -> track -> stop/rotate
        self.reached_threshold = robot_spec.get('reached_threshold', 0.3)
        self.current_goal_index = 0

        # Physics model (inherits from DoubleIntegrator3D_DPCBF)
        self.model = DoubleIntegrator3D_DPCBF(dt, robot_spec)

        # --- NEW FIX FOR AGENT_BARRIER ---
        # The base CBFQP solver might not pass 'robot_radius'. We inject it here.
        original_agent_barrier = self.model.agent_barrier
        
        def patched_agent_barrier(X, obs=None, *args, **kwargs):
            if obs is None:
                import casadi as ca
                if isinstance(X, (ca.SX, ca.MX, ca.DM)):
                    return ca.DM([1]), ca.DM.zeros(1, X.shape[0]) # <--- Now 2 values!
                else:
                    return 1.0, np.zeros((1, X.shape[0]))         # <--- Now 2 values!
                    
            return original_agent_barrier(X, obs, self.robot_spec['radius'], *args, **kwargs)
            
        self.model.agent_barrier = patched_agent_barrier

        # Robot state
        self.X = X0.reshape(-1, 1)
        self.yaw = 0.0
        self.u_att = np.zeros((1, 1))
        self.u_pos = np.zeros((3, 1))

        # Goal and waypoint tracking
        self.waypoints = None
        self.goal = None

        # Obstacles
        self.obs = np.array([]).reshape(0, 7)
        self.unknown_obs = np.array([])

        # Visualization
        self.rerun_logger = RerunLogger3D(app_name="SafeControl3D_DPCBF")
        self.trajectory = []
        self.step_count = 0

        # Controller (initialized lazily)
        self.pos_controller = None

    def setup_cbf_qp_controller(self):
        """Initialize CBF-QP position controller."""
        if self.pos_controller_type == 'cbf_qp':
            from safe_control.position_control.cbf_qp import CBFQP
            self.pos_controller = CBFQP(self.model, self.robot_spec, num_obs=10)

    def set_waypoints(self, waypoints: np.ndarray):
        """Set target waypoints for navigation."""
        self.waypoints = np.array(waypoints, dtype=np.float64)
        self.current_goal_index = 0
        self.state_machine = 'idle'

    def set_obstacles(self, obstacles: np.ndarray):
        """
        Set known obstacles.

        Args:
            obstacles: Array of shape (N, 7) where each row is [ox, oy, oz, r, vx, vy, vz]
        """
        self.obs = np.array(obstacles, dtype=np.float64).reshape(-1, 7)

    def update_goal(self) -> Optional[np.ndarray]:
        """
        Update goal based on waypoint reaching.
        """
        if self.waypoints is None or len(self.waypoints) == 0:
            return None

        # ---> THE FIX <---
        # Check if we've already reached all waypoints BEFORE trying to index
        if self.current_goal_index >= len(self.waypoints):
            return None

        current_goal = self.waypoints[self.current_goal_index]

        # Check if current goal is reached
        pos = self.X[0:3, 0]
        distance_to_goal = np.linalg.norm(current_goal - pos)

        if distance_to_goal < self.reached_threshold:
            self.current_goal_index += 1
            if self.current_goal_index >= len(self.waypoints):
                return None  # All waypoints reached
            return self.waypoints[self.current_goal_index].reshape(-1, 1)

        return current_goal.reshape(-1, 1)

    def step_dyn_obs(self):
        """Update dynamic obstacle positions based on their velocities."""
        if self.obs.shape[0] > 0 and self.obs.shape[1] >= 7:
            self.obs[:, 0] += self.obs[:, 4] * self.dt  # vx
            self.obs[:, 1] += self.obs[:, 5] * self.dt  # vy
            self.obs[:, 2] += self.obs[:, 6] * self.dt  # vz

    def step(self):
        """
        Execute one simulation step.

        Performs:
        1. State machine update
        2. Goal update (waypoint reaching)
        3. Obstacle dynamics
        4. Control computation
        5. Robot dynamics step
        6. Logging to Rerun
        """
        self.step_count += 1
        self.rerun_logger.set_time_step(self.step_count)

        # 1. Update state machine
        if self.state_machine == 'stop':
            if self._has_stopped():
                if self.enable_rotation:
                    self.state_machine = 'rotate'
                else:
                    self.state_machine = 'track'
                self.goal = self.update_goal()
        else:
            self.goal = self.update_goal()

        # 2. Update dynamic obstacles
        self.step_dyn_obs()

        # 3. Compute nominal control
        if self.state_machine == 'idle':
            self.state_machine = 'track'
            self.goal = self.update_goal()

        u_ref = self._compute_nominal_input()

        # 4. Solve control problem (if controller available)
        if self.pos_controller is None:
            self.setup_cbf_qp_controller()

        if self.pos_controller is not None:
            try:
                control_ref = {
                    'state_machine': self.state_machine,
                    'u_ref': u_ref,
                    'goal': self.goal
                }
                u = self.pos_controller.solve_control_problem(
                    self.X, control_ref, self._get_nearest_obs(num_obs=5)
                )
                self.u_pos = u
            except Exception as e:
                print(f"Control problem failed: {e}")
                u = u_ref

        else:
            u = u_ref

        # 5. Step robot dynamics
        self.X = self.model.step(self.X, u)

        # 6. Clamp velocities
        v_mag = np.linalg.norm(self.X[3:6, 0])
        v_max = self.robot_spec.get('v_max', 3.0)
        if v_mag > v_max:
            self.X[3:6, 0] *= v_max / v_mag

        # 7. Record trajectory
        self.trajectory.append(self.X[0:3, 0].copy())

        # 8. Log to Rerun
        self._log_frame()

        if self.goal is None and self.state_machine != 'idle':
            return -1  # All waypoints reached

        return 0  # Normal execution

    def run(self, max_steps: int = 1000, verbose: bool = True):
        """
        Run simulation for specified number of steps.

        Args:
            max_steps: Maximum simulation steps
            verbose: Print progress
        """
        for step_idx in range(max_steps):
            try:
                result = self.step()
                if result == -1:
                    if verbose:
                        print(f"All waypoints reached at step {step_idx}")
                    break
                if verbose and step_idx % 50 == 0:
                    pos = self.X[0:3, 0]
                    print(f"Step {step_idx}: Robot at {pos}, State: {self.state_machine}")
            except Exception as e:
                print(f"Error at step {step_idx}: {e}")
                if self.raise_error:
                    raise
                break

        if verbose:
            print(f"Simulation completed. Total steps: {self.step_count}")

    def _has_stopped(self, tol: float = 0.05) -> bool:
        """Check if robot has stopped (velocity near zero)."""
        return np.linalg.norm(self.X[3:6, 0]) < tol

    def _compute_nominal_input(self) -> np.ndarray:
        """
        Compute nominal control input for tracking.

        Returns:
            Acceleration command [ax, ay, az]
        """
        if self.goal is None:
            return self.model.stop(self.X)

        if self.state_machine == 'rotate':
            # Rotation behavior (simplified for 3D)
            return self.model.stop(self.X)

        # Normal tracking
        return self.model.nominal_input(self.X, self.goal)

    def _get_nearest_obs(self, num_obs: int = 5) -> Optional[np.ndarray]:
        """
        Get nearest obstacles.

        Args:
            num_obs: Number of nearest obstacles to return

        Returns:
            Array of nearest obstacles or None if no obstacles
        """
        if self.obs.shape[0] == 0:
            return None

        # Compute distances
        distances = np.linalg.norm(self.obs[:, 0:3] - self.X[0:3, 0], axis=1)
        nearest_indices = np.argsort(distances)[:min(num_obs, len(self.obs))]

        return self.obs[nearest_indices]

    def _log_frame(self):
        """Log current state to Rerun."""
        robot_pos = self.X[0:3, 0]
        robot_vel = self.X[3:6, 0]

        # Log robot
        self.rerun_logger.log_robot_state(robot_pos, robot_vel, robot_radius=0.25)

        # Log obstacles
        if self.obs.shape[0] > 0:
            self.rerun_logger.log_obstacles(self.obs)

        # Log current goal
        if self.goal is not None:
            self.rerun_logger.log_goal(self.goal.reshape(-1))

        # Log DPCBF paraboloid for nearest obstacles
        nearest_obs = self._get_nearest_obs(num_obs=3)
        if nearest_obs is not None:
            for i, obs in enumerate(nearest_obs):
                try:
                    # Extract DPCBF parameters
                    obs_pos = obs[0:3]
                    obs_vel = obs[4:7] if len(obs) > 4 else np.zeros(3)
                    obs_radius = obs[3]

                    # Compute relative position and velocity
                    p_rel = obs_pos - robot_pos
                    v_rel = obs_vel - robot_vel

                    p_rel_mag = np.linalg.norm(p_rel)
                    v_rel_mag = np.linalg.norm(v_rel)

                    if p_rel_mag < 1e-3 or v_rel_mag < 1e-3:
                        continue

                    # Combine safety radius
                    ego_dim = (obs_radius + self.robot_spec['radius']) * 1.05
                    d_safe = max(p_rel_mag**2 - ego_dim**2, 1e-6)

                    # Compute DPCBF parameters
                    k_lambda = self.model.k_lambda * np.sqrt(1.05**2 - 1) / ego_dim
                    k_mu = self.model.k_mu * np.sqrt(1.05**2 - 1) / ego_dim

                    func_lambda = k_lambda * np.sqrt(d_safe) / (v_rel_mag + 1e-6)
                    func_mu = k_mu * np.sqrt(d_safe)

                    self.rerun_logger.log_dpcbf_paraboloid(
                        robot_pos=robot_pos,
                        robot_vel=robot_vel,
                        obstacle=obs,
                        func_lambda=func_lambda,
                        func_mu=func_mu,
                        robot_radius=self.robot_spec['radius'],
                        color=(150, 150, 255),
                        alpha=80,
                        mesh_density=20
                    )
                except Exception as e:
                    pass  # Skip paraboloid if computation fails

        # Log trajectory (sparse update)
        if self.step_count % 10 == 0 and len(self.trajectory) > 0:
            self.rerun_logger.log_trajectory(self.trajectory[-100:])

        # Log waypoints (sparse update)
        if self.step_count % 100 == 0 and self.waypoints is not None:
            self.rerun_logger.log_waypoints(self.waypoints)
