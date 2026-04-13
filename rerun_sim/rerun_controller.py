"""
Created on Apr 09
@author: Kane Weng

@description: 
This code implements a simulation controller (RerunControllerDyn) for 3D robot navigation using Control Barrier Functions (CBF) and Quadratic Programming (QP).
It is specifically designed for a 3D point-mass model (Double Integrator) and implements Dynamic Paraboloid CBF (DPCBF) for advanced 3D dynamic obstacle avoidance.
The controller features a robust state machine for sequential waypoint tracking, active collision detection, and entirely bypasses traditional Matplotlib rendering to stream high-performance, real-time 3D visualizations directly to the Rerun SDK.

@required-scripts: safe_control/rerun_sim/rerun_logger.py, safe_control/robots/robot.py, safe_control/position_control/cbf_qp.py
"""

import numpy as np
from typing import Optional, List, Tuple
from safe_control.rerun_sim.rerun_logger import RerunLogger3D
from safe_control.utils.headless_plot import NullAxes, NullFigure
from safe_control.robots.robot import BaseRobot 

class RerunControllerDyn:
    """
    Main simulation controller for managing robot dynamics, obstacle avoidance, 
    and waypoint navigation while streaming telemetry and visualizations to Rerun.
    """
    # ========================================================================= #
    # INITIALIZATION & CONFIGURATION
    # ========================================================================= #

    def __init__(self,
                 X0: np.ndarray,
                 robot_spec: dict,
                 dt: float = 0.05,
                 controller_type: Optional[dict] = None,
                 enable_rotation: bool = True,
                 raise_error: bool = False):
        
        self.robot_spec = robot_spec
        self.pos_controller_type = controller_type.get('pos', 'cbf_qp')  
        self.att_controller_type = controller_type.get('att', 'velocity_tracking_yaw')  
        self.dt = dt
        self.enable_rotation = enable_rotation
        self.raise_error = raise_error

        # State machine initialization
        self.state_machine = 'idle'     # 'idle', 'track', 'stop', 'rotate', 'crashed'
        self.reached_threshold = robot_spec.get('reached_threshold', 0.3)
        self.current_goal_index = 0

        # Initialize BaseRobot wrapper with dummy axes for headless operation
        fig = NullFigure()
        ax = NullAxes(fig)
        self.robot = BaseRobot(X0, robot_spec, dt, ax)
        
        # Internal control states
        self.yaw = 0.0
        self.u_att = np.zeros((1, 1))
        self.u_pos = np.zeros((3, 1))

        # Environment data structures
        self.waypoints = None
        self.goal = None
        self.obs = np.array([]).reshape(0, 7) # [x, y, z, radius, vx, vy, vz]
        self.unknown_obs = np.array([])

        # Initialize the custom Rerun logging module
        self.rerun_logger = RerunLogger3D(app_name="SafeControl3D_DPCBF")
        self.trajectory = []
        self.step_count = 0
        self.pos_controller = None

    def setup_cbf_qp_controller(self):
        """Initializes the Control Barrier Function Quadratic Program solver."""
        if self.pos_controller_type == 'cbf_qp':
            from safe_control.position_control.cbf_qp import CBFQP
            self.pos_controller = CBFQP(self.robot, self.robot_spec, num_obs=10)

    def set_waypoints(self, waypoints: np.ndarray):
        """Loads a sequence of 3D coordinates for the robot to follow."""
        self.waypoints = np.array(waypoints, dtype=np.float64)
        self.current_goal_index = 0
        self.state_machine = 'idle'

    def set_obstacles(self, obstacles: np.ndarray):
        """Loads dynamic/static obstacles into the environment."""
        self.obs = np.array(obstacles, dtype=np.float64).reshape(-1, 7)

    # ========================================================================= #
    # MAIN EXECUTION LOOP
    # ========================================================================= #

    def run(self, max_steps: int = 1000, verbose: bool = True):
        """Executes the simulation loop for a maximum number of steps."""
        for step_idx in range(max_steps):
            try:
                result = self.step()
                
                # Handle Crash
                if result == -2:
                    if verbose:
                        print(f"CRASH DETECTED at step {step_idx}! Simulation stopped.")
                    break
                # Handle Success
                if result == -1:
                    if verbose:
                        print(f"All waypoints reached at step {step_idx}")
                    break
                
                if verbose and step_idx % 50 == 0:
                    pos = self.robot.X[0:3, 0]
                    print(f"Step {step_idx}: Robot at {pos}, State: {self.state_machine}")
            except Exception as e:
                print(f"Error at step {step_idx}: {e}")
                if self.raise_error:
                    raise
                break

        if verbose:
            print(f"Simulation completed. Total steps: {self.step_count}")

    def step(self):
        """
        Advances state machine, computes control inputs, applies safety filters, 
        updates kinematics, and logs to Rerun.
        """
        self.step_count += 1
        self.rerun_logger.set_time_step(self.step_count)

        # Evaluate State Machine logic
        if self.state_machine == 'stop':
            if self._has_stopped():
                if self.enable_rotation:
                    self.state_machine = 'rotate'
                else:
                    self.state_machine = 'track'
                self.goal = self.update_goal()
        else:
            self.goal = self.update_goal()

        # Advance environment
        self.step_dyn_obs()

        # Collision logic
        if self._check_collisions():
            self.state_machine = 'crashed'
            self._log_frame() 
            return -2

        if self.state_machine == 'idle':
            self.state_machine = 'track'
            self.goal = self.update_goal()

        # Get unconstrained control input
        u_ref = self._compute_nominal_input()

        # Initialize and apply Safety Filter (CBF-QP)
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
                    self.robot.X, control_ref, self._get_nearest_obs(num_obs=5)
                )
                self.u_pos = u
            except Exception as e:
                print(f"Control problem failed: {e}")
                u = u_ref 
        else:
            u = u_ref

        # Apply physical control
        self.robot.step(u)

        # Post-step Safety: Clamp velocities
        v_mag = np.linalg.norm(self.robot.X[3:6, 0])
        v_max = self.robot_spec.get('v_max', 3.0)
        if v_mag > v_max:
            self.robot.X[3:6, 0] *= v_max / v_mag

        # Record and Log Frame
        self.trajectory.append(self.robot.X[0:3, 0].copy())
        self._log_frame()

        if self.goal is None and self.state_machine != 'idle':
            return -1

        return 0

    # ========================================================================= #
    # STATE & ENVIRONMENT HELPERS
    # ========================================================================= #

    def update_goal(self) -> Optional[np.ndarray]:
        """Checks distance to waypoint and advances if within threshold."""
        if self.waypoints is None or len(self.waypoints) == 0:
            return None

        if self.current_goal_index >= len(self.waypoints):
            return None

        current_goal = self.waypoints[self.current_goal_index]
        pos = self.robot.X[0:3, 0]
        distance_to_goal = np.linalg.norm(current_goal - pos)

        if distance_to_goal < self.reached_threshold:
            self.current_goal_index += 1
            if self.current_goal_index >= len(self.waypoints):
                return None
            return self.waypoints[self.current_goal_index].reshape(-1, 1)

        return current_goal.reshape(-1, 1)

    def step_dyn_obs(self):
        """Applies simple Euler integration to update moving obstacle positions."""
        if self.obs.shape[0] > 0 and self.obs.shape[1] >= 7:
            self.obs[:, 0] += self.obs[:, 4] * self.dt 
            self.obs[:, 1] += self.obs[:, 5] * self.dt 
            self.obs[:, 2] += self.obs[:, 6] * self.dt 

    def _get_nearest_obs(self, num_obs: int = 5) -> Optional[np.ndarray]:
        """Finds and returns the 'n' closest obstacles to the robot."""
        if self.obs.shape[0] == 0:
            return None

        distances = np.linalg.norm(self.obs[:, 0:3] - self.robot.X[0:3, 0], axis=1)
        nearest_indices = np.argsort(distances)[:min(num_obs, len(self.obs))]

        return self.obs[nearest_indices]

    def _check_collisions(self) -> bool:
        """Returns True if the robot's sphere overlaps with any obstacle's sphere."""
        if self.obs.shape[0] == 0:
            return False
            
        robot_pos = self.robot.X[0:3, 0]
        robot_radius = self.robot_spec.get('radius', 0.25)
        
        distances = np.linalg.norm(self.obs[:, 0:3] - robot_pos, axis=1)
        collision_mask = distances <= (robot_radius + self.obs[:, 3])
        
        return np.any(collision_mask)

    # ========================================================================= #
    # CONTROL LOGIC HELPERS
    # ========================================================================= #

    def _has_stopped(self, tol: float = 0.05) -> bool:
        """Returns True if the robot's speed is below the tolerance threshold."""
        return np.linalg.norm(self.robot.X[3:6, 0]) < tol

    def _compute_nominal_input(self) -> np.ndarray:
        """Calculates standard control input before safety filtering."""
        if self.goal is None or self.state_machine == 'rotate':
            return self.robot.stop()

        return self.robot.nominal_input(self.goal)

    # ========================================================================= #
    # VISUALIZATION & LOGGING
    # ========================================================================= #

    def _log_frame(self):
        """Dispatches all data for the current time step to the Rerun viewer."""
        robot_pos = self.robot.X[0:3, 0]
        robot_vel = self.robot.X[3:6, 0]

        # Log basic entities
        self.rerun_logger.log_robot_state(robot_pos, robot_vel, robot_radius=0.25)

        if self.obs.shape[0] > 0:
            self.rerun_logger.log_obstacles(self.obs)

        if self.goal is not None:
            self.rerun_logger.log_goal(self.goal.reshape(-1))

        # Log advanced DPCBF visualizer for nearby obstacles
        nearest_obs = self._get_nearest_obs(num_obs=5)
        if nearest_obs is not None:
            for i, obs in enumerate(nearest_obs):
                try:
                    obs_pos = obs[0:3]
                    obs_vel = obs[4:7] if len(obs) > 4 else np.zeros(3)
                    obs_radius = obs[3]

                    p_rel = obs_pos - robot_pos
                    v_rel = obs_vel - robot_vel

                    p_rel_mag = np.linalg.norm(p_rel)
                    v_rel_mag = np.linalg.norm(v_rel)

                    if p_rel_mag < 1e-3 or v_rel_mag < 1e-3:
                        continue
                    
                    beta = 1.05     # safety margin
                    ego_dim = (obs_radius + self.robot_spec['radius']) * beta
                    eps = 1e-6
                    d_safe = max(p_rel_mag**2 - ego_dim**2, eps)
                    
                    # DPCBF functions
                    k_lambda = self.robot.robot.k_lambda * np.sqrt(beta**2 - 1) / ego_dim
                    k_mu = self.robot.robot.k_mu * np.sqrt(beta**2 - 1) / ego_dim
                    func_lambda = k_lambda * np.sqrt(d_safe) / (v_rel_mag + eps)
                    func_mu = k_mu * np.sqrt(d_safe)

                    self.rerun_logger.log_dpcbf_paraboloid(
                        robot_pos=robot_pos,
                        robot_vel=robot_vel,
                        obstacle=obs,
                        func_lambda=func_lambda,
                        func_mu=func_mu,
                        robot_radius=self.robot_spec['radius'],
                        obs_id=i,
                        alpha=80,
                        mesh_density=20
                    )
                except Exception as e:
                    print(f"Paraboloid render failed for obs {i}: {e}")

        # Throttled logging for performance
        if self.step_count % 10 == 0 and len(self.trajectory) > 0:
            self.rerun_logger.log_trajectory(self.trajectory[-100:])

        if self.step_count % 100 == 0 and self.waypoints is not None:
            self.rerun_logger.log_waypoints(self.waypoints)