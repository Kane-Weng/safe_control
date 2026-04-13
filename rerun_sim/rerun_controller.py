import numpy as np
from typing import Optional, List, Tuple
from safe_control.rerun_sim.rerun_logger import RerunLogger3D
from safe_control.utils.headless_plot import NullAxes, NullFigure

# IMPORTANT: Check this import path matches your actual BaseRobot location
from safe_control.robots.robot import BaseRobot 

class RerunControllerDyn:
    def __init__(self,
                 X0: np.ndarray,
                 robot_spec: dict,
                 dt: float = 0.05,
                 controller_type: Optional[dict] = None,
                 enable_rotation: bool = True,
                 raise_error: bool = False):
        
        self.robot_spec = robot_spec
        self.dt = dt
        self.enable_rotation = enable_rotation
        self.raise_error = raise_error

        if controller_type is None:
            controller_type = {'pos': 'cbf_qp'}
        self.pos_controller_type = controller_type.get('pos', 'cbf_qp')
        self.att_controller_type = controller_type.get('att', 'velocity_tracking_yaw')

        self.state_machine = 'idle'
        self.reached_threshold = robot_spec.get('reached_threshold', 0.3)
        self.current_goal_index = 0

        # --- THE REFACTOR ---
        # Initialize BaseRobot wrapper with dummy axes
        fig = NullFigure()
        ax = NullAxes(fig)
        self.robot = BaseRobot(X0, robot_spec, dt, ax)
        
        self.yaw = 0.0
        self.u_att = np.zeros((1, 1))
        self.u_pos = np.zeros((3, 1))

        self.waypoints = None
        self.goal = None
        self.obs = np.array([]).reshape(0, 7)
        self.unknown_obs = np.array([])

        self.rerun_logger = RerunLogger3D(app_name="SafeControl3D_DPCBF")
        self.trajectory = []
        self.step_count = 0
        self.pos_controller = None

    def setup_cbf_qp_controller(self):
        if self.pos_controller_type == 'cbf_qp':
            from safe_control.position_control.cbf_qp import CBFQP
            # Pass the BaseRobot wrapper into the solver
            self.pos_controller = CBFQP(self.robot, self.robot_spec, num_obs=10)

    def set_waypoints(self, waypoints: np.ndarray):
        self.waypoints = np.array(waypoints, dtype=np.float64)
        self.current_goal_index = 0
        self.state_machine = 'idle'

    def set_obstacles(self, obstacles: np.ndarray):
        self.obs = np.array(obstacles, dtype=np.float64).reshape(-1, 7)

    def update_goal(self) -> Optional[np.ndarray]:
        if self.waypoints is None or len(self.waypoints) == 0:
            return None

        if self.current_goal_index >= len(self.waypoints):
            return None

        current_goal = self.waypoints[self.current_goal_index]
        
        # Access state through the wrapper
        pos = self.robot.X[0:3, 0]
        distance_to_goal = np.linalg.norm(current_goal - pos)

        if distance_to_goal < self.reached_threshold:
            self.current_goal_index += 1
            if self.current_goal_index >= len(self.waypoints):
                return None
            return self.waypoints[self.current_goal_index].reshape(-1, 1)

        return current_goal.reshape(-1, 1)

    def step_dyn_obs(self):
        if self.obs.shape[0] > 0 and self.obs.shape[1] >= 7:
            self.obs[:, 0] += self.obs[:, 4] * self.dt
            self.obs[:, 1] += self.obs[:, 5] * self.dt
            self.obs[:, 2] += self.obs[:, 6] * self.dt

    def step(self):
        self.step_count += 1
        self.rerun_logger.set_time_step(self.step_count)

        if self.state_machine == 'stop':
            if self._has_stopped():
                if self.enable_rotation:
                    self.state_machine = 'rotate'
                else:
                    self.state_machine = 'track'
                self.goal = self.update_goal()
        else:
            self.goal = self.update_goal()

        self.step_dyn_obs()

        if self.state_machine == 'idle':
            self.state_machine = 'track'
            self.goal = self.update_goal()

        u_ref = self._compute_nominal_input()

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

        # Step the wrapper
        self.robot.step(u)

        # Clamp velocities directly on the wrapper's state
        v_mag = np.linalg.norm(self.robot.X[3:6, 0])
        v_max = self.robot_spec.get('v_max', 3.0)
        if v_mag > v_max:
            self.robot.X[3:6, 0] *= v_max / v_mag

        self.trajectory.append(self.robot.X[0:3, 0].copy())
        self._log_frame()

        if self.goal is None and self.state_machine != 'idle':
            return -1

        return 0

    def run(self, max_steps: int = 1000, verbose: bool = True):
        for step_idx in range(max_steps):
            try:
                result = self.step()
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

    def _has_stopped(self, tol: float = 0.05) -> bool:
        return np.linalg.norm(self.robot.X[3:6, 0]) < tol

    def _compute_nominal_input(self) -> np.ndarray:
        if self.goal is None:
            return self.robot.stop()

        if self.state_machine == 'rotate':
            return self.robot.stop()

        # We pass self.goal (BaseRobot handles passing state to the math model)
        return self.robot.nominal_input(self.goal)

    def _get_nearest_obs(self, num_obs: int = 5) -> Optional[np.ndarray]:
        if self.obs.shape[0] == 0:
            return None

        distances = np.linalg.norm(self.obs[:, 0:3] - self.robot.X[0:3, 0], axis=1)
        nearest_indices = np.argsort(distances)[:min(num_obs, len(self.obs))]

        return self.obs[nearest_indices]

    def _log_frame(self):
        robot_pos = self.robot.X[0:3, 0]
        robot_vel = self.robot.X[3:6, 0]

        self.rerun_logger.log_robot_state(robot_pos, robot_vel, robot_radius=0.25)

        if self.obs.shape[0] > 0:
            self.rerun_logger.log_obstacles(self.obs)

        if self.goal is not None:
            self.rerun_logger.log_goal(self.goal.reshape(-1))

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

                    ego_dim = (obs_radius + self.robot_spec['radius']) * 1.05
                    d_safe = max(p_rel_mag**2 - ego_dim**2, 1e-6)

                    # Extract parameters from the raw math model nested inside BaseRobot
                    k_lambda = self.robot.robot.k_lambda * np.sqrt(1.05**2 - 1) / ego_dim
                    k_mu = self.robot.robot.k_mu * np.sqrt(1.05**2 - 1) / ego_dim

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
                        obs_id=i,
                        alpha=80,
                        mesh_density=20
                    )
                except Exception as e:
                    print(f"Paraboloid render failed for obs {i}: {e}")

        if self.step_count % 10 == 0 and len(self.trajectory) > 0:
            self.rerun_logger.log_trajectory(self.trajectory[-100:])

        if self.step_count % 100 == 0 and self.waypoints is not None:
            self.rerun_logger.log_waypoints(self.waypoints)