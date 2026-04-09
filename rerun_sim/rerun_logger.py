"""
Rerun Logger Module

Handles all rr.log() calls for 3D visualization of robot dynamics, obstacles,
and DPCBF paraboloid surfaces.

Uses modern Rerun API (v0.15+) with rr.set_time() for timeline tracking.
"""

import numpy as np
import rerun as rr
from typing import Optional, List, Tuple


class RerunLogger3D:
    """Centralized logging handler for Rerun 3D visualization."""

    def __init__(self, app_name: str = "SafeControl3D", recording_id: Optional[str] = None):
        """
        Initialize Rerun logger.

        Args:
            app_name: Name of the Rerun application
            recording_id: Optional recording ID for persistence
        """
        self.app_name = app_name
        rr.init(self.app_name, spawn=True)

    def set_time_step(self, step: int):
        """Set the current simulation time step."""
        rr.set_time("step", sequence=step)

    def log_robot_state(self,
                       robot_pos: np.ndarray,
                       robot_vel: np.ndarray,
                       robot_radius: float = 0.25,
                       color: Tuple[int, int, int] = (100, 150, 200)):
        """
        Log robot position and velocity in 3D space.

        Args:
            robot_pos: 3D position [x, y, z]
            robot_vel: 3D velocity [vx, vy, vz]
            robot_radius: Sphere radius
            color: RGB color tuple
        """
        rr.log(
            "world/robot/position",
            rr.Points3D(
                positions=[robot_pos],
                radii=[robot_radius],
                colors=[color]
            ),
        )

        # Log velocity as arrow
        if np.linalg.norm(robot_vel) > 1e-3:
            rr.log(
                "world/robot/velocity",
                rr.Arrows3D(
                    origins=[robot_pos],
                    vectors=[robot_vel],
                    colors=[color],
                ),
            )

    def log_obstacles(self,
                     obstacles: np.ndarray,
                     color: Tuple[int, int, int] = (200, 100, 100)):
        """
        Log all dynamic obstacles as 3D spheres with velocities.

        Args:
            obstacles: Array of shape (N, 7) where each row is [ox, oy, oz, r, vx, vy, vz]
            color: RGB color tuple for obstacle visualization
        """
        if obstacles.shape[0] == 0:
            return

        positions = obstacles[:, 0:3]  # [x, y, z]
        radii = obstacles[:, 3:4].flatten()  # radius

        rr.log(
            "world/obstacles",
            rr.Points3D(
                positions=positions,
                radii=radii,
                colors=[color] * len(obstacles),
            ),
        )

        # Log velocities as arrows
        velocities = obstacles[:, 4:7]  # [vx, vy, vz]
        vel_magnitudes = np.linalg.norm(velocities, axis=1)

        # Only log arrows for obstacles with meaningful velocity
        if np.any(vel_magnitudes > 1e-3):
            non_static_mask = vel_magnitudes > 1e-3
            origins = positions[non_static_mask]
            directions = velocities[non_static_mask]

            rr.log(
                "world/obstacles/velocities",
                rr.Arrows3D(
                    origins=origins,
                    vectors=directions,
                    colors=[(255, 150, 100)] * np.sum(non_static_mask),
                ),
            )

    def log_waypoints(self,
                     waypoints: np.ndarray,
                     color: Tuple[int, int, int] = (50, 200, 50)):
        """
        Log 3D waypoint targets.

        Args:
            waypoints: Array of shape (M, 3) where each row is [x, y, z]
            color: RGB color tuple
        """
        if waypoints.shape[0] == 0:
            return

        rr.log(
            "world/waypoints",
            rr.Points3D(
                positions=waypoints,
                radii=[0.3] * len(waypoints),
                colors=[color] * len(waypoints),
                labels=[f"WP{i}" for i in range(len(waypoints))],
            ),
        )

    def log_dpcbf_paraboloid(self,
                            robot_pos: np.ndarray,
                            robot_vel: np.ndarray,
                            obstacle: np.ndarray,
                            func_lambda: float,
                            func_mu: float,
                            robot_radius: float,
                            color: Tuple[int, int, int] = (150, 150, 255),
                            alpha: int = 100,
                            mesh_density: int = 25):
        """
        Log 3D DPCBF paraboloid surface as a point cloud.

        The paraboloid boundary surface is defined by:
            h(v_rel) = v_rel_x + lambda * (v_rel_y^2 + v_rel_z^2) + mu = 0

        We generate a mesh in relative velocity space, solve for the boundary,
        and display it in world coordinates.

        Args:
            robot_pos: Robot 3D position [x, y, z]
            robot_vel: Robot 3D velocity [vx, vy, vz]
            obstacle: Obstacle array [ox, oy, oz, r, vx, vy, vz, ...]
            func_lambda: DPCBF lambda parameter
            func_mu: DPCBF mu parameter
            robot_radius: Robot radius
            color: RGB color tuple
            alpha: Transparency (0-255)
            mesh_density: Density of surface mesh points
        """
        # Extract obstacle properties
        obs_pos = obstacle[0:3]
        obs_vel = obstacle[4:7] if len(obstacle) > 4 else np.zeros(3)

        # Compute relative position and velocity
        p_rel = obs_pos - robot_pos
        v_rel = obs_vel - robot_vel

        p_rel_mag = np.linalg.norm(p_rel)
        v_rel_mag = np.linalg.norm(v_rel)

        # Skip if too close or too far
        if p_rel_mag < 0.5 or p_rel_mag > 30.0:
            return

        # Compute rotation (yaw and pitch) to align with relative position
        yaw = np.arctan2(p_rel[1], p_rel[0])
        pitch = np.arctan2(p_rel[2], np.sqrt(p_rel[0]**2 + p_rel[1]**2))

        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)

        # Rotation matrix from relative frame to world frame
        R = np.array([
            [cp * cy,  cp * sy,  sp],
            [-sy,      cy,       0],
            [-sp * cy, -sp * sy, cp]
        ])

        # Generate mesh in relative velocity space (parabola surface)
        # Range: cover ±L in y and z dimensions
        L = 1.5
        mesh_size = mesh_density

        y_vals = np.linspace(-L, L, mesh_size)
        z_vals = np.linspace(-L, L, mesh_size)
        Y, Z = np.meshgrid(y_vals, z_vals)

        # From h = v_rel_x + lambda * (y^2 + z^2) + mu = 0
        # Solve for v_rel_x at the boundary
        eps = 1e-6
        X_bound = -(func_lambda * (Y**2 + Z**2) + func_mu)

        # Stack into 3D relative velocity points
        points_rel = np.stack([
            X_bound.flatten(),
            Y.flatten(),
            Z.flatten()
        ], axis=0)  # Shape: (3, mesh_size^2)

        # Transform to world frame (velocity field vectors in world coordinates)
        # These represent actual velocity vectors in world frame
        points_world = R @ points_rel

        # Center the parabola at the robot position (visualization reference)
        # The actual rendering shows relative velocity surfaces emanating from robot
        points_display = robot_pos.reshape(3, 1) + points_world

        # Create point cloud with multiple colors for depth perception
        colors = self._generate_gradient_colors(
            points_world.shape[1],
            base_color=color,
            alpha=alpha
        )

        rr.log(
            "world/dpcbf/paraboloid",
            rr.Points3D(
                positions=points_display.T,
                radii=[0.05] * points_display.shape[1],
                colors=colors,
            ),
        )

    def log_trajectory(self,
                      trajectory: List[np.ndarray],
                      color: Tuple[int, int, int] = (255, 200, 100)):
        """
        Log robot trajectory as a line or sparse points.

        Args:
            trajectory: List of 3D positions [x, y, z]
            color: RGB color tuple
        """
        if len(trajectory) < 2:
            return

        traj_array = np.array(trajectory)

        # Subsample for visualization if trajectory is very long
        if len(trajectory) > 500:
            indices = np.linspace(0, len(trajectory) - 1, 500, dtype=int)
            traj_array = traj_array[indices]

        rr.log(
            "world/trajectory",
            rr.Points3D(
                positions=traj_array,
                radii=[0.03] * len(traj_array),
                colors=[color] * len(traj_array),
            ),
        )

    def log_goal(self,
                goal_pos: np.ndarray,
                goal_radius: float = 0.5,
                color: Tuple[int, int, int] = (50, 255, 50)):
        """
        Log current goal/target position.

        Args:
            goal_pos: 3D goal position [x, y, z]
            goal_radius: Visualization radius
            color: RGB color tuple
        """
        rr.log(
            "world/current_goal",
            rr.Points3D(
                positions=[goal_pos],
                radii=[goal_radius],
                colors=[color],
            ),
        )

    @staticmethod
    def _generate_gradient_colors(n: int,
                                 base_color: Tuple[int, int, int],
                                 alpha: int = 255) -> List[Tuple[int, int, int, int]]:
        """
        Generate a gradient of colors from dark to light for depth perception.

        Returns list of RGBA tuples.
        """
        colors = []
        for i in range(n):
            factor = i / max(n - 1, 1)

            # Blend base_color towards white for gradient effect
            r = int(base_color[0] + (255 - base_color[0]) * factor * 0.3)
            g = int(base_color[1] + (255 - base_color[1]) * factor * 0.3)
            b = int(base_color[2] + (255 - base_color[2]) * factor * 0.3)

            colors.append((r, g, b, alpha))

        return colors
