from safe_control.robots.double_integrator3D import DoubleIntegrator3D
import numpy as np
import casadi as ca

"""
Created on Apr 06, 2026
@author: Kane Weng

@description: 
Overrides the continuous and discrete-time CBF functions for the 3D Dynamic Parabolic CBF (DPCBF).
Adapted from 2D Kinematic Bicycle to 3D Double Integrator.
"""

class DoubleIntegrator3D_DPCBF(DoubleIntegrator3D):
    def __init__(self, dt, robot_spec, k_lambda=0.1, k_mu=0.5):
        super().__init__(dt, robot_spec)
        self.k_lambda = k_lambda
        self.k_mu = k_mu

    def _compute_h(self, X, obs, robot_radius, s):
        """
        Helper function to compute the 3D DPCBF value using Numpy
        """
        p = X[0:3, 0]
        v = X[3:6, 0]

        # Check if obstacles have velocity components (static or moving)
        obs_p = obs[0:3]
        if obs.shape[0] > 4:
            obs_v = obs[4:7]
        else:
            obs_v = np.zeros(3)

        # Combine safety radius (= r_obs + r_rob) with a safe margin (s)
        ego_dim = (obs[3] + robot_radius) * s

        # Compute relative position and velocity
        p_rel = obs_p - p
        v_rel = obs_v - v

        # Compute norms
        p_rel_mag = np.linalg.norm(p_rel)
        v_rel_mag = np.linalg.norm(v_rel)

        # Rotation angle and transformation
        yaw = np.arctan2(p_rel[1], p_rel[0])
        pitch = np.arctan2(p_rel[2], np.sqrt(p_rel[0]**2 + p_rel[1]**2))

        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        R = np.array([
            [cp * cy,  cp * sy,  sp],
            [-sy,      cy,       0],
            [-sp * cy, -sp * sy, cp]
        ])

        # Transform v_rel into the new coordinate frame
        v_rel_new = R @ v_rel
        v_rel_new_x = v_rel_new[0]
        v_rel_new_y = v_rel_new[1]
        v_rel_new_z = v_rel_new[2]

        # Compute clearance safely
        eps = 1e-6
        d_safe = np.maximum(p_rel_mag**2 - ego_dim**2, eps)

        # DPCBF functions
        func_lamda = self.k_lambda * np.sqrt(d_safe) / (v_rel_mag + eps) * np.sqrt(s**2 - 1)/ego_dim # using adaptive parameter
        func_mu = self.k_mu * np.sqrt(d_safe) * np.sqrt(s**2 - 1)/ego_dim 

        # Barrier function h(x)
        h = v_rel_new_x + func_lamda * (v_rel_new_y**2 + v_rel_new_z**2) + func_mu

        return h


    def agent_barrier(self, X, obs, robot_radius, s=1.05):
        """
        '''Continuous Time DPCBF'''
        Compute a Dynamic Parabolic Control Barrier Function for the Kinematic Bicycle2D.
        The barrier's relative degree is "1"
            h_dot = ∂h/∂x ⋅ f(x) + ∂h/∂x ⋅ g(x) ⋅ u
        Define h from the collision cone idea:
            p_rel = [obs_x - x, obs_y - y]
            v_rel = [obs_x_dot-v_cos(theta), obs_y_dot-v_sin(theta)]
            dist = ||p_rel||
            R = robot_radius + obs_r
        """
        h = self._compute_h(X, obs, robot_radius, s)

        # Compute dh_dx for DPCBF using Central Finite Difference
        dh_dx = np.zeros((1, 6))
        eps = 1e-5

        for i in range(6):
            X_up = X.copy()
            X_up[i, 0] += eps
            h_up = self._compute_h(X_up, obs, robot_radius, s)

            X_dn = X.copy()
            X_dn[i, 0] -= eps
            h_dn = self._compute_h(X_dn, obs, robot_radius, s)

            dh_dx[0, i] = (h_up - h_dn) / (2 * eps)

        return h, dh_dx

    def agent_barrier_dt(self, x_k, u_k, obs, robot_radius, s=1.05):
        '''Discrete Time DPCBF'''
        # Dynamics equations for the next states
        x_k1 = self.step(x_k, u_k)

        def h(x, obs, robot_radius, s=1.05):
            p = x[0:3, 0]
            v = x[3:6, 0]

            # Check if obstacles have velocity components (static or moving)
            obs_p = obs[0:3]
            if obs.shape[0] > 4:
                obs_v = obs[4:7]
            else:
                obs_v = ca.DM.zeros(3)

            # Combine radius R
            ego_dim = (obs[3] + robot_radius) * s

            # Compute relative position and velocity
            p_rel = obs_p - p
            v_rel = obs_v - v

            # Compute the rotation angle
            yaw = ca.atan2(p_rel[1], p_rel[0])
            pitch = ca.atan2(p_rel[2], ca.sqrt(p_rel[0]**2 + p_rel[1]**2))

            # Rotation matrix for transforming to the new coordinate frame:
            cy, sy = ca.cos(yaw), ca.sin(yaw)
            cp, sp = ca.cos(pitch), ca.sin(pitch)

            R = ca.vertcat(
                ca.horzcat(cp * cy, cp * sy, sp),
                ca.horzcat(-sy,     cy,      0),
                ca.horzcat(-sp * cy, -sp * sy, cp)
            )

            # Transform v_rel into the new coordinate frame
            v_rel_new = ca.mtimes(R, v_rel)

            p_rel_mag = ca.norm_2(p_rel)
            v_rel_mag = ca.norm_2(v_rel)

            eps = 1e-6
            d_safe = ca.fmax(p_rel_mag**2 - ego_dim**2, eps)

            k_lamda, k_mu = 0.1 * ca.sqrt(s**2 - 1)/ego_dim, 0.5 * ca.sqrt(s**2 - 1)/ego_dim

            lamda = k_lamda * ca.sqrt(d_safe) / (v_rel_mag + eps) # prevent 0 division
            mu = k_mu * ca.sqrt(d_safe)

            # Compute h
            h = v_rel_new[0] + lamda * (v_rel_new[1]**2 + v_rel_new[2]**2) + mu

            return h

        h_k1 = h(x_k1, obs, robot_radius, s)
        h_k = h(x_k, obs, robot_radius, s)

        d_h = h_k1 - h_k

        return h_k, d_h
    
    def draw_collision_parabola(self, X, obs_list, ax):
        '''
        Render the 2D slice of the 3D collision paraboloid.
        '''
        import matplotlib.pyplot as plt

        # Initialize lists to track matplotlib patches safely
        if not hasattr(self, 'collision_parabola_patches'):
            self.collision_parabola_patches = []
        if not hasattr(self, 'rel_vel_patches'):
            self.rel_vel_patches = []

        # Clear previous frame's drawings
        for line in self.collision_parabola_patches:
            try: line.remove()
            except: pass
        self.collision_parabola_patches.clear()

        for arrow in self.rel_vel_patches:
            try: arrow.remove()
            except: pass
        self.rel_vel_patches.clear()

        # 1. Extract 3D states, but we will plot from the 2D projection
        robot_pos_3d = X[0:3, 0]
        robot_vel_3d = X[3:6, 0]
        robot_pos_2d = robot_pos_3d[0:2] # For matplotlib

        # 2. Sort obstacles by true 3D distance
        obstacles_with_dist = []
        for obs in obs_list:
            obs_pos_3d = obs[0:3]
            distance = np.linalg.norm(obs_pos_3d - robot_pos_3d)
            obstacles_with_dist.append((distance, obs))

        obstacles_with_dist.sort(key=lambda item: item[0])
        num_to_plot = min(20, len(obstacles_with_dist))
        closest_obs_list = [item[1] for item in obstacles_with_dist[:num_to_plot]]
        
        if num_to_plot > 0:
            colors = plt.get_cmap('viridis')(np.linspace(0, 1, num_to_plot))
        else:
            colors = []

        s = 1.05 # Safety margin beta
        
        for i, obs in enumerate(closest_obs_list):
            obs = np.array(obs).flatten()
            obs_pos_3d = obs[0:3]
            obs_radius = obs[3]
            if len(obs) > 4:
                obs_vel_3d = obs[4:7]
            else:
                obs_vel_3d = np.zeros(3)

            # Combine radius
            ego_dim = (obs_radius + self.robot_spec['radius']) * s 

            # 3. Calculate True 3D Relative Vectors
            p_rel = obs_pos_3d - robot_pos_3d
            v_rel = obs_vel_3d - robot_vel_3d

            p_rel_mag = np.linalg.norm(p_rel)
            v_rel_mag = np.linalg.norm(v_rel)

            eps = 1e-6
            d_safe = np.maximum(p_rel_mag**2 - ego_dim**2, eps)

            # 4. Compute 3D DPCBF Parameters exactly as the QP sees them
            k_lambda = self.k_lambda * np.sqrt(s**2 - 1) / ego_dim
            k_mu = self.k_mu * np.sqrt(s**2 - 1) / ego_dim

            func_lambda = k_lambda * np.sqrt(d_safe) / (v_rel_mag + eps)
            func_mu = k_mu * np.sqrt(d_safe)

            # 5. Extract 2D components for Drawing on XY Plane
            p_rel_2d = p_rel[0:2]
            v_rel_2d = v_rel[0:2]
            
            # Avoid plotting errors if directly overhead (unlikely, but safe)
            if np.linalg.norm(p_rel_2d) < 1e-3:
                continue

            # Find 2D yaw angle for the XY slice
            rot_angle = np.arctan2(p_rel_2d[1], p_rel_2d[0])
            R_2d = np.array([
                [np.cos(rot_angle),  np.sin(rot_angle)],
                [-np.sin(rot_angle), np.cos(rot_angle)]
            ])

            # Generate the Parabola Curve
            L = 1.5
            y_disp = np.linspace(-L, L, 100)
            x_disp = (-func_lambda * (y_disp**2) - func_mu)

            # Rotate curve back to World Frame and shift to Robot Position
            pts_world = robot_pos_2d.reshape(2,1) + R_2d.T @ np.vstack([x_disp, y_disp])
            
            line, = ax.plot(pts_world[0,:], pts_world[1, :],
                            color=colors[i], linestyle='-', linewidth=2.0,
                            label=f"Quadratic Obs {i}")
            self.collision_parabola_patches.append(line)

            # Plot the Relative Velocity Vector
            offset_angle = 0.02 * (i - (len(obs_list)//2))
            R_offset = np.array([
                [np.cos(offset_angle), -np.sin(offset_angle)],
                [np.sin(offset_angle),  np.cos(offset_angle)]
            ])
            v_rel_offset = R_offset @ v_rel_2d
        
            arrow = ax.arrow(float(robot_pos_2d[0]), float(robot_pos_2d[1]),
                             float(v_rel_offset[0]), float(v_rel_offset[1]),
                             color=colors[i], width=0.02, alpha=1.0, zorder=10)
            self.rel_vel_patches.append(arrow)
