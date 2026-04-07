import numpy as np
import casadi as ca

"""
Created on Apr 04, 2026
@author: Kane Weng

@description: 
3D Double Integrator model for CBF-QP and MPC-CBF (casadi) with separated position and attitude states
# CHANGED: Adapted from 2D to 3D for spatial DPCBF formulation.
"""


def angle_normalize(x):
    if isinstance(x, (np.ndarray, float, int)):
        # NumPy implementation
        return (((x + np.pi) % (2 * np.pi)) - np.pi)
    elif isinstance(x, (ca.SX, ca.MX, ca.DM)):
        # CasADi implementation
        return ca.fmod(x + ca.pi, 2 * ca.pi) - ca.pi
    else:
        raise TypeError(f"Unsupported input type: {type(x)}")


class DoubleIntegrator3D:

    def __init__(self, dt, robot_spec):
        '''
            X: [x, y, z, vx, vy, vz] 
            theta: [roll, pitch, yaw]
            U: [ax, ay, az]
            U_attitude: [wx, wy, wz]
            cbf: h(x) = ||x-x_obs||^2 - beta*d_min^2
            relative degree: 2
        '''
        self.dt = dt
        self.robot_spec = robot_spec
        
        self.robot_spec.setdefault('model', 'DoubleIntegrator3D')

        self.robot_spec.setdefault('a_max', 1.0)
        self.robot_spec.setdefault('v_max', 1.0)
        self.robot_spec.setdefault('ax_max', self.robot_spec['a_max'])
        self.robot_spec.setdefault('ay_max', self.robot_spec['a_max'])
        self.robot_spec.setdefault('az_max', self.robot_spec['a_max'])
        self.robot_spec.setdefault('w_max', 0.5)

    def f(self, X, casadi=False):
        if casadi:
            return ca.vertcat(
                X[3, 0], # vx
                X[4, 0], # vy
                X[5, 0], # vz
                0,
                0,
                0
            )
        else:
            return np.array([X[3, 0],
                             X[4, 0],
                             X[5, 0],
                             0,
                             0,
                             0]).reshape(-1, 1)

    def df_dx(self, X):
        return np.array([
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ])

    def g(self, X, casadi=False):
        if casadi:
            return ca.DM([
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ])
        else:
            return np.array([
                [0, 0, 0], 
                [0, 0, 0], 
                [0, 0, 0], 
                [1, 0, 0], 
                [0, 1, 0], 
                [0, 0, 1]
            ])

    def step(self, X, U):
        X = X + (self.f(X) + self.g(X) @ U) * self.dt
        
        v_max = self.robot_spec.get('v_max')
        if v_max is not None:
            if isinstance(X, (ca.SX, ca.MX, ca.DM)):
                # CasADi symbolic implementation
                vx, vy, vz = X[3, 0], X[4, 0], X[5, 0]
                v_mag = ca.sqrt(vx**2 + vy**2 + vz**2)
                scale = ca.if_else(v_mag > v_max, v_max / v_mag, 1.0)
                # Apply scaling. Note: In CasADi, we can't modify X in place effectively if it's symbolic, 
                # but we can return a new expression. 
                X_new_3 = X[3, 0] * scale
                X_new_4 = X[4, 0] * scale
                X_new_5 = X[5, 0] * scale
                
                # Reconstruct X. If X is SX/MX, we can't do slice assignment efficiently the same way as numpy sometimes.
                # But CasADi supports slice assignment.
                X[3, 0] = X_new_3
                X[4, 0] = X_new_4
                X[5, 0] = X_new_5
            else:
                # NumPy implementation
                vx, vy, vz = X[3, 0], X[4, 0], X[5, 0]
                v_mag = np.sqrt(vx**2 + vy**2 + vz**2)
                if v_mag > v_max:
                    scale = v_max / v_mag
                    X[3, 0] *= scale
                    X[4, 0] *= scale
                    X[5, 0] *= scale
                
        return X

    def step_rotate(self, theta, U_attitude):
        theta = angle_normalize(theta + U_attitude * self.dt)
        return theta

    def nominal_input(self, X, G, d_min=0.05, k_v=1.0, k_a=1.0):
        '''
        nominal input for CBF-QP (position control)
        '''
        k_v = self.robot_spec.get('nominal_k_v', k_v)
        k_a = self.robot_spec.get('nominal_k_a', k_a)
        G = np.copy(G.reshape(-1, 1))  # goal state

        # Pad 2D goal to 3D (To match tracking.py)
        if G.shape[0] < 3:
            G = np.vstack((G, np.zeros((3 - G.shape[0], 1))))
        
        v_max = self.robot_spec['v_max']  # Maximum velocity (x+y)
        a_max = self.robot_spec['a_max']  # Maximum acceleration

        pos_errors = G[0:3, 0] - X[0:3, 0]
        pos_errors = np.sign(pos_errors) * \
            np.maximum(np.abs(pos_errors) - d_min, 0.0)

        # Compute desired velocities for x and y
        v_des = k_v * pos_errors
        v_mag = np.linalg.norm(v_des)
        if v_mag > v_max:
            v_des = v_des * v_max / v_mag

        # Compute accelerations
        current_v = X[3:6, 0]
        a = k_a * (v_des - current_v)
        a_mag = np.linalg.norm(a)
        if a_mag > a_max:
            a = a * a_max / a_mag

        return a.reshape(-1, 1)

    def nominal_attitude_input(self, theta, theta_des, k_theta=1.0):
        '''
        nominal input for attitude control
        '''
        error_theta = angle_normalize(theta_des - theta)
        yaw_rate = k_theta * error_theta
        return yaw_rate.reshape(-1, 1)

    def stop(self, X, k_a=1.0):
        # Set desired velocity to zero
        k_a = self.robot_spec.get('nominal_k_a', k_a)
        vx_des, vy_des, vz_des = 0.0, 0.0, 0.0
        ax = k_a * (vx_des - X[3, 0])
        ay = k_a * (vy_des - X[4, 0])
        az = k_a * (vz_des - X[5, 0])
        return np.array([ax, ay, az]).reshape(-1, 1)

    def has_stopped(self, X, tol=0.05):
        return np.linalg.norm(X[3:6, 0]) < tol

    def rotate_to(self, theta, theta_des, k_omega=2.0):
        error_theta = angle_normalize(theta_des - theta)
        yaw_rate = k_omega * error_theta
        yaw_rate = np.clip(yaw_rate, -self.robot_spec['w_max'], self.robot_spec['w_max'])
        return np.array([yaw_rate]).reshape(-1, 1)

    def agent_barrier(self, X, obs, robot_radius, beta=1.01):
        '''Continuous Time High Order CBF'''
        h = 0
        h_dot = 0
        dh_dot_dx = 0
        if obs[-1] == 0:
            obsX = obs[0:3].reshape(-1, 1)
            d_min = obs[3] + robot_radius  # obs radius + robot radius

            h = np.linalg.norm(X[0:3] - obsX[0:3])**2 - beta*d_min**2
            # Lgh is zero => relative degree is 2, f(x)[0:3] actually equals to X[3:6]
            h_dot = 2 * (X[0:3] - obsX[0:3]).T @ (self.f(X)[0:3])

            # these two options are the same
            # df_dx = self.df_dx(X)
            # dh_dot_dx = np.append( ( 2 * self.f(X)[0:3] ).T, np.array([[0,0]]), axis = 1 ) + 2 * ( X[0:3] - obsX[0:3] ).T @ df_dx[0:3,:]
            dh_dot_dx = np.append(2 * X[3:6].T, 2 * (X[0:3] - obsX[0:3]).T, axis=1)
        elif obs[-1] == 1:
            # Complex math skipped, not strictly required for DPCBF initial testing
            raise NotImplementedError("3D Superellipsoid analytical barrier not yet implemented. Use spheres (obs[-1] == 0).")

        return h, h_dot, dh_dot_dx

    def agent_barrier_dt(self, x_k, u_k, obs, robot_radius, beta=1.01):
        '''Discrete Time High Order CBF'''
        # Dynamics equations for the next states
        x_k1 = self.step(x_k, u_k)
        x_k2 = self.step(x_k1, u_k)

        def _h_sphere(x, obs, robot_radius, beta):
            '''Computes CBF h(x) = ||x-x_obs||^2 - beta*d_min^2'''
            x_obs = obs[0]
            y_obs = obs[1]
            z_obs = obs[2]
            r_obs = obs[3] 
            d_min = robot_radius + r_obs

            h = (x[0, 0] - x_obs)**2 + (x[1, 0] - y_obs)**2 + (x[2, 0] - z_obs)**2 - beta*d_min**2
            return h
        
        def _h_superellipsoid(x, obs, robot_radius, beta):
            # Complex math skipped, not strictly required for DPCBF initial testing
            raise NotImplementedError("3D Superellipsoid discrete barrier not yet implemented.")
        
        def h(x, obs, robot_radius, beta=1.01):
            
            is_sphere = (obs[-1] == 0)
            
            return ca.if_else(is_sphere,
                                _h_sphere(x, obs, robot_radius, beta),
                                _h_superellipsoid(x, obs, robot_radius, beta))

        h_k2 = h(x_k2, obs, robot_radius, beta)
        h_k1 = h(x_k1, obs, robot_radius, beta)
        h_k = h(x_k, obs, robot_radius, beta)

        d_h = h_k1 - h_k
        dd_h = h_k2 - 2 * h_k1 + h_k
        # hocbf_2nd_order = h_ddot + (gamma1 + gamma2) * h_dot + (gamma1 * gamma2) * h_k

        return h_k, d_h, dd_h
