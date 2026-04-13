import rerun as rr
import numpy as np
import time

# 1. Initialize Rerun and spawn the Rust-based 3D viewer alongside your script
rr.init("DPCBF_3D_Sim", spawn=True)

# Mock robot and obstacle data for demonstration
robot_radius = 0.25
obs_radius = 0.5

# 2. The Simulation Loop
for step in range(200):
    # Tell Rerun what "time" it is. Everything logged after this is attached to this step.
    rr.set_time_sequence("step", step)
    
    # --- MOCK DYNAMICS ---
    robot_pos = np.array([step * 0.1, 7.5, 2.0]) # Moving forward
    obs_pos = np.array([
        [8.0, 9.0 - (step*0.02), 2.0], 
        [10.0, 4.0 + (step*0.02), 2.0]
    ])
    obs_vel = np.array([[0.0, -0.5, 0.0], [0.0, 0.5, 0.0]])
    
    # --- RERUN LOGGING ---
    
    # Log the Robot (Green Sphere)
    rr.log(
        "world/robot", 
        rr.Points3D(robot_pos, radii=robot_radius, colors=[0, 255, 0])
    )
    
    # Log the Dynamic Obstacles (Orange Spheres)
    rr.log(
        "world/obstacles", 
        rr.Points3D(obs_pos, radii=obs_radius, colors=[255, 165, 0])
    )
    
    # Log the Obstacle Velocity Vectors (Red Arrows)
    rr.log(
        "world/obstacles/velocities", 
        rr.Arrows3D(origins=obs_pos, vectors=obs_vel, colors=[255, 0, 0])
    )

    # --- LOGGING THE 3D PARABOLOID (Point Cloud Approach) ---
    # Instead of Matplotlib's plot_surface, we can easily blast thousands of points 
    # to visualize the boundary of the DPCBF without tanking the frame rate.
    
    # (Mocking a paraboloid point cloud in front of the robot)
    y_grid, z_grid = np.meshgrid(np.linspace(-1.5, 1.5, 20), np.linspace(-1.5, 1.5, 20))
    x_curve = 0.5 * (y_grid**2 + z_grid**2) + 0.5 # Parabola opening forward
    
    parabola_pts = np.vstack([x_curve.flatten(), y_grid.flatten(), z_grid.flatten()]).T
    parabola_pts += robot_pos # Shift to robot position
    
    # Log the paraboloid as a semi-transparent blue point cloud
    rr.log(
        "world/robot/dpcbf_boundary", 
        rr.Points3D(parabola_pts, radii=0.02, colors=[0, 100, 255, 150])
    )

    time.sleep(0.05) # Simulate compute time