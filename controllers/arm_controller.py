import mujoco
import mujoco.viewer
import numpy as np
import time

m = mujoco.MjModel.from_xml_path('models/parts/robot_arm.xml')
d = mujoco.MjData(m)

# Define target positions as joint angles in radians
waypoints = [
    [0.0, 0.0],    # home position
    [1.0, 0.5],    # position 1
    [-1.0, 0.8],   # position 2
    [0.5, -0.5],   # position 3
    [0.0, 0.0],    # back to home
]

with mujoco.viewer.launch_passive(m, d) as v:
    for waypoint in waypoints:
        # Move to each waypoint and hold for 2 seconds
        start_time = time.time()
        while time.time() - start_time < 2.0:
            d.ctrl[0] = waypoint[0]
            d.ctrl[1] = waypoint[1]
            mujoco.mj_step(m, d)
            v.sync()
            time.sleep(0.01)