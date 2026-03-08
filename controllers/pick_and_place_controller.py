import mujoco
import mujoco.viewer
import numpy as np
import time

m = mujoco.MjModel.from_xml_path('models/parts/pick_and_place.xml')
d = mujoco.MjData(m)

tip_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "tip")
object_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")

def move_to(v, joint_angles, duration=2.0, carry=False):
    start_time = time.time()
    while time.time() - start_time < duration:
        d.ctrl[0] = joint_angles[0]
        d.ctrl[1] = joint_angles[1]
        d.ctrl[2] = joint_angles[2]

        if carry:
            tip_pos = d.xpos[tip_id].copy()
            d.qpos[3] = tip_pos[0]
            d.qpos[4] = tip_pos[1]
            d.qpos[5] = tip_pos[2] + 0.05
            d.qpos[6:10] = [1, 0, 0, 0]
            d.qvel[3:10] = 0

        mujoco.mj_step(m, d)
        v.sync()
        time.sleep(0.01)
    print(f"Tip after move: {d.xpos[tip_id]}")

def attach_object():
    tip_pos = d.xpos[tip_id].copy()
    d.qpos[3] = tip_pos[0]
    d.qpos[4] = tip_pos[1]
    d.qpos[5] = tip_pos[2] + 0.05
    d.qpos[6:10] = [1, 0, 0, 0]
    d.qvel[3:10] = 0
    mujoco.mj_forward(m, d)
    print("Object attached")

def detach_object():
    d.qpos[3:6] = [-0.5, 0, 0.05]
    d.qpos[6:10] = [1, 0, 0, 0]
    d.qvel[3:10] = 0
    mujoco.mj_forward(m, d)
    print("Object released")

with mujoco.viewer.launch_passive(m, d) as v:

    print("Step 1: Home")
    move_to(v, [0.0, 0.0, 0.0], duration=2.0)

    print("Step 2: Descending to object")
    move_to(v, [0.0, -0.175, 1.4], duration=3.0)
"""
    print("Step 3: Attaching object")
    attach_object()
    time.sleep(1.0)

    print("Step 4: Lifting object")
    move_to(v, [0.0, 0.0, 0.0], duration=5.0, carry=True)
"""
    # Keep viewer open
while v.is_running():
        mujoco.mj_step(m, d)
        v.sync()
        time.sleep(0.01)