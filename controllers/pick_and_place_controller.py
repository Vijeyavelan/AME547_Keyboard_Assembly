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

def attach_object(v, duration=1.0):
    start_time = time.time()
    while time.time() - start_time < duration:
        tip_pos = d.xpos[tip_id].copy()
        d.qpos[3] = tip_pos[0]
        d.qpos[4] = tip_pos[1]
        d.qpos[5] = tip_pos[2] - 0.05
        d.qpos[6:10] = [1, 0, 0, 0]
        d.qvel[3:10] = 0
        mujoco.mj_step(m, d)
        v.sync()
        time.sleep(0.01)
    print(f"Tip pos: {d.xpos[tip_id]}")
    print(f"Box pos: {d.xpos[object_id]}")
    print("Object attached")

def detach_object():
    d.qvel[3:10] = 0
    mujoco.mj_forward(m, d)
    print("Object released")

with mujoco.viewer.launch_passive(m, d) as v:

    print("Step 1: Home")
    move_to(v, [0.0, 0.0, 0.0], duration=2.0)

    print("Step 2: Descending to object")
    move_to(v, [0.0, -0.175, 1.4], duration=3.0)

    print("Step 3: Attaching object")
    attach_object(v, duration=1.0)
    time.sleep(1.0)

    print("Step 4: Lifting object")
    move_to(v, [0.0, 0.3, 1.0], duration=5.0, carry=True)

    print("Step 5: Rotating to target")
    move_to(v, [-3.2, 0.0, 0.0], duration=5.0, carry=True)

    print("Step 5b: Descending to target")
    move_to(v, [-3.2, -0.175, 1.4], duration=5.0, carry=True)

    print("Step 5c: Holding position without carry")
    move_to(v, [-3.2, -0.175, 1.4], duration=1.0, carry=False)

    print("Step 6: Placing object")
    detach_object()

    print("Step 7: Return home")
    move_to(v, [0.0, 0.0, 0.0], duration=5.0, carry=False)

    while v.is_running():
        mujoco.mj_step(m, d)
        v.sync()
        time.sleep(0.01)