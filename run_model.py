import mujoco
import mujoco.viewer
import sys

model_path = sys.argv[1]

m = mujoco.MjModel.from_xml_path(model_path)
d = mujoco.MjData(m)

# Load the starting keyframe
mujoco.mj_resetDataKeyframe(m, d, 0)

with mujoco.viewer.launch_passive(m, d) as v:
    while v.is_running():
        mujoco.mj_step(m, d)
        v.sync()