import mujoco
import numpy as np

m = mujoco.MjModel.from_xml_path('models/parts/pick_and_place.xml')
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)

print("Full qpos:", d.qpos)
print("qpos size:", m.nq)

# Print all body positions
for i in range(m.nbody):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
    print(f"Body {i} '{name}': pos={d.xpos[i]}")