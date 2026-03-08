import mujoco
import numpy as np

m = mujoco.MjModel.from_xml_path('models/parts/pick_and_place.xml')
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)

tip_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "tip")
print("Tip position at home:", d.xpos[tip_id])
print("Object position:", d.xpos[5])