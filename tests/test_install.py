import mujoco
import numpy as np

xml = """
<mujoco>
  <worldbody>
    <body name="box" pos="0 0 1">
      <geom type="box" size="0.1 0.1 0.1" rgba="1 0 0 1"/>
      <freejoint/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
mujoco.mj_step(model, data)
print("MuJoCo version:", mujoco.__version__)
print("Installation successful!")