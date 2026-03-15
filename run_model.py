import mujoco
import mujoco.viewer
import sys
model = mujoco.MjModel.from_xml_path(sys.argv[1])
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.azimuth = 45
    viewer.cam.elevation = -20
    viewer.cam.distance = 0.6
    viewer.cam.lookat = [0, 0, 0.011]

    import time
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.002)