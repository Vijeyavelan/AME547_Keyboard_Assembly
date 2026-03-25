import mujoco
import mujoco.viewer
import sys

# Usage: mjpython run_station.py <scene.xml> [keyframe_name]
# Default keyframe: "home"

model_path = sys.argv[1]
key_name   = sys.argv[2] if len(sys.argv) > 2 else "home"

model = mujoco.MjModel.from_xml_path(model_path)
data  = mujoco.MjData(model)

# Load keyframe by name if it exists
key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key_name)
if key_id >= 0:
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    print(f"Loaded keyframe '{key_name}' (id={key_id})")
else:
    print(f"Keyframe '{key_name}' not found — starting at default pose")

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
