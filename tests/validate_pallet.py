"""
Validate assembly_pallet.xml
Checks:
  1. Model loads without error
  2. Base plate dimensions correct (340x130x8mm)
  3. Dowel pin positions match FL and RR boss coordinates
  4. Pallet body origin at z=0.750m (table height)
  5. Pin tops above plate surface
"""
import sys
import mujoco

model_path = sys.argv[1] if len(sys.argv) > 1 else "assembly_pallet.xml"
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

print(f"Model loaded: {model.ngeom} geoms, {model.nbody} bodies")
print()

# --- Check base plate ---
plate_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "base_plate")
plate_size = model.geom_size[plate_id]
plate_pos = model.geom_pos[plate_id]
print("=== Base Plate ===")
print(f"  Half-sizes (XYZ): {plate_size}")
print(f"  Full dims: {plate_size[0]*2*1000:.1f} x {plate_size[1]*2*1000:.1f} x {plate_size[2]*2*1000:.1f} mm")
print(f"  Local pos: {plate_pos}")
assert abs(plate_size[0] - 0.170) < 1e-4, f"Plate X half-size wrong: {plate_size[0]}"
assert abs(plate_size[1] - 0.065) < 1e-4, f"Plate Y half-size wrong: {plate_size[1]}"
assert abs(plate_size[2] - 0.004) < 1e-4, f"Plate Z half-size wrong: {plate_size[2]}"
print("  ✅ Plate dimensions correct (340x130x8mm)")
print()

# --- Check pallet body world position ---
body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pallet_base")
body_pos = data.xpos[body_id]
print("=== Pallet Body ===")
print(f"  World pos: {body_pos}")
assert abs(body_pos[2] - 0.750) < 1e-4, f"Body Z wrong: {body_pos[2]}"
print("  ✅ Body origin at z=0.750m (table height)")
print()

# --- Check dowel pins ---
expected_pins = {
    "dowel_pin_FL": (-0.1515, -0.0495),
    "dowel_pin_RR": ( 0.1545,  0.0495),
}

print("=== Dowel Pins ===")
for pin_name, (ex, ey) in expected_pins.items():
    pin_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, pin_name)
    pin_pos = model.geom_pos[pin_id]
    pin_size = model.geom_size[pin_id]
    print(f"  {pin_name}:")
    print(f"    Local pos: ({pin_pos[0]:.4f}, {pin_pos[1]:.4f}, {pin_pos[2]:.4f})")
    print(f"    Radius: {pin_size[0]*1000:.1f}mm, Half-height: {pin_size[1]*1000:.1f}mm")
    assert abs(pin_pos[0] - ex) < 1e-4, f"{pin_name} X wrong: {pin_pos[0]} vs {ex}"
    assert abs(pin_pos[1] - ey) < 1e-4, f"{pin_name} Y wrong: {pin_pos[1]} vs {ey}"
    assert pin_pos[2] > 0, f"{pin_name} should be above plate surface"

    # Pin top in world frame = body_z + pin_local_z + half_height
    pin_top_world = body_pos[2] + pin_pos[2] + pin_size[1]
    plate_top_world = body_pos[2]  # body origin is at plate top
    print(f"    Pin top (world): {pin_top_world:.4f}m")
    print(f"    Plate top (world): {plate_top_world:.4f}m")
    assert pin_top_world > plate_top_world, f"{pin_name} doesn't protrude above plate!"
    print(f"    ✅ Pin protrudes {(pin_top_world - plate_top_world)*1000:.1f}mm above plate")
    print()

print("=" * 40)
print("ALL CHECKS PASSED ✅")
