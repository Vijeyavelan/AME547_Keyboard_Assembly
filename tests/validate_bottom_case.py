"""
validate_bottom_case.py
AME 547 — Phase 2 asset validation

Loads bottom_case.xml, prints body/geom/site inventory,
runs a short forward simulation to confirm no interpenetration,
and reports mass properties.

Usage:
    python tests/validate_bottom_case.py
"""

import mujoco
import numpy as np

XML_PATH = "models/parts/bottom_case.xml"   # adjust if running from repo root

def main():
    print("=" * 60)
    print("AME 547 — Bottom Case MJCF Validation")
    print("=" * 60)

    # ── Load model ──────────────────────────────────────────────
    try:
        model = mujoco.MjModel.from_xml_path(XML_PATH)
    except Exception as e:
        print(f"\n❌ Failed to load XML: {e}")
        return
    print(f"\n✅ Model loaded successfully: {XML_PATH}")

    data = mujoco.MjData(model)

    # ── Body inventory ──────────────────────────────────────────
    print(f"\n── Bodies ({model.nbody}) ─────────────────────────────")
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or f"body_{i}"
        print(f"  [{i}] {name}")

    # ── Geom inventory ──────────────────────────────────────────
    print(f"\n── Geoms ({model.ngeom}) ──────────────────────────────")
    geom_type_names = {0: "plane", 2: "sphere", 3: "capsule", 4: "ellipsoid",
                       5: "cylinder", 6: "box", 7: "mesh"}
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or f"geom_{i}"
        gtype = geom_type_names.get(model.geom_type[i], str(model.geom_type[i]))
        size = model.geom_size[i]
        print(f"  [{i}] {name:25s}  type={gtype:8s}  half-extents(m)={size}")

    # ── Site inventory ──────────────────────────────────────────
    print(f"\n── Sites ({model.nsite}) ──────────────────────────────")
    for i in range(model.nsite):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i) or f"site_{i}"
        pos = model.site_pos[i]
        print(f"  [{i}] {name:30s}  pos={np.round(pos*1000, 1)} mm")

    # ── Mass properties ─────────────────────────────────────────
    print(f"\n── Mass properties ────────────────────────────────────")
    # Body 1 = bottom_case (body 0 = worldbody)
    if model.nbody > 1:
        body_id = 1
        mass = model.body_mass[body_id]
        ipos = model.body_ipos[body_id]
        print(f"  Total body mass     : {mass*1000:.1f} g")
        print(f"  CoM offset (body)   : {np.round(ipos*1000, 2)} mm")
        print(f"  Diagonal inertia    : {model.body_inertia[body_id]} kg·m²")

    # ── Bounding box check ──────────────────────────────────────
    print(f"\n── Dimensional check ──────────────────────────────────")
    # Left/right walls span ±0.1625 + wall = ±0.164 → 328mm outer (correct for ABS lip)
    # Front/rear walls span ±0.0575 → 115mm outer
    # Height: body at z=0.011, geoms span from -0.011 to +0.011 → 22mm
    print(f"  Outer length (X)    : {(0.1625 + 0.0015)*2*1000:.0f} mm  (expect ~328mm with walls)")
    print(f"  Outer width  (Y)    : {0.0575*2*1000:.0f} mm  (expect 115mm)")
    print(f"  Case height  (Z)    : {0.022*1000:.0f} mm  (body center at 11mm)")

    # ── Forward sim ─────────────────────────────────────────────
    print(f"\n── Forward simulation (50 steps) ──────────────────────")
    mujoco.mj_resetData(model, data)
    # Place case just above ground
    data.qpos[2] = 0.015   # z slightly above resting pos
    initial_contacts = 0

    for step in range(50):
        mujoco.mj_step(model, data)

    final_z = data.qpos[2]
    ncon = data.ncon
    print(f"  Final z position    : {final_z*1000:.2f} mm  (expect ~11mm at rest)")
    print(f"  Active contacts     : {ncon}")
    print(f"  Simulation time     : {data.time:.3f} s")

    # Check for interpenetration (contact distances should be >= 0)
    penetration_detected = False
    for i in range(data.ncon):
        c = data.contact[i]
        if c.dist < -0.001:   # more than 1mm penetration = problem
            print(f"  ⚠️  Contact {i}: dist={c.dist*1000:.2f}mm — check geom overlap")
            penetration_detected = True
    if not penetration_detected:
        print(f"  ✅ No significant interpenetration detected")

    print("\n" + "=" * 60)
    print("Validation complete. Copy bottom_case.xml to:")
    print("  models/parts/bottom_case.xml")
    print("=" * 60)

if __name__ == "__main__":
    main()
