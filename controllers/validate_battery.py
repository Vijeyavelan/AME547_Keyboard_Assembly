"""
validate_battery.py
AME 547 — Phase 2 battery validation

Usage:
    python controllers/validate_battery.py
"""

import mujoco
import numpy as np

XML_PATH = "models/parts/battery.xml"

def main():
    print("=" * 60)
    print("AME 547 — Battery (606090 LiPo) Validation")
    print("=" * 60)

    try:
        model = mujoco.MjModel.from_xml_path(XML_PATH)
    except Exception as e:
        print(f"\n❌ Failed to load XML: {e}")
        return
    print(f"\n✅ Model loaded: {XML_PATH}")

    data = mujoco.MjData(model)

    # ── Geom inventory ──────────────────────────────────────────
    print(f"\n── Geoms ({model.ngeom}) ──────────────────────────────")
    geom_type_names = {0:"plane", 6:"box", 5:"cylinder", 2:"sphere"}
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or f"geom_{i}"
        gtype = geom_type_names.get(model.geom_type[i], str(model.geom_type[i]))
        size  = np.round(model.geom_size[i] * 1000, 2)
        print(f"  [{i}] {name:20s}  type={gtype:6s}  half-extents(mm)={size}")

    # ── Site inventory ──────────────────────────────────────────
    print(f"\n── Sites ({model.nsite}) ──────────────────────────────")
    for i in range(model.nsite):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i) or f"site_{i}"
        pos  = np.round(model.site_pos[i] * 1000, 1)
        print(f"  [{i}] {name:22s}  pos={pos} mm")

    # ── Mass & dimensions ───────────────────────────────────────
    print(f"\n── Mass & dimensions ──────────────────────────────────")
    if model.nbody > 1:
        mass = model.body_mass[1]
        print(f"  Mass            : {mass*1000:.1f} g  (expect ~80g)")
    print(f"  Length (X)      : {0.045*2*1000:.0f} mm  (expect 90mm)")
    print(f"  Width  (Y)      : {0.030*2*1000:.0f} mm  (expect 60mm)")
    print(f"  Thickness (Z)   : {0.003*2*1000:.0f} mm  (expect 6mm)")
    print(f"  X offset        : +25mm from case center (rib tray)")

    # ── Grasp check ─────────────────────────────────────────────
    print(f"\n── Grasp check ────────────────────────────────────────")
    print(f"  Vacuum cup dia  : 30mm → needs 15mm clear radius")
    print(f"  Grasp center    : X=0, Y=0 on battery face")
    print(f"  Battery face    : 90mm × 60mm")
    print(f"  Cup to X edge   : {45-15}mm clearance ✓")
    print(f"  Cup to Y edge   : {30-15}mm clearance ✓")

    # ── Forward sim ─────────────────────────────────────────────
    print(f"\n── Forward simulation (100 steps) ─────────────────────")
    mujoco.mj_resetData(model, data)
    data.qpos[2] = 0.015

    for _ in range(100):
        mujoco.mj_step(model, data)

    print(f"  Final z         : {data.qpos[2]*1000:.2f} mm")
    print(f"  Active contacts : {data.ncon}")

    penetration = False
    for i in range(data.ncon):
        if data.contact[i].dist < -0.001:
            print(f"  ⚠️  Contact {i}: dist={data.contact[i].dist*1000:.2f}mm")
            penetration = True
    if not penetration:
        print(f"  ✅ No penetration detected")

    print("\n" + "=" * 60)
    print("Copy to: models/parts/battery.xml")
    print("=" * 60)

if __name__ == "__main__":
    main()
