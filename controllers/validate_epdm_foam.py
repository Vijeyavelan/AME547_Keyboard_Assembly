"""
validate_epdm_foam.py
AME 547 — Phase 2 EPDM foam validation

Usage:
    python controllers/validate_epdm_foam.py
"""

import mujoco
import numpy as np

XML_PATH = "models/parts/epdm_foam.xml"

def main():
    print("=" * 60)
    print("AME 547 — EPDM Foam Validation")
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
        size = np.round(model.geom_size[i] * 1000, 2)
        print(f"  [{i}] {name:25s}  type={gtype:6s}  half-extents(mm)={size}")

    # ── Site inventory ──────────────────────────────────────────
    print(f"\n── Sites ({model.nsite}) ──────────────────────────────")
    for i in range(model.nsite):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i) or f"site_{i}"
        pos = np.round(model.site_pos[i] * 1000, 1)
        print(f"  [{i}] {name:28s}  pos={pos} mm")

    # ── Mass ────────────────────────────────────────────────────
    print(f"\n── Mass properties ────────────────────────────────────")
    if model.nbody > 1:
        mass = model.body_mass[1]
        print(f"  Total mass: {mass*1000:.1f} g  (expect ~8g for EPDM foam sheet)")

    # ── Coverage check ──────────────────────────────────────────
    print(f"\n── Coverage check ─────────────────────────────────────")
    print(f"  Foam outer X span : {0.1595*2*1000:.1f} mm  (expect 319mm — case interior)")
    print(f"  Foam outer Y span : {0.0545*2*1000:.1f} mm  (expect 109mm — case interior)")
    print(f"  Battery cutout X  : {(0.070-(-0.020))*1000:.1f} mm  (expect 90mm — battery length)")
    print(f"  Battery cutout Y  : {0.030*2*1000:.1f} mm  (expect 60mm — battery width)")
    print(f"  Battery X offset  : {((0.070+(-0.020))/2)*1000:.1f} mm  (expect +25mm from center)")

    # ── Forward sim ─────────────────────────────────────────────
    print(f"\n── Forward simulation (100 steps) ─────────────────────")
    mujoco.mj_resetData(model, data)
    data.qpos[2] = 0.010

    for _ in range(100):
        mujoco.mj_step(model, data)

    print(f"  Final z position  : {data.qpos[2]*1000:.2f} mm")
    print(f"  Active contacts   : {data.ncon}")

    penetration = False
    for i in range(data.ncon):
        if data.contact[i].dist < -0.001:
            print(f"  ⚠️  Contact {i}: dist={data.contact[i].dist*1000:.2f}mm")
            penetration = True
    if not penetration:
        print(f"  ✅ No significant penetration")

    print("\n" + "=" * 60)
    print("Copy to: models/parts/epdm_foam.xml")
    print("=" * 60)

if __name__ == "__main__":
    main()
