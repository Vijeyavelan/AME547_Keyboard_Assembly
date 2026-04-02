"""
tune_keyframes_v2.py — updated geometry:
  raiser=600mm, robot y=-0.35, table_y=0.150,
  conveyor y=0.28 z=0.820, belt half=0.920

Run from repo root:
    mjpython tune_keyframes_v2.py
"""
import mujoco
import numpy as np

MODEL_PATH = "models/stations/station1.xml"

# EPDM grasp site world pos:
#   body(-0.18, 0.28, 0.82575) + offset(-0.06, 0, 0.00075) = (-0.24, 0.28, 0.8265)
EPDM_GRASP = np.array([-0.06, 0.28, 0.8265])
PALLET_CTR = np.array([-0.06, 0.03, 0.810])
CLEARANCE  = 0.080

targets = {
    "pick_ready":  EPDM_GRASP + np.array([0, 0, CLEARANCE]),
    "pick_down":   EPDM_GRASP - np.array([0, 0, 0.005]),  # 5mm penetration
    "place_ready": PALLET_CTR + np.array([0, 0, CLEARANCE]),
    "place_1":     PALLET_CTR + np.array([0, 0, 0.004]),
    "place_2":     PALLET_CTR + np.array([0, 0, 0.008]),
    "place_3":     PALLET_CTR + np.array([0, 0, 0.012]),
    "place_4":     PALLET_CTR + np.array([0, 0, 0.013]),
    "place_5":     PALLET_CTR + np.array([0, 0, 0.014]),
}

GOOD_SEED = np.array([-1.5708, -1.20, 1.80, -2.20, -1.5708, 0.0])

def solve_ik(m, d, site_name, target_pos, seed=None, max_iter=2000, tol=5e-5):
    site_id      = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site_name)
    arm_jnt_ids  = list(range(6))
    arm_qpos_adr = [m.jnt_qposadr[i] for i in arm_jnt_ids]
    arm_dof_adr  = [m.jnt_dofadr[i]  for i in arm_jnt_ids]

    mujoco.mj_resetDataKeyframe(m, d, 0)
    s = seed if seed is not None else GOOD_SEED
    for i, adr in enumerate(arm_qpos_adr):
        d.qpos[adr] = s[i]
    mujoco.mj_forward(m, d)

    jacp = np.zeros((3, m.nv))
    jacr = np.zeros((3, m.nv))
    for _ in range(max_iter):
        mujoco.mj_forward(m, d)
        err = target_pos - d.site_xpos[site_id]
        if np.linalg.norm(err) < tol:
            break
        mujoco.mj_jacSite(m, d, jacp, jacr, site_id)
        J   = jacp[:, arm_dof_adr]
        lam = 1e-4
        dq  = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(3), err)
        for i, adr in enumerate(arm_qpos_adr):
            d.qpos[adr] = np.clip(
                d.qpos[adr] + dq[i] * 0.3,
                m.jnt_range[arm_jnt_ids[i], 0],
                m.jnt_range[arm_jnt_ids[i], 1]
            )

    mujoco.mj_forward(m, d)
    err_mm = np.linalg.norm(target_pos - d.site_xpos[site_id]) * 1000
    qpos   = [round(float(d.qpos[adr]), 4) for adr in arm_qpos_adr]
    return qpos, err_mm

def main():
    print(f"Loading: {MODEL_PATH}\n")
    m = mujoco.MjModel.from_xml_path(MODEL_PATH)
    d = mujoco.MjData(m)

    tip_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripper_tip")
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    print(f"Home gripper_tip: {np.round(d.site_xpos[tip_id], 4)}\n")

    results   = {}
    prev_qpos = GOOD_SEED.copy()

    for name, target in targets.items():
        print(f"  {name}: target={np.round(target, 4)}")
        qpos, err = solve_ik(m, d, "gripper_tip", target, seed=prev_qpos)
        arm_qpos_adr = [m.jnt_qposadr[i] for i in range(6)]
        for i, adr in enumerate(arm_qpos_adr):
            d.qpos[adr] = qpos[i]
        mujoco.mj_forward(m, d)
        elbow  = float(d.qpos[m.jnt_qposadr[2]])
        config = f"elbow={elbow:.3f}rad {'✓ good' if abs(elbow) > 0.5 else '⚠ NEAR SINGULAR'}"
        print(f"    joints: {qpos}")
        print(f"    error:  {err:.2f}mm  {config}")
        if err > 1.0:
            print(f"    WARNING: retrying from GOOD_SEED")
            qpos, err = solve_ik(m, d, "gripper_tip", target, seed=GOOD_SEED)
            print(f"    retry:  {qpos}, error: {err:.2f}mm")
        results[name] = qpos
        prev_qpos = np.array(qpos)
        print()

    parts    = "0.00 0.28 0.82575 1 0 0 0  -0.43 0.28 0.828 1 0 0 0  -0.68 0.28 0.8258 1 0 0 0  -1.04 0.28 0.8255 1 0 0 0  -1.40 0.28 0.82575 1 0 0 0"
    home_arm = "-1.5708 -1.5708 1.5708 -1.5708 -1.5708 0"

    print("\n" + "="*60)
    print("PASTE INTO station1.xml <keyframe> block:")
    print("="*60)
    print(f'\n    <key name="home"')
    print(f'         qpos="{home_arm}  {parts}"')
    print(f'         ctrl="{home_arm}"/>')
    for name, qpos in results.items():
        arm = " ".join(str(v) for v in qpos)
        print(f'\n    <key name="{name}"')
        print(f'         qpos="{arm}  {parts}"')
        print(f'         ctrl="{arm}"/>')

if __name__ == "__main__":
    main()