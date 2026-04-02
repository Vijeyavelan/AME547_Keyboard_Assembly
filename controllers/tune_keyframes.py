"""
tune_keyframes.py
Computes UR5e joint angles for S1 pick and place targets using
MuJoCo's numerical IK (mj_invpos via finite-difference Jacobian).

Run from repo root:
    python tune_keyframes.py

Outputs joint angles for each target pose — paste into station1.xml keyframes.
"""

import mujoco
import numpy as np

MODEL_PATH = "models/stations/station1.xml"

# ── Target world positions for gripper_tip site ─────────────────────────────
# Pick point: EPDM on conveyor at x=-0.18, y=0.40
# Clearance height: 80mm above belt (z=0.805) → z=0.885
# Contact height: belt surface z=0.805

targets = {
    "pick_ready":  np.array([-0.18,  0.40, 0.885]),  # 80mm above EPDM
    "pick_down":   np.array([-0.18,  0.40, 0.808]),  # gripper tip at belt
    "place_ready": np.array([ 0.00,  0.03, 0.890]),  # 80mm above pallet
    "place_1":     np.array([ 0.00,  0.03, 0.821]),  # EPDM layer
    "place_2":     np.array([ 0.00,  0.025,0.825]),  # Battery (x+0.025 offset)
    "place_3":     np.array([ 0.00,  0.03, 0.829]),  # PCB
    "place_4":     np.array([ 0.00,  0.03, 0.830]),  # Sound foam
    "place_5":     np.array([ 0.00,  0.03, 0.831]),  # Alu plate
}

def solve_ik(m, d, site_name, target_pos, init_qpos=None, max_iter=1000, tol=1e-4):
    """
    Simple position IK using Jacobian pseudoinverse.
    Moves gripper_tip site to target_pos.
    Returns joint angles (6,) or None if failed.
    """
    site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        print(f"  ERROR: site '{site_name}' not found")
        return None

    # Get arm joint qpos indices (joints 0-5 in this model)
    arm_joint_ids = list(range(6))  # shoulder_pan..wrist_3
    arm_qpos_adr = [m.jnt_qposadr[i] for i in arm_joint_ids]

    # Set initial pose
    mujoco.mj_resetDataKeyframe(m, d, 0)  # start from home
    if init_qpos is not None:
        for i, adr in enumerate(arm_qpos_adr):
            d.qpos[adr] = init_qpos[i]
    mujoco.mj_forward(m, d)

    jacp = np.zeros((3, m.nv))
    jacr = np.zeros((3, m.nv))

    for iteration in range(max_iter):
        mujoco.mj_forward(m, d)
        site_pos = d.site_xpos[site_id].copy()
        err = target_pos - site_pos
        if np.linalg.norm(err) < tol:
            break

        mujoco.mj_jacSite(m, d, jacp, jacr, site_id)

        # Extract columns for arm joints only
        arm_vadr = [m.jnt_dofadr[i] for i in arm_joint_ids]
        J = jacp[:, arm_vadr]  # 3x6

        # Pseudoinverse step with damping
        lam = 1e-4
        dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(3), err)

        # Apply step
        for i, adr in enumerate(arm_qpos_adr):
            d.qpos[adr] += dq[i] * 0.5

        # Clamp to joint limits
        for ji in arm_joint_ids:
            adr = m.jnt_qposadr[ji]
            lo, hi = m.jnt_range[ji]
            d.qpos[adr] = np.clip(d.qpos[adr], lo, hi)

    mujoco.mj_forward(m, d)
    final_err = np.linalg.norm(target_pos - d.site_xpos[site_id])
    return [round(d.qpos[adr], 4) for adr in arm_qpos_adr], final_err


def main():
    print(f"Loading model: {MODEL_PATH}\n")
    m = mujoco.MjModel.from_xml_path(MODEL_PATH)
    d = mujoco.MjData(m)

    # Verify site exists
    site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripper_tip")
    print(f"gripper_tip site id: {site_id}")
    if site_id < 0:
        print("ERROR: gripper_tip site not found — check XML")
        return

    # Solve for home first to verify
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    home_tip = d.site_xpos[site_id].copy()
    print(f"Home gripper_tip world pos: {np.round(home_tip, 4)}\n")

    results = {}
    prev_qpos = None

    print("Solving IK for each target...\n")
    for name, target in targets.items():
        print(f"  {name}: target={np.round(target,3)}")
        qpos, err = solve_ik(m, d, "gripper_tip", target, init_qpos=prev_qpos)
        if qpos is not None:
            print(f"    joints: {qpos}")
            print(f"    error:  {err*1000:.2f}mm")
            results[name] = qpos
            prev_qpos = qpos
        else:
            print(f"    FAILED")
        print()

    # Print keyframe XML
    parts_qpos = "-1.5708 -1.5708 1.5708 -1.5708 -1.5708 0  -0.18 0.4 0.80575 1 0 0 0  -0.43 0.4 0.808 1 0 0 0  -0.68 0.4 0.8058 1 0 0 0  -1.04 0.4 0.8055 1 0 0 0  -1.40 0.4 0.80575 1 0 0 0"

    print("\n" + "="*60)
    print("PASTE THESE INTO station1.xml <keyframe> block:")
    print("="*60)

    # Home
    home_arm = "-1.5708 -1.5708 1.5708 -1.5708 -1.5708 0"
    print(f'\n    <key name="home"')
    print(f'         qpos="{home_arm}  -0.18 0.4 0.80575 1 0 0 0  -0.43 0.4 0.808 1 0 0 0  -0.68 0.4 0.8058 1 0 0 0  -1.04 0.4 0.8055 1 0 0 0  -1.40 0.4 0.80575 1 0 0 0"')
    print(f'         ctrl="{home_arm}"/>')

    for name, qpos in results.items():
        arm_str = " ".join(str(v) for v in qpos)
        full_qpos = f"{arm_str}  -0.18 0.4 0.80575 1 0 0 0  -0.43 0.4 0.808 1 0 0 0  -0.68 0.4 0.8058 1 0 0 0  -1.04 0.4 0.8055 1 0 0 0  -1.40 0.4 0.80575 1 0 0 0"
        print(f'\n    <key name="{name}"')
        print(f'         qpos="{full_qpos}"')
        print(f'         ctrl="{arm_str}"/>')


if __name__ == "__main__":
    main()
