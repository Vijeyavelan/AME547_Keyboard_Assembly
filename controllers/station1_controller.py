"""
station1_controller.py — Step 7: Release and retract
Full sequence: home → pick → attach → carry → place → release → home

Run from repo root:
    mjpython controllers/station1_controller.py
"""

import mujoco
import mujoco.viewer
import numpy as np

MODEL_PATH   = "models/stations/station1.xml"
SETTLE_TOL   = 0.015
SETTLE_STEPS = 50

EPDM_PLACE_POS = np.array([0.0, 0.030, 0.81275])


def get_ctrl(m, kf_index):
    ctrl = np.zeros(m.nu)
    ctrl[:] = m.key_ctrl[kf_index, :m.nu]
    return ctrl


def get_site_pos(m, d, site_name):
    site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site_name)
    mujoco.mj_forward(m, d)
    return d.site_xpos[site_id].copy()


def move_to(m, d, v, target_ctrl, label):
    d.ctrl[:] = target_ctrl
    consecutive = 0
    steps = 0
    while consecutive < SETTLE_STEPS:
        mujoco.mj_step(m, d)
        v.sync()
        err = np.max(np.abs(d.qpos[:6] - target_ctrl[:6]))
        consecutive = consecutive + 1 if err < SETTLE_TOL else 0
        steps += 1
        if steps > 8000:
            print(f"  WARNING: {label} timed out")
            break
    print(f"  ✓ {label} (err={np.max(np.abs(d.qpos[:6]-target_ctrl[:6])):.4f}rad)")


def carry_to(m, d, v, fj_adr, fv_adr, carry_offset, target_ctrl, label):
    d.ctrl[:] = target_ctrl
    consecutive = 0
    steps = 0
    while consecutive < SETTLE_STEPS:
        tip_pos = get_site_pos(m, d, "gripper_tip")
        d.qpos[fj_adr:fj_adr+3]   = tip_pos + carry_offset
        d.qpos[fj_adr+3:fj_adr+7] = [1, 0, 0, 0]
        d.qvel[fv_adr:fv_adr+6]   = 0
        mujoco.mj_step(m, d)
        v.sync()
        err = np.max(np.abs(d.qpos[:6] - target_ctrl[:6]))
        consecutive = consecutive + 1 if err < SETTLE_TOL else 0
        steps += 1
        if steps > 8000:
            print(f"  WARNING: {label} timed out")
            break
    print(f"  ✓ {label} (err={np.max(np.abs(d.qpos[:6]-target_ctrl[:6])):.4f}rad)")


def main():
    print(f"Loading: {MODEL_PATH}\n")
    m = mujoco.MjModel.from_xml_path(MODEL_PATH)
    d = mujoco.MjData(m)

    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    kf = {i: get_ctrl(m, i) for i in range(m.nkey)}

    fj_id  = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "s1_epdm_jnt")
    fj_adr = m.jnt_qposadr[fj_id]
    fv_adr = m.jnt_dofadr[fj_id]

    with mujoco.viewer.launch_passive(m, d) as v:
        v.cam.lookat[:] = [0.0, 0.05, 0.83]
        v.cam.distance  = 1.0
        v.cam.elevation = -20
        v.cam.azimuth   = 150

        # ── STEPS 1-2 ─────────────────────────────────────────────────────
        print("[STEP 1] home → pick_ready")
        move_to(m, d, v, kf[0], "home")
        move_to(m, d, v, kf[1], "pick_ready")

        print("\n[STEP 2] pick_ready → pick_down")
        move_to(m, d, v, kf[2], "pick_down")

        # ── STEP 3: Attach ────────────────────────────────────────────────
        print("\n[STEP 3] Attach")
        tip      = get_site_pos(m, d, "gripper_tip")
        epdm_ctr = d.qpos[fj_adr:fj_adr+3].copy()
        carry_offset = epdm_ctr - tip
        print(f"  carry_offset: {np.round(carry_offset, 4)}")
        for _ in range(60):
            d.ctrl[:] = kf[2]
            tip_pos = get_site_pos(m, d, "gripper_tip")
            d.qpos[fj_adr:fj_adr+3]   = tip_pos + carry_offset
            d.qpos[fj_adr+3:fj_adr+7] = [1, 0, 0, 0]
            d.qvel[fv_adr:fv_adr+6]   = 0
            mujoco.mj_step(m, d)
            v.sync()
        print(f"  ✓ attached")

        # ── STEPS 4-6: Carry to place_1 ───────────────────────────────────
        print("\n[STEP 4] Lift → pick_ready")
        carry_to(m, d, v, fj_adr, fv_adr, carry_offset, kf[1], "pick_ready+part")

        print("\n[STEP 5] Swing → place_ready")
        carry_to(m, d, v, fj_adr, fv_adr, carry_offset, kf[3], "place_ready+part")

        print("\n[STEP 6] Descend → place_1")
        carry_to(m, d, v, fj_adr, fv_adr, carry_offset, kf[4], "place_1+part")

        # ── STEP 7: Release ───────────────────────────────────────────────
        print("\n[STEP 7] Release — clamping EPDM to final position")

        # Snap to exact place position and hold while arm retracts
        for _ in range(100):
            d.ctrl[:] = kf[4]          # keep arm at place_1
            d.qpos[fj_adr:fj_adr+3]   = EPDM_PLACE_POS
            d.qpos[fj_adr+3:fj_adr+7] = [1, 0, 0, 0]
            d.qvel[fv_adr:fv_adr+6]   = 0
            mujoco.mj_step(m, d)
            v.sync()

        # Retract arm — EPDM stays pinned for first 100 steps then free
        print("  Retracting to place_ready...")
        for i in range(500):
            d.ctrl[:] = kf[3]          # move arm to place_ready
            # Keep pinning EPDM for first 200 steps then let physics take over
            if i < 200:
                d.qpos[fj_adr:fj_adr+3]   = EPDM_PLACE_POS
                d.qpos[fj_adr+3:fj_adr+7] = [1, 0, 0, 0]
                d.qvel[fv_adr:fv_adr+6]   = 0
            mujoco.mj_step(m, d)
            v.sync()

        print("  Returning to home...")
        move_to(m, d, v, kf[0], "home")

        epdm_final = d.qpos[fj_adr:fj_adr+3].copy()
        print(f"\n  EPDM final position: {np.round(epdm_final, 4)}")
        print(f"  Target:              {EPDM_PLACE_POS}")
        print(f"  Error:               {np.round(epdm_final - EPDM_PLACE_POS, 4)}")

        print("\n[STEP 7 COMPLETE] Full sequence done!")
        print("  - Is EPDM sitting flat inside the case?")
        print("  - Did the arm retract cleanly?")
        print("  Holding... close viewer to exit.")

        while v.is_running():
            mujoco.mj_step(m, d)
            v.sync()


if __name__ == "__main__":
    main()