"""
station1_controller.py — All 5 parts: EPDM → Battery → PCB → Sound Foam → Alu Plate

Key behaviour: conveyor advances ALL trailing parts simultaneously while the arm
carries the current part, so the next part arrives at the grasp site by the time
the arm returns to pick_ready. No idle belt pauses.

Run from repo root:
    mjpython controllers/station1_controller.py
"""

import mujoco
import mujoco.viewer
import numpy as np

MODEL_PATH   = "models/stations/station1.xml"
SETTLE_TOL   = 0.015
SETTLE_STEPS = 50

# ── Part definitions (assembly order) ─────────────────────────────────────
PARTS = [
    dict(name="EPDM",       joint="s1_epdm_jnt",
         belt_z=0.82575, grasp_z_offset=0.00075,
         place_xy=np.array([0.000, 0.030]), place_kf="place_1",
         pick_kf="pick_down"),
    dict(name="Battery",    joint="s1_battery_jnt",
         belt_z=0.828,   grasp_z_offset=0.003,
         place_xy=np.array([0.025, 0.030]), place_kf="place_2",
         pick_kf=None),
    dict(name="PCB",        joint="s1_pcb_jnt",
         belt_z=0.8258,  grasp_z_offset=0.0008,
         place_xy=np.array([0.000, 0.030]), place_kf="place_3",
         pick_kf=None),
    dict(name="Sound Foam", joint="s1_sound_foam_jnt",
         belt_z=0.8255,  grasp_z_offset=0.00075,
         place_xy=np.array([0.000, 0.030]), place_kf="place_4",
         pick_kf=None),
    dict(name="Alu Plate",  joint="s1_alu_plate_jnt",
         belt_z=0.82575, grasp_z_offset=0.001,
         place_xy=np.array([0.000, 0.030]), place_kf="place_5",
         pick_kf=None),
]

INITIAL_BELT_X = {
    "s1_epdm_jnt":        0.00,
    "s1_battery_jnt":    -0.43,
    "s1_pcb_jnt":        -0.68,
    "s1_sound_foam_jnt": -1.04,
    "s1_alu_plate_jnt":  -1.40,
}

PICK_IK_SEED  = np.array([-1.684, -1.1889, 1.7115, -2.2357, -1.5915, 0.0])
BELT_STEPS    = 450    # smooth-step slide duration (~0.9s at 2ms timestep)


# ── BeltSlide ──────────────────────────────────────────────────────────────

class BeltSlide:
    """Smooth-step slide of one part along the belt X axis."""
    def __init__(self, fj_adr, fv_adr, start_x, target_x, belt_y, belt_z, total_steps, name):
        self.fj_adr = fj_adr;  self.fv_adr = fv_adr
        self.start_x = start_x;  self.target_x = target_x
        self.belt_y = belt_y;  self.belt_z = belt_z
        self.total_steps = total_steps;  self.name = name
        self.step = 0;  self.done = False

    def tick(self, d):
        if self.done:
            return
        t      = min(self.step / max(self.total_steps - 1, 1), 1.0)
        t_ease = t * t * (3 - 2 * t)
        d.qpos[self.fj_adr]     = self.start_x + (self.target_x - self.start_x) * t_ease
        d.qpos[self.fj_adr + 1] = self.belt_y
        d.qpos[self.fj_adr + 2] = self.belt_z
        d.qpos[self.fj_adr+3:self.fj_adr+7] = [1, 0, 0, 0]
        d.qvel[self.fv_adr:self.fv_adr+6]   = 0
        self.step += 1
        if self.step >= self.total_steps:
            self.done = True


# ── Helpers ────────────────────────────────────────────────────────────────

def get_ctrl(m, kf_index):
    ctrl = np.zeros(m.nu); ctrl[:] = m.key_ctrl[kf_index, :m.nu]; return ctrl

def get_joint_info(m, name):
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
    return m.jnt_qposadr[jid], m.jnt_dofadr[jid]

def get_tip(m, d):
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripper_tip")
    mujoco.mj_forward(m, d)
    return d.site_xpos[sid].copy()

def place_tip_z_fk(m, ctrl6):
    """FK tip Z for given arm joints (uses a throwaway MjData)."""
    d_ref = mujoco.MjData(m); d_ref.qpos[:6] = ctrl6
    mujoco.mj_forward(m, d_ref)
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripper_tip")
    return d_ref.site_xpos[sid][2]

def solve_ik(m, d, target_xyz, seed_joints, max_iter=500, tol=5e-4):
    d_ik = mujoco.MjData(m); d_ik.qpos[:] = d.qpos[:]; d_ik.qpos[:6] = seed_joints
    sid  = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripper_tip")
    jacp = np.zeros((3, m.nv)); err = 999.0
    for i in range(max_iter):
        mujoco.mj_forward(m, d_ik)
        ev = target_xyz - d_ik.site_xpos[sid]; err = np.linalg.norm(ev)
        if err < tol: print(f"    IK converged {i} iters, err={err*1000:.2f}mm"); break
        mujoco.mj_jacSite(m, d_ik, jacp, None, sid)
        J = jacp[:, :6]; d_ik.qpos[:6] += J.T @ np.linalg.solve(J@J.T + 1e-4*np.eye(3), ev) * 0.5
        d_ik.qpos[:6] = np.clip(d_ik.qpos[:6], -2*np.pi, 2*np.pi)
    else:
        print(f"    IK did not converge (err={err*1000:.2f}mm)")
    return d_ik.qpos[:6].copy()


def slerp_quat(q0, q1, t):
    """Spherical linear interpolation between two unit quaternions."""
    dot = np.clip(np.dot(q0, q1), -1.0, 1.0)
    if dot < 0:          # ensure shortest path
        q1 = -q1; dot = -dot
    if dot > 0.9995:     # nearly identical — linear interpolation is fine
        return q0 + t * (q1 - q0)
    theta0 = np.arccos(dot)
    theta  = theta0 * t
    sin0   = np.sin(theta0)
    return (np.sin(theta0 - theta) / sin0) * q0 + (np.sin(theta) / sin0) * q1


def pcb_tilt_and_insert(m, d, v, fj_adr, fv_adr, carry_offset,
                         hover_ctrl, place_ctrl, slides=None):
    """
    PCB-specific placement:
      1. Carry to hover pose (100mm above case, flat)
      2. Tilt PCB 20° around Y (-X edge down) over 60 steps
      3. Dwell 5s for JST connector handoff
      4. Insert: simultaneously descend arm + rotate PCB back to flat (200 steps)
    carry_offset held constant throughout (Option A).
    """
    Q_FLAT = np.array([1.0, 0.0, 0.0, 0.0])
    Q_TILT = np.array([0.9962, 0.0, -0.0872, 0.0])   # -10° around Y

    # ── Phase 1: carry flat to hover pose ─────────────────────────────
    carry_to(m, d, v, fj_adr, fv_adr, carry_offset,
             hover_ctrl, "  carry → PCB hover (flat)", slides=slides)

    # ── Phase 2: tilt PCB to 20° while arm holds at hover ─────────────
    print("  Tilting PCB 20° (-X edge down)...")
    for i in range(60):
        t      = (i + 1) / 60
        t_ease = t * t * (3 - 2 * t)
        q      = slerp_quat(Q_FLAT, Q_TILT, t_ease)
        d.ctrl[:] = hover_ctrl
        tip = get_tip(m, d)
        d.qpos[fj_adr:fj_adr+3]   = tip + carry_offset
        d.qpos[fj_adr+3:fj_adr+7] = q
        d.qvel[fv_adr:fv_adr+6]   = 0
        mujoco.mj_step(m, d); v.sync()
    print("  ✓ PCB tilted")

    # ── Phase 3: dwell 5 seconds ───────────────────────────────────────
    DWELL = 2500
    print(f"  [HOLD] Waiting for JST connector — 5s dwell ({DWELL} steps)...")
    for _ in range(DWELL):
        d.ctrl[:] = hover_ctrl
        tip = get_tip(m, d)
        d.qpos[fj_adr:fj_adr+3]   = tip + carry_offset
        d.qpos[fj_adr+3:fj_adr+7] = Q_TILT
        d.qvel[fv_adr:fv_adr+6]   = 0
        mujoco.mj_step(m, d); v.sync()
    print("  ✓ JST dwell complete")

    # ── Phase 4: insertion — descend + rotate to flat simultaneously ───
    INSERT = 200
    print(f"  Inserting PCB ({INSERT} steps, tilt → flat + descend)...")
    start_joints = d.qpos[:6].copy()
    end_joints   = place_ctrl[:6]
    for i in range(INSERT):
        t      = (i + 1) / INSERT
        t_ease = t * t * (3 - 2 * t)
        # Interpolate arm joints
        interp_joints = start_joints + (end_joints - start_joints) * t_ease
        d.ctrl[:6] = interp_joints
        tip = get_tip(m, d)
        d.qpos[fj_adr:fj_adr+3]   = tip + carry_offset
        # Rotate PCB from tilt back to flat
        q = slerp_quat(Q_TILT, Q_FLAT, t_ease)
        d.qpos[fj_adr+3:fj_adr+7] = q
        d.qvel[fv_adr:fv_adr+6]   = 0
        mujoco.mj_step(m, d); v.sync()
    print("  ✓ PCB inserted flat")

def move_to(m, d, v, ctrl, label, slides=None):
    d.ctrl[:] = ctrl; consecutive = 0; steps = 0
    while consecutive < SETTLE_STEPS:
        if slides:
            for s in slides: s.tick(d)
        mujoco.mj_step(m, d); v.sync()
        err = np.max(np.abs(d.qpos[:6] - ctrl[:6]))
        consecutive = consecutive + 1 if err < SETTLE_TOL else 0
        steps += 1
        if steps > 8000: print(f"  WARNING: {label} timed out"); break
    print(f"  ✓ {label} (err={np.max(np.abs(d.qpos[:6]-ctrl[:6])):.4f}rad, {steps} steps)")

def carry_to(m, d, v, fj_adr, fv_adr, carry_offset, ctrl, label, slides=None):
    d.ctrl[:] = ctrl; consecutive = 0; steps = 0
    while consecutive < SETTLE_STEPS:
        if slides:
            for s in slides: s.tick(d)
        tip = get_tip(m, d)
        d.qpos[fj_adr:fj_adr+3]   = tip + carry_offset
        d.qpos[fj_adr+3:fj_adr+7] = [1, 0, 0, 0]
        d.qvel[fv_adr:fv_adr+6]   = 0
        mujoco.mj_step(m, d); v.sync()
        err = np.max(np.abs(d.qpos[:6] - ctrl[:6]))
        consecutive = consecutive + 1 if err < SETTLE_TOL else 0
        steps += 1
        if steps > 8000: print(f"  WARNING: {label} timed out"); break
    print(f"  ✓ {label} (err={np.max(np.abs(d.qpos[:6]-ctrl[:6])):.4f}rad, {steps} steps)")

def release_and_retract(m, d, v, fj_adr, fv_adr, carry_offset, hold_ctrl, retract_ctrl, label):
    for _ in range(100):
        d.ctrl[:] = hold_ctrl
        tip = get_tip(m, d)
        d.qpos[fj_adr:fj_adr+3]   = tip + carry_offset
        d.qpos[fj_adr+3:fj_adr+7] = [1, 0, 0, 0]
        d.qvel[fv_adr:fv_adr+6]   = 0
        mujoco.mj_step(m, d); v.sync()
    d.qvel[fv_adr:fv_adr+6] = 0
    for _ in range(500):
        d.ctrl[:] = retract_ctrl
        mujoco.mj_step(m, d); v.sync()
    print(f"  ✓ {label} released, arm retracted")

def make_slides(trailing_parts, advance_dx):
    slides = []
    for pi in trailing_parts:
        cur_x = pi["current_belt_x"]
        tgt_x = cur_x + advance_dx
        slides.append(BeltSlide(
            fj_adr=pi["fj_adr"], fv_adr=pi["fv_adr"],
            start_x=cur_x, target_x=tgt_x,
            belt_y=pi["belt_y"], belt_z=pi["belt_z"],
            total_steps=BELT_STEPS, name=pi["name"]
        ))
        pi["current_belt_x"] = tgt_x
    return slides


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print(f"Loading: {MODEL_PATH}\n")
    m = mujoco.MjModel.from_xml_path(MODEL_PATH)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)

    kf     = {i: get_ctrl(m, i) for i in range(m.nkey)}
    kf_idx = {m.key(i).name: i for i in range(m.nkey)}
    print("Keyframes:", list(kf_idx.keys()))

    # Resolve joint addresses and belt Y from initial state
    parts_rt = []
    for p in PARTS:
        fj_adr, fv_adr = get_joint_info(m, p["joint"])
        parts_rt.append({
            **p,
            "fj_adr":         fj_adr,
            "fv_adr":         fv_adr,
            "belt_y":         d.qpos[fj_adr + 1],
            "current_belt_x": INITIAL_BELT_X[p["joint"]],
        })

    with mujoco.viewer.launch_passive(m, d) as v:
        v.cam.lookat[:] = [0.0, 0.05, 0.83]
        v.cam.distance  = 1.0
        v.cam.elevation = -20
        v.cam.azimuth   = 150

        move_to(m, d, v, kf[kf_idx["home"]], "home")

        for step_idx, part in enumerate(parts_rt):
            fj_adr = part["fj_adr"]; fv_adr = part["fv_adr"]
            print()
            print("=" * 60)
            print(f"[STEP {step_idx+1}/{len(parts_rt)}] {part['name']}")
            print("=" * 60)

            # ── PICK ──────────────────────────────────────────────────────
            move_to(m, d, v, kf[kf_idx["pick_ready"]], "  pick_ready")

            if part["pick_kf"]:
                pick_ctrl = kf[kf_idx[part["pick_kf"]]]
                move_to(m, d, v, pick_ctrl, f"  pick_down (kf)")
            else:
                grasp_xyz = np.array([
                    part["current_belt_x"],
                    part["belt_y"],
                    part["belt_z"] + part["grasp_z_offset"]
                ])
                print(f"  IK pick → {np.round(grasp_xyz, 4)}")
                pick_joints  = solve_ik(m, d, grasp_xyz, seed_joints=PICK_IK_SEED)
                pick_ctrl    = np.zeros(m.nu); pick_ctrl[:6] = pick_joints
                move_to(m, d, v, pick_ctrl, "  pick_down (IK)")

            # ── ATTACH ────────────────────────────────────────────────────
            tip          = get_tip(m, d)
            carry_offset = d.qpos[fj_adr:fj_adr+3].copy() - tip
            print(f"  carry_offset: {np.round(carry_offset, 4)}")
            attach_steps = 60
            part_start = d.qpos[fj_adr:fj_adr+3].copy()  # part's position before attach
            for i in range(attach_steps):
                t = (i + 1) / attach_steps
                t_ease = t * t * (3 - 2 * t)              # smooth-step: no snap on frame 1
                d.ctrl[:] = pick_ctrl
                tip = get_tip(m, d)
                target = tip + carry_offset
                d.qpos[fj_adr:fj_adr+3]   = part_start + (target - part_start) * t_ease
                d.qpos[fj_adr+3:fj_adr+7] = [1, 0, 0, 0]
                d.qvel[fv_adr:fv_adr+6]   = 0
                mujoco.mj_step(m, d); v.sync()
            print(f"  ✓ {part['name']} attached")

            # ── BELT ADVANCE (starts now, runs during carry) ───────────────
            trailing = parts_rt[step_idx + 1:]
            if trailing:
                advance_dx = 0.00 - trailing[0]["current_belt_x"]
                print(f"  [BELT] {len(trailing)} part(s) → +{advance_dx:.3f}m  ({', '.join(p['name'] for p in trailing)})")
                slides = make_slides(trailing, advance_dx)
            else:
                slides = []

            # ── CARRY + PLACE (belt slides in parallel) ────────────────────
            carry_to(m, d, v, fj_adr, fv_adr, carry_offset,
                     kf[kf_idx["pick_ready"]],  "  lift → pick_ready",  slides=slides)
            carry_to(m, d, v, fj_adr, fv_adr, carry_offset,
                     kf[kf_idx["place_ready"]], "  swing → place_ready", slides=slides)

            ref_z      = place_tip_z_fk(m, kf[kf_idx[part["place_kf"]]][:6])
            place_tgt  = np.array([
                part["place_xy"][0] - carry_offset[0],
                part["place_xy"][1] - carry_offset[1],
                ref_z
            ])
            print(f"  IK place tip → {np.round(place_tgt, 4)}")
            place_joints = solve_ik(m, d, place_tgt, seed_joints=kf[kf_idx[part["place_kf"]]][:6])
            place_ctrl   = np.zeros(m.nu); place_ctrl[:6] = place_joints

            if part["name"] == "PCB":
                # Hover 100mm above place, +20mm X for -X edge clearance during tilt
                hover_tgt = np.array([place_tgt[0] + 0.02, place_tgt[1], ref_z + 0.05])
                print(f"  IK hover tip → {np.round(hover_tgt, 4)}")
                hover_joints = solve_ik(m, d, hover_tgt,
                                        seed_joints=kf[kf_idx["place_ready"]][:6])
                hover_ctrl   = np.zeros(m.nu); hover_ctrl[:6] = hover_joints

                pcb_tilt_and_insert(m, d, v, fj_adr, fv_adr, carry_offset,
                                    hover_ctrl=hover_ctrl,
                                    place_ctrl=place_ctrl,
                                    slides=slides)
            else:
                carry_to(m, d, v, fj_adr, fv_adr, carry_offset,
                         place_ctrl, f"  descend → place_{step_idx+1}", slides=slides)

            # ── RELEASE ───────────────────────────────────────────────────
            release_and_retract(m, d, v, fj_adr, fv_adr, carry_offset,
                                hold_ctrl=place_ctrl,
                                retract_ctrl=kf[kf_idx["place_ready"]],
                                label=part["name"])
            print(f"  {part['name']} final: {np.round(d.qpos[fj_adr:fj_adr+3], 4)}")

        print()
        print("[DONE] Returning home...")
        move_to(m, d, v, kf[kf_idx["home"]], "home")
        print("=" * 60)
        print("All 5 parts placed. Holding viewer — close to exit.")
        print("=" * 60)

        while v.is_running():
            mujoco.mj_step(m, d); v.sync()


if __name__ == "__main__":
    main()