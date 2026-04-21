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
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cycle_timer import CycleTimer
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
    # Alu Plate placed by pin gripper after tool change (see Phase 7)
]

# Alu plate definition used in Phase 7
ALU_PLATE = dict(name="Alu Plate", joint="s1_alu_plate_jnt",
                 belt_z=0.82575, grasp_z_offset=0.001,
                 place_xy=np.array([0.000, 0.030]), place_kf="place_5",
                 pick_kf=None)

INITIAL_BELT_X = {
    "s1_epdm_jnt":        0.00,
    "s1_battery_jnt":    -0.43,
    "s1_pcb_jnt":        -0.68,
    "s1_sound_foam_jnt": -1.04,
    "s1_alu_plate_jnt":  -1.40,
}

PICK_IK_SEED  = np.array([-1.684, -1.1889, 1.7115, -2.2357, -1.5915, 0.0])
BELT_STEPS    = 450    # smooth-step slide duration (~0.9s at 2ms timestep)

# ── Tool change & screwdriving constants ───────────────────────────────────
SCREW_NAMES = ["FL", "FR", "RL", "RR"]

SCREW_INSTALL_POS = {
    "FL": np.array([-0.1515, -0.0195, 0.822]),
    "FR": np.array([ 0.1545, -0.0195, 0.822]),
    "RL": np.array([-0.1515,  0.0795, 0.822]),
    "RR": np.array([ 0.1545,  0.0795, 0.822]),
}

DOCK_VACUUM_W2  = np.array([-0.38, -0.240, 0.882])  # gripper_tip at body origin
DOCK_PIN_W2     = np.array([-0.38, -0.150, 0.882])  # gripper_tip at body origin
DOCK_SCREW_W2   = np.array([-0.38, -0.060, 0.882])  # gripper_tip at SD body origin
WINGMAN_SLIDE_X = 0.040
DOCK_IK_SEED    = np.array([-0.8, -1.8, 2.0, -1.8, -1.5, 0.0])
SCREW_IK_SEED   = np.array([-1.5, -1.5, 2.0, -2.0, -1.5, 0.0])
ROTATE_STEPS    = 300


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



def get_body_jnt(m, body_name):
    """Return (fj_adr, fv_adr) for the freejoint of a body."""
    bid  = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
    jid  = m.body_jntadr[bid]
    return m.jnt_qposadr[jid], m.jnt_dofadr[jid]


def get_site_world(m, d, site_name):
    mujoco.mj_forward(m, d)
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site_name)
    return d.site_xpos[sid].copy()


def ik_move(m, d, v, target_xyz, seed, label):
    """IK to target, then move_to."""
    joints = solve_ik(m, d, target_xyz, seed_joints=seed)
    ctrl   = np.zeros(m.nu); ctrl[:6] = joints
    move_to(m, d, v, ctrl, label)
    return ctrl


def wingman_put(m, d, v, W2_world, holder_world_pos, tool_fj_adr, tool_fv_adr, label):
    """
    Put current tool into holder:
      W4 (current) → slide back to W2 → lift to W1 → tool teleports to holder.
    Simplified: just IK to W2, then IK to W1 (W2+100mm Z), then teleport.
    """
    print(f"  [PUT {label}] Moving to W2...")
    # W1: 80mm above W2
    W1_world = W2_world + np.array([0, 0, 0.080])
    ik_move(m, d, v, W1_world, DOCK_IK_SEED, f"  {label} → W1 approach")
    # Descend to W2
    W2_ctrl = ik_move(m, d, v, W2_world, DOCK_IK_SEED, f"  {label} → W2 engage")
    # Slide to W4 (put: slide −X to re-enter holder, actually just hold W2)
    # For put: hold W2 briefly, then lift away
    for _ in range(60):
        d.ctrl[:] = W2_ctrl
        mujoco.mj_step(m, d); v.sync()
    # Teleport tool to holder resting position
    d.qpos[tool_fj_adr:tool_fj_adr+3] = holder_world_pos
    d.qpos[tool_fj_adr+3:tool_fj_adr+7] = [1, 0, 0, 0]
    d.qvel[tool_fv_adr:tool_fv_adr+6] = 0
    mujoco.mj_forward(m, d)
    print(f"  ✓ {label} returned to dock")


def wingman_get(m, d, v, W2_world, tool_fj_adr, tool_fv_adr, label):
    """
    Get tool from holder:
      Approach W1 → descend W2 → slide to W4 → tool teleports to gripper tip.
    Returns the ctrl for W4 position.
    """
    print(f"  [GET {label}] Approaching...")
    W1_world = W2_world + np.array([0, 0, 0.080])
    ik_move(m, d, v, W1_world, DOCK_IK_SEED, f"  {label} → W1")
    W2_ctrl = ik_move(m, d, v, W2_world, DOCK_IK_SEED, f"  {label} → W2")
    # Slide to W4 (+X)
    W4_world = W2_world + np.array([WINGMAN_SLIDE_X, 0, 0])
    W4_ctrl  = ik_move(m, d, v, W4_world, DOCK_IK_SEED, f"  {label} → W4 lock")
    # Tool is now "attached" — keep it at gripper tip via carry
    print(f"  ✓ {label} picked up")
    return W4_ctrl


def carry_tool_to(m, d, v, tool_fj_adr, tool_fv_adr, carry_offset, ctrl, label):
    """Like carry_to but for tool bodies (same pattern)."""
    carry_to(m, d, v, tool_fj_adr, tool_fv_adr, carry_offset, ctrl, label)


def drive_screw(m, d, v, screw_name, screw_fj_adr, screw_fv_adr):
    """
    Full screw driving cycle:
      1. IK to screw box pick site
      2. Screw teleports to screwdriver tip
      3. IK carry to hover above boss
      4. IK carry descend to boss
      5. Rotate wrist 3 turns
      6. Screw teleports to installed position
      7. Retract
    """
    boss_world = SCREW_INSTALL_POS[screw_name]
    hover_z    = boss_world[2] + 0.050

    print(f"\n  [SCREW {screw_name}]")

    # ── Pick from box ──────────────────────────────────────────────
    pick_target = np.array([0.28, -0.18, 0.854])   # gripper_tip 0.046m above screwdriver_tip at box
    pick_ctrl = ik_move(m, d, v, pick_target, SCREW_IK_SEED,
                        f"  SD → screw box pick ({screw_name})")

    # Teleport screw to screwdriver tip
    tip = get_tip(m, d) if True else None
    mujoco.mj_forward(m, d)
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "screwdriver_tip")
    tip_pos = d.site_xpos[sid].copy()
    # Screw sits 8mm below tip (shaft pointing down)
    # SD tip (screwdriver_tip site) is at body_origin + [0,0,-0.102]
    # Use screwdriver_tip site directly
    mujoco.mj_forward(m, d)
    sd_tip_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "screwdriver_tip")
    sd_tip_pos = d.site_xpos[sd_tip_sid].copy()
    # screwdriver_tip aligns with screw head top (z+0.012 above body origin)
    # screw body origin = sd_tip_pos - [0,0,0.012]
    screw_carried_pos = sd_tip_pos - np.array([0, 0, 0.012])
    d.qpos[screw_fj_adr:screw_fj_adr+3] = screw_carried_pos
    d.qpos[screw_fj_adr+3:screw_fj_adr+7] = [1, 0, 0, 0]
    d.qvel[screw_fv_adr:screw_fv_adr+6] = 0
    screw_offset = screw_carried_pos - sd_tip_pos  # = [0,0,-0.012]
    print(f"  ✓ {screw_name} picked from box")

    # ── Carry to hover above boss ───────────────────────────────────
    # screwdriver_tip is 0.046m further in +Y than gripper_tip (world -Z)
    # To place SD tip at hover_z: gripper_tip at hover_z + 0.046
    hover_target = np.array([boss_world[0],
                              boss_world[1],
                              hover_z + 0.046])
    hover_ctrl = ik_move(m, d, v, hover_target, SCREW_IK_SEED,
                         f"  SD → hover {screw_name}")
    # Keep screw + SD together during move
    for _ in range(80):
        d.ctrl[:] = hover_ctrl
        mujoco.mj_forward(m, d)
        sid2 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "screwdriver_tip")
        tip2 = d.site_xpos[sid2].copy()
        d.qpos[screw_fj_adr:screw_fj_adr+3] = tip2 + screw_offset
        d.qvel[screw_fv_adr:screw_fv_adr+6] = 0
        mujoco.mj_step(m, d); v.sync()

    # ── Descend to boss ─────────────────────────────────────────────
    # Want SD tip at boss: gripper_tip 0.046m above boss
    insert_target = np.array([boss_world[0],
                               boss_world[1],
                               boss_world[2] + 0.046])
    insert_ctrl = ik_move(m, d, v, insert_target, SCREW_IK_SEED,
                          f"  SD → insert {screw_name}")
    for _ in range(80):
        d.ctrl[:] = insert_ctrl
        mujoco.mj_forward(m, d)
        sid2 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "screwdriver_tip")
        tip2 = d.site_xpos[sid2].copy()
        d.qpos[screw_fj_adr:screw_fj_adr+3] = tip2 + screw_offset
        d.qvel[screw_fv_adr:screw_fv_adr+6] = 0
        mujoco.mj_step(m, d); v.sync()

    # ── Rotate wrist 3 turns ────────────────────────────────────────
    print(f"  Rotating {screw_name}...")
    start_w3 = d.qpos[5]
    for i in range(ROTATE_STEPS):
        t = i / ROTATE_STEPS
        d.ctrl[5] = start_w3 + t * (3 * 2 * np.pi)
        mujoco.mj_forward(m, d)
        sid2 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "screwdriver_tip")
        tip2 = d.site_xpos[sid2].copy()
        d.qpos[screw_fj_adr:screw_fj_adr+3] = tip2 + screw_offset
        d.qvel[screw_fv_adr:screw_fv_adr+6] = 0
        mujoco.mj_step(m, d); v.sync()

    # ── Snap screw to installed position ────────────────────────────
    # Installed: shaft tip 5mm into boss, head sits 2mm proud of plate
    d.qpos[screw_fj_adr:screw_fj_adr+3] = boss_world - np.array([0, 0, 0.005])
    d.qpos[screw_fj_adr+3:screw_fj_adr+7] = [1, 0, 0, 0]
    d.qvel[screw_fv_adr:screw_fv_adr+6] = 0
    print(f"  ✓ {screw_name} installed")

    # ── Retract to hover ────────────────────────────────────────────
    d.ctrl[5] = start_w3   # reset wrist rotation
    move_to(m, d, v, hover_ctrl, f"  SD retract from {screw_name}")


def set_geom_rgba(m, d, geom_name, rgba):
    """Set a geom's rgba (alpha=0 hides it, alpha=1 shows it)."""
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    m.geom_rgba[gid] = rgba

VACUUM_GEOMS     = ["gripper_flange","gripper_body","gripper_accent","gripper_cup"]
PIN_GEOMS        = ["pin_body_blk","pin_left","pin_right"]
SD_GEOMS         = ["sd_flange","sd_spring","sd_body","sd_shaft","sd_bit"]
TOOL_SWAP_STEPS  = 200   # ~0.4s fade for tool swap animation

def show_tool(m, tool, d=None, v=None):
    """Fade-swap gripper visuals. If d+v provided, animates over TOOL_SWAP_STEPS."""
    TOOLS = {"vacuum": VACUUM_GEOMS, "pin": PIN_GEOMS, "screwdriver": SD_GEOMS}
    show_geoms = TOOLS[tool]
    hide_geoms = [g for geoms in TOOLS.values() for g in geoms if g not in show_geoms]

    if d is None or v is None:
        # Instant toggle (no viewer)
        for g in show_geoms:
            gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g)
            m.geom_rgba[gid][3] = 1.0
        for g in hide_geoms:
            gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g)
            m.geom_rgba[gid][3] = 0.0
        return

    # Animated cross-fade
    for step in range(TOOL_SWAP_STEPS):
        t = (step + 1) / TOOL_SWAP_STEPS
        t_e = t * t * (3 - 2 * t)   # smooth-step
        for g in show_geoms:
            gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g)
            m.geom_rgba[gid][3] = t_e
        for g in hide_geoms:
            gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g)
            m.geom_rgba[gid][3] = 1.0 - t_e
        mujoco.mj_step(m, d); v.sync()

def hide_dock_visual(m, tool):
    """Hide the static dock visual when robot picks up that tool."""
    if tool == "pin":
        for g in ["dpv_body","dpv_pin_l","dpv_pin_r"]:
            set_geom_rgba(m, None, g, [0.15, 0.15, 0.15, 0.0])
    elif tool == "screwdriver":
        for g in ["dsv_flange","dsv_spring","dsv_body","dsv_shaft","dsv_bit"]:
            set_geom_rgba(m, None, g, [0.5, 0.5, 0.5, 0.0])

def show_dock_visual(m, tool):
    """Show the static dock visual when robot returns that tool."""
    if tool == "pin":
        set_geom_rgba(m, None, "dpv_body",  [0.15, 0.15, 0.15, 1.0])
        set_geom_rgba(m, None, "dpv_pin_l", [0.7,  0.7,  0.7,  1.0])
        set_geom_rgba(m, None, "dpv_pin_r", [0.7,  0.7,  0.7,  1.0])
    elif tool == "screwdriver":
        for g, c in [("dsv_flange",[0.278,0.278,0.278,1]),("dsv_spring",[0.7,0.7,0.7,1]),
                     ("dsv_body",[1,0.45,0,1]),("dsv_shaft",[0.8,0.8,0.82,1]),("dsv_bit",[0.5,0.5,0.5,1])]:
            set_geom_rgba(m, None, g, c)


def tool_change_and_screw(m, d, v, kf, kf_idx, parts_rt, timer):
    """
    Phases 6-9:
      6. Tool change: vacuum → pin gripper
      7. Pick & place aluminum plate (reuse carry logic)
      8. Tool change: pin → screwdriver
      9. Drive 4 screws
    """
    # ── Resolve freejoint addresses for screws ────────────────────────────
    screws = {name: get_body_jnt(m, f"screw_{name}") for name in SCREW_NAMES}

    # ── Phase 6: PUT vacuum gripper, GET pin gripper ──────────────────────
    print()
    print("=" * 60)
    print("[PHASE 6] Tool change: Vacuum → Pin Gripper")
    print("=" * 60)

    W1 = DOCK_VACUUM_W2 + np.array([0, 0, 0.080])
    ik_move(m, d, v, W1, DOCK_IK_SEED, "  vacuum W1 approach")
    ik_move(m, d, v, DOCK_VACUUM_W2, DOCK_IK_SEED, "  vacuum W2 engage")
    W4v = DOCK_VACUUM_W2 + np.array([WINGMAN_SLIDE_X, 0, 0])
    ik_move(m, d, v, W4v, DOCK_IK_SEED, "  vacuum W4 slide out")

    W1p = DOCK_PIN_W2 + np.array([0, 0, 0.080])
    ik_move(m, d, v, W1p, DOCK_IK_SEED, "  pin W1 approach")
    ik_move(m, d, v, DOCK_PIN_W2, DOCK_IK_SEED, "  pin W2 engage")
    W4p = DOCK_PIN_W2 + np.array([WINGMAN_SLIDE_X, 0, 0])
    ik_move(m, d, v, W4p, DOCK_IK_SEED, "  pin W4 lock")

    # Toggle visuals: hide vacuum, show pin gripper, hide dock pin visual
    show_tool(m, "pin", d, v)
    hide_dock_visual(m, "pin")
    print("  ✓ Pin gripper active")
    timer.mark(d, "Tool change: vacuum → pin")

    # ── Phase 7: Pick & place aluminum plate with pin gripper ─────────────
    print()
    print("=" * 60)
    print("[PHASE 7] Pick & place aluminum plate (pin gripper)")
    print("=" * 60)

    # Alu plate = parts_rt[4], belt-advanced to x=0 during 4-part assembly
    plate = parts_rt[4]
    fj_adr = plate["fj_adr"]; fv_adr = plate["fv_adr"]

    move_to(m, d, v, kf[kf_idx["pick_ready"]], "  pick_ready")

    grasp_xyz = np.array([plate["current_belt_x"],
                          plate["belt_y"],
                          plate["belt_z"] + plate["grasp_z_offset"]])
    print(f"  IK pick plate → {np.round(grasp_xyz, 4)}")
    pick_joints = solve_ik(m, d, grasp_xyz, seed_joints=PICK_IK_SEED)
    pick_ctrl   = np.zeros(m.nu); pick_ctrl[:6] = pick_joints
    move_to(m, d, v, pick_ctrl, "  pick plate (IK)")

    # Attach plate
    tip = get_tip(m, d)
    carry_offset = d.qpos[fj_adr:fj_adr+3].copy() - tip
    for i in range(60):
        t = (i+1)/60; t_e = t*t*(3-2*t)
        d.ctrl[:] = pick_ctrl; tip = get_tip(m, d)
        d.qpos[fj_adr:fj_adr+3] = d.qpos[fj_adr:fj_adr+3]*(1-t_e) + (tip+carry_offset)*t_e
        d.qpos[fj_adr+3:fj_adr+7] = [1,0,0,0]
        d.qvel[fv_adr:fv_adr+6] = 0
        # Keep pin gripper at tip too
        mujoco.mj_step(m, d); v.sync()

    # Carry to place
    carry_to(m, d, v, fj_adr, fv_adr, carry_offset,
             kf[kf_idx["pick_ready"]], "  lift → pick_ready")
    carry_to(m, d, v, fj_adr, fv_adr, carry_offset,
             kf[kf_idx["place_ready"]], "  swing → place_ready")

    ref_z     = place_tip_z_fk(m, kf[kf_idx["place_5"]][:6])
    place_tgt = np.array([plate["place_xy"][0] - carry_offset[0],
                          plate["place_xy"][1] - carry_offset[1], ref_z])
    place_joints = solve_ik(m, d, place_tgt, seed_joints=kf[kf_idx["place_5"]][:6])
    place_ctrl   = np.zeros(m.nu); place_ctrl[:6] = place_joints

    carry_to(m, d, v, fj_adr, fv_adr, carry_offset,
             place_ctrl, "  descend → place_5")
    release_and_retract(m, d, v, fj_adr, fv_adr, carry_offset,
                        hold_ctrl=place_ctrl,
                        retract_ctrl=kf[kf_idx["place_ready"]],
                        label="Alu Plate (pin)")
    print(f"  Plate final: {np.round(d.qpos[fj_adr:fj_adr+3], 4)}")
    timer.mark(d, "Plate placed (pin gripper)")

    # ── Phase 8: PUT pin gripper, GET screwdriver ─────────────────────────
    print()
    print("=" * 60)
    print("[PHASE 8] Tool change: Pin → Screwdriver")
    print("=" * 60)

    move_to(m, d, v, kf[kf_idx["home"]], "  home before tool change")

    # Put pin gripper
    W1p2 = DOCK_PIN_W2 + np.array([0, 0, 0.080])
    ik_move(m, d, v, W1p2, DOCK_IK_SEED, "  pin return W1")
    ik_move(m, d, v, DOCK_PIN_W2, DOCK_IK_SEED, "  pin return W2")
    W4pr = DOCK_PIN_W2 - np.array([WINGMAN_SLIDE_X, 0, 0])
    ik_move(m, d, v, W4pr, DOCK_IK_SEED, "  pin W4 slide back (put)")
    show_dock_visual(m, "pin")

    # Get screwdriver
    W1s = DOCK_SCREW_W2 + np.array([0, 0, 0.080])
    ik_move(m, d, v, W1s, DOCK_IK_SEED, "  SD W1 approach")
    ik_move(m, d, v, DOCK_SCREW_W2, DOCK_IK_SEED, "  SD W2 engage")
    W4s = DOCK_SCREW_W2 + np.array([WINGMAN_SLIDE_X, 0, 0])
    ik_move(m, d, v, W4s, DOCK_IK_SEED, "  SD W4 lock")

    # Toggle visuals: hide pin, show screwdriver, hide dock SD visual
    show_tool(m, "screwdriver", d, v)
    hide_dock_visual(m, "screwdriver")
    print("  ✓ Screwdriver active")
    timer.mark(d, "Tool change: pin → screwdriver")

    # ── Phase 9: Drive 4 screws ───────────────────────────────────────────
    print()
    print("=" * 60)
    print("[PHASE 9] Screwdriving — 4 screws")
    print("=" * 60)

    for screw_name in SCREW_NAMES:
        sfj, sfv = screws[screw_name]
        drive_screw(m, d, v, screw_name, sfj, sfv)
        timer.mark(d, f"Screw {screw_name} installed")

    # ── Return screwdriver to dock ────────────────────────────────────────
    print()
    print("  Returning screwdriver to dock...")
    W1s2 = DOCK_SCREW_W2 + np.array([0, 0, 0.080])
    ik_move(m, d, v, W1s2, DOCK_IK_SEED, "  SD return W1")
    W2s2_ctrl = ik_move(m, d, v, DOCK_SCREW_W2, DOCK_IK_SEED, "  SD return W2")
    for _ in range(80):
        d.ctrl[:] = W2s2_ctrl
        mujoco.mj_step(m, d); v.sync()
    show_tool(m, "vacuum", d, v)
    show_dock_visual(m, "screwdriver")
    print("  ✓ Screwdriver returned to dock")

    move_to(m, d, v, kf[kf_idx["home"]], "  final home")
    print("\n✓ All phases complete.")


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
    # Add alu plate to belt tracking (advances with belt but placed by pin gripper)
    fj_alu0, fv_alu0 = get_joint_info(m, ALU_PLATE["joint"])
    alu_belt_entry = {
        **ALU_PLATE, "fj_adr": fj_alu0, "fv_adr": fv_alu0,
        "belt_y": d.qpos[fj_alu0 + 1],
        "current_belt_x": INITIAL_BELT_X[ALU_PLATE["joint"]],
    }
    parts_rt.append(alu_belt_entry)  # index 4 — belt advance only, not placed

    with mujoco.viewer.launch_passive(m, d) as v:
        v.cam.lookat[:] = [0.0, 0.05, 0.83]
        v.cam.distance  = 1.0
        v.cam.elevation = -20
        v.cam.azimuth   = 150

        move_to(m, d, v, kf[kf_idx["home"]], "home")
        timer = CycleTimer("Station 1")
        timer.start(d)
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
            timer.mark(d, f"Place & release — {part['name']}")
        print()
        move_to(m, d, v, kf[kf_idx["home"]], "home after 5-part assembly")
        timer.mark(d, "5-part assembly complete")

        # ══════════════════════════════════════════════════════════════
        # PHASES 6-9: Tool change + plate + screwdriving
        # ══════════════════════════════════════════════════════════════
        tool_change_and_screw(m, d, v, kf, kf_idx, parts_rt, timer)

        print()
        print("=" * 60)
        print("Station 1 complete. Close viewer to exit.")
        print("=" * 60)
        timer.finish(d)
        timer.print_report(ref_key="pick_place_fast")

        while v.is_running():
            mujoco.mj_step(m, d); v.sync()



if __name__ == "__main__":
    main()