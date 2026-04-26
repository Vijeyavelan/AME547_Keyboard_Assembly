"""
station1_controller.py — EPDM -> Battery -> PCB -> Alu Plate -> Screws
Updated for new station1.xml layout (kit tray + pre-stage, no belt conveyor).

Key fix: parts are gravity-locked at their keyframe positions until picked.
carry_offset = pick_world - tip_at_pick  (not from d.qpos after physics moved things)

Actual qpos ordering (from XML body declaration order in worldbody):
  [7:14]  kit_tray_pickzone   [14:21] kit_tray_prestage
  [21:28] ps_epdm  [28:35] ps_battery  [35:42] ps_pcb  [42:49] ps_plate
  [49:56] s1_epdm  [56:63] s1_battery  [63:70] s1_pcb  [70:77] s1_alu_plate
  [77:84] screw_FL [84:91] screw_FR  [91:98] screw_RL  [98:105] screw_RR
NOTE: The XML keyframe comment has s1_ and ps_ ordering REVERSED.
The correct order is determined by body declaration order in worldbody.
ps_ bodies are declared before s1_ bodies in the XML.

Run from repo root:
    mjpython controllers/station1_controller.py
"""

import mujoco
import mujoco.viewer
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cycle_timer import CycleTimer

MODEL_PATH   = "models/stations/station1_B.xml"
SETTLE_TOL   = 0.020   # v7: avoid false timeouts at harmless ~0.018-0.019 rad residual
SETTLE_STEPS = 50
# j1 faces the dock, j2/j3 create a high 'elbow-up' arch, j4/j5 point the flange down
DOCK_VAC_SEED = np.array([-1.57, -1.57, 1.57, -1.57, -1.57, 0.0])
DOCK_PIN_SEED = np.array([-1.57, -1.57, 1.57, -1.57, -1.57, 0.0])
DOCK_SD_SEED  = np.array([-1.57, -1.57, 1.57, -1.57, -1.57, 0.0])
# ---------------------------------------------------------------------------
# PART DEFINITIONS
# ---------------------------------------------------------------------------
# World positions from keyframe (s1_ bodies at qpos indices 49-76)
PICK_WORLD = {
    "s1_epdm":      np.array([-0.3000,  0.31915, 0.81375]),  # lifted to sit on pocket
    "s1_battery":   np.array([-0.3000,  0.43630, 0.81300]),
    "s1_pcb":       np.array([-0.3000,  0.54830, 0.81200]),
    "s1_alu_plate": np.array([-0.3000,  0.68745, 0.81775]),
}

# Flange perfectly horizontal, tool pointing straight down along -Z
STRICT_DOWN_QUAT = np.array([0.70710678, -0.70710678, 0.0, 0.0])
# If the dock is rotated or the arm approach needs a specific twist:
# You can use mujoco.mju_euler2Quat to find the exact alignment.
# Tip (cup face) 3mm above part top surfaceshow
GRASP_Z_OFFSET = {
    "s1_epdm":      0.006,
    "s1_battery":   0.011,
    "s1_pcb":       0.007,
    "s1_alu_plate": 0.000,   # plate uses internal L-hook sequence instead of suction-style top grasp
}

# Where the part body origin should land in the case (world frame)
# Case: pos=(0,0,0.821), euler Z=pi  -> world_x = -body_x, world_y = -body_y
# Battery body-local X=-0.025 -> world X=+0.025
# XY offset applied to grasp target only (body stays at PICK_WORLD)
GRASP_XY_OFFSET = {
    "s1_epdm": np.array([-0.087, 0.0]),   # pick over left piece, away from cutout
}
PLACE_PART_WORLD = {
    # EPDM+Battery parallel on case floor (Z=0.813); PCB on battery top; Plate on PCB
    "s1_epdm":      np.array([ 0.000,  0.000, 0.81175]),
    "s1_battery":   np.array([ 0.025,  0.000, 0.81300]),
    "s1_pcb":       np.array([ 0.000,  0.000, 0.82200]),
    "s1_alu_plate": np.array([ -0.006061,  -0.002, 0.82900]),  # v8: exact case datum; visual mesh offset stays inside geom
}

PARTS = ["s1_epdm", "s1_battery", "s1_pcb", "s1_alu_plate"]

# Body orientation targets. The aluminum plate is placed with a 180 deg Z rotation
# so the function-row cutouts end up toward world -Y in the bottom case.
PLATE_TRAY_QUAT  = np.array([0.0, 0.0, 0.0, 1.0])  # MuJoCo wxyz, Rz(pi) in feeder tray
PLATE_FINAL_QUAT = np.array([0.0, 0.0, 0.0, 1.0])  # MuJoCo wxyz, Rz(pi) at final placement
PLACE_PART_QUAT = {
    "s1_epdm":      None,
    "s1_battery":   None,
    "s1_pcb":       None,
    "s1_alu_plate": PLATE_FINAL_QUAT,
}


# Internal cutout-locking gripper for aluminum plate.
# Pins are 6 keyboard pitches apart: +/-57.15 mm about the plate center.
PLATE_LOCK_SITE       = "plate_lock_tcp"
PLATE_PIN_LEFT_X      = -0.05715
PLATE_PIN_RIGHT_X     =  0.05715
PLATE_PIN_Y           =  0.0
PLATE_APPROACH_CLEAR  =  0.070
PLATE_INSERT_CLEAR    =  0.002   # version A logical lock: only go just under the thin plate
PLATE_RELEASE_CLEAR   =  0.000   # place at final seating height before pin retract
PLATE_SEAT_EXTRA_DOWN =  0.000   # final-pose backsolve: do not drive the carried plate below its tuned seated pose


# Visual-only internal pin actuation. We animate pin/hook geoms directly in
# model-local coordinates; plate carrying remains deterministic/logical.
PIN_LOCK_TRAVEL = 0.004
PIN_LOCK_GEOM_BASE = {
    # Dock-matched robot values: dock local Z -> robot local +Y.
    "ilock_L_stem": np.array([-0.05715, 0.0540, 0.0]),
    "ilock_L_hook": np.array([-0.06065, 0.0610, 0.0]),
    "ilock_R_stem": np.array([ 0.05715, 0.0540, 0.0]),
    "ilock_R_hook": np.array([ 0.06065, 0.0610, 0.0]),
}
PIN_LOCK_SITE_BASE = {
    # Keep the logical lock/contact sites on the visible hook-tip line, not up at the flange.
    "pin_L_tip":          np.array([-0.05715, 0.0610, 0.0]),
    "pin_R_tip":          np.array([ 0.05715, 0.0610, 0.0]),
    "pin_L_hook_contact": np.array([-0.06065, 0.0610, 0.0]),
    "pin_R_hook_contact": np.array([ 0.06065, 0.0610, 0.0]),
}

# ---------------------------------------------------------------------------
# IK SEEDS
# Robot base: (0.200, 0.400, 0.600), euler Z=pi
# pan=0 -> faces -X (tray). pan~1.107 -> faces pallet.
# ---------------------------------------------------------------------------
PICK_IK_SEED  = np.array([ 0.10, -1.60,  1.80, -1.80, -1.5708, 0.0])
PLACE_IK_SEED = np.array([ 1.10, -1.60,  1.80, -1.80, -1.5708, 0.0])
DOCK_VAC_SEED = np.array([-2.30, -1.80, 1.60, -1.40, -1.5708, 0.0])  # v7: rear dock, slots arranged along X
DOCK_PIN_SEED = np.array([-2.15, -1.80, 1.60, -1.40, -1.5708, 0.0])
DOCK_SD_SEED  = np.array([-2.00, -1.80, 1.60, -1.40, -1.5708, 0.0])
# Z targets use gripper_tip world Z. Dock cup/site top is at world Z≈0.960
# because tool_dock is at 0.870 and dock sites are local z=0.090.
DOCK_HOVER_Z  = 1.018   # tip hover above cup; wrist has clearance
DOCK_SEAT_Z   = 0.886   # tip Z when flange/tool base visually seats into dock cup
SCREW_IK_SEED  = np.array([-2.45, -1.65, 1.55, -1.45, -1.5708, 0.0])  # v7: screw box near main conveyor/right side
# Screwdriver geometry: gripper_tip is the vacuum-cup reference site at local Y=0.074.
# screwdriver_tip is at local Y=0.102. With STRICT_DOWN_QUAT, +Y points world -Z,
# so the screwdriver tip is 28 mm below gripper_tip.
SD_TIP_FROM_GRIPPER_Z = 0.028
SCREW_ENGAGE_Z        = 0.012   # v8: M3 screw body origin -> screw head/bit engagement site
SCREW_BOX_HOVER_CLEAR = 0.100
SCREW_DRIVE_CLEAR     = 0.070
SCREW_BOX_Z           = 0.806   # screw body origin when sitting in the screw box
SCREW_INSTALLED_Z     = 0.8145  # v8: head top sits at the aluminum plate top surface
# Per-boss IK seeds: j1 tuned to face each boss position from robot base
SCREW_BOSS_SEED = np.array([1.10, -1.50, 1.80, -1.85, -1.5708, 0.0])

# ---------------------------------------------------------------------------
# TOOL DOCK + SCREW BOX WORLD POSITIONS
# Layout v7:
# - Tool dock is behind/+Y of the UR5e, reserved only for tool exchange.
# - Screw box is moved close to the main conveyor/pallet side so the repeated
#   screw task follows a short feeder -> boss -> feeder path.
# Dock body: (0.280, 0.680, 0.870)
# dock_vac_W2 site body-local (-0.070,0,0.090) -> world (0.210, 0.680, 0.960)
# dock_pin_W2 site body-local ( 0.000,0,0.090) -> world (0.280, 0.680, 0.960)
# dock_sd_W2  site body-local (+0.070,0,0.090) -> world (0.350, 0.680, 0.960)
# ---------------------------------------------------------------------------
DOCK_VAC_XY     = np.array([0.160, 0.680])   # v7: wider dock slot spacing to clear pin gripper
DOCK_PIN_XY     = np.array([0.280, 0.680])   # internal-lock slot remains centered behind robot
DOCK_SD_XY      = np.array([0.400, 0.680])   # wider spacing keeps screwdriver clear too

SCREW_BOX_XY    = np.array([0.520, 0.200])   # v7: shifted +Y to clear main conveyor
SCREW_PICK_POS = {
    "FL": np.array([0.510, 0.190, SCREW_BOX_Z]),
    "FR": np.array([0.530, 0.190, SCREW_BOX_Z]),
    "RL": np.array([0.510, 0.210, SCREW_BOX_Z]),
    "RR": np.array([0.530, 0.210, SCREW_BOX_Z]),
}

# Local screw-transfer waypoint: keeps the screwdriver path between feeder and pallet,
# instead of swinging around the dock/feeder tray on the long IK branch. This is a
# gripper_tip target, not screwdriver_tip; it is intentionally high and inside the
# direct screw-box -> pallet corridor.
SCREW_LOCAL_TRANSFER = np.array([0.360, 0.120, 0.980])

# ---------------------------------------------------------------------------
# SCREW BOSS WORLD POSITIONS
# Case euler Z=pi: world_x=-body_x, world_y=-body_y
# Body-local: FL(-0.1484,-0.05315) -> world (+0.1484,+0.05315)
#             FR(+0.1484,-0.05315) -> world (-0.1484,+0.05315)
#             RL(-0.1484,+0.05315) -> world (+0.1484,-0.05315)
#             RR(+0.1484,+0.05315) -> world (-0.1484,-0.05315)
# Boss Z body-local -0.0085 -> world 0.821-0.0085=0.8125
# ---------------------------------------------------------------------------
BOSS_Z  = 0.8125
SCREW_NAMES = ["FL", "FR", "RL", "RR"]
# v8: screw targets now come from aluminum_plate.xml screw sites, with the
# plate placed at Rz(pi).  World XY = - plate-local XY.
# aluminum_plate.xml sites:
#   screw_FL=(-0.1515,-0.0495), screw_FR=(+0.1545,-0.0495),
#   screw_RL=(-0.1515,+0.0495), screw_RR=(+0.1545,+0.0495).
SCREW_INSTALL_POS = {
    "FL": np.array([ 0.128,  0.0485, BOSS_Z]),
    "FR": np.array([-0.1339,  0.0476, BOSS_Z]),
    "RL": np.array([ 0.1329, -0.0468, BOSS_Z]),
    "RR": np.array([-0.1338, -0.0472, BOSS_Z]),
}

ROTATE_STEPS = 300


def gripper_target_for_sd_tip(sd_tip_world):
    """Return gripper_tip target that places screwdriver_tip at sd_tip_world."""
    return np.array(sd_tip_world) + np.array([0.0, 0.0, SD_TIP_FROM_GRIPPER_Z])


def screw_origin_for_sd_tip(sd_tip_world):
    """Screw origin when its head engagement point is exactly on screwdriver_tip."""
    return np.array(sd_tip_world) - np.array([0.0, 0.0, SCREW_ENGAGE_Z])



# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def get_body_jnt(m, body_name):
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
    jid = m.body_jntadr[bid]
    return m.jnt_qposadr[jid], m.jnt_dofadr[jid]

def lock_part(d, fj, fv, pos, quat=None):
    d.qpos[fj:fj+3]   = pos
    d.qpos[fj+3:fj+7] = quat if quat is not None else [1, 0, 0, 0]
    d.qvel[fv:fv+6]   = 0

def solve_ik(m, d, target, seed, max_iter=600, tol=5e-4,
             target_quat=None):
    """6-DOF IK: position + orientation.

    gripper_tip site offset is pos="0 0.056 0" — tool axis is body local +Y.
    Straight-down means local +Y maps to world -Z.
    That is R_x(-90deg) = quat [0.7071, -0.7071, 0, 0] (wxyz).

    target_quat=None  → use straight-down default (normal pick/place)
    target_quat=False → position-only IK (avoid for tool dock seating)
    Orientation weighted 0.3x to avoid fighting position convergence.
    """
    ori_enabled = (target_quat is not False)
    if ori_enabled and target_quat is None:
        # R_x(-90deg): site local +Y -> world -Z (gripper straight down)
        target_quat = np.array([0.70710678, -0.70710678, 0.0, 0.0])  # wxyz

    d2 = mujoco.MjData(m)
    d2.qpos[:] = d.qpos[:]
    d2.qpos[:6] = seed
    sid  = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripper_tip")
    jacp = np.zeros((3, m.nv))
    jacr = np.zeros((3, m.nv))
    tgt_mat = None
    if ori_enabled:
        tgt_mat = np.zeros(9)
        mujoco.mju_quat2Mat(tgt_mat, target_quat)
        tgt_mat = tgt_mat.reshape(3, 3)
    # Increase orientation priority so it doesn't sacrifice tilt for position
    ORI_W = 1.0  

    # Inside the loop, tighten the orientation error check
    
    err = 999.0
    for i in range(max_iter):
        mujoco.mj_forward(m, d2)
        ep  = target - d2.site_xpos[sid]
        err = np.linalg.norm(ep)
        if ori_enabled:
            site_mat = d2.site_xmat[sid].reshape(3, 3)
            R_err = tgt_mat @ site_mat.T
            eo = 0.5 * np.array([R_err[2,1] - R_err[1,2],
                                  R_err[0,2] - R_err[2,0],
                                  R_err[1,0] - R_err[0,1]])
            ori_err = np.linalg.norm(eo)
            # Increase orientation priority so it doesn't sacrifice tilt for position

# Inside the loop, tighten the orientation error check
            if err < tol and ori_err < 0.0001:  # 0.001 radians is much stricter
                print(f"    IK ok {i+1} iters err={err*1000:.2f}mm ori={np.degrees(ori_err):.2f}deg")
                break
            mujoco.mj_jacSite(m, d2, jacp, jacr, sid)
            Jp = jacp[:, :6]
            Jr = jacr[:, :6]
            J  = np.vstack([Jp, ORI_W * Jr])
            e  = np.concatenate([ep, ORI_W * eo])
            d2.qpos[:6] += J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e) * 0.2
        else:
            if err < tol:
                print(f"    IK ok {i+1} iters err={err*1000:.2f}mm"); break
            mujoco.mj_jacSite(m, d2, jacp, None, sid)
            J = jacp[:, :6]
            d2.qpos[:6] += J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), ep) * 0.2
        d2.qpos[:6] = np.clip(d2.qpos[:6], -2*np.pi, 2*np.pi)
    else:
        print(f"    IK no-converge err={err*1000:.2f}mm")
    return d2.qpos[:6].copy()

def slerp(q0, q1, t):
    dot = np.clip(np.dot(q0, q1), -1.0, 1.0)
    if dot < 0: q1 = -q1; dot = -dot
    if dot > 0.9995: return q0 + t*(q1-q0)
    th = np.arccos(dot); s = np.sin(th)
    return (np.sin(th*(1-t))/s)*q0 + (np.sin(th*t)/s)*q1

def get_tip(m, d):
    mujoco.mj_forward(m, d)
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripper_tip")
    return d.site_xpos[sid].copy()

def get_site_pos(m, d, site_name):
    """World position of a named MuJoCo site."""
    mujoco.mj_forward(m, d)
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if sid < 0:
        raise ValueError(f"Site not found: {site_name}")
    return d.site_xpos[sid].copy()

def ikc_site(m, d, target, seed, site_name="gripper_tip", target_quat=None):
    """IK to place an arbitrary tool site at target.

    solve_ik() is hard-wired to gripper_tip. For the internal plate gripper we
    need to command plate_lock_tcp, which sits between the two L-hooks. This
    small wrapper temporarily aliases the requested site by solving a local
    equivalent target for gripper_tip based on the current tool geometry.
    """
    if site_name == "gripper_tip":
        return ikc(m, d, target, seed, target_quat=target_quat)
    mujoco.mj_forward(m, d)
    grip = get_site_pos(m, d, "gripper_tip")
    site = get_site_pos(m, d, site_name)
    # Under a fixed straight-down tool orientation, this site-to-gripper offset
    # is stable in world during the motion.
    equiv_gripper_target = np.array(target) - (site - grip)
    return ikc(m, d, equiv_gripper_target, seed, target_quat=target_quat)

def move_to(m, d, v, ctrl, label, locked=None):
    d.ctrl[:] = ctrl
    cons = 0; steps = 0
    while cons < SETTLE_STEPS:
        if locked:
            for fj,fv,pos,q in locked: lock_part(d,fj,fv,pos,q)
        mujoco.mj_step(m, d); v.sync()
        err = np.max(np.abs(d.qpos[:6]-ctrl[:6]))
        cons = cons+1 if err < SETTLE_TOL else 0
        steps += 1
        if steps > 8000: print(f"  TIMEOUT {label}"); break
    print(f"  ✓ {label} ({steps}st err={np.max(np.abs(d.qpos[:6]-ctrl[:6])):.4f}rad)")


def dwell_seconds(m, d, v, seconds=1.0, locked=None, label="dwell"):
    """Hold the current robot pose for a fixed time while staged parts stay locked."""
    steps = max(1, int(seconds / m.opt.timestep))
    for _ in range(steps):
        if locked:
            for fj, fv, pos, q in locked:
                lock_part(d, fj, fv, pos, q)
        mujoco.mj_step(m, d)
        v.sync()
    print(f"  ✓ {label} ({seconds:.1f}s hold)")


def set_pin_lock_visual(m, amount):
    """Set visual L-hook lock amount: 0=inward/unlocked, 1=outward/locked."""
    a = float(np.clip(amount, 0.0, 1.0))
    for name, base in PIN_LOCK_GEOM_BASE.items():
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid >= 0:
            direction = -1.0 if "_L_" in name else 1.0
            pos = base.copy()
            pos[0] += direction * PIN_LOCK_TRAVEL * a
            m.geom_pos[gid] = pos
    for name, base in PIN_LOCK_SITE_BASE.items():
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid >= 0:
            direction = -1.0 if "_L_" in name else 1.0
            pos = base.copy()
            pos[0] += direction * PIN_LOCK_TRAVEL * a
            m.site_pos[sid] = pos


def animate_pin_lock(m, d, v, start, end, seconds=1.0, locked=None, label="animate pins"):
    """Animate visual L-hooks outward/inward while the robot holds position."""
    steps = max(1, int(seconds / m.opt.timestep))
    ctrl_hold = d.ctrl.copy()
    for i in range(steps):
        t = (i + 1) / steps
        s = t * t * (3.0 - 2.0 * t)
        set_pin_lock_visual(m, start + (end - start) * s)
        d.ctrl[:] = ctrl_hold
        if locked:
            for fj, fv, pos, q in locked:
                lock_part(d, fj, fv, pos, q)
        mujoco.mj_step(m, d)
        v.sync()
    print(f"  ✓ {label} ({seconds:.1f}s)")


def animate_pin_lock_carry(m, d, v, fj, fv, co, start, end, seconds=1.0,
                           site_name=PLATE_LOCK_SITE, locked=None,
                           label="animate pins while carrying"):
    """Animate L-hooks while the plate is still carried by plate_lock_tcp.

    This prevents the plate from snapping/dropping to its final freejoint pose
    before the visual pin retraction has completed.
    """
    steps = max(1, int(seconds / m.opt.timestep))
    ctrl_hold = d.ctrl.copy()
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site_name)
    for i in range(steps):
        t = (i + 1) / steps
        s = t * t * (3.0 - 2.0 * t)
        set_pin_lock_visual(m, start + (end - start) * s)
        d.ctrl[:] = ctrl_hold
        if locked:
            for lfj, lfv, pos, q in locked:
                lock_part(d, lfj, lfv, pos, q)
        mujoco.mj_forward(m, d)
        site = d.site_xpos[sid].copy()
        d.qpos[fj:fj+3]   = site + co
        d.qpos[fj+3:fj+7] = PLATE_FINAL_QUAT
        d.qvel[fv:fv+6]   = 0
        mujoco.mj_step(m, d)
        v.sync()
    print(f"  ✓ {label} ({seconds:.1f}s)")


def carry_to(m, d, v, fj, fv, co, ctrl, label, locked=None):
    d.ctrl[:] = ctrl
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripper_tip")
    cons = 0; steps = 0
    while cons < SETTLE_STEPS:
        if locked:
            for lfj,lfv,pos,q in locked: lock_part(d,lfj,lfv,pos,q)
        mujoco.mj_forward(m, d)
        tip = d.site_xpos[sid].copy()
        d.qpos[fj:fj+3]   = tip + co

        d.qpos[fj+3:fj+7] = PLATE_FINAL_QUAT
        d.qvel[fv:fv+6]   = 0
        mujoco.mj_step(m, d); v.sync()
        err = np.max(np.abs(d.qpos[:6]-ctrl[:6]))
        cons = cons+1 if err < SETTLE_TOL else 0
        steps += 1
        if steps > 8000: print(f"  TIMEOUT {label}"); break
    print(f"  ✓ {label} ({steps}st err={np.max(np.abs(d.qpos[:6]-ctrl[:6])):.4f}rad)")
        
def carry_to_site(m, d, v, fj, fv, co, ctrl, label, site_name, locked=None):
    """Carry a free body using any tool site, not only gripper_tip."""
    d.ctrl[:] = ctrl
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site_name)
    cons = 0; steps = 0
    while cons < SETTLE_STEPS:
        if locked:
            for lfj,lfv,pos,q in locked: lock_part(d,lfj,lfv,pos,q)
        mujoco.mj_forward(m, d)
        site = d.site_xpos[sid].copy()
        d.qpos[fj:fj+3]   = site + co
        d.qpos[fj+3:fj+7] = PLATE_FINAL_QUAT
        d.qvel[fv:fv+6]   = 0
        mujoco.mj_step(m, d); v.sync()
        err = np.max(np.abs(d.qpos[:6]-ctrl[:6]))
        cons = cons+1 if err < SETTLE_TOL else 0
        steps += 1
        if steps > 8000: print(f"  TIMEOUT {label}"); break
    print(f"  ✓ {label} ({steps}st err={np.max(np.abs(d.qpos[:6]-ctrl[:6])):.4f}rad)")

def release_retract(m, d, v, fj, fv, co, hold_ctrl, ret_ctrl, label, locked=None,
                    place_pos=None):
    """Hold at tip+co, snap to place_pos, retract with part hard-locked every step."""
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripper_tip")
    for _ in range(100):
        d.ctrl[:] = hold_ctrl
        if locked:
            for lfj,lfv,pos,q in locked: lock_part(d,lfj,lfv,pos,q)
        mujoco.mj_forward(m, d)
        tip = d.site_xpos[sid].copy()
        d.qpos[fj:fj+3]   = tip + co
        d.qpos[fj+3:fj+7] = [1,0,0,0]
        d.qvel[fv:fv+6]   = 0
        mujoco.mj_step(m, d); v.sync()
    if place_pos is not None:
        d.qpos[fj:fj+3]   = place_pos
        d.qpos[fj+3:fj+7] = [1,0,0,0]
    d.qvel[fv:fv+6] = 0
    for _ in range(500):
        d.ctrl[:] = ret_ctrl
        if place_pos is not None:
            d.qpos[fj:fj+3]   = place_pos
            d.qpos[fj+3:fj+7] = [1,0,0,0]
            d.qvel[fv:fv+6]   = 0
        if locked:
            for lfj,lfv,pos,q in locked: lock_part(d,lfj,lfv,pos,q)
        mujoco.mj_step(m, d); v.sync()
    print(f"  ✓ {label} released")
def ikc(m, d, tgt, seed, pos_only=False, target_quat=None):
    tq = False if pos_only else target_quat
    j = solve_ik(m, d, tgt, seed, target_quat=tq)
    c = np.zeros(m.nu); c[:6] = j; return c

def ik_move(m, d, v, tgt, seed, label, locked=None, pos_only=False):
    c = ikc(m, d, tgt, seed, pos_only=pos_only)
    move_to(m, d, v, c, label, locked=locked)
    return c


# ---------------------------------------------------------------------------
# PCB TILT-AND-INSERT
# ---------------------------------------------------------------------------

def pcb_tilt_and_insert(m, d, v, fj, fv, co, hover_ctrl, place_ctrl, locked=None):
    Q0 = np.array([1.0, 0.0, 0.0, 0.0])
    Qt = np.array([0.9962, 0.0, -0.0872, 0.0])  # -10 deg around Y
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripper_tip")

    carry_to(m, d, v, fj, fv, co, hover_ctrl, "  PCB hover", locked=locked)

    print("  Tilt 10 deg...")
    for i in range(60):
        te = ((i+1)/60)**2*(3-2*(i+1)/60)
        d.ctrl[:] = hover_ctrl
        if locked:
            for lfj,lfv,pos,q in locked: lock_part(d,lfj,lfv,pos,q)
        mujoco.mj_forward(m, d)
        tip = d.site_xpos[sid].copy()
        d.qpos[fj:fj+3]   = tip + co
        d.qpos[fj+3:fj+7] = slerp(Q0, Qt, te)
        d.qvel[fv:fv+6]   = 0
        mujoco.mj_step(m, d); v.sync()
    print("  ✓ tilted")

    print("  JST dwell 2500 steps...")
    for _ in range(2500):
        d.ctrl[:] = hover_ctrl
        if locked:
            for lfj,lfv,pos,q in locked: lock_part(d,lfj,lfv,pos,q)
        mujoco.mj_forward(m, d)
        tip = d.site_xpos[sid].copy()
        d.qpos[fj:fj+3]   = tip + co
        d.qpos[fj+3:fj+7] = Qt
        d.qvel[fv:fv+6]   = 0
        mujoco.mj_step(m, d); v.sync()
    print("  ✓ dwell done")

    print("  Insert 200 steps...")
    sj = d.qpos[:6].copy(); ej = place_ctrl[:6]
    for i in range(200):
        te = ((i+1)/200)**2*(3-2*(i+1)/200)
        d.ctrl[:6] = sj + (ej-sj)*te
        if locked:
            for lfj,lfv,pos,q in locked: lock_part(d,lfj,lfv,pos,q)
        mujoco.mj_forward(m, d)
        tip = d.site_xpos[sid].copy()
        d.qpos[fj:fj+3]   = tip + co
        d.qpos[fj+3:fj+7] = slerp(Qt, Q0, te)
        d.qvel[fv:fv+6]   = 0
        mujoco.mj_step(m, d); v.sync()
    print("  ✓ PCB inserted")


# ---------------------------------------------------------------------------
# TOOL VISUAL SWAP
# ---------------------------------------------------------------------------

def _body_alpha(m, bname, a):
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, bname)
    if bid < 0: return
    for i in range(m.body_geomnum[bid]):
        m.geom_rgba[m.body_geomadr[bid]+i][3] = a

def hide_dock_vis(m, tool, d=None, v=None):
    body = {"vac": "dock_vac_vis", "pin": "dock_pin_vis", "screwdriver": "dock_sd_vis"}[tool]
    _body_alpha(m, body, 0.0)
    if d is not None:
        mujoco.mj_forward(m, d)
    if v is not None:
        v.sync()


def show_dock_vis(m, tool, d=None, v=None):
    body = {"vac": "dock_vac_vis", "pin": "dock_pin_vis", "screwdriver": "dock_sd_vis"}[tool]
    _body_alpha(m, body, 1.0)
    if d is not None:
        mujoco.mj_forward(m, d)
    if v is not None:
        v.sync()

VAC_G = ["g_mount_flange","g_base_body","g_base_ring","g_neck","g_bellows","g_bellows2","g_cup_rim"]
PIN_G = ["pin_mount_flange","pin_base_body","pin_base_ring",
         "ilock_base_block", "ilock_L_stem", "ilock_L_hook",
         "ilock_R_stem", "ilock_R_hook"]
SD_G  = ["sd_flange","sd_spring","sd_body","sd_shaft","sd_bit"]
# Version A uses a deterministic logical attachment, not physical hook contact.
# Keep the L-hook geometry visual only to avoid MuJoCo contact explosions when
# the pins pass through the thin STL/mesh plate.
PIN_COLLISION_G = set()

def _set_geom_alpha(m, geom_names, alpha):
    for g in geom_names:
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g)
        if gid >= 0:
            m.geom_rgba[gid][3] = alpha
            if g in PIN_COLLISION_G and alpha <= 0.0:
                m.geom_contype[gid] = 0
                m.geom_conaffinity[gid] = 0
            elif g in PIN_COLLISION_G and alpha > 0.0:
                m.geom_contype[gid] = 1
                m.geom_conaffinity[gid] = 1


def hide_all_robot_tools(m, d=None, v=None, locked=None):
    """Instantly hide all robot-mounted tool geoms.

    This avoids one-frame flashes during tool swaps. Fading inactive tools can
    briefly reveal the wrong gripper in MuJoCo's viewer; a hard hide gives a
    cleaner tool-change visual.
    """
    all_robot_tool_geoms = VAC_G + PIN_G + SD_G
    _set_geom_alpha(m, all_robot_tool_geoms, 0.0)
    if d is not None:
        if locked:
            for fj, fv, pos, q in locked:
                lock_part(d, fj, fv, pos, q)
        mujoco.mj_forward(m, d)
    if v is not None:
        v.sync()


def show_tool(m, tool, d=None, v=None, locked=None):
    """Show exactly one robot-mounted tool, or none, with no transition flash.

    Valid tool values: 'vacuum', 'pin', 'screwdriver', 'none'.
    """
    all_g = {"vacuum": VAC_G, "pin": PIN_G, "screwdriver": SD_G}
    hide_all_robot_tools(m, d=None, v=None, locked=None)

    if tool != "none":
        if tool == "pin":
            set_pin_lock_visual(m, 0.0)
        _set_geom_alpha(m, all_g[tool], 1.0)

    if d is not None:
        if locked:
            for fj, fv, pos, q in locked:
                lock_part(d, fj, fv, pos, q)
        mujoco.mj_forward(m, d)
    if v is not None:
        v.sync()

def drive_screw(m, d, v, name, sfj, sfv, placed_locks=None, home_ctrl=None, screw_box_locks=None):
    """Pick one screw from the screw box and drive it into the matching boss.

    Visual/robust behavior:
      - all screws remain locked in the screw box until the selected screw is picked
      - the selected screw is snapped to the real screwdriver_tip site while traveling
      - no plain move_to()/ik_move() is used while the selected screw is attached
      - after driving, the screw is left at its installed pose and welded to bottom case
        so it visually fastens the aluminum plate to the bottom case (Option C).
    """
    placed_locks = placed_locks or []
    screw_box_locks = screw_box_locks or []

    pick_origin = SCREW_PICK_POS[name]
    install_origin = np.array([SCREW_INSTALL_POS[name][0], SCREW_INSTALL_POS[name][1], SCREW_INSTALLED_Z])
    sid_sd = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "screwdriver_tip")
    print(f"\n  [SCREW {name}]")

    current_lock = (sfj, sfv, pick_origin, None)
    future_box_locks = [lk for lk in screw_box_locks if lk[0] != sfj]
    pre_pick_locks = placed_locks + future_box_locks + [current_lock]
    carry_locks    = placed_locks + future_box_locks

    pick_sd_tip = pick_origin + np.array([0.0, 0.0, SCREW_ENGAGE_Z])
    pick_hover  = gripper_target_for_sd_tip(pick_sd_tip + np.array([0.0, 0.0, SCREW_BOX_HOVER_CLEAR]))
    pick_target = gripper_target_for_sd_tip(pick_sd_tip)

    # Seed from the current posture so the arm chooses the nearby/right-side branch,
    # not a long rotation through the dock/feeder-tray side of the cell.
    pick_hover_ctrl = ikc(m, d, pick_hover, d.qpos[:6].copy(), pos_only=False)
    move_to(m, d, v, pick_hover_ctrl, f"  SD->box hover ({name})", locked=pre_pick_locks)
    pick_ctrl = ikc(m, d, pick_target, d.qpos[:6].copy(), pos_only=False)
    move_to(m, d, v, pick_ctrl, f"  SD->box lower ({name})", locked=pre_pick_locks)

    box_wid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f"weld_box_screw_{name}")
    if box_wid >= 0:
        d.eq_active[box_wid] = 0

    mujoco.mj_forward(m, d)
    sd_tip = d.site_xpos[sid_sd].copy()
    d.qpos[sfj:sfj+3]   = screw_origin_for_sd_tip(sd_tip)
    d.qpos[sfj+3:sfj+7] = [1, 0, 0, 0]
    d.qvel[sfv:sfv+6]   = 0
    print(f"  attached {name} to screwdriver_tip")

    def carry_screw_to(ctrl, label, max_steps=8000):
        """Move robot to ctrl while deterministically carrying the screw at the bit."""
        d.ctrl[:] = ctrl
        cons = 0; steps = 0
        while cons < SETTLE_STEPS:
            if carry_locks:
                for lfj, lfv, pos, q in carry_locks:
                    lock_part(d, lfj, lfv, pos, q)
            mujoco.mj_forward(m, d)
            sd_t = d.site_xpos[sid_sd].copy()
            d.qpos[sfj:sfj+3]   = screw_origin_for_sd_tip(sd_t)
            d.qpos[sfj+3:sfj+7] = [1, 0, 0, 0]
            d.qvel[sfv:sfv+6]   = 0
            mujoco.mj_step(m, d); v.sync()
            err = np.max(np.abs(d.qpos[:6]-ctrl[:6]))
            cons = cons+1 if err < SETTLE_TOL else 0
            steps += 1
            if steps > max_steps:
                print(f"  TIMEOUT {label}"); break
        print(f"  ✓ {label} ({steps}st err={np.max(np.abs(d.qpos[:6]-ctrl[:6])):.4f}rad)")

    raise_ctrl = ikc(m, d, pick_hover, d.qpos[:6].copy(), pos_only=False)
    carry_screw_to(raise_ctrl, f"  lift screw {name}")

    # v7 motion plan: use a local high transfer waypoint between screw box and pallet.
    # This prevents the solver from taking the long ~270 deg branch over the tool dock.
    transfer_ctrl = ikc(m, d, SCREW_LOCAL_TRANSFER, d.qpos[:6].copy(), pos_only=False)
    carry_screw_to(transfer_ctrl, f"  local transfer to pallet corridor {name}")

    drive_sd_tip = install_origin + np.array([0.0, 0.0, SCREW_ENGAGE_Z])
    boss_hover   = gripper_target_for_sd_tip(drive_sd_tip + np.array([0.0, 0.0, SCREW_DRIVE_CLEAR]))
    boss_drive   = gripper_target_for_sd_tip(drive_sd_tip)

    hover_ctrl = ikc(m, d, boss_hover, d.qpos[:6].copy(), pos_only=False)
    carry_screw_to(hover_ctrl, f"  hover {name}")

    drive_ctrl = ikc(m, d, boss_drive, d.qpos[:6].copy(), pos_only=False)
    carry_screw_to(drive_ctrl, f"  lower to screw height {name}")

    print(f"  Rotating {name} at drive height...")
    w3_0 = d.qpos[5]
    for i in range(ROTATE_STEPS):
        d.ctrl[:] = drive_ctrl
        d.ctrl[5] = w3_0 + (i / ROTATE_STEPS) * (3 * 2 * np.pi)
        if carry_locks:
            for lfj, lfv, pos, q in carry_locks:
                lock_part(d, lfj, lfv, pos, q)
        mujoco.mj_forward(m, d)
        sd_t = d.site_xpos[sid_sd].copy()
        d.qpos[sfj:sfj+3]   = screw_origin_for_sd_tip(sd_t)
        d.qpos[sfj+3:sfj+7] = [1, 0, 0, 0]
        d.qvel[sfv:sfv+6]   = 0
        mujoco.mj_step(m, d); v.sync()

    d.qpos[sfj:sfj+3]   = install_origin
    d.qpos[sfj+3:sfj+7] = [1, 0, 0, 0]
    d.qvel[sfv:sfv+6]   = 0
    case_wid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f"weld_screw_{name}")
    if case_wid >= 0:
        d.eq_active[case_wid] = 1
    d.ctrl[5] = w3_0
    print(f"  {name} installed through plate + welded to bottom case")

    move_to(m, d, v, hover_ctrl, f"  retract {name}", locked=carry_locks)

    # Return to the local corridor before the next screw-box approach. This keeps the
    # next IK solve on the short branch instead of letting the shoulder swing around.
    transfer_back_ctrl = ikc(m, d, SCREW_LOCAL_TRANSFER, d.qpos[:6].copy(), pos_only=False)
    move_to(m, d, v, transfer_back_ctrl, f"  transfer back to screw-box corridor {name}", locked=placed_locks + future_box_locks)


# ---------------------------------------------------------------------------
# TOOL CHANGE PHASES
# ---------------------------------------------------------------------------

# def dock_ik_move(m, d, v, xy, z, seed, label, locked=None):
   # """IK move to dock position. Seed determines arm configuration (vertical approach).
    #Uses orientation IK (not pos_only) so arm arrives with gripper straight down."""
    #tgt = np.array([xy[0], xy[1], z])
    #return ik_move(m, d, v, tgt, seed, label, locked=locked)

def dock_ik_move(m, d, v, xy, z, seed, label, locked=None):
    """
    Force strict orientation during tool change.
    """
    tgt = np.array([xy[0], xy[1], z])
    # Instead of letting solve_ik decide, we force STRICT_DOWN_QUAT
    j = solve_ik(m, d, tgt, seed, target_quat=STRICT_DOWN_QUAT)
    
    c = np.zeros(m.nu)
    c[:6] = j
    move_to(m, d, v, c, label, locked=locked)
    return c

def dock_lower(m, d, v, xy, z, label, locked=None):
    """Vertical dock descent/retract with the flange kept square to the dock.

    Do NOT use position-only IK here. During tool changes, XYZ is not enough:
    position-only IK can let wrist_1/2/3 rotate while the tool lowers into the
    dock, which makes the flange look tilted relative to the docked gripper
    base. Using the current qpos as the seed preserves the same elbow posture,
    while STRICT_DOWN_QUAT keeps the flange/tool axis flat and vertical.
    """
    tgt = np.array([xy[0], xy[1], z])
    seed = d.qpos[:6].copy()
    j = solve_ik(m, d, tgt, seed, target_quat=STRICT_DOWN_QUAT)
    c = np.zeros(m.nu)
    c[:6] = j
    move_to(m, d, v, c, label, locked=locked)
    return c


def set_equality_active(m, d, weld_name, active):
    """Enable/disable a named equality weld if it exists.

    The internal plate gripper still updates the free body pose deterministically
    every step for robustness. This weld flag is used as an explicit logical
    attach/release marker and is safe if the weld is absent.
    """
    wid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, weld_name)
    if wid >= 0:
        m.eq_active0[wid] = 1 if active else 0
        d.eq_active[wid] = 1 if active else 0
        return True
    return False


def phase_place_plate_internal_lock(m, d, v, fj, fv, pw, plw, waiting, timer):
    """Pick and place the aluminum keyboard plate with the internal L-hook gripper.

    State sequence:
      1. approach above two symmetric third-row switch cutouts
      2. insert pins through cutouts
      3. logical lock: hooks are considered shifted outward under the plate
      4. attach/carry plate by deterministic snap to plate_lock_tcp
      5. lift
      6. move to bottom case
      7. lower near seating height
      8. detach/release slightly above final seating height
      9. unlock
     10. retract upward
    """
    print("  [plate gripper] internal two-pin L-hook sequence")
    print(f"  selected cutouts: L=({PLATE_PIN_LEFT_X:.5f}, {PLATE_PIN_Y:.5f}) m, "
          f"R=({PLATE_PIN_RIGHT_X:.5f}, {PLATE_PIN_Y:.5f}) m")

    # The lock TCP is between the two hook contact sites. Target it at the
    # plate centerline; the hook site is lowered just below the plate underside.
    insert_site_world = pw + np.array([0.0, 0.0, -PLATE_INSERT_CLEAR])
    approach_site_world = insert_site_world + np.array([0.0, 0.0, PLATE_APPROACH_CLEAR])

    print(f"  approach lock TCP -> {approach_site_world.round(4)}")
    approach_ctrl = ikc_site(m, d, approach_site_world, PICK_IK_SEED,
                             site_name=PLATE_LOCK_SITE, target_quat=STRICT_DOWN_QUAT)
    move_to(m, d, v, approach_ctrl, "  approach above plate cutouts",
            locked=[(fj, fv, pw, PLATE_TRAY_QUAT)] + waiting)

    print(f"  insert pins through cutouts -> {insert_site_world.round(4)}")
    insert_ctrl = ikc_site(m, d, insert_site_world, PICK_IK_SEED,
                           site_name=PLATE_LOCK_SITE, target_quat=STRICT_DOWN_QUAT)
    move_to(m, d, v, insert_ctrl, "  insert L-hook pins",
            locked=[(fj, fv, pw, PLATE_TRAY_QUAT)] + waiting)

    # Visual lock: pins slide outward inside the cutouts, then hold so the
    # mechanical capture is visible before the robot lifts the plate.
    animate_pin_lock(m, d, v, 0.0, 1.0, seconds=1.0,
                     locked=[(fj, fv, pw, PLATE_TRAY_QUAT)] + waiting,
                     label="engage L-hook pins outward")
    dwell_seconds(m, d, v, 1.0, locked=[(fj, fv, pw, PLATE_TRAY_QUAT)] + waiting,
                  label="hold after pin engagement")
    print("  ✓ hooks locked outward under plate cutout bridges")

    # Attach plate to the dock-matched hook-tip TCP, not to the flange.
    # This carry offset is intentionally computed from plate_lock_tcp after insertion,
    # so the plate visually follows the internal pins/hooks while final placement remains tuned.
    lock_site_at_pick = get_site_pos(m, d, PLATE_LOCK_SITE)
    co = pw - lock_site_at_pick
    print(f"  lock_site_at_pick: {lock_site_at_pick.round(4)}")
    print(f"  plate carry_offset from lock TCP: {co.round(4)}  mag={np.linalg.norm(co)*1000:.1f}mm")

    # Do NOT enable the MJCF weld here. A precompiled equality weld stores a fixed
    # relative pose from model load time; turning it on after the gripper is at
    # the plate can create a large constraint impulse and make the arm/plate
    # explode or the joint controller timeout. We carry the plate deterministically
    # by snapping the plate freejoint to plate_lock_tcp + carry_offset each step.
    # set_equality_active(m, d, "weld_alu_plate", True)
    for i in range(60):
        te = ((i+1)/60)**2*(3-2*(i+1)/60)
        d.ctrl[:] = insert_ctrl
        if waiting:
            for lfj,lfv,pos,q in waiting: lock_part(d,lfj,lfv,pos,q)
        site = get_site_pos(m, d, PLATE_LOCK_SITE)
        tgt = site + co
        d.qpos[fj:fj+3]   = pw + (tgt-pw)*te
        d.qpos[fj+3:fj+7] = PLATE_TRAY_QUAT
        d.qvel[fv:fv+6]   = 0
        mujoco.mj_step(m, d); v.sync()
    print("  ✓ plate attached to internal L-hook gripper")

    # Lift while carrying from the internal lock TCP.
    lift_ctrl = ikc_site(m, d, insert_site_world + np.array([0,0,0.090]),
                         PICK_IK_SEED, site_name=PLATE_LOCK_SITE,
                         target_quat=STRICT_DOWN_QUAT)
    carry_to_site(m, d, v, fj, fv, co, lift_ctrl, "  lift locked plate",
                  PLATE_LOCK_SITE, locked=waiting)

    # Move to the bottom case and lower. Release slightly above final seating so
    # the hooks can unlock before the tool retracts through the switch cutouts.
    PLATE_PLACE_Z_BIAS = 0.007
    place_lock_site = plw - co + np.array([0.0, 0.0, PLATE_PLACE_Z_BIAS])
    hover_ctrl = ikc_site(m, d, place_lock_site + np.array([0,0,0.085]),
                          PLACE_IK_SEED, site_name=PLATE_LOCK_SITE,
                          target_quat=STRICT_DOWN_QUAT)
    carry_to_site(m, d, v, fj, fv, co, hover_ctrl, "  move locked plate over case",
                  PLATE_LOCK_SITE, locked=waiting)

    # Backsolve the gripper/TCP placement from the tuned final plate pose:
    #   carried_plate_pos = plate_lock_tcp_world + co
    # therefore:
    #   plate_lock_tcp_world = final_plate_pos - co
    # With PLATE_RELEASE_CLEAR=0, release_site == place_lock_site and the carried
    # plate stays exactly at PLATE_FINAL_POS while the pins retract.
    release_site = place_lock_site + np.array([0, 0, PLATE_RELEASE_CLEAR])
    release_ctrl = ikc_site(m, d, release_site, PLACE_IK_SEED,
                            site_name=PLATE_LOCK_SITE,
                            target_quat=STRICT_DOWN_QUAT)
    carry_to_site(m, d, v, fj, fv, co, release_ctrl, "  lower carried plate to final seated pose",
                  PLATE_LOCK_SITE, locked=waiting)

    # Do not lower below final pose. Retract the pins while the plate is still held
    # at the backsolved final height, then release/snap to the same final pose.
    animate_pin_lock_carry(m, d, v, fj, fv, co, 1.0, 0.0, seconds=1.0,
                           site_name=PLATE_LOCK_SITE, locked=waiting,
                           label="unlock L-hook pins inward while plate is still held at final pose")

    print("  ✓ hooks unlocked inward; releasing plate at final seated pose")
    set_equality_active(m, d, "weld_alu_plate", False)
    d.qpos[fj:fj+3] = plw
    d.qpos[fj+3:fj+7] = PLATE_FINAL_QUAT
    d.qvel[fv:fv+6] = 0
    dwell_seconds(m, d, v, 0.5, locked=waiting + [(fj, fv, plw, PLATE_FINAL_QUAT)],
                  label="hold after plate release")

    print("  ✓ plate released; retracting internal gripper")
    retract_ctrl = ikc_site(m, d, release_site + np.array([0,0,0.075]),
                            PLACE_IK_SEED, site_name=PLATE_LOCK_SITE,
                            target_quat=STRICT_DOWN_QUAT)
    move_to(m, d, v, retract_ctrl, "  retract internal gripper",
            locked=waiting + [(fj, fv, plw, PLATE_FINAL_QUAT)])
    timer.mark(d, "Place s1_alu_plate")
    return co


def phase6_vac_to_pin(m, d, v, placed_locks, timer):
    print(); print("="*60); print("[PHASE 6] Vacuum -> Pin Gripper"); print("="*60)

    # 1. Swing to vertical above vac slot
    dock_ik_move(m, d, v, DOCK_VAC_XY, DOCK_HOVER_Z, DOCK_VAC_SEED,
                 "  swing → vac hover", locked=placed_locks)

    # 2. Lower straight down onto vac cup
    dock_lower(m, d, v, DOCK_VAC_XY, DOCK_SEAT_Z, "  lower → vac seat", locked=placed_locks)
    dwell_seconds(m, d, v, 1.0, locked=placed_locks, label="hold at vacuum dock before release")

    # 3. Detach vacuum — show empty flange (no tool geoms visible)
    show_tool(m, "none", d, v, locked=placed_locks)
    show_dock_vis(m, "vac", d, v)
    print("  ✓ Vacuum detached — docked")

    # 4. Raise straight up from vac slot
    dock_lower(m, d, v, DOCK_VAC_XY, DOCK_HOVER_Z, "  raise from vac", locked=placed_locks)

    # 5. Swing laterally to pin slot (stays at hover height, arm stays vertical)
    dock_ik_move(m, d, v, DOCK_PIN_XY, DOCK_HOVER_Z, DOCK_PIN_SEED,
                 "  swing → pin hover", locked=placed_locks)

    # 6. Lower straight down onto pin cup
    dock_lower(m, d, v, DOCK_PIN_XY, DOCK_SEAT_Z, "  lower → pin seat", locked=placed_locks)
    dwell_seconds(m, d, v, 1.0, locked=placed_locks, label="hold at pin dock before pickup")

    # 7. Attach pin gripper: hide docked visual first, then show robot-mounted tool.
    hide_dock_vis(m, "pin", d, v)
    show_tool(m, "pin", d, v, locked=placed_locks)
    print("  ✓ Pin gripper attached")

    # 8. Raise straight up with pin gripper
    dock_lower(m, d, v, DOCK_PIN_XY, DOCK_HOVER_Z, "  raise with pin", locked=placed_locks)

    timer.mark(d, "Tool change: vac->pin")


def phase8_pin_to_sd(m, d, v, placed_locks, timer):
    print(); print("="*60); print("[PHASE 8] Pin -> Screwdriver"); print("="*60)

    # 1. Swing to vertical above pin slot
    dock_ik_move(m, d, v, DOCK_PIN_XY, DOCK_HOVER_Z, DOCK_PIN_SEED,
                 "  swing → pin hover", locked=placed_locks)

    # 2. Lower onto pin cup
    dock_lower(m, d, v, DOCK_PIN_XY, DOCK_SEAT_Z, "  lower → pin seat", locked=placed_locks)
    dwell_seconds(m, d, v, 1.0, locked=placed_locks, label="hold at pin dock before release")

    # 3. Detach pin
    show_tool(m, "none", d, v, locked=placed_locks)
    show_dock_vis(m, "pin", d, v)
    print("  ✓ Pin detached — docked")

    # 4. Raise from pin slot
    dock_lower(m, d, v, DOCK_PIN_XY, DOCK_HOVER_Z, "  raise from pin", locked=placed_locks)

    # 5. Swing to SD slot
    dock_ik_move(m, d, v, DOCK_SD_XY, DOCK_HOVER_Z, DOCK_SD_SEED,
                 "  swing → SD hover", locked=placed_locks)

    # 6. Lower onto SD cup
    dock_lower(m, d, v, DOCK_SD_XY, DOCK_SEAT_Z, "  lower → SD seat", locked=placed_locks)
    dwell_seconds(m, d, v, 1.0, locked=placed_locks, label="hold at screwdriver dock before pickup")

    # 7. Attach screwdriver: hide docked visual first, then show robot-mounted tool.
    hide_dock_vis(m, "screwdriver", d, v)
    show_tool(m, "screwdriver", d, v, locked=placed_locks)
    print("  ✓ Screwdriver attached")

    # 8. Raise with screwdriver
    dock_lower(m, d, v, DOCK_SD_XY, DOCK_HOVER_Z, "  raise with SD", locked=placed_locks)

    timer.mark(d, "Tool change: pin->SD")

def phase9_screws(m, d, v, home_ctrl, placed_locks, timer):
    print(); print("="*60); print("[PHASE 9] Screwdriving"); print("="*60)
    screws = {n: get_body_jnt(m, f"screw_{n}") for n in SCREW_NAMES}
    for i, name in enumerate(SCREW_NAMES):
        sfj, sfv = screws[name]
        remaining_box_locks = [(screws[n][0], screws[n][1], SCREW_PICK_POS[n], None)
                               for n in SCREW_NAMES[i:]]
        drive_screw(m, d, v, name, sfj, sfv, placed_locks, home_ctrl,
                    screw_box_locks=remaining_box_locks)
        timer.mark(d, f"Screw {name}")

    # Return SD to dock, then pick vacuum gripper before going home.
    # This leaves the cell in a realistic ready state: robot at home + vacuum attached.
    print("\n  Returning SD...")
    dock_ik_move(m, d, v, DOCK_SD_XY, DOCK_HOVER_Z, DOCK_SD_SEED,
                 "  swing → SD return hover", locked=placed_locks)
    dock_lower(m, d, v, DOCK_SD_XY, DOCK_SEAT_Z, "  lower → SD seat", locked=placed_locks)
    dwell_seconds(m, d, v, 1.0, locked=placed_locks, label="hold at screwdriver dock before release")
    show_tool(m, "none", d, v, locked=placed_locks)
    show_dock_vis(m, "screwdriver", d, v)
    print("  ✓ SD returned to dock")
    dock_lower(m, d, v, DOCK_SD_XY, DOCK_HOVER_Z, "  raise from SD", locked=placed_locks)

    print("\n  Re-attaching vacuum gripper before home...")
    dock_ik_move(m, d, v, DOCK_VAC_XY, DOCK_HOVER_Z, DOCK_VAC_SEED,
                 "  swing → vac pickup hover", locked=placed_locks)
    dock_lower(m, d, v, DOCK_VAC_XY, DOCK_SEAT_Z, "  lower → vac pickup seat", locked=placed_locks)
    dwell_seconds(m, d, v, 1.0, locked=placed_locks, label="hold at vacuum dock before pickup")
    hide_dock_vis(m, "vac", d, v)
    show_tool(m, "vacuum", d, v, locked=placed_locks)
    print("  ✓ Vacuum gripper attached")
    dock_lower(m, d, v, DOCK_VAC_XY, DOCK_HOVER_Z, "  raise with vacuum", locked=placed_locks)

    move_to(m, d, v, home_ctrl, "  final home with vacuum", locked=placed_locks)
    print("✓ All phases complete.")



def tray_lock_for_part(b, part_jnts):
    """Return a tray lock tuple; aluminum plate waits rotated in the feeder tray."""
    q = PLATE_TRAY_QUAT if b == "s1_alu_plate" else None
    return (part_jnts[b][0], part_jnts[b][1], PICK_WORLD[b], q)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print(f"Loading: {MODEL_PATH}\n")
    m = mujoco.MjModel.from_xml_path(MODEL_PATH)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)

    kf_idx = {m.key(i).name: i for i in range(m.nkey)}
    print("Keyframes:", list(kf_idx.keys()))

    home_ctrl    = np.zeros(m.nu)
    home_ctrl[:] = m.key_ctrl[kf_idx["home"], :m.nu]

    # Disable all plate/tray welds — tray locked via lock_part every step
    for wname in ["weld_plate_pz", "weld_alu_plate", "weld_tray_pz"]:
        wid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, wname)
        if wid >= 0:
            m.eq_active0[wid] = 0
            d.eq_active[wid]  = 0
    tray_fj, tray_fv = get_body_jnt(m, "kit_tray_pickzone")
    tray_home_pos    = d.qpos[tray_fj:tray_fj+3].copy()
    TRAY_LOCK        = (tray_fj, tray_fv, tray_home_pos, None)

    # Resolve freejoint addresses and verify against expected PICK_WORLD
    part_jnts = {}
    print("\nFreejoint address verification:")
    for bname in PARTS:
        fj, fv = get_body_jnt(m, bname)
        actual = d.qpos[fj:fj+3]
        expect = PICK_WORLD[bname]
        ok = np.allclose(actual, expect, atol=1e-4)
        print(f"  {bname}: [{fj}] = {actual.round(5)}  expected {expect.round(5)}"
              f"  {'OK' if ok else 'MISMATCH - check XML body order!'}")
        part_jnts[bname] = (fj, fv)

    screw_jnts = {n: get_body_jnt(m, f"screw_{n}") for n in SCREW_NAMES}
    screw_box_locks = [(screw_jnts[n][0], screw_jnts[n][1], SCREW_PICK_POS[n], None)
                       for n in SCREW_NAMES]

    with mujoco.viewer.launch_passive(m, d) as v:
        v.cam.lookat[:] = [0.0, 0.3, 0.83]
        v.cam.distance  = 1.5
        v.cam.elevation = -22
        v.cam.azimuth   = 145

        # Startup state: vacuum is on the robot, pin and SD are in the dock
        show_tool(m, "vacuum")   # show vacuum on robot
        hide_dock_vis(m, "vac", d, v)        # vac slot in dock starts empty
        # pin and SD dock visuals start visible (set in XML already)

        all_locks = [(part_jnts[b][0], part_jnts[b][1], PICK_WORLD[b], None)
                     for b in PARTS] + [TRAY_LOCK] + screw_box_locks
        move_to(m, d, v, home_ctrl, "home", locked=all_locks)

        timer = CycleTimer("Station 1")
        timer.start(d)

        remaining = list(PARTS)  # pops as each part is picked

        for step_idx, bname in enumerate(PARTS):
            fj_adr, fv_adr = part_jnts[bname]

            # Parts still in tray (excluding current)
            waiting = [tray_lock_for_part(b, part_jnts)
                       for b in remaining if b != bname] + [TRAY_LOCK] + screw_box_locks

            use_pin = (bname == "s1_alu_plate")
            print(); print("="*60)
            print(f"[STEP {step_idx+1}/{len(PARTS)}] {bname.upper()}"
                  f"  ({'pin' if use_pin else 'vacuum'})")
            print("="*60)

            if use_pin:
                # Build locks for the 3 parts already placed (epdm, battery, pcb)
                already_placed = PARTS[:step_idx]  # all before alu_plate
                pre_locks = [(part_jnts[b][0], part_jnts[b][1], PLACE_PART_WORLD[b], None)
                            for b in already_placed]
                # Lock plate at its tray position — weld_plate_pz is disabled,
                # weld_plate_world not yet active, so plate falls without this
                plate_fj, plate_fv = part_jnts["s1_alu_plate"]
                pre_locks += [(plate_fj, plate_fv, PICK_WORLD["s1_alu_plate"], PLATE_TRAY_QUAT)]
                pre_locks += [TRAY_LOCK] + screw_box_locks
                phase6_vac_to_pin(m, d, v, pre_locks, timer)

            pw  = PICK_WORLD[bname]
            plw = PLACE_PART_WORLD[bname]

            if use_pin:
                phase_place_plate_internal_lock(m, d, v, fj_adr, fv_adr, pw, plw, waiting, timer)
                remaining.remove(bname)
                continue

            # -- PICK -------------------------------------------------------
            xy_off = GRASP_XY_OFFSET.get(bname, np.zeros(2))
            grasp  = pw + np.array([xy_off[0], xy_off[1], GRASP_Z_OFFSET[bname]])
            print(f"  IK pick -> {grasp.round(4)}")
            pick_ctrl = ikc(m, d, grasp, PICK_IK_SEED)
            # Lock current part AND waiting parts during arm approach
            move_to(m, d, v, pick_ctrl, "  pick_down",
                    locked=[(fj_adr, fv_adr, pw, None)] + waiting)

            # carry_offset from LOCKED position (not d.qpos after physics)
            mujoco.mj_forward(m, d)
            sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripper_tip")
            tip_at_pick  = d.site_xpos[sid].copy()
            co           = pw - tip_at_pick   # carry_offset
            print(f"  tip_at_pick:  {tip_at_pick.round(4)}")
            print(f"  carry_offset: {co.round(4)}  mag={np.linalg.norm(co)*1000:.1f}mm")

            # -- ATTACH: smooth snap 60 steps --------------------------------
            p0 = pw.copy()
            for i in range(60):
                te = ((i+1)/60)**2*(3-2*(i+1)/60)
                d.ctrl[:] = pick_ctrl
                if waiting:
                    for lfj,lfv,pos,q in waiting: lock_part(d,lfj,lfv,pos,q)
                mujoco.mj_forward(m, d)
                tip = d.site_xpos[sid].copy()
                tgt = tip + co
                d.qpos[fj_adr:fj_adr+3]   = p0 + (tgt-p0)*te
                d.qpos[fj_adr+3:fj_adr+7] = [1,0,0,0]
                d.qvel[fv_adr:fv_adr+6]   = 0
                mujoco.mj_step(m, d); v.sync()
            print(f"  ✓ attached")

            remaining.remove(bname)

            # -- LIFT --------------------------------------------------------
            lift_ctrl = ikc(m, d, grasp+[0,0,0.08], PICK_IK_SEED)
            carry_to(m, d, v, fj_adr, fv_adr, co, lift_ctrl, "  lift", locked=waiting)

            # -- PLACE -------------------------------------------------------
            # tip_target = place_world - carry_offset
            # because: part_pos = tip + co => tip = part_pos - co = plw - co
            tip_tgt = plw - co

            if bname == "s1_pcb":
                hover_ctrl = ikc(m, d, tip_tgt+[0.020, 0, 0.060], PLACE_IK_SEED)
                place_ctrl = ikc(m, d, tip_tgt, PLACE_IK_SEED)
                swing_ctrl = ikc(m, d, tip_tgt+[0, 0, 0.130], PLACE_IK_SEED)
                carry_to(m, d, v, fj_adr, fv_adr, co,
                         swing_ctrl, "  swing pallet", locked=waiting)
                pcb_tilt_and_insert(m, d, v, fj_adr, fv_adr, co,
                                    hover_ctrl, place_ctrl, locked=waiting)
                ret_ctrl = ikc(m, d, tip_tgt+[0,0,0.070], PLACE_IK_SEED)
            else:
                hover_ctrl = ikc(m, d, tip_tgt+[0, 0, 0.080], PLACE_IK_SEED)
                carry_to(m, d, v, fj_adr, fv_adr, co,
                         hover_ctrl, "  swing+hover", locked=waiting)
                place_ctrl = ikc(m, d, tip_tgt, PLACE_IK_SEED)
                carry_to(m, d, v, fj_adr, fv_adr, co,
                         place_ctrl, f"  place {step_idx+1}", locked=waiting)
                ret_ctrl = hover_ctrl

            # -- RELEASE -----------------------------------------------------
            release_retract(m, d, v, fj_adr, fv_adr, co,
                hold_ctrl=place_ctrl, ret_ctrl=ret_ctrl,
                label=bname, locked=waiting,
                place_pos=plw)
            print(f"  landed: {d.qpos[fj_adr:fj_adr+3].round(4)}")
            timer.mark(d, f"Place {bname}")
        

        print()
        # Activate permanent world-weld for alu_plate — holds it absolutely, no per-step lock needed
        wid_pw = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, "weld_plate_world")
        if wid_pw >= 0:
            d.eq_active[wid_pw] = 1
        # Base placed locks for assembled keyboard stack + tray.
        # Add screw_box_locks only during the pin->screwdriver tool change, so screws
        # stay visually in the screw box. During screwdriving, phase9 manages locks per screw.
        placed_locks = [(part_jnts[b][0], part_jnts[b][1], PLACE_PART_WORLD[b], PLACE_PART_QUAT.get(b))
                for b in PARTS] + [TRAY_LOCK]
        station_locks = placed_locks + screw_box_locks
        timer.mark(d, "Assembly complete")

        # Go directly to tool change — no intermediate home swing
        phase8_pin_to_sd(m, d, v, station_locks, timer)
        phase9_screws(m, d, v, home_ctrl, placed_locks, timer)

        print(); print("="*60)
        print("Station 1 complete. Close viewer to exit.")
        print("="*60)
        timer.finish(d)
        timer.print_report()

        while v.is_running():
            mujoco.mj_step(m, d); v.sync()


if __name__ == "__main__":
    main()