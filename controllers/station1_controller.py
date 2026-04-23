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

MODEL_PATH   = "models/stations/station1.xml"
SETTLE_TOL   = 0.015
SETTLE_STEPS = 50

# ---------------------------------------------------------------------------
# PART DEFINITIONS
# ---------------------------------------------------------------------------
# World positions from keyframe (s1_ bodies at qpos indices 49-76)
PICK_WORLD = {
    "s1_epdm":      np.array([-0.3000,  0.31915, 0.81075]),
    "s1_battery":   np.array([-0.3000,  0.43630, 0.81300]),
    "s1_pcb":       np.array([-0.3000,  0.54830, 0.81080]),
    "s1_alu_plate": np.array([-0.3000,  0.68745, 0.81375]),
}

# Gripper tip Z above part body origin at pick
GRASP_Z_OFFSET = {
    "s1_epdm":      0.005,
    "s1_battery":   0.010,
    "s1_pcb":       0.006,
    "s1_alu_plate": 0.004,
}

# Where the part body origin should land in the case (world frame)
# Case: pos=(0,0,0.821), euler Z=pi  -> world_x = -body_x, world_y = -body_y
# Battery body-local X=-0.025 -> world X=+0.025
PLACE_PART_WORLD = {
    # EPDM+Battery parallel on case floor (Z=0.813); PCB on battery top; Plate on PCB
    "s1_epdm":      np.array([ 0.000,  0.000, 0.81175]),
    "s1_battery":   np.array([ 0.025,  0.000, 0.81300]),
    "s1_pcb":       np.array([ 0.000,  0.000, 0.82100]),
    "s1_alu_plate": np.array([ 0.000,  0.000, 0.82500]),
}

PARTS = ["s1_epdm", "s1_battery", "s1_pcb", "s1_alu_plate"]

# ---------------------------------------------------------------------------
# IK SEEDS
# Robot base: (0.200, 0.400, 0.600), euler Z=pi
# pan=0 -> faces -X (tray). pan~1.107 -> faces pallet.
# ---------------------------------------------------------------------------
PICK_IK_SEED  = np.array([ 0.10, -1.60,  1.80, -1.80, -1.5708, 0.0])
PLACE_IK_SEED = np.array([ 1.10, -1.60,  1.80, -1.80, -1.5708, 0.0])
DOCK_IK_SEED  = np.array([ 1.60, -1.50,  1.80, -1.85, -1.5708, 0.0])
SCREW_IK_SEED = np.array([ 1.10, -1.50,  1.80, -1.85, -1.5708, 0.0])

# ---------------------------------------------------------------------------
# TOOL DOCK WORLD POSITIONS
# Dock body: (0.650, 0.400, 0.800)
# dock_vac_W2 site body-local (0,-0.090,0.090) -> world (0.650, 0.310, 0.890)
# dock_pin_W2 site body-local (0, 0.000,0.090) -> world (0.650, 0.400, 0.890)
# dock_sd_W2  site body-local (0,+0.090,0.090) -> world (0.650, 0.490, 0.890)
# ---------------------------------------------------------------------------
DOCK_VAC_W2     = np.array([0.650, 0.310, 0.890])
DOCK_PIN_W2     = np.array([0.650, 0.400, 0.890])
DOCK_SD_W2      = np.array([0.650, 0.490, 0.890])
WINGMAN_SLIDE_X = 0.040

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
SCREW_INSTALL_POS = {
    "FL": np.array([ 0.1484,  0.05315, BOSS_Z]),
    "FR": np.array([-0.1484,  0.05315, BOSS_Z]),
    "RL": np.array([ 0.1484, -0.05315, BOSS_Z]),
    "RR": np.array([-0.1484, -0.05315, BOSS_Z]),
}

ROTATE_STEPS = 300
SD_Z_OFFSET  = 0.046  # Z from gripper_tip down to screwdriver cup tip



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

def solve_ik(m, d, target, seed, max_iter=600, tol=5e-4):
    d2 = mujoco.MjData(m)
    d2.qpos[:] = d.qpos[:]
    d2.qpos[:6] = seed
    sid  = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripper_tip")
    jacp = np.zeros((3, m.nv))
    err  = 999.0
    for i in range(max_iter):
        mujoco.mj_forward(m, d2)
        ev  = target - d2.site_xpos[sid]
        err = np.linalg.norm(ev)
        if err < tol:
            print(f"    IK ok {i+1} iters err={err*1000:.2f}mm"); break
        mujoco.mj_jacSite(m, d2, jacp, None, sid)
        J = jacp[:, :6]
        d2.qpos[:6] += J.T @ np.linalg.solve(J@J.T + 1e-4*np.eye(3), ev) * 0.5
        d2.qpos[:6]  = np.clip(d2.qpos[:6], -2*np.pi, 2*np.pi)
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
        d.qpos[fj+3:fj+7] = [1,0,0,0]
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
def ikc(m, d, tgt, seed):
    j = solve_ik(m, d, tgt, seed)
    c = np.zeros(m.nu); c[:6] = j; return c

def ik_move(m, d, v, tgt, seed, label, locked=None):
    c = ikc(m, d, tgt, seed)
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

def hide_dock_vis(m, tool):
    _body_alpha(m, "dock_pin_vis" if tool=="pin" else "dock_sd_vis", 0.0)

def show_dock_vis(m, tool):
    _body_alpha(m, "dock_pin_vis" if tool=="pin" else "dock_sd_vis", 1.0)

VAC_G = ["g_flange","g_body","g_accent","g_cup"]
PIN_G = ["pin_body","pin_l","pin_r"]
SD_G  = ["sd_flange","sd_spring","sd_body","sd_shaft","sd_bit"]

def show_tool(m, tool, d=None, v=None):
    all_g = {"vacuum": VAC_G, "pin": PIN_G, "screwdriver": SD_G}
    sg = all_g[tool]
    hg = [g for k, glist in all_g.items() if k != tool for g in glist]
    N  = 200 if d is not None else 1
    for s in range(N):
        te = ((s+1)/N)**2*(3-2*(s+1)/N)
        for g in sg:
            gid = mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,g)
            if gid>=0: m.geom_rgba[gid][3] = te
        for g in hg:
            gid = mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,g)
            if gid>=0: m.geom_rgba[gid][3] = 1.0-te
        if d is not None: mujoco.mj_step(m,d); v.sync()

def drive_screw(m, d, v, name, sfj, sfv):
    boss = SCREW_INSTALL_POS[name]
    sid  = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripper_tip")
    print(f"\n  [SCREW {name}]")

    # Pick from box
    pick_tgt  = np.array([0.650, 0.200, 0.819 + SD_Z_OFFSET])
    pick_ctrl = ik_move(m, d, v, pick_tgt, SCREW_IK_SEED, f"  SD->box ({name})")

    # Attach screw to SD tip
    mujoco.mj_forward(m, d)
    g_tip        = d.site_xpos[sid].copy()
    sd_tip       = g_tip - np.array([0, 0, SD_Z_OFFSET])
    screw_origin = sd_tip - np.array([0, 0, 0.012])
    screw_off    = screw_origin - sd_tip   # [0,0,-0.012]
    d.qpos[sfj:sfj+3]   = screw_origin
    d.qpos[sfj+3:sfj+7] = [1,0,0,0]
    d.qvel[sfv:sfv+6]   = 0
    print(f"  ✓ {name} attached to SD")

    def carry_screw(ctrl, n=80):
        for _ in range(n):
            d.ctrl[:] = ctrl
            mujoco.mj_forward(m, d)
            g_t = d.site_xpos[sid].copy()
            sd_t = g_t - np.array([0,0,SD_Z_OFFSET])
            d.qpos[sfj:sfj+3] = sd_t + screw_off
            d.qvel[sfv:sfv+6] = 0
            mujoco.mj_step(m, d); v.sync()

    # Hover
    hover_tgt  = np.array([boss[0], boss[1], boss[2]+0.050+SD_Z_OFFSET])
    hover_ctrl = ik_move(m, d, v, hover_tgt, SCREW_IK_SEED, f"  hover {name}")
    carry_screw(hover_ctrl)

    # Insert
    ins_tgt  = np.array([boss[0], boss[1], boss[2]+SD_Z_OFFSET])
    ins_ctrl = ik_move(m, d, v, ins_tgt, SCREW_IK_SEED, f"  insert {name}")
    carry_screw(ins_ctrl)

    # Rotate 3 turns
    print(f"  Rotating {name}...")
    w3_0 = d.qpos[5]
    for i in range(ROTATE_STEPS):
        d.ctrl[5] = w3_0 + (i/ROTATE_STEPS)*(3*2*np.pi)
        mujoco.mj_forward(m, d)
        g_t = d.site_xpos[sid].copy()
        sd_t = g_t - np.array([0,0,SD_Z_OFFSET])
        d.qpos[sfj:sfj+3] = sd_t + screw_off
        d.qvel[sfv:sfv+6] = 0
        mujoco.mj_step(m, d); v.sync()

    # Snap to installed
    d.qpos[sfj:sfj+3]   = boss - np.array([0,0,0.005])
    d.qpos[sfj+3:sfj+7] = [1,0,0,0]
    d.qvel[sfv:sfv+6]   = 0
    print(f"  ✓ {name} installed")

    d.ctrl[5] = w3_0
    move_to(m, d, v, hover_ctrl, f"  retract {name}")


# ---------------------------------------------------------------------------
# TOOL CHANGE PHASES
# ---------------------------------------------------------------------------

def phase6_vac_to_pin(m, d, v, timer):
    print(); print("="*60); print("[PHASE 6] Vacuum -> Pin Gripper"); print("="*60)
    ik_move(m, d, v, DOCK_VAC_W2+[0,0,0.08],         DOCK_IK_SEED, "  vac W1")
    ik_move(m, d, v, DOCK_VAC_W2,                     DOCK_IK_SEED, "  vac W2")
    ik_move(m, d, v, DOCK_VAC_W2+[WINGMAN_SLIDE_X,0,0], DOCK_IK_SEED, "  vac W4")
    ik_move(m, d, v, DOCK_PIN_W2+[0,0,0.08],         DOCK_IK_SEED, "  pin W1")
    ik_move(m, d, v, DOCK_PIN_W2,                     DOCK_IK_SEED, "  pin W2")
    ik_move(m, d, v, DOCK_PIN_W2+[WINGMAN_SLIDE_X,0,0], DOCK_IK_SEED, "  pin W4")
    show_tool(m, "pin", d, v)
    hide_dock_vis(m, "pin")
    print("  ✓ Pin gripper active")
    timer.mark(d, "Tool change: vac->pin")

def phase8_pin_to_sd(m, d, v, home_ctrl, timer):
    print(); print("="*60); print("[PHASE 8] Pin -> Screwdriver"); print("="*60)
    move_to(m, d, v, home_ctrl, "  home")
    ik_move(m, d, v, DOCK_PIN_W2+[0,0,0.08],          DOCK_IK_SEED, "  pin return W1")
    ik_move(m, d, v, DOCK_PIN_W2,                      DOCK_IK_SEED, "  pin return W2")
    ik_move(m, d, v, DOCK_PIN_W2-[WINGMAN_SLIDE_X,0,0], DOCK_IK_SEED, "  pin W4 back")
    show_dock_vis(m, "pin")
    ik_move(m, d, v, DOCK_SD_W2+[0,0,0.08],           DOCK_IK_SEED, "  SD W1")
    ik_move(m, d, v, DOCK_SD_W2,                       DOCK_IK_SEED, "  SD W2")
    ik_move(m, d, v, DOCK_SD_W2+[WINGMAN_SLIDE_X,0,0],  DOCK_IK_SEED, "  SD W4")
    show_tool(m, "screwdriver", d, v)
    hide_dock_vis(m, "screwdriver")
    print("  ✓ Screwdriver active")
    timer.mark(d, "Tool change: pin->SD")

def phase9_screws(m, d, v, home_ctrl, timer):
    print(); print("="*60); print("[PHASE 9] Screwdriving"); print("="*60)
    screws = {n: get_body_jnt(m, f"screw_{n}") for n in SCREW_NAMES}
    for name in SCREW_NAMES:
        sfj, sfv = screws[name]
        drive_screw(m, d, v, name, sfj, sfv)
        timer.mark(d, f"Screw {name}")
    print("\n  Returning SD...")
    ik_move(m, d, v, DOCK_SD_W2+[0,0,0.08], DOCK_IK_SEED, "  SD return W1")
    ik_move(m, d, v, DOCK_SD_W2,             DOCK_IK_SEED, "  SD return W2")
    show_dock_vis(m, "screwdriver")
    move_to(m, d, v, home_ctrl, "  final home")
    print("✓ All phases complete.")


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

    with mujoco.viewer.launch_passive(m, d) as v:
        v.cam.lookat[:] = [0.0, 0.3, 0.83]
        v.cam.distance  = 1.5
        v.cam.elevation = -22
        v.cam.azimuth   = 145

        all_locks = [(part_jnts[b][0], part_jnts[b][1], PICK_WORLD[b], None)
                     for b in PARTS] + [TRAY_LOCK]
        move_to(m, d, v, home_ctrl, "home", locked=all_locks)

        timer = CycleTimer("Station 1")
        timer.start(d)

        remaining = list(PARTS)  # pops as each part is picked

        for step_idx, bname in enumerate(PARTS):
            fj_adr, fv_adr = part_jnts[bname]

            # Parts still in tray (excluding current)
            waiting = [(part_jnts[b][0], part_jnts[b][1], PICK_WORLD[b], None)
                       for b in remaining if b != bname] + [TRAY_LOCK]

            use_pin = (bname == "s1_alu_plate")
            print(); print("="*60)
            print(f"[STEP {step_idx+1}/{len(PARTS)}] {bname.upper()}"
                  f"  ({'pin' if use_pin else 'vacuum'})")
            print("="*60)

            if use_pin:
                phase6_vac_to_pin(m, d, v, timer)
                # Atomically release tray weld and arm gripper weld in the same
                # sim step so the plate never enters free-fall between the two.

            pw  = PICK_WORLD[bname]
            plw = PLACE_PART_WORLD[bname]

            # -- PICK -------------------------------------------------------
            grasp = pw + np.array([0, 0, GRASP_Z_OFFSET[bname]])
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
        placed_locks = [(part_jnts[b][0], part_jnts[b][1], PLACE_PART_WORLD[b], None)
                        for b in PARTS] + [TRAY_LOCK]
        move_to(m, d, v, home_ctrl, "home after assembly", locked=placed_locks)
        timer.mark(d, "Assembly complete")

        # Phases 8-9: pin->SD, drive screws
        phase8_pin_to_sd(m, d, v, home_ctrl, timer)
        phase9_screws(m, d, v, home_ctrl, timer)

        print(); print("="*60)
        print("Station 1 complete. Close viewer to exit.")
        print("="*60)
        timer.finish(d)
        timer.print_report()

        while v.is_running():
            mujoco.mj_step(m, d); v.sync()


if __name__ == "__main__":
    main()