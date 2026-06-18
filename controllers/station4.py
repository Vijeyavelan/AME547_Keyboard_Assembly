"""
station4_controller.py — Keycap installation, 84 keycaps
Identical architecture to station3_controller.py with these differences:
  - Keycap bodies (kc_*) instead of switch bodies (sw_*)
  - kc_* sites at switch stem top (z_local=0.012 above sw_* site)
  - No MESH_OFFSET_Z — keycap body origin = bottom face
  - SWITCH_TOP_OFFSET=0.012: keycap bottom seats on switch stem top
  - CARRY_Z_OFFSET=-0.0036: keycap center (half of 7.23mm height)
  - Same tray belt advance, same rotate+XY pattern, same dual-head

Run from repo root:
    mjpython controllers/station4.py
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cycle_timer import CycleTimer
MODEL_PATH = "models/stations/station4.xml"

TIMESTEP       = 0.002
HEAD_Z_STEPS   = 150
ROTATE_STEPS   = 200
ADVANCE_STEPS  = 200
SETTLE_STEPS   = 30

HEAD_Z_INSERT  = -0.015   # head descends 15mm to press keycap onto stem
HEAD_Z_RETRACT =  0.000

# Keycap geometry
SWITCH_TOP_OFFSET = 0.012   # kc_* site is 12mm above plate site
                             # keycap bottom = kc_site_world_z (no extra offset)
CARRY_Z_OFFSET    = -0.0036 # keycap center above tip (half of 7.23mm height)

TRAY_WORLD_Y  = 0.300
TRAY_WORLD_Z  = 0.8411      # keycap bottom face at tray surface
PITCH         = 0.01905     # 19.05mm slot pitch

# 84 keycaps — same layout as switches
KEYCAPS = [
    ("Esc",-0.1429,0.0480),("F1",-0.1238,0.0480),("F2",-0.1048,0.0480),
    ("F3",-0.0857,0.0480),("F4",-0.0667,0.0480),("F5",-0.0476,0.0480),
    ("F6",-0.0286,0.0480),("F7",-0.0095,0.0480),("F8",0.0095,0.0480),
    ("F9",0.0286,0.0480),("F10",0.0476,0.0480),("F11",0.0667,0.0480),
    ("F12",0.0857,0.0480),("PrtSc",0.1048,0.0480),("Pause",0.1238,0.0480),
    ("Del",0.1429,0.0480),("Grave",-0.1429,0.0290),("1",-0.1238,0.0290),
    ("2",-0.1048,0.0290),("3",-0.0857,0.0290),("4",-0.0667,0.0290),
    ("5",-0.0476,0.0290),("6",-0.0286,0.0290),("7",-0.0095,0.0290),
    ("8",0.0095,0.0290),("9",0.0286,0.0290),("0",0.0476,0.0290),
    ("Minus",0.0667,0.0290),("Equal",0.0857,0.0290),("Bksp",0.1143,0.0290),
    ("PgUp",0.1429,0.0290),("Tab",-0.1381,0.0099),("Q",-0.1143,0.0099),
    ("W",-0.0953,0.0099),("E",-0.0762,0.0099),("R",-0.0572,0.0099),
    ("T",-0.0381,0.0099),("Y",-0.0191,0.0099),("U",0.0000,0.0099),
    ("I",0.0191,0.0099),("O",0.0381,0.0099),("P",0.0572,0.0099),
    ("LBrace",0.0762,0.0099),("RBrace",0.0953,0.0099),
    ("Bkslash",0.1191,0.0099),("PgDn",0.1429,0.0099),
    ("Caps",-0.1357,-0.0092),("A",-0.1095,-0.0092),("S",-0.0905,-0.0092),
    ("D",-0.0714,-0.0092),("F",-0.0524,-0.0092),("G",-0.0333,-0.0092),
    ("H",-0.0143,-0.0092),("J",0.0048,-0.0092),("K",0.0238,-0.0092),
    ("L",0.0429,-0.0092),("Semicol",0.0619,-0.0092),("Quote",0.0810,-0.0092),
    ("Enter",0.1119,-0.0092),("Home",0.1429,-0.0092),
    ("LShift",-0.1310,-0.0282),("Z",-0.1000,-0.0282),("X",-0.0810,-0.0282),
    ("C",-0.0619,-0.0282),("V",-0.0429,-0.0282),("B",-0.0238,-0.0282),
    ("N",-0.0048,-0.0282),("M",0.0143,-0.0282),("Comma",0.0333,-0.0282),
    ("Period",0.0524,-0.0282),("Slash",0.0714,-0.0282),
    ("RShift",0.0976,-0.0282),("Up",0.1238,-0.0282),("End",0.1429,-0.0282),
    ("LCtrl",-0.1405,-0.0473),("LWin",-0.1167,-0.0473),
    ("LAlt",-0.0929,-0.0473),("Space",-0.0214,-0.0473),
    ("RAlt",0.0476,-0.0473),("Fn",0.0667,-0.0473),
    ("Left",0.0857,-0.0473),("Down",0.1048,-0.0473),
    ("Right",0.1238,-0.0473),("End2",0.1429,-0.0473),
]
assert len(KEYCAPS) == 84
N = len(KEYCAPS)
KEYCAP_MAP  = {k: (sx, sy) for k, sx, sy in KEYCAPS}
KEYCAP_KEYS = [k for k, _, _ in KEYCAPS]

# Keycap widths in metres (measured from STL)
KEYCAP_WIDTHS = {
    "Esc":0.018,"F1":0.018,"F2":0.018,"F3":0.018,"F4":0.018,
    "F5":0.018,"F6":0.018,"F7":0.018,"F8":0.018,"F9":0.018,
    "F10":0.018,"F11":0.018,"F12":0.018,"PrtSc":0.018,
    "Pause":0.018,"Del":0.018,"Grave":0.018,"1":0.018,"2":0.018,
    "3":0.018,"4":0.018,"5":0.018,"6":0.018,"7":0.018,"8":0.018,
    "9":0.018,"0":0.018,"Minus":0.018,"Equal":0.018,
    "Bksp":0.0370,"PgUp":0.018,"Tab":0.0275,"Q":0.018,"W":0.018,
    "E":0.018,"R":0.018,"T":0.018,"Y":0.018,"U":0.018,"I":0.018,
    "O":0.018,"P":0.018,"LBrace":0.018,"RBrace":0.018,
    "Bkslash":0.0275,"PgDn":0.018,"Caps":0.0323,"A":0.018,
    "S":0.018,"D":0.018,"F":0.018,"G":0.018,"H":0.018,"J":0.018,
    "K":0.018,"L":0.018,"Semicol":0.018,"Quote":0.018,
    "Enter":0.0418,"Home":0.018,"LShift":0.0418,"Z":0.018,
    "X":0.018,"C":0.018,"V":0.018,"B":0.018,"N":0.018,"M":0.018,
    "Comma":0.018,"Period":0.018,"Slash":0.018,
    "RShift":0.0323,"Up":0.018,"End":0.018,
    "LCtrl":0.0228,"LWin":0.0228,"LAlt":0.0228,
    "Space":0.1180,"RAlt":0.0228,"Fn":0.0228,
    "Left":0.018,"Down":0.018,"Right":0.018,"End2":0.018,
}

GAP = 0.002  # 2mm gap between keycaps in tray


def build_keycap_tray_x(keys, widths, gap):
    """
    Return initial tray X positions for a variable-width keycap row.

    The first keycap center starts exactly at the pickup X reference. Each
    following keycap is placed one center-to-center step behind it:

        previous_width/2 + gap + current_width/2

    This matches the belt advance used before every pick, so after advancing
    from key i-1 to key i, the target key's center lands under the pickup tip.
    """
    tray_x = {}
    current_center_x = 0.0

    for i, key in enumerate(keys):
        if i == 0:
            tray_x[key] = current_center_x
            continue

        prev_key = keys[i - 1]
        step = widths[prev_key] / 2.0 + gap + widths[key] / 2.0
        current_center_x -= step
        tray_x[key] = current_center_x

    return tray_x


def keycap_advance_amount(prev_key, current_key):
    """Center-to-center belt step from prev_key to current_key."""
    return KEYCAP_WIDTHS[prev_key] / 2.0 + GAP + KEYCAP_WIDTHS[current_key] / 2.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_aid(m, name):
    i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if i < 0: raise ValueError(f"Actuator '{name}' not found")
    return i

def get_jnt(m, name):
    i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
    if i < 0: raise ValueError(f"Joint '{name}' not found")
    return m.jnt_qposadr[i], m.jnt_dofadr[i]

def get_eid(m, name):
    i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, name)
    if i < 0: raise ValueError(f"Weld '{name}' not found")
    return i

def get_body_jnt(m, bname):
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, bname)
    jid = m.body_jntadr[bid]
    return m.jnt_qposadr[jid], m.jnt_dofadr[jid]

def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def socket_to_ctrl(sx, sy):
    """Table ctrl to bring plate socket (sx,sy) under insert head (world x=0, y=-0.050)."""
    xm = float(np.clip(-sx,        -0.160, 0.160))
    ym = float(np.clip(0.010 - sy, -0.115, 0.115))
    return xm, ym

def snap_to_site(m, d, key):
    """Snap keycap body so its bottom face sits exactly on the kc_* site."""
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"kc_{key}")
    mujoco.mj_forward(m, d)
    sp = d.site_xpos[sid].copy()
    qa, da = get_body_jnt(m, f"kc_{key}")
    # Keycap body origin = bottom face, so qpos z = site world z directly
    d.qpos[qa+0]      = sp[0]
    d.qpos[qa+1]      = sp[1]
    d.qpos[qa+2]      = sp[2]
    d.qpos[qa+3]      = 1.0
    d.qpos[qa+4:qa+7] = 0.0
    d.qvel[da:da+6]   = 0.0

def place_on_tray(d, qa, da, x):
    """Pin a keycap body at tray position x."""
    d.qpos[qa+0] = x
    d.qpos[qa+1] = TRAY_WORLD_Y
    d.qpos[qa+2] = TRAY_WORLD_Z
    d.qpos[qa+3] = 1.0
    d.qpos[qa+4] = 0.0
    d.qpos[qa+5] = 0.0
    d.qpos[qa+6] = 0.0
    d.qvel[da:da+6] = 0.0


# ── Simulation step ───────────────────────────────────────────────────────────

def sim_step(m, d, v, tbl, inserted_map, tray_state, carried_map):
    """
    Single physics step with four bypass operations:
    1. Table X/Y teleport
    2. Inserted keycap re-snap (follow plate as table moves)
    3. Tray keycap gravity lock
    4. Carried keycap re-snap (pin to head tip during carry and rotation)
    """
    x_qa, y_qa, x_doa, y_doa, x_aid, y_aid = tbl

    # 1. Table teleport
    d.qpos[x_qa]  = d.ctrl[x_aid]
    d.qpos[y_qa]  = d.ctrl[y_aid]
    d.qvel[x_doa] = 0.0
    d.qvel[y_doa] = 0.0

    # 2. Re-snap installed keycaps to their plate sites
    for (qa, da, sid) in inserted_map.values():
        sp = d.site_xpos[sid]
        d.qpos[qa+0] = sp[0]
        d.qpos[qa+1] = sp[1]
        d.qpos[qa+2] = sp[2]   # body origin = bottom face, sits at site Z
        d.qpos[qa+3] = 1.0
        d.qpos[qa+4] = 0.0
        d.qpos[qa+5] = 0.0
        d.qpos[qa+6] = 0.0
        d.qvel[da:da+6] = 0.0

    # 3. Gravity lock all tray keycaps
    for key, (qa, da) in tray_state.items():
        d.qpos[qa+1] = TRAY_WORLD_Y
        d.qpos[qa+2] = TRAY_WORLD_Z
        d.qpos[qa+3] = 1.0
        d.qpos[qa+4] = 0.0
        d.qpos[qa+5] = 0.0
        d.qpos[qa+6] = 0.0
        d.qvel[da:da+6] = 0.0

    # 4. Pin carried keycaps to head tip
    for (qa, da, tip_sid) in carried_map.values():
        tip = d.site_xpos[tip_sid]
        d.qpos[qa+0] = tip[0]
        d.qpos[qa+1] = tip[1]
        d.qpos[qa+2] = tip[2] + CARRY_Z_OFFSET
        d.qpos[qa+3] = 1.0
        d.qpos[qa+4] = 0.0
        d.qpos[qa+5] = 0.0
        d.qpos[qa+6] = 0.0
        d.qvel[da:da+6] = 0.0

    t0 = time.time()
    mujoco.mj_step(m, d)
    elapsed = time.time() - t0
    time.sleep(max(0, TIMESTEP - elapsed))
    v.sync()


def drive_to(m, d, v, targets, steps, label, tbl, inserted_map, tray_state, carried_map):
    """Smoothly drive actuators to target values over N steps."""
    start = {name: d.ctrl[get_aid(m, name)] for name in targets}
    for i in range(steps):
        t = smoothstep((i + 1) / steps)
        for name, target in targets.items():
            d.ctrl[get_aid(m, name)] = start[name] + (target - start[name]) * t
        sim_step(m, d, v, tbl, inserted_map, tray_state, carried_map)
    if label:
        print(f"  ✓ {label}")


def settle(m, d, v, steps, tbl, inserted_map, tray_state, carried_map):
    for _ in range(steps):
        sim_step(m, d, v, tbl, inserted_map, tray_state, carried_map)


def tray_advance_step(d, tray_state, tray_x, advance_start_x, step, total_steps, advance_amount):
    t = smoothstep((step + 1) / total_steps)
    for key, (qa, da) in tray_state.items():
        new_x = advance_start_x[key] + advance_amount * t
        tray_x[key] = new_x
        d.qpos[qa+0] = new_x


def do_pick_and_insert(m, d, v,
                       ins_head_idx, ins_key,
                       pick_head_idx, pick_key,
                       tbl, inserted_map, tray_state, tray_x, carried_map):
    """
    Simultaneous insert (press keycap onto stem) + pick (lift from tray).
    Mirrors S3 do_pick_and_insert exactly with keycap naming.
    """
    ins_head  = f"head_{ins_head_idx+1}"
    pick_head = f"head_{pick_head_idx+1}"
    ins_drive  = f"{ins_head}_drive"
    pick_drive = f"{pick_head}_drive"
    ins_tip_sid  = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{ins_head}_tip_site")
    pick_tip_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{pick_head}_tip_site")

    has_ins  = ins_key is not None
    has_pick = pick_key is not None

    # ── Both heads descend simultaneously ────────────────────────────────────
    targets = {}
    if has_ins:
        targets[ins_drive]  = HEAD_Z_INSERT
    if has_pick:
        targets[pick_drive] = HEAD_Z_INSERT
    if targets:
        drive_to(m, d, v, targets, HEAD_Z_STEPS, "", tbl, inserted_map, tray_state, carried_map)

    mujoco.mj_forward(m, d)

    # ── INSERT: snap keycap to plate site, activate weld ────────────────────
    if has_ins:
        snap_to_site(m, d, ins_key)
        mujoco.mj_forward(m, d)
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"kc_{ins_key}")
        qa, da = get_body_jnt(m, f"kc_{ins_key}")
        inserted_map[ins_key] = (qa, da, sid)
        # Deactivate pick welds, activate insert weld
        for h in range(2):
            eid = get_eid(m, f"pick{h+1}_{ins_key}")
            m.eq_active0[eid] = 0
        eid = get_eid(m, f"ins_{ins_key}")
        m.eq_active0[eid] = 1
        # Remove from carried_map
        if ins_head_idx in carried_map:
            del carried_map[ins_head_idx]
        print(f"    → kc_{ins_key} snapped to plate")

    # ── PICK: teleport keycap to pick head tip, activate pick weld ──────────
    if has_pick and pick_key in tray_state:
        mujoco.mj_forward(m, d)
        tip = d.site_xpos[pick_tip_sid].copy()
        qa, da = get_body_jnt(m, f"kc_{pick_key}")
        # Teleport keycap center to head tip
        d.qpos[qa+0] = tip[0]
        d.qpos[qa+1] = tip[1]
        d.qpos[qa+2] = tip[2] + CARRY_Z_OFFSET
        d.qpos[qa+3] = 1.0
        d.qpos[qa+4:qa+7] = 0.0
        d.qvel[da:da+6]   = 0.0
        mujoco.mj_forward(m, d)
        # Activate pick weld
        eid = get_eid(m, f"pick{pick_head_idx+1}_{pick_key}")
        m.eq_active0[eid] = 1
        # Add to carried_map, remove from tray
        carried_map[pick_head_idx] = (qa, da, pick_tip_sid)
        del tray_state[pick_key]
        print(f"    → kc_{pick_key} picked from tray")

    settle(m, d, v, SETTLE_STEPS, tbl, inserted_map, tray_state, carried_map)

    # ── Both heads retract simultaneously ────────────────────────────────────
    targets = {}
    if has_ins:
        targets[ins_drive]  = HEAD_Z_RETRACT
    if has_pick:
        targets[pick_drive] = HEAD_Z_RETRACT
    if targets:
        drive_to(m, d, v, targets, HEAD_Z_STEPS, "", tbl, inserted_map, tray_state, carried_map)

# Add this before the rotation loop:

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    m = mujoco.MjModel.from_xml_path(MODEL_PATH)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)

    # ── Verify welds & sites ──────────────────────────────────────────────────
    print("Verifying welds (252)...")
    try:
        for key, _, _ in KEYCAPS:
            get_eid(m, f"pick1_{key}")
            get_eid(m, f"pick2_{key}")
            get_eid(m, f"ins_{key}")
        print("  All 252 welds ✓")
    except ValueError as e:
        print(f"  ERROR: {e}"); return

    print("Verifying sites (84)...")
    missing = [k for k, _, _ in KEYCAPS
               if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"kc_{k}") < 0]
    if missing:
        print(f"  ERROR: missing sites: {missing}"); return
    print("  All 84 sites ✓")

    # ── Resolve tray keycap joints ────────────────────────────────────────────
    print("Resolving tray keycap joints...")
    tray_state = {}
    tray_x     = {}

    tray_x = build_keycap_tray_x(KEYCAP_KEYS, KEYCAP_WIDTHS, GAP)
    for key in KEYCAP_KEYS:
        qa, da = get_body_jnt(m, f"kc_{key}")
        tray_state[key] = (qa, da)
    print(f"  {len(tray_state)} keycaps in tray ✓")
    print("  Tray pickup reference: first keycap center starts at x=0.00mm")

    # Place all keycaps at tray positions
    print("Loading tray...")
    for key, (qa, da) in tray_state.items():
        place_on_tray(d, qa, da, tray_x[key])
    mujoco.mj_forward(m, d)
    print("  Tray loaded ✓")

    # ── Joint / actuator addresses ────────────────────────────────────────────
    x_qa,  x_doa  = get_jnt(m, "x_joint")
    y_qa,  y_doa  = get_jnt(m, "y_joint")
    x_aid_v = get_aid(m, "x_drive")
    y_aid_v = get_aid(m, "y_drive")
    col_aid = get_aid(m, "col_rotate")
    tbl = (x_qa, y_qa, x_doa, y_doa, x_aid_v, y_aid_v)

    inserted_map = {}   # key → (qa, da, site_id) for installed keycaps
    carried_map  = {}   # head_idx → (qa, da, tip_site_id) for in-transit keycaps

    def ss():
        sim_step(m, d, v, tbl, inserted_map, tray_state, carried_map)
    def dt(targets, steps, label=""):
        drive_to(m, d, v, targets, steps, label, tbl, inserted_map, tray_state, carried_map)
    def stl(steps):
        settle(m, d, v, steps, tbl, inserted_map, tray_state, carried_map)

    installed    = 0
    col_angle    = 0.0
    col_sign     = +1
    head_carry   = [None, None]
    kc_queue_idx = 0

    with mujoco.viewer.launch_passive(m, d) as v:
        v.cam.lookat[:] = [0.0, 0.05, 0.83]
        v.cam.distance  = 1.5
        v.cam.elevation = -25
        v.cam.azimuth   = 150

        # ── HOME ─────────────────────────────────────────────────────────────
        print("\n[INIT] Homing...")
        d.ctrl[x_aid_v] = 0.0
        d.ctrl[y_aid_v] = 0.0
        d.ctrl[col_aid] = 0.0
        dt({"head_1_drive": HEAD_Z_RETRACT,
            "head_2_drive": HEAD_Z_RETRACT}, 150, "heads home")
        stl(SETTLE_STEPS)

        mujoco.mj_forward(m, d)
        for h, site in enumerate(["head_1_tip_site", "head_2_tip_site"]):
            sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site)
            wp  = d.site_xpos[sid]
            print(f"  Head {h+1} tip @ home: "
                  f"x={wp[0]:.4f}  y={wp[1]:.4f}  z={wp[2]:.4f}")

        # ── PRIME: Head 2 picks kc_Esc ───────────────────────────────────────
        print("\n[PRIME] Head 2 picks kc_Esc from tray...")
        do_pick_and_insert(m, d, v,
                           ins_head_idx=0,  ins_key=None,
                           pick_head_idx=1, pick_key="Esc",
                           tbl=tbl, inserted_map=inserted_map,
                           tray_state=tray_state, tray_x=tray_x,
                           carried_map=carried_map)
        head_carry[1] = "Esc"
        kc_queue_idx  = 1
        print(f"  ✓ Head 2 carries kc_Esc")

        insert_head = 1
        cycle = 0
        timer = CycleTimer("Station 4")
        timer.start(d)
        # ── MAIN LOOP ─────────────────────────────────────────────────────────
        while installed < N:
            pick_head = 1 - insert_head
            ins_key   = head_carry[insert_head]
            has_ins   = ins_key is not None
            pick_key  = KEYCAPS[kc_queue_idx][0] if kc_queue_idx < N else None

            cycle += 1
            print()
            print("=" * 60)
            print(f"[CYCLE {cycle:02d}]  "
                  f"Install: kc_{ins_key or 'none':12s}  "
                  f"Pick: kc_{pick_key or 'none'}")
            print("=" * 60)

            # ── Rotate + XY + tray belt advance ──────────────────────────────
            col_angle += col_sign * np.pi
            col_angle  = (col_angle + np.pi) % (2 * np.pi) - np.pi
            col_sign   = -col_sign

            if has_ins:
                ins_sx, ins_sy = KEYCAP_MAP[ins_key]
                ins_xc, ins_yc = socket_to_ctrl(ins_sx, ins_sy)
            else:
                ins_xc, ins_yc = 0.0, 0.0

            if pick_key is not None and kc_queue_idx > 0:
                prev_key = KEYCAP_KEYS[kc_queue_idx - 1]
                advance_amount = keycap_advance_amount(prev_key, pick_key)
            else:
                advance_amount = 0.0

            print(f"  Rotate → {np.degrees(col_angle):.0f}°  "
                  f"XY → ({ins_xc*1000:.1f}, {ins_yc*1000:.1f}) mm  "
                  f"Belt advance +{advance_amount*1000:.2f} mm")

            col_start = d.ctrl[col_aid]
            x_start   = d.ctrl[x_aid_v]
            y_start   = d.ctrl[y_aid_v]
            XY_STEPS  = int(ROTATE_STEPS * 0.8)

            advance_start_x = {k: tray_x[k] for k in tray_state}

            for step in range(ROTATE_STEPS):
                tc = smoothstep((step + 1) / ROTATE_STEPS)
                d.ctrl[col_aid] = col_start + (col_angle - col_start) * tc

                if step < XY_STEPS:
                    tx = smoothstep((step + 1) / XY_STEPS)
                    d.ctrl[x_aid_v] = x_start + (ins_xc - x_start) * tx
                    d.ctrl[y_aid_v] = y_start + (ins_yc - y_start) * tx

                    
                if step < ADVANCE_STEPS and pick_key is not None:
                    tray_advance_step(d, tray_state, tray_x,
                                      advance_start_x, step, ADVANCE_STEPS, advance_amount)
                ss()

            stl(SETTLE_STEPS)
            print(f"  ✓ rotation + XY + belt advance done")
            timer.mark(d, f"C{cycle:02d} rotate+XY+belt ({ins_key or 'none'}→{pick_key or 'none'})")
            # ── Simultaneous install + pick ───────────────────────────────────
            if pick_key is not None:
                px = tray_x.get(pick_key, None)
                if px is not None:
                    pick_tip_site = f"head_{pick_head+1}_tip_site"
                    pick_tip_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, pick_tip_site)
                    mujoco.mj_forward(m, d)
                    pick_tip_x = d.site_xpos[pick_tip_sid][0]
                    print(f"  Tray: kc_{pick_key} at x={px*1000:.2f}mm; "
                          f"{pick_tip_site} x={pick_tip_x*1000:.2f}mm; "
                          f"pickup offset={(px-pick_tip_x)*1000:.2f}mm")

            do_pick_and_insert(m, d, v,
                               ins_head_idx=insert_head, ins_key=ins_key if has_ins else None,
                               pick_head_idx=pick_head,  pick_key=pick_key,
                               tbl=tbl, inserted_map=inserted_map,
                               tray_state=tray_state, tray_x=tray_x,
                               carried_map=carried_map)
            timer.mark(d, f"C{cycle:02d} insert+pick ({ins_key or 'none'}+{pick_key or 'none'})")
            if has_ins:
                head_carry[insert_head] = None
                installed += 1
                print(f"  ✓ [{installed}/{N}] kc_{ins_key} installed")

            if pick_key is not None:
                head_carry[pick_head] = pick_key
                kc_queue_idx += 1
                print(f"  ✓ kc_{pick_key} picked  [queue → {kc_queue_idx}/{N}]")

            insert_head = pick_head

        # ── DONE ─────────────────────────────────────────────────────────────
        print()
        print("=" * 60)
        print(f"[DONE] All {installed}/{N} keycaps installed.")
        print("=" * 60)
        timer.finish(d)
        timer.print_report(ref_key="keycap_fast")
        d.ctrl[x_aid_v] = 0.0
        d.ctrl[y_aid_v] = 0.0
        d.ctrl[col_aid] = 0.0
        dt({"head_1_drive": HEAD_Z_RETRACT,
            "head_2_drive": HEAD_Z_RETRACT}, 300, "final home")

        print("Holding — close window to exit.")
        while v.is_running():
            sim_step(m, d, v, tbl, inserted_map, tray_state, carried_map)


if __name__ == "__main__":
    main()
