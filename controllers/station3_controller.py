"""
station3_controller.py — Sequential insert → pick with tray belt advance, 84 switches

Tray system:
  All 84 sw_* bodies start at XML positions spread along the tray
  (sw_Esc at x=0, sw_F1 at x=-0.01905, ..., sw_End2 at x=-1.581).
  tray_queue tracks which switch is currently at each tray slot.
  tray_x[key] tracks the current world X of each un-picked switch.

  During the rotate+XY phase each cycle, the belt advances concurrently:
  every un-picked tray switch slides +PITCH in X over ADVANCE_STEPS steps,
  bringing the next switch to x=0 under the pick head.

  Gravity lock: every sim_step pins Y, Z, quat, vel of all tray switches
  so they don't drift between advances.

Pick sequence:
  Head descends to tray depth (switch is already at x=0, y=0.300).
  Switch is teleported precisely to head tip, pick weld activates, head rises.
  Switch visually lifts off the tray.

Insert sequence (unchanged):
  Head descends, snap_to_site, weld swap, head rises.
  inserted_map re-snaps every placed switch to its plate site every step.

Cycle structure (after prime):
  [Rotate 180° + XY to insert + tray belt advance (concurrent)]
  [Sub-A] Insert head DOWN → snap → weld swap → UP
  [Sub-B] Pick head DOWN   → teleport to tip → pick weld → UP
  Swap heads. Repeat.

Run from repo root:
    mjpython controllers/station3_controller.py
"""

import mujoco
import mujoco.viewer
import numpy as np
import time

MODEL_PATH = "models/stations/station3.xml"

TIMESTEP      = 0.002
HEAD_Z_STEPS  = 150
ROTATE_STEPS  = 350
ADVANCE_STEPS = 350   # belt advance concurrent with rotation — same step count
SETTLE_STEPS  = 30

HEAD_Z_INSERT  = -0.015
HEAD_Z_RETRACT =  0.000

MESH_OFFSET_Z  = 0.00945

CARRY_Z_OFFSET = -0.013  # push switch 10mm below tip so 75% is visible
TRAY_WORLD_Y  = 0.300
TRAY_WORLD_Z  = 0.834
TRAY_SWITCH_Z = TRAY_WORLD_Z - MESH_OFFSET_Z   # 0.82455

PITCH = 0.01905   # 19.05 mm switch pitch

# Switch sequence — 84 keys (name, plate-local-x, plate-local-y)
SWITCHES = [
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
assert len(SWITCHES) == 84
N = len(SWITCHES)
SWITCH_MAP = {k: (sx, sy) for k, sx, sy in SWITCHES}
SWITCH_KEYS = [k for k, _, _ in SWITCHES]


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
    """Table ctrl to bring plate socket (sx,sy) under insert head (world x=0, y=0.040)."""
    xm = float(np.clip(-sx,        -0.160, 0.160))
    ym = float(np.clip(0.010 - sy, -0.115, 0.115))
    return xm, ym

def snap_to_site(m, d, key):
    """Snap switch body flush to its plate socket site."""
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"sw_{key}")
    mujoco.mj_forward(m, d)
    sp = d.site_xpos[sid].copy()
    qa, da = get_body_jnt(m, f"sw_{key}")
    d.qpos[qa+0]      = sp[0]
    d.qpos[qa+1]      = sp[1]
    d.qpos[qa+2]      = sp[2] - MESH_OFFSET_Z
    d.qpos[qa+3]      = 1.0
    d.qpos[qa+4:qa+7] = 0.0
    d.qvel[da:da+6]   = 0.0

def place_on_tray(d, qa, da, x):
    """Pin a switch body at tray position x, locking Y/Z/quat/vel."""
    d.qpos[qa+0] = x
    d.qpos[qa+1] = TRAY_WORLD_Y
    d.qpos[qa+2] = TRAY_SWITCH_Z
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
    2. Inserted switch re-snap (follow plate as table moves)
    3. Tray switch gravity lock (pin Y/Z/quat/vel of all un-picked switches)
    4. Carried switch re-snap (pin switch to head tip during carry and rotation)
    """
    x_qa, y_qa, x_doa, y_doa, x_aid, y_aid = tbl

    # 1. Table teleport
    d.qpos[x_qa]  = d.ctrl[x_aid]
    d.qpos[y_qa]  = d.ctrl[y_aid]
    d.qvel[x_doa] = 0.0
    d.qvel[y_doa] = 0.0

    # 2. Re-snap inserted switches to their plate sites
    for (qa, da, sid) in inserted_map.values():
        sp = d.site_xpos[sid]
        d.qpos[qa+0] = sp[0]
        d.qpos[qa+1] = sp[1]
        d.qpos[qa+2] = sp[2] - MESH_OFFSET_Z
        d.qpos[qa+3] = 1.0
        d.qpos[qa+4] = 0.0
        d.qpos[qa+5] = 0.0
        d.qpos[qa+6] = 0.0
        d.qvel[da:da+6] = 0.0

    # 3. Gravity lock all tray switches (pin Y/Z/quat/vel, leave X alone)
    for key, (qa, da) in tray_state.items():
        d.qpos[qa+1] = TRAY_WORLD_Y
        d.qpos[qa+2] = TRAY_SWITCH_Z
        d.qpos[qa+3] = 1.0
        d.qpos[qa+4] = 0.0
        d.qpos[qa+5] = 0.0
        d.qpos[qa+6] = 0.0
        d.qvel[da:da+6] = 0.0

    # 4. Pin carried switches exactly to their head tip (smooth arc during rotation)
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

    t0 = time.perf_counter()
    mujoco.mj_step(m, d)
    v.sync()
    rem = TIMESTEP - (time.perf_counter() - t0)
    if rem > 0:
        time.sleep(rem)


def drive_to(m, d, v, targets, steps, label, tbl, inserted_map, tray_state, carried_map):
    act_ids = {n: get_aid(m, n) for n in targets}
    starts  = {n: d.ctrl[act_ids[n]] for n in targets}
    for i in range(steps):
        t = smoothstep((i + 1) / steps)
        for n, tgt in targets.items():
            d.ctrl[act_ids[n]] = starts[n] + (tgt - starts[n]) * t
        sim_step(m, d, v, tbl, inserted_map, tray_state, carried_map)
    if label:
        print(f"  ✓ {label}")

def settle(m, d, v, steps, tbl, inserted_map, tray_state, carried_map):
    for _ in range(steps):
        sim_step(m, d, v, tbl, inserted_map, tray_state, carried_map)


# ── Tray belt advance ─────────────────────────────────────────────────────────

def tray_advance_step(d, tray_state, tray_x, start_x, step, total_steps):
    """
    Advance all tray switches one pitch toward pick position (+X direction).
    Called once per sim step during the rotate+XY phase.
    tray_x: dict key → current world X (updated in place as animation progresses)
    start_x: dict key → world X at start of this advance
    """
    t = smoothstep((step + 1) / total_steps)
    for key, (qa, da) in tray_state.items():
        new_x = start_x[key] + PITCH * t
        d.qpos[qa+0] = new_x
        tray_x[key]  = new_x


# ── Combined simultaneous pick + insert ──────────────────────────────────────

def do_pick_and_insert(m, d, v,
                       ins_head_idx, ins_key,
                       pick_head_idx, pick_key,
                       tbl, inserted_map, tray_state, tray_x, carried_map):
    """
    Both heads operate simultaneously:
      ins_head  — inserts ins_key into its plate socket
      pick_head — picks pick_key from the tray (already at x=0)

    Either argument may be None to run only one head (prime / last cycle).

    Sequence:
      1. Both heads DOWN together (single drive_to with both actuators)
      2. Weld operations (instantaneous qpos writes):
           Insert: snap_to_site → deactivate pick weld → activate ins weld
           Pick:   teleport to tip → activate pick weld
      3. mj_forward + sync
      4. Both heads UP together
      5. Settle
    """
    targets_down = {}
    targets_up   = {}

    if ins_key is not None:
        targets_down[f"head_{ins_head_idx+1}_drive"] = HEAD_Z_INSERT
        targets_up  [f"head_{ins_head_idx+1}_drive"] = HEAD_Z_RETRACT
    if pick_key is not None:
        targets_down[f"head_{pick_head_idx+1}_drive"] = HEAD_Z_INSERT
        targets_up  [f"head_{pick_head_idx+1}_drive"] = HEAD_Z_RETRACT

    label_down = (
        f"Head {ins_head_idx+1} insert + Head {pick_head_idx+1} pick"
        if ins_key and pick_key
        else ("insert head down" if ins_key else "pick head down")
    )

    # ── 1. Both heads DOWN ────────────────────────────────────────────────────
    drive_to(m, d, v, targets_down, HEAD_Z_STEPS,
             f"both down ({label_down})", tbl, inserted_map, tray_state, carried_map)
    settle(m, d, v, SETTLE_STEPS, tbl, inserted_map, tray_state, carried_map)

    # ── 2. Weld operations ────────────────────────────────────────────────────
    if ins_key is not None:
        # Snap switch to socket, swap welds
        snap_to_site(m, d, ins_key)
        d.eq_active[get_eid(m, f"pick{ins_head_idx+1}_{ins_key}")] = 0
        d.eq_active[get_eid(m, f"ins_{ins_key}")]                  = 1
        # Register for sim_step re-snap; remove from carried_map
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"sw_{ins_key}")
        qa, da = get_body_jnt(m, f"sw_{ins_key}")
        inserted_map[ins_key] = (qa, da, sid)
        carried_map.pop(ins_head_idx, None)

    if pick_key is not None:
        # Switch is at x=0 in tray — teleport to pick head tip
        tip_site = f"head_{pick_head_idx+1}_tip_site"
        tip_sid  = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, tip_site)
        mujoco.mj_forward(m, d)
        tip = d.site_xpos[tip_sid].copy()
        qa, da = get_body_jnt(m, f"sw_{pick_key}")
        d.qpos[qa+0] = tip[0]
        d.qpos[qa+1] = tip[1]
        d.qpos[qa+2] = tip[2] + CARRY_Z_OFFSET
        d.qpos[qa+3]      = 1.0
        d.qpos[qa+4:qa+7] = 0.0
        d.qvel[da:da+6]   = 0.0
        d.eq_active[get_eid(m, f"pick{pick_head_idx+1}_{pick_key}")] = 1
        # Remove from tray, register in carried_map
        tray_state.pop(pick_key, None)
        tray_x.pop(pick_key, None)
        carried_map[pick_head_idx] = (qa, da, tip_sid)

    mujoco.mj_forward(m, d)
    v.sync()

    # ── 3. Both heads UP ──────────────────────────────────────────────────────
    drive_to(m, d, v, targets_up, HEAD_Z_STEPS,
             "both up", tbl, inserted_map, tray_state, carried_map)
    settle(m, d, v, SETTLE_STEPS, tbl, inserted_map, tray_state, carried_map)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading: {MODEL_PATH}\n")
    m = mujoco.MjModel.from_xml_path(MODEL_PATH)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)

    # ── Verify welds & sites ──────────────────────────────────────────────────
    print("Verifying welds (252)...")
    try:
        for key, _, _ in SWITCHES:
            get_eid(m, f"pick1_{key}")
            get_eid(m, f"pick2_{key}")
            get_eid(m, f"ins_{key}")
        print("  All 252 welds ✓")
    except ValueError as e:
        print(f"  ERROR: {e}"); return

    print("Verifying sites (84)...")
    missing = [k for k, _, _ in SWITCHES
               if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"sw_{k}") < 0]
    if missing:
        print(f"  ERROR: missing sites: {missing}"); return
    print("  All 84 sites ✓")

    # ── Resolve tray switch joints ────────────────────────────────────────────
    print("Resolving tray switch joints...")
    # tray_state: key → (qa, da) for every switch still in the tray
    # tray_x:     key → current world X
    tray_state = {}
    tray_x     = {}
    for i, key in enumerate(SWITCH_KEYS):
        qa, da = get_body_jnt(m, f"sw_{key}")
        tray_state[key] = (qa, da)
        tray_x[key]     = -i * PITCH   # sw_Esc at x=0, sw_F1 at -0.01905, etc.
    print(f"  {len(tray_state)} switches in tray ✓")

    # Place all switches at their tray positions (overrides XML initial pos
    # with the controller's canonical positions)
    print("Placing switches in tray...")
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

    inserted_map = {}   # key → (qa, da, site_id) for placed switches
    carried_map  = {}   # head_idx → (qa, da, tip_site_id) for in-transit switches

    def ss():
        sim_step(m, d, v, tbl, inserted_map, tray_state, carried_map)
    def dt(targets, steps, label=""):
        drive_to(m, d, v, targets, steps, label, tbl, inserted_map, tray_state, carried_map)
    def stl(steps):
        settle(m, d, v, steps, tbl, inserted_map, tray_state, carried_map)

    inserted     = 0
    col_angle    = 0.0
    col_sign     = +1
    head_carry   = [None, None]
    sw_queue_idx = 0

    with mujoco.viewer.launch_passive(m, d) as v:
        v.cam.lookat[:] = [0.0, 0.05, 0.83]
        v.cam.distance  = 1.5
        v.cam.elevation = -25
        v.cam.azimuth   = 150

        # ── HOME + DIAGNOSTICS ───────────────────────────────────────────────
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

        # ── PRIME: Head 2 picks sw_Esc (pick only, no insert yet) ───────────
        # sw_Esc is already at x=0 — the pick head is directly above it.
        print("\n[PRIME] Head 2 picks sw_Esc from tray...")
        do_pick_and_insert(m, d, v,
                           ins_head_idx=0,  ins_key=None,
                           pick_head_idx=1, pick_key="Esc",
                           tbl=tbl, inserted_map=inserted_map,
                           tray_state=tray_state, tray_x=tray_x,
                           carried_map=carried_map)
        head_carry[1] = "Esc"
        sw_queue_idx  = 1
        print(f"  ✓ Head 2 carries sw_Esc")

        insert_head = 1
        cycle = 0

        # ── MAIN LOOP ────────────────────────────────────────────────────────
        while inserted < N:
            pick_head = 1 - insert_head
            ins_key   = head_carry[insert_head]
            has_ins   = ins_key is not None
            pick_key  = SWITCHES[sw_queue_idx][0] if sw_queue_idx < N else None

            cycle += 1
            print()
            print("=" * 60)
            print(f"[CYCLE {cycle:02d}]  "
                  f"Insert: sw_{ins_key or 'none':12s}  "
                  f"Pick: sw_{pick_key or 'none'}")
            print("=" * 60)

            # ── Rotate + XY + tray belt advance (all concurrent) ─────────────
            col_angle += col_sign * np.pi
            col_angle  = (col_angle + np.pi) % (2 * np.pi) - np.pi
            col_sign   = -col_sign

            if has_ins:
                ins_sx, ins_sy = SWITCH_MAP[ins_key]
                ins_xc, ins_yc = socket_to_ctrl(ins_sx, ins_sy)
            else:
                ins_xc, ins_yc = 0.0, 0.0

            print(f"  Rotate → {np.degrees(col_angle):.0f}°  "
                  f"XY → ({ins_xc*1000:.1f}, {ins_yc*1000:.1f}) mm  "
                  f"Belt advance +{PITCH*1000:.2f} mm")

            col_start  = d.ctrl[col_aid]
            x_start    = d.ctrl[x_aid_v]
            y_start    = d.ctrl[y_aid_v]
            XY_STEPS   = int(ROTATE_STEPS * 0.8)

            # Snapshot tray X positions at start of advance
            advance_start_x = {k: tray_x[k] for k in tray_state}

            for step in range(ROTATE_STEPS):
                # Column rotation
                tc = smoothstep((step + 1) / ROTATE_STEPS)
                d.ctrl[col_aid] = col_start + (col_angle - col_start) * tc

                # XY table move (finishes at 80% of rotation)
                if step < XY_STEPS:
                    tx = smoothstep((step + 1) / XY_STEPS)
                    d.ctrl[x_aid_v] = x_start + (ins_xc - x_start) * tx
                    d.ctrl[y_aid_v] = y_start + (ins_yc - y_start) * tx

                # Belt advance (same step count as rotation)
                if step < ADVANCE_STEPS and pick_key is not None:
                    tray_advance_step(d, tray_state, tray_x,
                                      advance_start_x, step, ADVANCE_STEPS)

                ss()

            stl(SETTLE_STEPS)
            print(f"  ✓ rotation + XY + belt advance done")

            # ── Simultaneous insert + pick ────────────────────────────────────
            if pick_key is not None:
                px = tray_x.get(pick_key, None)
                if px is not None:
                    print(f"  Tray: sw_{pick_key} at x={px*1000:.2f}mm")

            do_pick_and_insert(m, d, v,
                               ins_head_idx=insert_head, ins_key=ins_key if has_ins else None,
                               pick_head_idx=pick_head,  pick_key=pick_key,
                               tbl=tbl, inserted_map=inserted_map,
                               tray_state=tray_state, tray_x=tray_x,
                               carried_map=carried_map)

            if has_ins:
                head_carry[insert_head] = None
                inserted += 1
                print(f"  ✓ [{inserted}/{N}] sw_{ins_key} inserted")

            if pick_key is not None:
                head_carry[pick_head] = pick_key
                sw_queue_idx += 1
                print(f"  ✓ sw_{pick_key} picked  [queue → {sw_queue_idx}/{N}]")

            insert_head = pick_head

        # ── DONE ─────────────────────────────────────────────────────────────
        print()
        print("=" * 60)
        print(f"[DONE] All {inserted}/{N} switches inserted.")
        print("=" * 60)

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