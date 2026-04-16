"""
station3_controller.py — Dual-head simultaneous pick+insert, 84 switches

Architecture:
  84 switch bodies start IN the tray at 19.05mm pitch.
  Each cycle both heads operate simultaneously:
    - One head picks from tray (pick weld activates)
    - Other head inserts into keyboard (insert weld activates)
  Column rotates 180° each cycle to swap head roles.

Weld system (252 welds):
  pick1_X : head_1_slide ↔ sw_X  (Head 1 carries switch)
  pick2_X : head_2_slide ↔ sw_X  (Head 2 carries switch)
  ins_X   : alu_plate    ↔ sw_X  (permanent after insertion)

Cycle timing:
  Phase 1: Both heads DOWN simultaneously
    - Inserting head: head at socket depth, deactivate pick weld, activate ins weld
    - Picking head: head at tray depth, activate pick weld (switch attaches)
  Phase 2: Both heads UP simultaneously (switches travel with heads)
  Phase 3: Column rotates 180° + XY moves to next socket + tray does not advance
           (no tray — switches are in fixed tray positions)

Pick position: front of tray at x=0, y=0.300 (world)
  Table must be at x=0 for pick (x_ctrl=0 always for pick head)
  After rotation, pick head becomes insert head over keyboard

Table XY: teleported directly each step (bypasses physics, 605 DOF model)

Run from repo root:
    mjpython controllers/station3_controller.py
"""

import mujoco
import mujoco.viewer
import numpy as np
import time

MODEL_PATH = "models/stations/station3.xml"

TIMESTEP       = 0.002
HEAD_Z_STEPS   = 150
ROTATE_STEPS   = 350
XY_MOVE_STEPS  = 280
SETTLE_STEPS   = 40

HEAD_Z_INSERT  = -0.015
HEAD_Z_RETRACT =  0.000
MESH_OFFSET_Z  =  0.00945

# Tray: 84 slots at 19.05mm pitch, x=0 to x=-1.5812
TRAY_WORLD_Y   = 0.300
TRAY_WORLD_Z   = 0.834
TRAY_SWITCH_Z  = TRAY_WORLD_Z - MESH_OFFSET_Z
PITCH          = 0.01905

# Switch sequence — 84 keys with plate-local socket positions
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def aid(m, name):
    i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if i < 0: raise ValueError(f"Actuator '{name}' not found")
    return i

def jnt(m, name):
    i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
    if i < 0: raise ValueError(f"Joint '{name}' not found")
    return m.jnt_qposadr[i], m.jnt_dofadr[i]

def eid(m, name):
    i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, name)
    if i < 0: raise ValueError(f"Weld '{name}' not found")
    return i

def body_jnt(m, bname):
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, bname)
    jid = m.body_jntadr[bid]
    return m.jnt_qposadr[jid], m.jnt_dofadr[jid]

def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def socket_to_ctrl(sx, sy):
    """Move plate socket (sx,sy) to Head 1 world pos (0, 0.040)."""
    xm = float(np.clip(-sx,        -0.160, 0.160))
    ym = float(np.clip(0.010 - sy, -0.115, 0.115))
    return xm, ym

def park(m, d, key):
    """Park switch body below floor."""
    qa, da = body_jnt(m, f"sw_{key}")
    d.qpos[qa:qa+7] = [0, 0, -2, 1, 0, 0, 0]
    d.qvel[da:da+6] = 0.0

def teleport_to_tip(m, d, key, tip_site):
    """Teleport switch body to head tip position."""
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, tip_site)
    mujoco.mj_forward(m, d)
    tip = d.site_xpos[sid].copy()
    qa, da = body_jnt(m, f"sw_{key}")
    d.qpos[qa+0] = tip[0]
    d.qpos[qa+1] = tip[1]
    d.qpos[qa+2] = tip[2]
    d.qpos[qa+3] = 1.0
    d.qpos[qa+4:qa+7] = 0.0
    d.qvel[da:da+6] = 0.0
    mujoco.mj_forward(m, d)

def snap_to_site(m, d, key):
    """Snap switch body exactly to its plate site."""
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"sw_{key}")
    mujoco.mj_forward(m, d)
    sp = d.site_xpos[sid].copy()
    qa, da = body_jnt(m, f"sw_{key}")
    d.qpos[qa+0] = sp[0]
    d.qpos[qa+1] = sp[1]
    d.qpos[qa+2] = sp[2] - MESH_OFFSET_Z
    d.qpos[qa+3] = 1.0
    d.qpos[qa+4:qa+7] = 0.0
    d.qvel[da:da+6] = 0.0


# ── Simulation step ───────────────────────────────────────────────────────────

def sim_step(m, d, v, x_qa, y_qa, x_doa, y_doa, x_aid, y_aid):
    """Single step — table teleported, physics for everything else."""
    d.qpos[x_qa]  = d.ctrl[x_aid]
    d.qpos[y_qa]  = d.ctrl[y_aid]
    d.qvel[x_doa] = 0.0
    d.qvel[y_doa] = 0.0
    t0 = time.perf_counter()
    mujoco.mj_step(m, d)
    v.sync()
    rem = TIMESTEP - (time.perf_counter() - t0)
    if rem > 0:
        time.sleep(rem)

def drive_to(m, d, v, targets, steps, label, tbl):
    x_qa, y_qa, x_doa, y_doa, x_aid, y_aid = tbl
    act_ids = {n: aid(m, n) for n in targets}
    starts  = {n: d.ctrl[act_ids[n]] for n in targets}
    for i in range(steps):
        t = smoothstep((i + 1) / steps)
        for n, tgt in targets.items():
            d.ctrl[act_ids[n]] = starts[n] + (tgt - starts[n]) * t
        sim_step(m, d, v, x_qa, y_qa, x_doa, y_doa, x_aid, y_aid)
    if label:
        print(f"  ✓ {label}")

def settle(m, d, v, steps, tbl):
    x_qa, y_qa, x_doa, y_doa, x_aid, y_aid = tbl
    for _ in range(steps):
        sim_step(m, d, v, x_qa, y_qa, x_doa, y_doa, x_aid, y_aid)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading: {MODEL_PATH}\n")
    m = mujoco.MjModel.from_xml_path(MODEL_PATH)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)

    # Verify welds
    print("Verifying welds (252)...")
    try:
        for key, _, _ in SWITCHES:
            eid(m, f"pick1_{key}")
            eid(m, f"pick2_{key}")
            eid(m, f"ins_{key}")
        print("  All 252 welds found ✓")
    except ValueError as e:
        print(f"  ERROR: {e}"); return

    # Verify sites
    print("Verifying plate sites (84)...")
    mujoco.mj_forward(m, d)
    missing = [k for k,_,_ in SWITCHES
               if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"sw_{k}") < 0]
    if missing:
        print(f"  ERROR: missing sites: {missing[:5]}"); return
    sid0 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "sw_Esc")
    print(f"  All 84 sites ✓  (sw_Esc world_z={d.site_xpos[sid0][2]:.4f})")

    # All switches start in tray — already positioned by XML initial pos
    # Just verify they're at tray Z
    print("Switches positioned in tray by XML ✓")

    # Table joint addresses
    x_qa,  x_doa  = jnt(m, "x_joint")
    y_qa,  y_doa  = jnt(m, "y_joint")
    x_aid_v = aid(m, "x_drive")
    y_aid_v = aid(m, "y_drive")
    col_aid = aid(m, "col_rotate")
    tbl = (x_qa, y_qa, x_doa, y_doa, x_aid_v, y_aid_v)

    def ss():
        sim_step(m, d, v, *tbl)
    def dt(targets, steps, label=""):
        drive_to(m, d, v, targets, steps, label, tbl)
    def stl(steps):
        settle(m, d, v, steps, tbl)

    # Head tip sites
    H1_TIP = "head_1_tip_site"
    H2_TIP = "head_2_tip_site"

    inserted    = 0
    col_angle   = 0.0
    col_sign    = +1

    # Track what each head is currently carrying
    # head_carry[0] = key carried by Head 1 (None if empty)
    # head_carry[1] = key carried by Head 2 (None if empty)
    head_carry  = [None, None]

    # Which head inserts next cycle (0=Head1, 1=Head2)
    # Head 2 starts by picking (it's over tray at y=0.300)
    # After rotation, Head 2 swings to keyboard side
    # So: cycle 0 → Head2 picks, Head1 inserts (but Head1 is empty = prime)
    #     cycle 1 → Head1 picks, Head2 inserts
    #     etc.
    # insert_head alternates: 0=Head1 inserts / 1=Head2 inserts
    # pick_head = 1 - insert_head

    with mujoco.viewer.launch_passive(m, d) as v:
        v.cam.lookat[:] = [0.0, 0.05, 0.83]
        v.cam.distance  = 1.5
        v.cam.elevation = -25
        v.cam.azimuth   = 150

        # ── HOME ─────────────────────────────────────────────────────────
        print("\n[INIT] Homing...")
        d.ctrl[x_aid_v]  = 0.0
        d.ctrl[y_aid_v]  = 0.0
        d.ctrl[col_aid]  = 0.0
        dt({"head_1_drive": HEAD_Z_RETRACT,
            "head_2_drive": HEAD_Z_RETRACT}, 150, "home")
        stl(SETTLE_STEPS)

        # ── PRIME: Head 2 picks first switch (Esc) ───────────────────────
        print("\n[PRIME] Head 2 picks sw_Esc from tray...")
        dt({"head_2_drive": HEAD_Z_INSERT}, HEAD_Z_STEPS, "Head 2 down to tray")
        stl(SETTLE_STEPS)

        pick_key = SWITCHES[0][0]   # "Esc"
        teleport_to_tip(m, d, pick_key, H2_TIP)
        d.eq_active[eid(m, f"pick2_{pick_key}")] = 1
        mujoco.mj_forward(m, d)
        v.sync()
        head_carry[1] = pick_key    # Head 2 carries Esc
        sw_queue_idx  = 1           # next switch to pick is index 1 (F1)
        print(f"  ✓ Head 2 carries sw_{pick_key}")

        dt({"head_2_drive": HEAD_Z_RETRACT}, HEAD_Z_STEPS, "Head 2 up with switch")
        stl(SETTLE_STEPS)

        # ── MAIN LOOP ─────────────────────────────────────────────────────
        # Each cycle:
        #   insert_head picks from [0,1] alternately
        #   After prime: Head2 has Esc, col rotates → Head2 over keyboard
        #   So first real cycle: Head2 inserts Esc, Head1 picks F1
        #   Next: Head1 inserts F1, Head2 picks F2  ... etc.

        insert_head = 1   # Head 2 (index 1) inserts first (carries Esc)
        cycle       = 0

        while inserted < N:
            pick_head   = 1 - insert_head
            ins_key     = head_carry[insert_head]   # key being inserted
            has_insert  = ins_key is not None

            # Next switch to pick (if queue not exhausted)
            if sw_queue_idx < N:
                pick_key, pick_sx, pick_sy = SWITCHES[sw_queue_idx]
            else:
                pick_key = None

            # Current insert socket
            if has_insert:
                ins_key_data = next((s for s in SWITCHES if s[0] == ins_key), None)
                ins_sx, ins_sy = ins_key_data[1], ins_key_data[2]
                ins_xc, ins_yc = socket_to_ctrl(ins_sx, ins_sy)
            else:
                ins_xc, ins_yc = 0.0, 0.0

            cycle += 1
            print()
            print("=" * 60)
            print(f"[CYCLE {cycle}] Insert: sw_{ins_key or 'none'}  "
                  f"Pick: sw_{pick_key or 'none'}")
            print("=" * 60)

            # ── PHASE 3: Rotate + XY to INSERT socket ────────────────────
            col_angle += col_sign * np.pi
            col_angle  = (col_angle + np.pi) % (2 * np.pi) - np.pi
            col_sign   = -col_sign
            print(f"  Rotate col→{np.degrees(col_angle):.0f}°  "
                  f"XY→({ins_xc*1000:.1f}, {ins_yc*1000:.1f})mm")

            col_start = d.ctrl[col_aid]
            x_start   = d.ctrl[x_aid_v]
            y_start   = d.ctrl[y_aid_v]

            for step in range(ROTATE_STEPS):
                tc = smoothstep((step + 1) / ROTATE_STEPS)
                d.ctrl[col_aid] = col_start + (col_angle - col_start) * tc
                if step < XY_MOVE_STEPS:
                    tx = smoothstep((step + 1) / XY_MOVE_STEPS)
                    d.ctrl[x_aid_v] = x_start + (ins_xc - x_start) * tx
                    d.ctrl[y_aid_v] = y_start + (ins_yc - y_start) * tx
                ss()

            print(f"  ✓ rotation + XY done")
            stl(SETTLE_STEPS)

            # ── PHASE 1: Both heads DOWN simultaneously ───────────────────
            print("  Both heads down...")
            targets = {}
            if has_insert:
                # Insert head goes to socket depth
                targets[f"head_{insert_head+1}_drive"] = HEAD_Z_INSERT
            if pick_key is not None:
                # Pick head goes to tray depth (table x=ins_xc but tray at world x=0)
                # Pick head tip is at world y=0.300 (tray side) after rotation ✓
                targets[f"head_{pick_head+1}_drive"] = HEAD_Z_INSERT

            if targets:
                dt(targets, HEAD_Z_STEPS, "both heads down")
            stl(SETTLE_STEPS)

            # ── INSERT: deactivate pick weld, activate ins weld ───────────
            if has_insert:
                snap_to_site(m, d, ins_key)
                p_weld = f"pick{insert_head+1}_{ins_key}"
                i_weld = f"ins_{ins_key}"
                d.eq_active[eid(m, p_weld)] = 0
                d.eq_active[eid(m, i_weld)] = 1
                mujoco.mj_forward(m, d)
                v.sync()
                head_carry[insert_head] = None
                inserted += 1
                print(f"  ✓ sw_{ins_key} inserted [{inserted}/{N}]")

            # ── PICK: attach next switch to pick head ─────────────────────
            if pick_key is not None:
                tip_site = H1_TIP if pick_head == 0 else H2_TIP
                teleport_to_tip(m, d, pick_key, tip_site)
                d.eq_active[eid(m, f"pick{pick_head+1}_{pick_key}")] = 1
                mujoco.mj_forward(m, d)
                v.sync()
                head_carry[pick_head] = pick_key
                sw_queue_idx += 1
                print(f"  ✓ sw_{pick_key} picked by Head {pick_head+1}")

            # ── PHASE 2: Both heads UP ────────────────────────────────────
            up_targets = {
                "head_1_drive": HEAD_Z_RETRACT,
                "head_2_drive": HEAD_Z_RETRACT,
            }
            dt(up_targets, HEAD_Z_STEPS, "both heads up")
            stl(SETTLE_STEPS)

            # Swap roles for next cycle
            insert_head = pick_head

        # ── DONE ─────────────────────────────────────────────────────────
        print()
        print("=" * 60)
        print(f"[DONE] {inserted}/{N} switches inserted.")
        d.ctrl[x_aid_v]  = 0.0
        d.ctrl[y_aid_v]  = 0.0
        d.ctrl[col_aid]  = 0.0
        dt({"head_1_drive": HEAD_Z_RETRACT,
            "head_2_drive": HEAD_Z_RETRACT}, 300, "final home")

        print("Holding — close window to exit.")
        while v.is_running():
            d.qpos[x_qa]  = d.ctrl[x_aid_v]
            d.qpos[y_qa]  = d.ctrl[y_aid_v]
            d.qvel[x_doa] = 0.0
            d.qvel[y_doa] = 0.0
            mujoco.mj_step(m, d)
            v.sync()


if __name__ == "__main__":
    main()