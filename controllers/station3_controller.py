"""
station3_controller.py — Switch insertion for all 84 keys (Station 3)

Fixes vs v1:
  - Switch bodies (swb_<key> / swj_<key>) with freejoints are defined in the XML.
    On insertion we teleport each body to the world position of its plate site
    by writing directly to qpos[fj_adr:fj_adr+7].
  - drive_to() now paces to real-time via time.sleep so the animation is visible.
  - Actuator ctrl units confirmed: x_drive / y_drive use mm (ctrlrange ±160/115 mm),
    col_rotate uses radians, head drives use mm.
  - Prime cycle picks first switch before main loop so Head 1 is loaded for SW 01.

Architecture recap:
  - XY table slides the keyboard so target socket is always under Head 1 (world X=0, Y=0.040)
  - Column alternates ±180° each cycle to swap Head 1 ↔ Head 2
  - Head 2 picks from tray feeder while Head 1 is inserting (sequential in code,
    logically simultaneous in the real machine)
  - Tray slots consumed visually (rgba → 0); reload every 16 picks

Run from repo root:
    mjpython controllers/station3_controller.py
"""

import mujoco
import mujoco.viewer
import numpy as np
import time

MODEL_PATH = "models/stations/station3.xml"

# ── Timing ────────────────────────────────────────────────────────────────────
TIMESTEP        = 0.002          # matches <option timestep="0.002"/>
XY_MOVE_STEPS   = 400
HEAD_Z_STEPS    = 200
ROTATE_STEPS    = 350
SETTLE_STEPS    = 80

# ── Geometry constants ────────────────────────────────────────────────────────
HEAD1_WORLD_Y   = 0.040
TABLE_BASE_Y    = 0.030
XY_Y_BIAS       = HEAD1_WORLD_Y - TABLE_BASE_Y   # = 0.010

HEAD_Z_INSERT   = -0.015         # metres
HEAD_Z_RETRACT  = 0.000

TRAY_SLOTS      = 16

# ── Switch insertion sequence (84 keys, row-by-row) ──────────────────────────
SWITCH_SEQUENCE = [
    # ROW 0 — Function row
    ("sw_Esc",     -0.1429,  0.0480), ("sw_F1",      -0.1238,  0.0480),
    ("sw_F2",      -0.1048,  0.0480), ("sw_F3",      -0.0857,  0.0480),
    ("sw_F4",      -0.0667,  0.0480), ("sw_F5",      -0.0476,  0.0480),
    ("sw_F6",      -0.0286,  0.0480), ("sw_F7",      -0.0095,  0.0480),
    ("sw_F8",       0.0095,  0.0480), ("sw_F9",       0.0286,  0.0480),
    ("sw_F10",      0.0476,  0.0480), ("sw_F11",      0.0667,  0.0480),
    ("sw_F12",      0.0857,  0.0480), ("sw_PrtSc",    0.1048,  0.0480),
    ("sw_Pause",    0.1238,  0.0480), ("sw_Del",      0.1429,  0.0480),
    # ROW 1 — Number row
    ("sw_Grave",   -0.1429,  0.0290), ("sw_1",       -0.1238,  0.0290),
    ("sw_2",       -0.1048,  0.0290), ("sw_3",       -0.0857,  0.0290),
    ("sw_4",       -0.0667,  0.0290), ("sw_5",       -0.0476,  0.0290),
    ("sw_6",       -0.0286,  0.0290), ("sw_7",       -0.0095,  0.0290),
    ("sw_8",        0.0095,  0.0290), ("sw_9",        0.0286,  0.0290),
    ("sw_0",        0.0476,  0.0290), ("sw_Minus",    0.0667,  0.0290),
    ("sw_Equal",    0.0857,  0.0290), ("sw_Bksp",     0.1143,  0.0290),
    ("sw_PgUp",     0.1429,  0.0290),
    # ROW 2 — QWERTY row
    ("sw_Tab",     -0.1381,  0.0099), ("sw_Q",       -0.1143,  0.0099),
    ("sw_W",       -0.0953,  0.0099), ("sw_E",       -0.0762,  0.0099),
    ("sw_R",       -0.0572,  0.0099), ("sw_T",       -0.0381,  0.0099),
    ("sw_Y",       -0.0191,  0.0099), ("sw_U",        0.0000,  0.0099),
    ("sw_I",        0.0191,  0.0099), ("sw_O",        0.0381,  0.0099),
    ("sw_P",        0.0572,  0.0099), ("sw_LBrace",   0.0762,  0.0099),
    ("sw_RBrace",   0.0953,  0.0099), ("sw_Bkslash",  0.1191,  0.0099),
    ("sw_PgDn",     0.1429,  0.0099),
    # ROW 3 — Home row
    ("sw_Caps",    -0.1357, -0.0092), ("sw_A",       -0.1095, -0.0092),
    ("sw_S",       -0.0905, -0.0092), ("sw_D",       -0.0714, -0.0092),
    ("sw_F",       -0.0524, -0.0092), ("sw_G",       -0.0333, -0.0092),
    ("sw_H",       -0.0143, -0.0092), ("sw_J",        0.0048, -0.0092),
    ("sw_K",        0.0238, -0.0092), ("sw_L",        0.0429, -0.0092),
    ("sw_Semicol",  0.0619, -0.0092), ("sw_Quote",    0.0810, -0.0092),
    ("sw_Enter",    0.1119, -0.0092), ("sw_Home",     0.1429, -0.0092),
    # ROW 4 — Shift row
    ("sw_LShift",  -0.1310, -0.0282), ("sw_Z",       -0.1000, -0.0282),
    ("sw_X",       -0.0810, -0.0282), ("sw_C",       -0.0619, -0.0282),
    ("sw_V",       -0.0429, -0.0282), ("sw_B",       -0.0238, -0.0282),
    ("sw_N",       -0.0048, -0.0282), ("sw_M",        0.0143, -0.0282),
    ("sw_Comma",    0.0333, -0.0282), ("sw_Period",   0.0524, -0.0282),
    ("sw_Slash",    0.0714, -0.0282), ("sw_RShift",   0.0976, -0.0282),
    ("sw_Up",       0.1238, -0.0282), ("sw_End",      0.1429, -0.0282),
    # ROW 5 — Bottom row
    ("sw_LCtrl",   -0.1405, -0.0473), ("sw_LWin",    -0.1167, -0.0473),
    ("sw_LAlt",    -0.0929, -0.0473), ("sw_Space",   -0.0214, -0.0473),
    ("sw_RAlt",     0.0476, -0.0473), ("sw_Fn",       0.0667, -0.0473),
    ("sw_Left",     0.0857, -0.0473), ("sw_Down",     0.1048, -0.0473),
    ("sw_Right",    0.1238, -0.0473), ("sw_End2",     0.1429, -0.0473),
]
assert len(SWITCH_SEQUENCE) == 84


# ── MuJoCo lookup helpers ─────────────────────────────────────────────────────

def act_id(m, name):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

def get_jnt_qadr(m, name):
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise ValueError(f"Joint not found: {name}")
    return m.jnt_qposadr[jid], m.jnt_dofadr[jid]

def get_geom_id(m, name):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)

def get_site_xpos(m, d, name):
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, name)
    mujoco.mj_forward(m, d)
    return d.site_xpos[sid].copy()


# ── Switch body registry ──────────────────────────────────────────────────────

def build_switch_registry(m):
    """
    Pre-resolve freejoint qpos/dof addresses for all 84 switch bodies.
    Returns dict: site_name → (fj_qadr, fj_dofadr)
    """
    registry = {}
    missing  = []
    for site_name, _, _ in SWITCH_SEQUENCE:
        key      = site_name[3:]       # "sw_Esc" → "Esc"
        jnt_name = f"swj_{key}"
        try:
            qa, da = get_jnt_qadr(m, jnt_name)
            registry[site_name] = (qa, da)
        except ValueError:
            missing.append(jnt_name)
    if missing:
        print(f"  WARNING: {len(missing)} joints not found: {missing[:5]}...")
    else:
        print(f"  Switch registry: all 84 joints resolved ✓")
    return registry


def park_all_switches(m, d, registry):
    """Park all switch bodies below the floor at z=-2."""
    for site_name, (qa, da) in registry.items():
        d.qpos[qa:qa+7] = [0.0, 0.0, -2.0, 1.0, 0.0, 0.0, 0.0]
        d.qvel[da:da+6] = 0.0


def teleport_switch(m, d, site_name: str, registry: dict, world_pos: np.ndarray):
    """
    Place switch body at world_pos (= plate site world XYZ = plate top face).
    Switch body origin is at its bottom face, so body pos = plate top face directly.
    """
    qa, da = registry[site_name]
    d.qpos[qa + 0] = world_pos[0]
    d.qpos[qa + 1] = world_pos[1]
    d.qpos[qa + 2] = world_pos[2]
    d.qpos[qa + 3] = 1.0   # identity quat (w,x,y,z)
    d.qpos[qa + 4] = 0.0
    d.qpos[qa + 5] = 0.0
    d.qpos[qa + 6] = 0.0
    d.qvel[da:da+6] = 0.0


# ── Tray visual helpers ───────────────────────────────────────────────────────

def consume_tray_slot(m, slot_index: int):
    slot_1indexed = (slot_index % TRAY_SLOTS) + 1
    gid = get_geom_id(m, f"ts_{slot_1indexed}")
    if gid >= 0:
        m.geom_rgba[gid] = [0, 0, 0, 0]

def restore_all_tray_slots(m):
    for i in range(1, TRAY_SLOTS + 1):
        gid = get_geom_id(m, f"ts_{i}")
        if gid >= 0:
            m.geom_rgba[gid] = [0.15, 0.15, 0.15, 1.0]


# ── Motion primitives ─────────────────────────────────────────────────────────

def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def drive_to(m, d, v, targets: dict, steps: int, label: str):
    """
    Smooth-step interpolation for actuator ctrl values over `steps` steps.
    Real-time paced via time.sleep(TIMESTEP) per step.
    """
    aids   = {n: act_id(m, n) for n in targets}
    starts = {n: d.ctrl[aids[n]] for n in targets}

    for i in range(steps):
        t_ease = smoothstep((i + 1) / steps)
        for n, tgt in targets.items():
            d.ctrl[aids[n]] = starts[n] + (tgt - starts[n]) * t_ease
        t0 = time.perf_counter()
        mujoco.mj_step(m, d)
        v.sync()
        elapsed = time.perf_counter() - t0
        rem = TIMESTEP - elapsed
        if rem > 0:
            time.sleep(rem)

    print(f"  ✓ {label}")


def settle(m, d, v, steps: int):
    for _ in range(steps):
        t0 = time.perf_counter()
        mujoco.mj_step(m, d)
        v.sync()
        rem = TIMESTEP - (time.perf_counter() - t0)
        if rem > 0:
            time.sleep(rem)


# ── Coordinate helper ─────────────────────────────────────────────────────────

def socket_to_ctrl(sx: float, sy: float):
    """
    Convert plate-local socket position (sx, sy) to actuator ctrl values (mm).
    x_joint = -sx  (metres) → ctrl = -sx * 1000 mm
    y_joint = XY_Y_BIAS - sy  (metres) → ctrl = (XY_Y_BIAS - sy) * 1000 mm
    Clamped to joint limits before conversion.
    """
    x_m = float(np.clip(-sx,            -0.160, 0.160))
    y_m = float(np.clip(XY_Y_BIAS - sy, -0.115, 0.115))
    return x_m * 1000.0, y_m * 1000.0


# ── Insertion state ───────────────────────────────────────────────────────────

class InsertionState:
    def __init__(self):
        self.col_angle     = 0.0
        self.rotate_sign   = +1
        self.tray_slot     = 0
        self.inserted      = 0

    def next_col_angle(self):
        target = self.col_angle + self.rotate_sign * np.pi
        target = (target + np.pi) % (2 * np.pi) - np.pi
        self.col_angle   = target
        self.rotate_sign = -self.rotate_sign
        return target

    def consume_slot(self):
        s = self.tray_slot
        self.tray_slot = (self.tray_slot + 1) % TRAY_SLOTS
        return s


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading: {MODEL_PATH}\n")
    m = mujoco.MjModel.from_xml_path(MODEL_PATH)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)

    print("Building switch registry...")
    registry = build_switch_registry(m)
    park_all_switches(m, d, registry)
    mujoco.mj_forward(m, d)

    state = InsertionState()

    with mujoco.viewer.launch_passive(m, d) as v:
        v.cam.lookat[:] = [0.0, 0.05, 0.83]
        v.cam.distance  = 1.3
        v.cam.elevation = -25
        v.cam.azimuth   = 150

        # ── HOME ─────────────────────────────────────────────────────────
        print("\n[INIT] Homing all axes...")
        drive_to(m, d, v, {
            "x_drive": 0.0, "y_drive": 0.0, "col_rotate": 0.0,
            "head_1_drive": 0.0, "head_2_drive": 0.0,
        }, steps=150, label="home")
        settle(m, d, v, 60)

        # ── PRIME: Head 2 picks first switch, column rotates ─────────────
        print("\n[PRIME] Head 2 picks first switch from tray...")
        slot0 = state.consume_slot()
        drive_to(m, d, v, {"head_2_drive": HEAD_Z_INSERT * 1000},
                 HEAD_Z_STEPS, "Head 2 down (prime pick)")
        consume_tray_slot(m, slot0)
        settle(m, d, v, SETTLE_STEPS)
        drive_to(m, d, v, {"head_2_drive": HEAD_Z_RETRACT * 1000},
                 HEAD_Z_STEPS, "Head 2 up (prime pick)")

        col0 = state.next_col_angle()
        drive_to(m, d, v, {"col_rotate": col0},
                 ROTATE_STEPS, f"column → {np.degrees(col0):.0f}° (prime)")
        settle(m, d, v, SETTLE_STEPS)
        print("  Head 1 now carries switch #1. Head 2 over feeder.")

        # ── MAIN LOOP ─────────────────────────────────────────────────────
        for sw_idx, (site_name, sx, sy) in enumerate(SWITCH_SEQUENCE):

            print()
            print("=" * 60)
            print(f"[SW {sw_idx+1:02d}/84] {site_name}  local=({sx:.4f}, {sy:.4f})")
            print("=" * 60)

            # 1. Table move: bring socket under Head 1
            x_ctrl, y_ctrl = socket_to_ctrl(sx, sy)
            print(f"  XY → x={x_ctrl:.1f}mm  y={y_ctrl:.1f}mm")
            drive_to(m, d, v,
                     {"x_drive": x_ctrl, "y_drive": y_ctrl},
                     XY_MOVE_STEPS, f"table → {site_name}")
            settle(m, d, v, SETTLE_STEPS)

            # 2. Head 1 presses down to insert
            drive_to(m, d, v,
                     {"head_1_drive": HEAD_Z_INSERT * 1000},
                     HEAD_Z_STEPS, f"Head 1 down → {site_name}")
            settle(m, d, v, SETTLE_STEPS)

            # 3. Teleport switch body to plate socket
            site_pos = get_site_xpos(m, d, site_name)
            teleport_switch(m, d, site_name, registry, site_pos)
            mujoco.mj_forward(m, d)
            v.sync()
            print(f"  ✓ switch @ {np.round(site_pos, 4)}")
            state.inserted += 1

            # 4. Head 1 retracts
            drive_to(m, d, v,
                     {"head_1_drive": HEAD_Z_RETRACT * 1000},
                     HEAD_Z_STEPS, "Head 1 up")

            # 5. Head 2 picks next switch from tray
            next_slot = state.consume_slot()
            print(f"  Head 2 picks slot {next_slot}")
            drive_to(m, d, v,
                     {"head_2_drive": HEAD_Z_INSERT * 1000},
                     HEAD_Z_STEPS, f"Head 2 down (slot {next_slot})")
            consume_tray_slot(m, next_slot)
            settle(m, d, v, SETTLE_STEPS)
            drive_to(m, d, v,
                     {"head_2_drive": HEAD_Z_RETRACT * 1000},
                     HEAD_Z_STEPS, "Head 2 up")

            # 6. Column rotates 180° alternating direction
            col_angle = state.next_col_angle()
            drive_to(m, d, v,
                     {"col_rotate": col_angle},
                     ROTATE_STEPS, f"column → {np.degrees(col_angle):.0f}°")
            settle(m, d, v, SETTLE_STEPS)

            # 7. Tray reload every 16 picks
            if (next_slot + 1) % TRAY_SLOTS == 0:
                print("  [TRAY] Reloading...")
                restore_all_tray_slots(m)

            print(f"  [{state.inserted:02d}/84 inserted]")

        # ── FINISH ───────────────────────────────────────────────────────
        print()
        print("=" * 60)
        print(f"[DONE] All {state.inserted} switches inserted.")
        drive_to(m, d, v, {
            "x_drive": 0.0, "y_drive": 0.0, "col_rotate": 0.0,
            "head_1_drive": 0.0, "head_2_drive": 0.0,
        }, steps=300, label="final home")
        print("Holding viewer — close window to exit.")
        while v.is_running():
            mujoco.mj_step(m, d)
            v.sync()


if __name__ == "__main__":
    main()