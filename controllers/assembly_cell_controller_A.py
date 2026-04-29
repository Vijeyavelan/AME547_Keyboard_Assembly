"""AME 547 — assembly-cell controller, Pass 2 S1→S3 with real S3 wrapper.

Run:
    mjpython controllers/assembly_cell_controller_A.py

Pass 2 scope:
- Deterministic one-pallet freejoint movement through S1 → S3.
- Uses XML checkpoint sites as controller ground truth.
- Replaces the Pass 1 S3 placeholder dwell with a prefix-aware S3 switch insertion runner.
- Pass 2F keeps the Pass 2E pallet follow fix, then restores the standalone
  Station 3 dual-head sequencing: Head 2 primes sw_Esc, then heads alternate.
- Pass 2G converts inserted switches to visual-only/static-follow: after each
  insertion, the switch free body is overwritten from its table site every step
  and the insertion weld is kept inactive to avoid accumulating solver constraints.
- Pass 2I adds payload-follow handoff: S3-installed switches are re-parented
visually to the pallet after S3 so they travel into S4, and installed S4 keycaps
can also follow the pallet after S4.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

try:
    import mujoco
    import mujoco.viewer
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Could not import mujoco. Run this with the project venv and mjpython, e.g.\n"
        "    mjpython controllers/assembly_cell_controller_A.py"
    ) from exc


# ----------------------------- conveyor constants -----------------------------
CDLR_SPEED = 0.300
STOP_THRESHOLD = 0.003
PALLET_QUAT = np.array([1.0, 0.0, 0.0, 0.0])
TIME_SCALE = float(os.environ.get("AME_CELL_TIME_SCALE", "1.0"))
# Pass 2E debug switch:
#   follow    = continuously overwrite pallet freejoint pose from s3_pallet_clamp_site; no weld
#   snap_only = continuously hold pallet at s3_pallet_clamp_site; skip S3 sequence
#   weld      = old equality-weld path, kept only for comparison/debug
S3_LOCK_MODE = os.environ.get("AME_S3_LOCK_MODE", "follow").strip().lower()
# Pass 2G speed optimization. When enabled, inserted switches are visual-only:
# they are kinematically held on their final table-relative sites and their
# insertion welds remain inactive, preventing 84 active equality constraints.
S3_STATIC_INSERT = os.environ.get("AME_S3_STATIC_INSERT", "1").strip().lower() not in {"0", "false", "no", "off"}
S4_LOCK_MODE = os.environ.get("AME_S4_LOCK_MODE", "follow").strip().lower()
S4_STATIC_INSERT = os.environ.get("AME_S4_STATIC_INSERT", "1").strip().lower() not in {"0", "false", "no", "off"}

CYCLE_TIMES = {
    "S1_PROCESS_OR_WAIT": 5.0,
    "S3_LOCK": 0.25,
    "S3_UNLOCK": 0.25,
    "S4_LOCK": 0.25,
    "S4_UNLOCK": 0.25,
}

CHECKPOINT_SITE_BY_STATE = {
    "MOVE_TO_S1": "cell_pallet_s1_stop_site",
    "MOVE_TO_S3_BUFFER": "cell_pallet_s3_buffer_site",
    "MOVE_TO_S3_STATION": "cell_pallet_s3_station_site",
    "MOVE_TO_S3_EXIT": "cell_pallet_s3_exit_site",
    "MOVE_TO_S4_BUFFER": "cell_pallet_s4_buffer_site",
    "MOVE_TO_S4_STATION": "cell_pallet_s4_station_site",
    "MOVE_TO_S4_EXIT": "cell_pallet_s4_exit_site",
}

NEXT_STATE_AFTER_MOVE = {
    "MOVE_TO_S1": "S1_PROCESS_OR_WAIT",
    "MOVE_TO_S3_BUFFER": "MOVE_TO_S3_STATION",
    "MOVE_TO_S3_STATION": "S3_LOCK",
    "MOVE_TO_S3_EXIT": "MOVE_TO_S4_BUFFER",
    "MOVE_TO_S4_BUFFER": "MOVE_TO_S4_STATION",
    "MOVE_TO_S4_STATION": "S4_LOCK",
    "MOVE_TO_S4_EXIT": "DONE_S1_S3_S4",
}


# ----------------------------- S3 sequence constants -----------------------------
TIMESTEP = 0.002
HEAD_Z_STEPS = int(os.environ.get("AME_S3_HEAD_Z_STEPS", "150"))
ROTATE_STEPS = int(os.environ.get("AME_S3_ROTATE_STEPS", "200"))
ADVANCE_STEPS = ROTATE_STEPS
SETTLE_STEPS = int(os.environ.get("AME_S3_SETTLE_STEPS", "30"))

HEAD_Z_INSERT = -0.015
HEAD_Z_RETRACT = 0.000
MESH_OFFSET_Z = 0.00945
CARRY_Z_OFFSET = -0.013
PITCH = 0.01905
S3_WORLD_X = 2.25
S3_TRAY_WORLD_Y = -0.300
S3_TRAY_WORLD_Z = 0.834
S3_TRAY_SWITCH_Z = S3_TRAY_WORLD_Z - MESH_OFFSET_Z
S3_INSERT_HEAD_WORLD_Y = -0.040  # active insertion head line in the integrated/rotated S3 cell

# Optional debug throttle. Default is full 84 switches.
S3_MAX_SWITCHES = int(os.environ.get("AME_S3_MAX_SWITCHES", "84"))

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
SWITCHES = SWITCHES[:S3_MAX_SWITCHES]
SWITCH_KEYS = [k for k, _, _ in SWITCHES]
SWITCH_MAP = {k: (sx, sy) for k, sx, sy in SWITCHES}


# ----------------------------- S4 sequence constants -----------------------------
S4_MAX_KEYCAPS = int(os.environ.get("AME_S4_MAX_KEYCAPS", "84"))
S4_WORLD_X = 5.25
S4_TRAY_WORLD_Y = -0.300
S4_TRAY_WORLD_Z = 0.8411
S4_GAP = 0.002
KEYCAPS = SWITCHES[:S4_MAX_KEYCAPS] if S4_MAX_KEYCAPS != 84 else [
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
KEYCAPS = KEYCAPS[:S4_MAX_KEYCAPS]
KEYCAP_KEYS = [k for k, _, _ in KEYCAPS]
KEYCAP_MAP = {k: (sx, sy) for k, sx, sy in KEYCAPS}
KEYCAP_WIDTHS = {k:0.018 for k in [k for k,_,_ in KEYCAPS]}
KEYCAP_WIDTHS.update({
    "Bksp":0.0370,"Tab":0.0275,"Bkslash":0.0275,"Caps":0.0323,
    "Enter":0.0418,"LShift":0.0418,"RShift":0.0323,
    "LCtrl":0.0228,"LWin":0.0228,"LAlt":0.0228,"Space":0.1180,"RAlt":0.0228,"Fn":0.0228,
})

def keycap_advance_amount(prev_key: str, current_key: str) -> float:
    return KEYCAP_WIDTHS[prev_key] / 2.0 + S4_GAP + KEYCAP_WIDTHS[current_key] / 2.0

def build_keycap_tray_x(keys) -> Dict[str, float]:
    tray_x: Dict[str, float] = {}
    current_center_x = S4_WORLD_X
    for i, key in enumerate(keys):
        if i == 0:
            tray_x[key] = current_center_x
            continue
        prev = keys[i - 1]
        current_center_x -= keycap_advance_amount(prev, key)
        tray_x[key] = current_center_x
    return tray_x


def smoothstep(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def quat_conj(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.asarray(a, dtype=float)
    bw, bx, by, bz = np.asarray(b, dtype=float)
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=float)


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qv = np.array([0.0, *np.asarray(v, dtype=float)], dtype=float)
    return quat_mul(quat_mul(q, qv), quat_conj(q))[1:]


def normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    n = float(np.linalg.norm(q))
    if n <= 1e-12:
        return PALLET_QUAT.copy()
    q = q / n
    return q if q[0] >= 0.0 else -q


def resolve_name(model: mujoco.MjModel, obj_type, name: str) -> int:
    idx = mujoco.mj_name2id(model, obj_type, name)
    if idx < 0:
        raise RuntimeError(f"Required MuJoCo object not found: {name}")
    return idx


class S3SwitchRunner:
    """Prefix-aware wrapper around the standalone Station 3 switch insertion pattern."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, prefix: str = "s3"):
        self.m = model
        self.d = data
        self.p = prefix
        self.pre_step_callback = None
        self.inserted_map: Dict[str, tuple[int, int, int]] = {}
        self.inserted_eq_map: Dict[str, int] = {}
        self.tray_state: Dict[str, tuple[int, int]] = {}
        self.tray_x: Dict[str, float] = {}
        self.carried_map: Dict[int, tuple[int, int, int]] = {}
        self.ran = False

        self.x_qa, self.x_doa = self.get_joint(f"{prefix}_x_joint")
        self.y_qa, self.y_doa = self.get_joint(f"{prefix}_y_joint")
        self.x_aid = self.get_actuator(f"{prefix}_x_drive")
        self.y_aid = self.get_actuator(f"{prefix}_y_drive")
        self.col_aid = self.get_actuator(f"{prefix}_col_rotate")
        self.h1_aid = self.get_actuator(f"{prefix}_head_1_drive")
        self.h2_aid = self.get_actuator(f"{prefix}_head_2_drive")

    def n(self, suffix: str) -> str:
        return f"{self.p}_{suffix}"

    def get_actuator(self, name: str) -> int:
        return resolve_name(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

    def get_joint(self, name: str) -> tuple[int, int]:
        jid = resolve_name(self.m, mujoco.mjtObj.mjOBJ_JOINT, name)
        return self.m.jnt_qposadr[jid], self.m.jnt_dofadr[jid]

    def get_eq(self, name: str) -> int:
        return resolve_name(self.m, mujoco.mjtObj.mjOBJ_EQUALITY, name)

    def get_body_jnt(self, body_name: str) -> tuple[int, int]:
        bid = resolve_name(self.m, mujoco.mjtObj.mjOBJ_BODY, body_name)
        jid = self.m.body_jntadr[bid]
        if jid < 0:
            raise RuntimeError(f"Body has no joint/freejoint: {body_name}")
        return self.m.jnt_qposadr[jid], self.m.jnt_dofadr[jid]

    def verify(self) -> None:
        missing = []
        for key in SWITCH_KEYS:
            for obj_type, name in [
                (mujoco.mjtObj.mjOBJ_BODY, self.n(f"sw_{key}")),
                (mujoco.mjtObj.mjOBJ_SITE, self.n(f"site_sw_{key}")),
                (mujoco.mjtObj.mjOBJ_EQUALITY, self.n(f"pick1_{key}")),
                (mujoco.mjtObj.mjOBJ_EQUALITY, self.n(f"pick2_{key}")),
                (mujoco.mjtObj.mjOBJ_EQUALITY, self.n(f"ins_{key}")),
            ]:
                if mujoco.mj_name2id(self.m, obj_type, name) < 0:
                    missing.append(name)
        if missing:
            raise RuntimeError("S3 wrapper missing XML support objects: " + ", ".join(missing[:20]))
        print(f"  S3 wrapper verified: {len(SWITCH_KEYS)} switches, {len(SWITCH_KEYS)} sites, {len(SWITCH_KEYS)*3} welds")

    def socket_to_ctrl(self, sx: float, sy: float) -> tuple[float, float]:
        # The integrated S3 table is rotated 180 deg about Z. These commands keep the selected
        # socket under the active insert head line while preserving the standalone station behavior.
        xm = float(np.clip(-sx, -0.160, 0.160))
        ym = float(np.clip(0.010 - sy, -0.115, 0.115))
        return xm, ym

    def place_on_tray(self, key: str, x: float) -> None:
        qa, da = self.get_body_jnt(self.n(f"sw_{key}"))
        self.d.qpos[qa + 0] = x
        self.d.qpos[qa + 1] = S3_TRAY_WORLD_Y
        self.d.qpos[qa + 2] = S3_TRAY_SWITCH_Z
        self.d.qpos[qa + 3] = 1.0
        self.d.qpos[qa + 4:qa + 7] = 0.0
        self.d.qvel[da:da + 6] = 0.0
        self.tray_state[key] = (qa, da)
        self.tray_x[key] = x

    def sim_step(self, viewer: Optional[mujoco.viewer.Handle]) -> None:
        # 1. Deterministic table tracking from control values.
        self.d.qpos[self.x_qa] = self.d.ctrl[self.x_aid]
        self.d.qpos[self.y_qa] = self.d.ctrl[self.y_aid]
        self.d.qvel[self.x_doa] = 0.0
        self.d.qvel[self.y_doa] = 0.0

        # 2. Keep inserted switches snapped to moving table sites.
        #    In static-follow mode, also keep their insertion welds inactive so
        #    the solver does not accumulate one equality constraint per switch.
        if S3_STATIC_INSERT:
            for eqid in self.inserted_eq_map.values():
                self.d.eq_active[eqid] = 0
        for qa, da, sid in self.inserted_map.values():
            sp = self.d.site_xpos[sid]
            self.d.qpos[qa + 0] = sp[0]
            self.d.qpos[qa + 1] = sp[1]
            self.d.qpos[qa + 2] = sp[2] - MESH_OFFSET_Z
            self.d.qpos[qa + 3] = 1.0
            self.d.qpos[qa + 4:qa + 7] = 0.0
            self.d.qvel[da:da + 6] = 0.0

        # 3. Keep unpicked tray switches on the tray line.
        for qa, da in self.tray_state.values():
            self.d.qpos[qa + 1] = S3_TRAY_WORLD_Y
            self.d.qpos[qa + 2] = S3_TRAY_SWITCH_Z
            self.d.qpos[qa + 3] = 1.0
            self.d.qpos[qa + 4:qa + 7] = 0.0
            self.d.qvel[da:da + 6] = 0.0

        # 4. Keep carried switches attached to their active tip sites.
        for qa, da, tip_sid in self.carried_map.values():
            tip = self.d.site_xpos[tip_sid]
            self.d.qpos[qa + 0] = tip[0]
            self.d.qpos[qa + 1] = tip[1]
            self.d.qpos[qa + 2] = tip[2] + CARRY_Z_OFFSET
            self.d.qpos[qa + 3] = 1.0
            self.d.qpos[qa + 4:qa + 7] = 0.0
            self.d.qvel[da:da + 6] = 0.0

        if self.pre_step_callback is not None:
            self.pre_step_callback()

        t0 = time.perf_counter()
        mujoco.mj_step(self.m, self.d)
        if viewer is not None:
            viewer.sync()
        rem = TIMESTEP - (time.perf_counter() - t0)
        if rem > 0:
            time.sleep(rem)

    def settle(self, viewer: Optional[mujoco.viewer.Handle], steps: int = SETTLE_STEPS) -> None:
        for _ in range(steps):
            self.sim_step(viewer)

    def drive_to(self, targets: Dict[str, float], steps: int, viewer: Optional[mujoco.viewer.Handle], label: str = "") -> None:
        act_ids = {name: self.get_actuator(self.n(name)) for name in targets}
        starts = {name: self.d.ctrl[aid] for name, aid in act_ids.items()}
        for i in range(steps):
            t = smoothstep((i + 1) / steps)
            for name, target in targets.items():
                aid = act_ids[name]
                self.d.ctrl[aid] = starts[name] + (target - starts[name]) * t
            self.sim_step(viewer)
        if label:
            print(f"  ✓ {label}")

    def snap_to_site(self, key: str) -> None:
        sid = resolve_name(self.m, mujoco.mjtObj.mjOBJ_SITE, self.n(f"site_sw_{key}"))
        mujoco.mj_forward(self.m, self.d)
        sp = self.d.site_xpos[sid].copy()
        qa, da = self.get_body_jnt(self.n(f"sw_{key}"))
        self.d.qpos[qa + 0] = sp[0]
        self.d.qpos[qa + 1] = sp[1]
        self.d.qpos[qa + 2] = sp[2] - MESH_OFFSET_Z
        self.d.qpos[qa + 3] = 1.0
        self.d.qpos[qa + 4:qa + 7] = 0.0
        self.d.qvel[da:da + 6] = 0.0

    def tray_advance_step(self, start_x: Dict[str, float], step: int) -> None:
        # Integrated S3 tray advances toward decreasing X to bring the next switch to x=2.25.
        t = smoothstep((step + 1) / ADVANCE_STEPS)
        for key, (qa, _da) in self.tray_state.items():
            new_x = start_x[key] - PITCH * t
            self.d.qpos[qa + 0] = new_x
            self.tray_x[key] = new_x

    def debug_head_tips(self, label: str) -> None:
        """Print current world positions for both S3 head tip sites."""
        mujoco.mj_forward(self.m, self.d)
        parts = []
        for h in range(2):
            sid = resolve_name(self.m, mujoco.mjtObj.mjOBJ_SITE, self.n(f"head_{h + 1}_tip_site"))
            p = self.d.site_xpos[sid]
            parts.append(f"H{h + 1}=({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f})")
        print(f"  [S3 debug] head tips {label}: " + "  ".join(parts))

    def debug_insert_alignment(self, head_idx: int, key: str, warn_mm: float = 35.0) -> None:
        """Warn if the selected insert head is far from the target switch site."""
        mujoco.mj_forward(self.m, self.d)
        tip_sid = resolve_name(self.m, mujoco.mjtObj.mjOBJ_SITE, self.n(f"head_{head_idx + 1}_tip_site"))
        site_sid = resolve_name(self.m, mujoco.mjtObj.mjOBJ_SITE, self.n(f"site_sw_{key}"))
        tip = self.d.site_xpos[tip_sid].copy()
        site = self.d.site_xpos[site_sid].copy()
        dxy_mm = float(np.linalg.norm((tip - site)[:2]) * 1000.0)
        print(f"  [S3 debug] insert head H{head_idx + 1} → sw_{key}: XY error {dxy_mm:.1f} mm")
        if dxy_mm > warn_mm:
            print(f"  [S3 warning] H{head_idx + 1} is far from site sw_{key}; check head sequencing/rotation phase")

    def do_pick_and_insert(
        self,
        viewer: Optional[mujoco.viewer.Handle],
        ins_head_idx: int,
        ins_key: Optional[str],
        pick_head_idx: int,
        pick_key: Optional[str],
    ) -> None:
        targets_down: Dict[str, float] = {}
        targets_up: Dict[str, float] = {}
        if ins_key is not None:
            targets_down[f"head_{ins_head_idx + 1}_drive"] = HEAD_Z_INSERT
            targets_up[f"head_{ins_head_idx + 1}_drive"] = HEAD_Z_RETRACT
        if pick_key is not None:
            targets_down[f"head_{pick_head_idx + 1}_drive"] = HEAD_Z_INSERT
            targets_up[f"head_{pick_head_idx + 1}_drive"] = HEAD_Z_RETRACT

        self.drive_to(targets_down, HEAD_Z_STEPS, viewer, "S3 heads down")
        self.settle(viewer)

        if ins_key is not None:
            self.debug_insert_alignment(ins_head_idx, ins_key)
            self.snap_to_site(ins_key)
            self.d.eq_active[self.get_eq(self.n(f"pick{ins_head_idx + 1}_{ins_key}"))] = 0
            ins_eq = self.get_eq(self.n(f"ins_{ins_key}"))
            self.d.eq_active[ins_eq] = 0 if S3_STATIC_INSERT else 1
            sid = resolve_name(self.m, mujoco.mjtObj.mjOBJ_SITE, self.n(f"site_sw_{ins_key}"))
            qa, da = self.get_body_jnt(self.n(f"sw_{ins_key}"))
            self.inserted_map[ins_key] = (qa, da, sid)
            self.inserted_eq_map[ins_key] = ins_eq
            self.carried_map.pop(ins_head_idx, None)

        if pick_key is not None:
            tip_sid = resolve_name(self.m, mujoco.mjtObj.mjOBJ_SITE, self.n(f"head_{pick_head_idx + 1}_tip_site"))
            mujoco.mj_forward(self.m, self.d)
            tip = self.d.site_xpos[tip_sid].copy()
            qa, da = self.get_body_jnt(self.n(f"sw_{pick_key}"))
            self.d.qpos[qa + 0] = tip[0]
            self.d.qpos[qa + 1] = tip[1]
            self.d.qpos[qa + 2] = tip[2] + CARRY_Z_OFFSET
            self.d.qpos[qa + 3] = 1.0
            self.d.qpos[qa + 4:qa + 7] = 0.0
            self.d.qvel[da:da + 6] = 0.0
            self.d.eq_active[self.get_eq(self.n(f"pick{pick_head_idx + 1}_{pick_key}"))] = 1
            self.tray_state.pop(pick_key, None)
            self.tray_x.pop(pick_key, None)
            self.carried_map[pick_head_idx] = (qa, da, tip_sid)

        mujoco.mj_forward(self.m, self.d)
        if viewer is not None:
            viewer.sync()

        self.drive_to(targets_up, HEAD_Z_STEPS, viewer, "S3 heads up")
        self.settle(viewer)

    def run(self, viewer: Optional[mujoco.viewer.Handle]) -> None:
        if self.ran:
            return
        print(f"\n[S3] Starting integrated switch insertion for {len(SWITCH_KEYS)} switches... (standalone head sequence)")
        print(f"  S3 static-insert visual follow: {'ON' if S3_STATIC_INSERT else 'OFF'}")
        self.verify()

        # Resolve and place tray switches. Esc starts at station pick x; following switches are +pitch.
        self.tray_state.clear()
        self.tray_x.clear()
        self.inserted_map.clear()
        self.inserted_eq_map.clear()
        self.carried_map.clear()
        for i, key in enumerate(SWITCH_KEYS):
            self.place_on_tray(key, S3_WORLD_X + i * PITCH)

        self.d.ctrl[self.x_aid] = 0.0
        self.d.ctrl[self.y_aid] = 0.0
        self.d.ctrl[self.col_aid] = 0.0
        self.drive_to({"head_1_drive": HEAD_Z_RETRACT, "head_2_drive": HEAD_Z_RETRACT}, 100, viewer, "S3 home")
        self.settle(viewer)

        # Match the standalone Station 3 controller exactly:
        # PRIME: Head 2 picks sw_Esc from the tray. The head carrying a switch
        # becomes the first insert head after the 180 deg column rotation.
        self.debug_head_tips("home before prime")
        print("[S3] Prime: Head 2 picks sw_Esc from tray")
        self.do_pick_and_insert(viewer, ins_head_idx=0, ins_key=None, pick_head_idx=1, pick_key="Esc")

        head_carry = [None, "Esc"]
        sw_queue_idx = 1
        insert_head = 1
        print("  ✓ Head 2 carries sw_Esc")
        inserted = 0
        col_angle = 0.0
        col_sign = +1
        cycle = 0

        while inserted < len(SWITCH_KEYS):
            pick_head = 1 - insert_head
            ins_key = head_carry[insert_head]
            pick_key = SWITCH_KEYS[sw_queue_idx] if sw_queue_idx < len(SWITCH_KEYS) else None
            cycle += 1

            print(f"[S3 C{cycle:02d}] insert={ins_key or 'none'} pick={pick_key or 'none'}")

            col_angle += col_sign * np.pi
            col_angle = (col_angle + np.pi) % (2 * np.pi) - np.pi
            col_sign = -col_sign

            if ins_key is not None:
                ins_sx, ins_sy = SWITCH_MAP[ins_key]
                ins_xc, ins_yc = self.socket_to_ctrl(ins_sx, ins_sy)
            else:
                ins_xc, ins_yc = 0.0, 0.0

            col_start = self.d.ctrl[self.col_aid]
            x_start = self.d.ctrl[self.x_aid]
            y_start = self.d.ctrl[self.y_aid]
            xy_steps = max(1, int(ROTATE_STEPS * 0.8))
            advance_start_x = {k: self.tray_x[k] for k in self.tray_state}

            for step in range(ROTATE_STEPS):
                tc = smoothstep((step + 1) / ROTATE_STEPS)
                self.d.ctrl[self.col_aid] = col_start + (col_angle - col_start) * tc

                if step < xy_steps:
                    tx = smoothstep((step + 1) / xy_steps)
                    self.d.ctrl[self.x_aid] = x_start + (ins_xc - x_start) * tx
                    self.d.ctrl[self.y_aid] = y_start + (ins_yc - y_start) * tx

                if step < ADVANCE_STEPS and pick_key is not None:
                    self.tray_advance_step(advance_start_x, step)

                self.sim_step(viewer)

            self.settle(viewer)
            if ins_key is not None:
                self.debug_insert_alignment(insert_head, ins_key)
            self.do_pick_and_insert(
                viewer,
                ins_head_idx=insert_head,
                ins_key=ins_key,
                pick_head_idx=pick_head,
                pick_key=pick_key,
            )

            if ins_key is not None:
                head_carry[insert_head] = None
                inserted += 1
                print(f"  ✓ S3 inserted {inserted}/{len(SWITCH_KEYS)}: sw_{ins_key}")
            if pick_key is not None:
                head_carry[pick_head] = pick_key
                sw_queue_idx += 1

            insert_head = pick_head

        print("[S3] Switch insertion complete")
        self.ran = True


class S4KeycapRunner:
    """Prefix-aware wrapper around the verified Station 4 keycap installation pattern."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, prefix: str = "s4"):
        self.m = model
        self.d = data
        self.p = prefix
        self.pre_step_callback = None
        self.inserted_map: Dict[str, tuple[int, int, int]] = {}
        self.inserted_eq_map: Dict[str, int] = {}
        self.tray_state: Dict[str, tuple[int, int]] = {}
        self.tray_x: Dict[str, float] = {}
        self.carried_map: Dict[int, tuple[int, int, int]] = {}
        self.ran = False

        self.x_qa, self.x_doa = self.get_joint(f"{prefix}_x_joint")
        self.y_qa, self.y_doa = self.get_joint(f"{prefix}_y_joint")
        self.x_aid = self.get_actuator(f"{prefix}_x_drive")
        self.y_aid = self.get_actuator(f"{prefix}_y_drive")
        self.col_aid = self.get_actuator(f"{prefix}_col_rotate")
        self.h1_aid = self.get_actuator(f"{prefix}_head_1_drive")
        self.h2_aid = self.get_actuator(f"{prefix}_head_2_drive")

    def n(self, suffix: str) -> str:
        return f"{self.p}_{suffix}"

    def get_actuator(self, name: str) -> int:
        return resolve_name(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

    def get_joint(self, name: str) -> tuple[int, int]:
        jid = resolve_name(self.m, mujoco.mjtObj.mjOBJ_JOINT, name)
        return self.m.jnt_qposadr[jid], self.m.jnt_dofadr[jid]

    def get_eq(self, name: str) -> int:
        return resolve_name(self.m, mujoco.mjtObj.mjOBJ_EQUALITY, name)

    def get_body_jnt(self, body_name: str) -> tuple[int, int]:
        bid = resolve_name(self.m, mujoco.mjtObj.mjOBJ_BODY, body_name)
        jid = self.m.body_jntadr[bid]
        if jid < 0:
            raise RuntimeError(f"Body has no joint/freejoint: {body_name}")
        return self.m.jnt_qposadr[jid], self.m.jnt_dofadr[jid]

    def verify(self) -> None:
        missing = []
        for key in KEYCAP_KEYS:
            for obj_type, name in [
                (mujoco.mjtObj.mjOBJ_BODY, self.n(f"kc_{key}")),
                (mujoco.mjtObj.mjOBJ_SITE, self.n(f"kc_{key}")),
                (mujoco.mjtObj.mjOBJ_EQUALITY, self.n(f"pick1_{key}")),
                (mujoco.mjtObj.mjOBJ_EQUALITY, self.n(f"pick2_{key}")),
                (mujoco.mjtObj.mjOBJ_EQUALITY, self.n(f"ins_{key}")),
            ]:
                if mujoco.mj_name2id(self.m, obj_type, name) < 0:
                    missing.append(name)
        if missing:
            raise RuntimeError("S4 wrapper missing XML support objects: " + ", ".join(missing[:20]))
        print(f"  S4 wrapper verified: {len(KEYCAP_KEYS)} keycaps, {len(KEYCAP_KEYS)} sites, {len(KEYCAP_KEYS)*3} welds")

    def socket_to_ctrl(self, sx: float, sy: float) -> tuple[float, float]:
        xm = float(np.clip(-sx, -0.160, 0.160))
        ym = float(np.clip(0.010 - sy, -0.115, 0.115))
        return xm, ym

    def place_on_tray(self, key: str, x: float) -> None:
        qa, da = self.get_body_jnt(self.n(f"kc_{key}"))
        self.d.qpos[qa + 0] = x
        self.d.qpos[qa + 1] = S4_TRAY_WORLD_Y
        self.d.qpos[qa + 2] = S4_TRAY_WORLD_Z
        self.d.qpos[qa + 3] = 1.0
        self.d.qpos[qa + 4:qa + 7] = 0.0
        self.d.qvel[da:da + 6] = 0.0
        self.tray_state[key] = (qa, da)
        self.tray_x[key] = x

    def sim_step(self, viewer: Optional[mujoco.viewer.Handle]) -> None:
        self.d.qpos[self.x_qa] = self.d.ctrl[self.x_aid]
        self.d.qpos[self.y_qa] = self.d.ctrl[self.y_aid]
        self.d.qvel[self.x_doa] = 0.0
        self.d.qvel[self.y_doa] = 0.0

        if S4_STATIC_INSERT:
            for eqid in self.inserted_eq_map.values():
                self.d.eq_active[eqid] = 0
        for qa, da, sid in self.inserted_map.values():
            sp = self.d.site_xpos[sid]
            self.d.qpos[qa + 0] = sp[0]
            self.d.qpos[qa + 1] = sp[1]
            self.d.qpos[qa + 2] = sp[2]
            self.d.qpos[qa + 3] = 1.0
            self.d.qpos[qa + 4:qa + 7] = 0.0
            self.d.qvel[da:da + 6] = 0.0

        for qa, da in self.tray_state.values():
            self.d.qpos[qa + 1] = S4_TRAY_WORLD_Y
            self.d.qpos[qa + 2] = S4_TRAY_WORLD_Z
            self.d.qpos[qa + 3] = 1.0
            self.d.qpos[qa + 4:qa + 7] = 0.0
            self.d.qvel[da:da + 6] = 0.0

        for qa, da, tip_sid in self.carried_map.values():
            tip = self.d.site_xpos[tip_sid]
            self.d.qpos[qa + 0] = tip[0]
            self.d.qpos[qa + 1] = tip[1]
            self.d.qpos[qa + 2] = tip[2] + CARRY_Z_OFFSET
            self.d.qpos[qa + 3] = 1.0
            self.d.qpos[qa + 4:qa + 7] = 0.0
            self.d.qvel[da:da + 6] = 0.0

        if self.pre_step_callback is not None:
            self.pre_step_callback()

        t0 = time.perf_counter()
        mujoco.mj_step(self.m, self.d)
        if viewer is not None:
            viewer.sync()
        rem = TIMESTEP - (time.perf_counter() - t0)
        if rem > 0:
            time.sleep(rem)

    def settle(self, viewer: Optional[mujoco.viewer.Handle], steps: int = SETTLE_STEPS) -> None:
        for _ in range(steps):
            self.sim_step(viewer)

    def drive_to(self, targets: Dict[str, float], steps: int, viewer: Optional[mujoco.viewer.Handle], label: str = "") -> None:
        act_ids = {name: self.get_actuator(self.n(name)) for name in targets}
        starts = {name: self.d.ctrl[aid] for name, aid in act_ids.items()}
        for i in range(steps):
            t = smoothstep((i + 1) / steps)
            for name, target in targets.items():
                aid = act_ids[name]
                self.d.ctrl[aid] = starts[name] + (target - starts[name]) * t
            self.sim_step(viewer)
        if label:
            print(f"  ✓ {label}")

    def snap_to_site(self, key: str) -> None:
        sid = resolve_name(self.m, mujoco.mjtObj.mjOBJ_SITE, self.n(f"kc_{key}"))
        mujoco.mj_forward(self.m, self.d)
        sp = self.d.site_xpos[sid].copy()
        qa, da = self.get_body_jnt(self.n(f"kc_{key}"))
        self.d.qpos[qa + 0] = sp[0]
        self.d.qpos[qa + 1] = sp[1]
        self.d.qpos[qa + 2] = sp[2]
        self.d.qpos[qa + 3] = 1.0
        self.d.qpos[qa + 4:qa + 7] = 0.0
        self.d.qvel[da:da + 6] = 0.0

    def tray_advance_step(self, start_x: Dict[str, float], step: int, advance_amount: float) -> None:
        t = smoothstep((step + 1) / ADVANCE_STEPS)
        for key, (qa, _da) in self.tray_state.items():
            new_x = start_x[key] + advance_amount * t
            self.d.qpos[qa + 0] = new_x
            self.tray_x[key] = new_x

    def debug_head_tips(self, label: str) -> None:
        mujoco.mj_forward(self.m, self.d)
        parts = []
        for h in range(2):
            sid = resolve_name(self.m, mujoco.mjtObj.mjOBJ_SITE, self.n(f"head_{h + 1}_tip_site"))
            p = self.d.site_xpos[sid]
            parts.append(f"H{h + 1}=({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f})")
        print(f"  [S4 debug] head tips {label}: " + "  ".join(parts))

    def debug_insert_alignment(self, head_idx: int, key: str) -> None:
        mujoco.mj_forward(self.m, self.d)
        tip_sid = resolve_name(self.m, mujoco.mjtObj.mjOBJ_SITE, self.n(f"head_{head_idx + 1}_tip_site"))
        site_sid = resolve_name(self.m, mujoco.mjtObj.mjOBJ_SITE, self.n(f"kc_{key}"))
        tip = self.d.site_xpos[tip_sid].copy()
        site = self.d.site_xpos[site_sid].copy()
        dxy_mm = float(np.linalg.norm((tip - site)[:2]) * 1000.0)
        print(f"  [S4 debug] insert head H{head_idx + 1} → kc_{key}: XY error {dxy_mm:.1f} mm")

    def do_pick_and_insert(self, viewer: Optional[mujoco.viewer.Handle], ins_head_idx: int, ins_key: Optional[str], pick_head_idx: int, pick_key: Optional[str]) -> None:
        targets_down: Dict[str, float] = {}
        targets_up: Dict[str, float] = {}
        if ins_key is not None:
            targets_down[f"head_{ins_head_idx + 1}_drive"] = HEAD_Z_INSERT
            targets_up[f"head_{ins_head_idx + 1}_drive"] = HEAD_Z_RETRACT
        if pick_key is not None:
            targets_down[f"head_{pick_head_idx + 1}_drive"] = HEAD_Z_INSERT
            targets_up[f"head_{pick_head_idx + 1}_drive"] = HEAD_Z_RETRACT

        self.drive_to(targets_down, HEAD_Z_STEPS, viewer, "S4 heads down")
        self.settle(viewer)

        if ins_key is not None:
            self.debug_insert_alignment(ins_head_idx, ins_key)
            self.snap_to_site(ins_key)
            self.d.eq_active[self.get_eq(self.n(f"pick{ins_head_idx + 1}_{ins_key}"))] = 0
            ins_eq = self.get_eq(self.n(f"ins_{ins_key}"))
            self.d.eq_active[ins_eq] = 0 if S4_STATIC_INSERT else 1
            sid = resolve_name(self.m, mujoco.mjtObj.mjOBJ_SITE, self.n(f"kc_{ins_key}"))
            qa, da = self.get_body_jnt(self.n(f"kc_{ins_key}"))
            self.inserted_map[ins_key] = (qa, da, sid)
            self.inserted_eq_map[ins_key] = ins_eq
            self.carried_map.pop(ins_head_idx, None)

        if pick_key is not None:
            tip_sid = resolve_name(self.m, mujoco.mjtObj.mjOBJ_SITE, self.n(f"head_{pick_head_idx + 1}_tip_site"))
            mujoco.mj_forward(self.m, self.d)
            tip = self.d.site_xpos[tip_sid].copy()
            qa, da = self.get_body_jnt(self.n(f"kc_{pick_key}"))
            self.d.qpos[qa + 0] = tip[0]
            self.d.qpos[qa + 1] = tip[1]
            self.d.qpos[qa + 2] = tip[2] + CARRY_Z_OFFSET
            self.d.qpos[qa + 3] = 1.0
            self.d.qpos[qa + 4:qa + 7] = 0.0
            self.d.qvel[da:da + 6] = 0.0
            self.d.eq_active[self.get_eq(self.n(f"pick{pick_head_idx + 1}_{pick_key}"))] = 1
            self.tray_state.pop(pick_key, None)
            self.tray_x.pop(pick_key, None)
            self.carried_map[pick_head_idx] = (qa, da, tip_sid)

        mujoco.mj_forward(self.m, self.d)
        if viewer is not None:
            viewer.sync()

        self.drive_to(targets_up, HEAD_Z_STEPS, viewer, "S4 heads up")
        self.settle(viewer)

    def run(self, viewer: Optional[mujoco.viewer.Handle]) -> None:
        if self.ran:
            return
        print(f"\n[S4] Starting integrated keycap insertion for {len(KEYCAP_KEYS)} keycaps...")
        print(f"  S4 static-insert visual follow: {'ON' if S4_STATIC_INSERT else 'OFF'}")
        self.verify()

        self.tray_state.clear()
        self.tray_x.clear()
        self.inserted_map.clear()
        self.inserted_eq_map.clear()
        self.carried_map.clear()
        tray_init = build_keycap_tray_x(KEYCAP_KEYS)
        for key in KEYCAP_KEYS:
            self.place_on_tray(key, tray_init[key])

        self.d.ctrl[self.x_aid] = 0.0
        self.d.ctrl[self.y_aid] = 0.0
        self.d.ctrl[self.col_aid] = 0.0
        self.drive_to({"head_1_drive": HEAD_Z_RETRACT, "head_2_drive": HEAD_Z_RETRACT}, 100, viewer, "S4 home")
        self.settle(viewer)
        self.debug_head_tips("home before prime")

        # In the integrated S4 cell, Head 1 is physically over the keycap tray at home
        # and Head 2 is over the insert line. This preserves the standalone alternating
        # mechanism but uses the correct integrated head index phase.
        print("[S4] Prime: Head 1 picks kc_Esc from tray")
        self.do_pick_and_insert(viewer, ins_head_idx=1, ins_key=None, pick_head_idx=0, pick_key="Esc")
        head_carry = ["Esc", None]
        kc_queue_idx = 1
        insert_head = 0
        print("  ✓ Head 1 carries kc_Esc")

        installed = 0
        col_angle = 0.0
        col_sign = +1
        cycle = 0
        while installed < len(KEYCAP_KEYS):
            pick_head = 1 - insert_head
            ins_key = head_carry[insert_head]
            pick_key = KEYCAP_KEYS[kc_queue_idx] if kc_queue_idx < len(KEYCAP_KEYS) else None
            cycle += 1
            print(f"[S4 C{cycle:02d}] install={ins_key or 'none'} pick={pick_key or 'none'}")

            col_angle += col_sign * np.pi
            col_angle = (col_angle + np.pi) % (2 * np.pi) - np.pi
            col_sign = -col_sign

            if ins_key is not None:
                ins_sx, ins_sy = KEYCAP_MAP[ins_key]
                ins_xc, ins_yc = self.socket_to_ctrl(ins_sx, ins_sy)
            else:
                ins_xc, ins_yc = 0.0, 0.0

            advance_amount = keycap_advance_amount(KEYCAP_KEYS[kc_queue_idx - 1], pick_key) if pick_key is not None and kc_queue_idx > 0 else 0.0
            col_start = self.d.ctrl[self.col_aid]
            x_start = self.d.ctrl[self.x_aid]
            y_start = self.d.ctrl[self.y_aid]
            xy_steps = max(1, int(ROTATE_STEPS * 0.8))
            advance_start_x = {k: self.tray_x[k] for k in self.tray_state}

            for step in range(ROTATE_STEPS):
                tc = smoothstep((step + 1) / ROTATE_STEPS)
                self.d.ctrl[self.col_aid] = col_start + (col_angle - col_start) * tc
                if step < xy_steps:
                    tx = smoothstep((step + 1) / xy_steps)
                    self.d.ctrl[self.x_aid] = x_start + (ins_xc - x_start) * tx
                    self.d.ctrl[self.y_aid] = y_start + (ins_yc - y_start) * tx
                if step < ADVANCE_STEPS and pick_key is not None:
                    self.tray_advance_step(advance_start_x, step, advance_amount)
                self.sim_step(viewer)

            self.settle(viewer)
            if ins_key is not None:
                self.debug_insert_alignment(insert_head, ins_key)
            self.do_pick_and_insert(viewer, insert_head, ins_key, pick_head, pick_key)

            if ins_key is not None:
                head_carry[insert_head] = None
                installed += 1
                print(f"  ✓ S4 installed {installed}/{len(KEYCAP_KEYS)}: kc_{ins_key}")
            if pick_key is not None:
                head_carry[pick_head] = pick_key
                kc_queue_idx += 1
            insert_head = pick_head

        print("[S4] Keycap insertion complete")
        self.ran = True


@dataclass
class CellRuntime:
    state: str = "INIT"
    state_t: float = 0.0
    last_print_state: str = ""


class AssemblyCellController:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.m = model
        self.d = data
        self.rt = CellRuntime()
        self.s3_runner = S3SwitchRunner(model, data, "s3")
        self.s4_runner = S4KeycapRunner(model, data, "s4")

        self.pallet_joint_id = resolve_name(model, mujoco.mjtObj.mjOBJ_JOINT, "pallet_joint")
        self.pallet_qadr = model.jnt_qposadr[self.pallet_joint_id]
        self.pallet_vadr = model.jnt_dofadr[self.pallet_joint_id]
        self.pallet_body_id = resolve_name(model, mujoco.mjtObj.mjOBJ_BODY, "pallet")
        self.s3_table_body_id = resolve_name(model, mujoco.mjtObj.mjOBJ_BODY, "s3_table_top_body")
        self.s3_pallet_clamp_site_id = resolve_name(model, mujoco.mjtObj.mjOBJ_SITE, "s3_pallet_clamp_site")
        self.s3_lock_eq_id = resolve_name(model, mujoco.mjtObj.mjOBJ_EQUALITY, "s3_pallet_lock_weld")
        self.s3_x_joint_id = resolve_name(model, mujoco.mjtObj.mjOBJ_JOINT, "s3_x_joint")
        self.s3_y_joint_id = resolve_name(model, mujoco.mjtObj.mjOBJ_JOINT, "s3_y_joint")
        self.s3_x_qadr = model.jnt_qposadr[self.s3_x_joint_id]
        self.s3_y_qadr = model.jnt_qposadr[self.s3_y_joint_id]
        self.s3_x_vadr = model.jnt_dofadr[self.s3_x_joint_id]
        self.s3_y_vadr = model.jnt_dofadr[self.s3_y_joint_id]
        self.s3_lock_engaged = False
        self.s3_follow_active = False

        # Bodies installed on the moving workpiece are converted to pallet-relative
        # visual payloads after a station completes. This prevents S3 switches from
        # being left behind at the S3 table when the pallet moves to S4.
        # name -> (qpos_adr, dof_adr, local_pos, local_quat) relative to pallet body.
        self.pallet_payloads: Dict[str, tuple[int, int, np.ndarray, np.ndarray]] = {}
        self.s3_payload_captured = False
        self.s4_payload_captured = False

        self.s3_runner.pre_step_callback = self.apply_all_kinematic_followers

        self.s4_pallet_clamp_site_id = resolve_name(model, mujoco.mjtObj.mjOBJ_SITE, "s4_pallet_clamp_site")
        self.s4_x_joint_id = resolve_name(model, mujoco.mjtObj.mjOBJ_JOINT, "s4_x_joint")
        self.s4_y_joint_id = resolve_name(model, mujoco.mjtObj.mjOBJ_JOINT, "s4_y_joint")
        self.s4_x_qadr = model.jnt_qposadr[self.s4_x_joint_id]
        self.s4_y_qadr = model.jnt_qposadr[self.s4_y_joint_id]
        self.s4_x_vadr = model.jnt_dofadr[self.s4_x_joint_id]
        self.s4_y_vadr = model.jnt_dofadr[self.s4_y_joint_id]
        self.s4_follow_active = False
        self.s4_runner.pre_step_callback = self.apply_all_kinematic_followers

        self.checkpoints: Dict[str, np.ndarray] = {}
        for name in sorted(set(CHECKPOINT_SITE_BY_STATE.values())):
            sid = resolve_name(model, mujoco.mjtObj.mjOBJ_SITE, name)
            self.checkpoints[name] = model.site_pos[sid].copy()

        self.transition("MOVE_TO_S1")
        self.snap_pallet_to_site("cell_pallet_s1_stop_site")

    def pallet_pos(self) -> np.ndarray:
        return self.d.qpos[self.pallet_qadr:self.pallet_qadr + 3].copy()

    def snap_pallet(self, pos: np.ndarray) -> None:
        self.d.qpos[self.pallet_qadr:self.pallet_qadr + 3] = np.asarray(pos, dtype=float)
        self.d.qpos[self.pallet_qadr + 3:self.pallet_qadr + 7] = PALLET_QUAT
        self.d.qvel[self.pallet_vadr:self.pallet_vadr + 6] = 0.0
        mujoco.mj_forward(self.m, self.d)

    def snap_pallet_to_site(self, site_name: str) -> None:
        self.snap_pallet(self.checkpoints[site_name])

    def snap_pallet_to_s3_clamp_site(self) -> None:
        """Snap pallet to the moving S3 table clamp site in world coordinates."""
        mujoco.mj_forward(self.m, self.d)
        self.snap_pallet(self.d.site_xpos[self.s3_pallet_clamp_site_id].copy())

    def apply_s3_pallet_follow_if_active(self) -> None:
        """Kinematically hold the pallet on the moving S3 table clamp site.

        This is intentionally not a weld. The pallet and loaded assembly are visual
        payloads, so continuous pose overwrite avoids contact/equality solver fights.
        """
        if not self.s3_follow_active:
            return
        mujoco.mj_forward(self.m, self.d)
        site_pos = self.d.site_xpos[self.s3_pallet_clamp_site_id].copy()
        self.d.qpos[self.pallet_qadr:self.pallet_qadr + 3] = site_pos
        self.d.qpos[self.pallet_qadr + 3:self.pallet_qadr + 7] = PALLET_QUAT
        self.d.qvel[self.pallet_vadr:self.pallet_vadr + 6] = 0.0

    def capture_body_as_pallet_payload(self, payload_name: str, body_name: str) -> None:
        """Store current body pose relative to pallet, then keep it riding with pallet."""
        bid = resolve_name(self.m, mujoco.mjtObj.mjOBJ_BODY, body_name)
        jid = self.m.body_jntadr[bid]
        if jid < 0:
            return
        qa = self.m.jnt_qposadr[jid]
        da = self.m.jnt_dofadr[jid]

        mujoco.mj_forward(self.m, self.d)
        pallet_pos = self.d.xpos[self.pallet_body_id].copy()
        pallet_quat = normalize_quat(self.d.xquat[self.pallet_body_id].copy())
        body_pos = self.d.xpos[bid].copy()
        body_quat = normalize_quat(self.d.xquat[bid].copy())

        inv_pallet_quat = quat_conj(pallet_quat)
        local_pos = quat_rotate(inv_pallet_quat, body_pos - pallet_pos)
        local_quat = normalize_quat(quat_mul(inv_pallet_quat, body_quat))
        self.pallet_payloads[payload_name] = (qa, da, local_pos, local_quat)

    def capture_s3_switch_payloads(self) -> None:
        """After S3, make installed switches travel with the pallet into S4."""
        if self.s3_payload_captured:
            return
        # Ensure final S3 table-home/pallet-follow pose is current before measuring offsets.
        self.apply_s3_pallet_follow_if_active()
        count = 0
        for key in SWITCH_KEYS:
            body_name = f"s3_sw_{key}"
            if mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, body_name) >= 0:
                self.capture_body_as_pallet_payload(f"s3_sw_{key}", body_name)
                count += 1
        self.s3_payload_captured = True
        print(f"[S3] captured {count} installed switches as pallet payloads for S4 handoff")

    def capture_s4_keycap_payloads(self) -> None:
        """After S4, make installed keycaps remain on the workpiece during exit."""
        if self.s4_payload_captured:
            return
        self.apply_s4_pallet_follow_if_active()
        count = 0
        for key in KEYCAP_KEYS:
            body_name = f"s4_kc_{key}"
            if mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, body_name) >= 0:
                self.capture_body_as_pallet_payload(f"s4_kc_{key}", body_name)
                count += 1
        self.s4_payload_captured = True
        print(f"[S4] captured {count} installed keycaps as pallet payloads for exit handoff")

    def apply_pallet_payload_follow(self) -> None:
        if not self.pallet_payloads:
            return
        mujoco.mj_forward(self.m, self.d)
        pallet_pos = self.d.qpos[self.pallet_qadr:self.pallet_qadr + 3].copy()
        pallet_quat = normalize_quat(self.d.qpos[self.pallet_qadr + 3:self.pallet_qadr + 7].copy())
        for _name, (qa, da, local_pos, local_quat) in self.pallet_payloads.items():
            world_pos = pallet_pos + quat_rotate(pallet_quat, local_pos)
            world_quat = normalize_quat(quat_mul(pallet_quat, local_quat))
            self.d.qpos[qa:qa + 3] = world_pos
            self.d.qpos[qa + 3:qa + 7] = world_quat
            self.d.qvel[da:da + 6] = 0.0

    def apply_all_kinematic_followers(self) -> None:
        # Order matters: first move the pallet to the active station clamp, then move
        # all already-installed payload bodies relative to that pallet.
        self.apply_s3_pallet_follow_if_active()
        self.apply_s4_pallet_follow_if_active()
        self.apply_pallet_payload_follow()

    def set_s3_table_home(self) -> None:
        """Kinematically home the S3 XY table before locking the pallet."""
        self.d.ctrl[self.s3_runner.x_aid] = 0.0
        self.d.ctrl[self.s3_runner.y_aid] = 0.0
        self.d.qpos[self.s3_x_qadr] = 0.0
        self.d.qpos[self.s3_y_qadr] = 0.0
        self.d.qvel[self.s3_x_vadr] = 0.0
        self.d.qvel[self.s3_y_vadr] = 0.0
        mujoco.mj_forward(self.m, self.d)

    def update_s3_lock_relpose_from_current_pose(self) -> None:
        """Make the weld lock the current table-to-pallet transform, not XML qpos0."""
        table_pos = self.d.xpos[self.s3_table_body_id].copy()
        table_quat = normalize_quat(self.d.xquat[self.s3_table_body_id].copy())
        pallet_pos = self.d.xpos[self.pallet_body_id].copy()
        pallet_quat = normalize_quat(self.d.xquat[self.pallet_body_id].copy())

        inv_table_quat = quat_conj(table_quat)
        rel_pos = quat_rotate(inv_table_quat, pallet_pos - table_pos)
        rel_quat = normalize_quat(quat_mul(inv_table_quat, pallet_quat))

        self.m.eq_data[self.s3_lock_eq_id, 0:3] = rel_pos
        self.m.eq_data[self.s3_lock_eq_id, 3:7] = rel_quat

    def engage_s3_pallet_lock(self) -> None:
        if self.s3_lock_engaged or self.s3_follow_active:
            return
        self.set_s3_table_home()
        # Pass 2E: always start from a clean clamp pose. In follow/snap_only modes,
        # the controller keeps overwriting the pallet pose every step instead of
        # letting contacts or equality constraints move the free body.
        self.snap_pallet_to_s3_clamp_site()

        if S3_LOCK_MODE in {"follow", "snap_only"}:
            self.s3_follow_active = True
            if S3_LOCK_MODE == "snap_only":
                print("[S3] snap_only mode: continuously holding pallet at s3_pallet_clamp_site; S3 sequence disabled")
            else:
                print("[S3] follow mode: pallet will continuously follow s3_pallet_clamp_site; weld disabled")
            return

        # Old weld mode is kept for comparison only.
        self.update_s3_lock_relpose_from_current_pose()
        self.d.eq_active[self.s3_lock_eq_id] = 1
        self.s3_lock_engaged = True
        mujoco.mj_forward(self.m, self.d)

    def disengage_s3_pallet_lock(self) -> None:
        if self.s3_follow_active:
            self.apply_s3_pallet_follow_if_active()
            self.s3_follow_active = False
            mujoco.mj_forward(self.m, self.d)
            return

        if not self.s3_lock_engaged:
            return
        mujoco.mj_forward(self.m, self.d)
        pallet_world_pos = self.d.xpos[self.pallet_body_id].copy()
        pallet_world_quat = normalize_quat(self.d.xquat[self.pallet_body_id].copy())
        self.d.eq_active[self.s3_lock_eq_id] = 0
        self.s3_lock_engaged = False
        self.d.qpos[self.pallet_qadr:self.pallet_qadr + 3] = pallet_world_pos
        self.d.qpos[self.pallet_qadr + 3:self.pallet_qadr + 7] = pallet_world_quat
        self.d.qvel[self.pallet_vadr:self.pallet_vadr + 6] = 0.0
        mujoco.mj_forward(self.m, self.d)

    def snap_pallet_to_s4_clamp_site(self) -> None:
        mujoco.mj_forward(self.m, self.d)
        self.snap_pallet(self.d.site_xpos[self.s4_pallet_clamp_site_id].copy())

    def apply_s4_pallet_follow_if_active(self) -> None:
        if not self.s4_follow_active:
            return
        mujoco.mj_forward(self.m, self.d)
        site_pos = self.d.site_xpos[self.s4_pallet_clamp_site_id].copy()
        self.d.qpos[self.pallet_qadr:self.pallet_qadr + 3] = site_pos
        self.d.qpos[self.pallet_qadr + 3:self.pallet_qadr + 7] = PALLET_QUAT
        self.d.qvel[self.pallet_vadr:self.pallet_vadr + 6] = 0.0

    def set_s4_table_home(self) -> None:
        self.d.ctrl[self.s4_runner.x_aid] = 0.0
        self.d.ctrl[self.s4_runner.y_aid] = 0.0
        self.d.qpos[self.s4_x_qadr] = 0.0
        self.d.qpos[self.s4_y_qadr] = 0.0
        self.d.qvel[self.s4_x_vadr] = 0.0
        self.d.qvel[self.s4_y_vadr] = 0.0
        mujoco.mj_forward(self.m, self.d)

    def engage_s4_pallet_lock(self) -> None:
        if self.s4_follow_active:
            return
        self.set_s4_table_home()
        self.snap_pallet_to_s4_clamp_site()
        self.s4_follow_active = True
        print("[S4] follow mode: pallet will continuously follow s4_pallet_clamp_site; weld disabled")

    def disengage_s4_pallet_lock(self) -> None:
        if self.s4_follow_active:
            self.apply_s4_pallet_follow_if_active()
            self.s4_follow_active = False
            mujoco.mj_forward(self.m, self.d)

    def advance_pallet_toward(self, target: np.ndarray, dt: float) -> bool:
        pos = self.pallet_pos()
        delta = target - pos
        dist_xy = float(np.linalg.norm(delta[:2]))
        if dist_xy <= STOP_THRESHOLD:
            self.snap_pallet(target)
            return True
        step = CDLR_SPEED * dt
        if step >= dist_xy:
            self.snap_pallet(target)
            return True
        direction_xy = delta[:2] / dist_xy
        new_pos = pos.copy()
        new_pos[:2] += direction_xy * step
        new_pos[2] = target[2]
        self.snap_pallet(new_pos)
        return False

    def transition(self, new_state: str) -> None:
        self.rt.state = new_state
        self.rt.state_t = 0.0

    def dwell_complete(self, state: str, dt: float) -> bool:
        self.rt.state_t += dt
        return self.rt.state_t >= CYCLE_TIMES[state] * TIME_SCALE

    def step(self, viewer: Optional[mujoco.viewer.Handle] = None) -> None:
        dt = self.m.opt.timestep
        state = self.rt.state

        if state != self.rt.last_print_state:
            print(f"[{self.d.time:8.3f}s] state -> {state}")
            self.rt.last_print_state = state

        if state in CHECKPOINT_SITE_BY_STATE:
            site_name = CHECKPOINT_SITE_BY_STATE[state]
            if self.advance_pallet_toward(self.checkpoints[site_name], dt):
                self.transition(NEXT_STATE_AFTER_MOVE[state])

        elif state == "S1_PROCESS_OR_WAIT":
            self.snap_pallet_to_site("cell_pallet_s1_stop_site")
            if self.dwell_complete(state, dt):
                self.transition("MOVE_TO_S3_BUFFER")

        elif state == "S3_LOCK":
            # Pass 2E: follow/snap-only are kinematic; weld is legacy debug.
            self.engage_s3_pallet_lock()
            if S3_LOCK_MODE == "snap_only":
                # Hold the visual pose for inspection; do not advance to S3.
                pass
            elif self.dwell_complete(state, dt):
                self.transition("S3_RUN_SWITCH_SEQUENCE")

        elif state == "S3_RUN_SWITCH_SEQUENCE":
            self.s3_runner.run(viewer)
            self.transition("S3_UNLOCK")

        elif state == "S3_UNLOCK":
            # Always home the table before unlocking; otherwise the released pallet can
            # inherit a strange XY offset from the last insertion target.
            self.set_s3_table_home()
            self.apply_s3_pallet_follow_if_active()
            self.capture_s3_switch_payloads()
            self.disengage_s3_pallet_lock()
            if self.dwell_complete(state, dt):
                self.transition("MOVE_TO_S3_EXIT")

        elif state == "S4_LOCK":
            self.engage_s4_pallet_lock()
            if S4_LOCK_MODE == "snap_only":
                pass
            elif self.dwell_complete(state, dt):
                self.transition("S4_RUN_KEYCAP_SEQUENCE")

        elif state == "S4_RUN_KEYCAP_SEQUENCE":
            self.s4_runner.run(viewer)
            self.transition("S4_UNLOCK")

        elif state == "S4_UNLOCK":
            self.set_s4_table_home()
            self.apply_s4_pallet_follow_if_active()
            self.capture_s4_keycap_payloads()
            self.disengage_s4_pallet_lock()
            if self.dwell_complete(state, dt):
                self.transition("MOVE_TO_S4_EXIT")

        elif state == "DONE_S1_S3_S4":
            self.snap_pallet_to_site("cell_pallet_s4_exit_site")

        else:
            raise RuntimeError(f"Unknown cell state: {state}")

        self.apply_all_kinematic_followers()
        mujoco.mj_step(self.m, self.d)


def resolve_model_path() -> Path:
    here = Path(__file__).resolve()
    candidates = [
        here.parents[1] / "models" / "stations" / "assembly_cellA.xml",
        Path.cwd() / "models" / "stations" / "assembly_cellA.xml",
        Path.cwd() / "assembly_cellA.xml",
        Path("/mnt/data/assembly_cellA_pass2I_s4_payload_follow.xml"),
        Path("/mnt/data/assembly_cellA_pass2G_s3_static_follow.xml"),
        Path("/mnt/data/assembly_cellA_pass2F_s3_standalone_logic.xml"),
        Path("/mnt/data/assembly_cellA_pass2E_s3_follow.xml"),
        Path("/mnt/data/assembly_cellA_pass2D_s3_safe_clamp.xml"),
        Path("/mnt/data/assembly_cellA_pass2C_s3_lock.xml"),
        Path("/mnt/data/assembly_cellA_pass2_s3.xml"),
        Path("/mnt/data/assembly_cellA_pass1.xml"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Could not find assembly_cellA.xml")


def main() -> None:
    model_path = resolve_model_path()
    print(f"Loading model: {model_path}")
    print(f"TIME_SCALE={TIME_SCALE}; S3_LOCK_MODE={S3_LOCK_MODE}; S3_STATIC_INSERT={int(S3_STATIC_INSERT)}; S3_MAX_SWITCHES={len(SWITCH_KEYS)}; S4_LOCK_MODE={S4_LOCK_MODE}; S4_STATIC_INSERT={int(S4_STATIC_INSERT)}; S4_MAX_KEYCAPS={len(KEYCAP_KEYS)}")
    m = mujoco.MjModel.from_xml_path(str(model_path))
    d = mujoco.MjData(m)
    ctrl = AssemblyCellController(m, d)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            ctrl.step(viewer)
            viewer.sync()


if __name__ == "__main__":
    main()
