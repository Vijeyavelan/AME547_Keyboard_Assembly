"""AME 547 — Block 3 assembly-cell controller.

Run:
    mjpython controllers/assembly_cell_controller.py

This first Block-3 integration pass validates the locked straight CDLR layout,
pallet freejoint architecture, proximity stopper detection, and full S1→S6
state machine. S1/S3/S4 are represented as timed station calls so the full
cell can be visually verified before layering in the detailed station modules.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np

try:
    import mujoco
    import mujoco.viewer
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Could not import mujoco. Run this with the project venv and mjpython, e.g.\n"
        "    mjpython controllers/assembly_cell_controller.py"
    ) from exc


# ----------------------------- locked constants -----------------------------
CDLR_SPEED = 0.300       # m/s
STOP_THRESHOLD = 0.005   # m
CDLR_Z = 0.800           # belt top from station1_A.xml
PALLET_Z = 0.810         # pallet body origin from station1_A.xml
PALLET_Y = 0.0
PALLET_QUAT = np.array([1.0, 0.0, 0.0, 0.0])

S1_STOP_X = 0.185
S3_STOP_X = 3.916
S4_STOP_X = 8.090
S5_STOP_X = 11.148
S6_STOP_X = 12.648

CYCLE_TIMES = {
    "S1_RUNNING": 90.0,
    "S3_RUNNING": 99.0,
    "S4_RUNNING": 94.0,
    "S5_DWELL": 3.0,
    "S6_DWELL": 3.0,
}

# For quick visual debugging only. Default preserves locked cycle times.
# Example: AME_CELL_TIME_SCALE=0.05 mjpython controllers/assembly_cell_controller.py
TIME_SCALE = float(os.environ.get("AME_CELL_TIME_SCALE", "1.0"))

STATES = [
    "CDLR_TO_S1",
    "S1_RUNNING",
    "CDLR_TO_S3",
    "S3_RUNNING",
    "CDLR_TO_S4",
    "S4_RUNNING",
    "CDLR_TO_S5",
    "S5_DWELL",
    "CDLR_TO_S6",
    "S6_DWELL",
    "COMPLETE",
]

STOP_X_BY_STATE = {
    "CDLR_TO_S1": S1_STOP_X,
    "CDLR_TO_S3": S3_STOP_X,
    "CDLR_TO_S4": S4_STOP_X,
    "CDLR_TO_S5": S5_STOP_X,
    "CDLR_TO_S6": S6_STOP_X,
}

NEXT_STATE_AFTER_STOP = {
    "CDLR_TO_S1": "S1_RUNNING",
    "CDLR_TO_S3": "S3_RUNNING",
    "CDLR_TO_S4": "S4_RUNNING",
    "CDLR_TO_S5": "S5_DWELL",
    "CDLR_TO_S6": "S6_DWELL",
}

NEXT_STATE_AFTER_RUN = {
    "S1_RUNNING": "CDLR_TO_S3",
    "S3_RUNNING": "CDLR_TO_S4",
    "S4_RUNNING": "CDLR_TO_S5",
    "S5_DWELL": "CDLR_TO_S6",
    "S6_DWELL": "COMPLETE",
}


@dataclass
class CellRuntime:
    state: str = "CDLR_TO_S1"
    state_t: float = 0.0
    last_print_state: str = ""


class AssemblyCellController:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.m = model
        self.d = data
        self.rt = CellRuntime()
        self.pallet_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pallet_joint")
        if self.pallet_joint_id < 0:
            raise RuntimeError("pallet_joint not found in assembly_cellA.xml")
        self.pallet_qadr = model.jnt_qposadr[self.pallet_joint_id]
        self.pallet_vadr = model.jnt_dofadr[self.pallet_joint_id]
        self.stop_site_ids: Dict[str, int] = {}
        for name in ["s1_stop", "s3_stop", "s4_stop", "s5_stop", "s6_stop"]:
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
            if sid < 0:
                raise RuntimeError(f"{name} site not found in assembly_cellA.xml")
            self.stop_site_ids[name] = sid
        self.snap_pallet(S1_STOP_X)

    # -------------------------- freejoint helpers --------------------------
    def pallet_pos(self) -> np.ndarray:
        return self.d.qpos[self.pallet_qadr:self.pallet_qadr + 3].copy()

    def snap_pallet(self, x: float, y: float = PALLET_Y, z: float = PALLET_Z) -> None:
        """Stopper/station re-snap for pallet registration."""
        self.d.qpos[self.pallet_qadr:self.pallet_qadr + 3] = np.array([x, y, z])
        self.d.qpos[self.pallet_qadr + 3:self.pallet_qadr + 7] = PALLET_QUAT
        self.d.qvel[self.pallet_vadr:self.pallet_vadr + 6] = 0.0
        mujoco.mj_forward(self.m, self.d)

    def advance_pallet(self, dt: float) -> None:
        q = self.d.qpos
        q[self.pallet_qadr] += CDLR_SPEED * dt
        q[self.pallet_qadr + 1] = PALLET_Y
        q[self.pallet_qadr + 2] = PALLET_Z
        q[self.pallet_qadr + 3:self.pallet_qadr + 7] = PALLET_QUAT
        self.d.qvel[self.pallet_vadr:self.pallet_vadr + 6] = 0.0

    def near_stop(self, stop_x: float) -> bool:
        pos = self.pallet_pos()
        return np.linalg.norm(pos[:2] - np.array([stop_x, PALLET_Y])) < STOP_THRESHOLD

    # --------------------------- station functions --------------------------
    def run_s1(self, dt: float) -> bool:
        # Placeholder for extracted station1 sequence.
        self.rt.state_t += dt
        return self.rt.state_t >= CYCLE_TIMES["S1_RUNNING"] * TIME_SCALE

    def run_s3(self, dt: float) -> bool:
        # Placeholder for extracted station3 switch insertion sequence.
        self.rt.state_t += dt
        return self.rt.state_t >= CYCLE_TIMES["S3_RUNNING"] * TIME_SCALE

    def run_s4(self, dt: float) -> bool:
        # Placeholder for extracted station4 keycap installation sequence.
        self.rt.state_t += dt
        return self.rt.state_t >= CYCLE_TIMES["S4_RUNNING"] * TIME_SCALE

    def dwell(self, state: str, dt: float) -> bool:
        self.rt.state_t += dt
        return self.rt.state_t >= CYCLE_TIMES[state] * TIME_SCALE

    # ------------------------------ state step ------------------------------
    def transition(self, new_state: str) -> None:
        self.rt.state = new_state
        self.rt.state_t = 0.0

    def step(self) -> None:
        dt = self.m.opt.timestep
        state = self.rt.state

        if state != self.rt.last_print_state:
            print(f"[{self.d.time:8.3f}s] state -> {state}")
            self.rt.last_print_state = state

        # 1. CDLR advance → proximity detection → stopper re-snap.
        if state in STOP_X_BY_STATE:
            target_x = STOP_X_BY_STATE[state]
            if self.pallet_pos()[0] < target_x:
                self.advance_pallet(dt)
            if self.near_stop(target_x) or self.pallet_pos()[0] >= target_x:
                self.snap_pallet(target_x)  # stopper re-snap / station lock re-snap
                self.transition(NEXT_STATE_AFTER_STOP[state])

        # 2. Timed station calls. Later pass can replace these with full inline logic.
        elif state == "S1_RUNNING":
            self.snap_pallet(S1_STOP_X)
            if self.run_s1(dt):
                self.transition("CDLR_TO_S3")

        elif state == "S3_RUNNING":
            self.snap_pallet(S3_STOP_X)
            if self.run_s3(dt):
                self.transition("CDLR_TO_S4")

        elif state == "S4_RUNNING":
            self.snap_pallet(S4_STOP_X)
            if self.run_s4(dt):
                self.transition("CDLR_TO_S5")

        elif state == "S5_DWELL":
            self.snap_pallet(S5_STOP_X)
            if self.dwell("S5_DWELL", dt):
                self.transition("CDLR_TO_S6")

        elif state == "S6_DWELL":
            self.snap_pallet(S6_STOP_X)
            if self.dwell("S6_DWELL", dt):
                self.transition("COMPLETE")

        elif state == "COMPLETE":
            self.snap_pallet(S6_STOP_X)

        else:
            raise RuntimeError(f"Unknown cell state: {state}")

        mujoco.mj_step(self.m, self.d)


def resolve_model_path() -> Path:
    here = Path(__file__).resolve()
    candidates = [
        here.parents[1] / "models" / "stations" / "assembly_cellA.xml",  # repo/controllers -> repo/models
        here.parents[1] / "models" / "stations" / "assembly_cellA.xml",
        Path.cwd() / "models" / "stations" / "assembly_cellA.xml",
        Path.cwd() / "models" / "stations" / "assembly_cellA.xml",
        Path.cwd() / "models" / "stations" / "assembly_cellA.xml",
        Path.cwd() / "stations" / "assembly_cellA.xml",
        Path.cwd() / "stations" / "assembly_cellA.xml",
        Path("/mnt/data/assembly_cellA.xml"),
        Path("/mnt/data/assembly_cellA.xml"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Could not find assembly_cell.xml or assembly_cellA.xml")


def main() -> None:
    model_path = resolve_model_path()
    print(f"Loading model: {model_path}")
    print(f"TIME_SCALE={TIME_SCALE}  (1.0 = locked cycle times)")
    m = mujoco.MjModel.from_xml_path(str(model_path))
    d = mujoco.MjData(m)
    ctrl = AssemblyCellController(m, d)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            ctrl.step()
            viewer.sync()
            if ctrl.rt.state == "COMPLETE":
                # Keep final pose visible.
                pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
