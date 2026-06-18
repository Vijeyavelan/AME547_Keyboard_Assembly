"""AME 547 — Pass 3A full-cell process overview controller.

This controller is intentionally a visual process-flow demo, not a full station
physics integration. One kinematic/mocap overview pallet moves through S0→S7
and staged visual parts are revealed at S0, S1, S3, and S4.

Recommended XML:
    models/stations/assembly_cell.xml

Run:
    mjpython controllers/assembly_cell.py
"""


import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import mujoco
    import mujoco.viewer
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Could not import mujoco. Run with the project venv and mjpython, e.g.\n"
        "    mjpython controllers/assembly_cell.py"
    ) from exc

PALLET_BODY = "p3a_overview_pallet"
PALLET_JOINT = "p3a_overview_pallet_freejoint"  # fallback for old XML only
PALLET_QUAT = np.array([1.0, 0.0, 0.0, 0.0])

TIME_SCALE = float(os.environ.get("AME_CELL_OVERVIEW_TIME_SCALE", "1.0"))
MOVE_SPEED = float(os.environ.get("AME_CELL_OVERVIEW_SPEED", "0.45"))  # m/s
MIN_MOVE_DURATION = float(os.environ.get("AME_CELL_OVERVIEW_MIN_MOVE", "1.0"))
REALTIME = os.environ.get("AME_REALTIME", "1").strip().lower() not in {"0", "false", "no", "off"}
LOOP_DEMO = os.environ.get("AME_CELL_OVERVIEW_LOOP", "0").strip().lower() in {"1", "true", "yes", "on"}

DWELL = {
    "S0_LOAD_BOTTOM_CASE": 0.75,
    "S1_PROCESS": 5.0,
    "S3_BUFFER_WAIT": 2.0,
    "S3_PROCESS": 5.0,
    "S4_BUFFER_WAIT": 2.0,
    "S4_PROCESS": 5.0,
    "S5_LASER_OVERVIEW": 2.0,
    "S6_PACKAGING_OVERVIEW": 2.0,
    "S7_EXIT_PAUSE": 1.0,
}

PATH_SITES = {
    "S0": "cell_pallet_s0_load_site",
    "S1": "cell_pallet_s1_stop_site",
    "S3_BUFFER": "cell_pallet_s3_buffer_site",
    "S3": "cell_pallet_s3_station_site",
    "S3_EXIT": "cell_pallet_s3_exit_site",
    "S4_BUFFER": "cell_pallet_s4_buffer_site",
    "S4": "cell_pallet_s4_station_site",
    "S4_EXIT": "cell_pallet_s4_exit_site",
    "S5": "cell_pallet_s5_station_site",
    "S6": "cell_pallet_s6_station_site",
    "S7": "cell_pallet_s7_exit_site",
}

STAGE_PREFIXES = {
    "s0": "p3a_stage_s0_",
    "s1": "p3a_stage_s1_",
    "s3": "p3a_stage_s3_",
    "s4": "p3a_stage_s4_",
}


def project_root() -> Path:
    here = Path(__file__).resolve()
    return here.parent.parent if here.parent.name == "controllers" else here.parent


def model_path() -> Path:
    env_path = os.environ.get("AME_CELL_OVERVIEW_XML")
    if env_path:
        return Path(env_path).expanduser().resolve()

    root = project_root()
    candidates = [
        root / "models" / "stations" / "assembly_cell.xml",
        root / "models" / "stations" / "assembly_cellA_pass3A.xml",
        root / "models" / "stations" / "assembly_cellA.xml",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


@dataclass
class MoveSegment:
    start: np.ndarray
    target: np.ndarray
    duration: float
    elapsed: float = 0.0

    def step(self, dt: float) -> tuple[np.ndarray, bool]:
        self.elapsed += dt
        u = min(self.elapsed / max(self.duration, 1e-6), 1.0)
        s = u * u * (3.0 - 2.0 * u)  # smoothstep
        return (1.0 - s) * self.start + s * self.target, u >= 1.0


class OverviewController:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.model = model
        self.data = data

        self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, PALLET_BODY)
        if self.body_id < 0:
            raise RuntimeError(f"Missing body {PALLET_BODY!r}. Use the Pass 3A overview XML.")

        self.mocap_id = int(model.body_mocapid[self.body_id])
        self.joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, PALLET_JOINT)
        if self.mocap_id < 0 and self.joint_id < 0:
            raise RuntimeError(
                f"{PALLET_BODY!r} must either be mocap='true' or have freejoint {PALLET_JOINT!r}."
            )

        self.qpos_adr = model.jnt_qposadr[self.joint_id] if self.joint_id >= 0 else -1
        self.dof_adr = model.jnt_dofadr[self.joint_id] if self.joint_id >= 0 else -1

        self.site_pos = self._load_site_positions_from_model()
        self.stage_geom_ids = self._find_stage_geoms()

        self.state = "INIT"
        self.state_time = 0.0
        self.move: MoveSegment | None = None
        self.move_next_state = ""

        self._hide_all_stages()
        self._set_pallet_pose(self.site_pos["S0"])
        mujoco.mj_forward(model, data)

    def _load_site_positions_from_model(self) -> dict[str, np.ndarray]:
        """Read site local positions directly.

        The Pass 3A path sites are placed directly in worldbody. Using model.site_pos
        avoids the bad 30 m jumps that can happen if world xforms are read before
        all free bodies are initialized.
        """
        sites: dict[str, np.ndarray] = {}
        for key, site_name in PATH_SITES.items():
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            if site_id < 0:
                raise RuntimeError(f"Missing overview path site {site_name!r} for {key}.")
            sites[key] = self.model.site_pos[site_id].copy()
        return sites

    def _find_stage_geoms(self) -> dict[str, list[int]]:
        found = {name: [] for name in STAGE_PREFIXES}
        for geom_id in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            for stage, prefix in STAGE_PREFIXES.items():
                if name.startswith(prefix):
                    found[stage].append(geom_id)
        missing = [stage for stage, ids in found.items() if not ids]
        if missing:
            raise RuntimeError(f"Missing stage geoms for {missing}. Use the Pass 3A overview XML.")
        return found

    def _set_alpha(self, geom_ids: Iterable[int], alpha: float) -> None:
        for geom_id in geom_ids:
            self.model.geom_rgba[geom_id, 3] = alpha

    def _hide_all_stages(self) -> None:
        for ids in self.stage_geom_ids.values():
            self._set_alpha(ids, 0.0)

    def _show_stage(self, stage: str) -> None:
        self._set_alpha(self.stage_geom_ids[stage], 1.0)

    def _current_pallet_pos(self) -> np.ndarray:
        if self.mocap_id >= 0:
            return self.data.mocap_pos[self.mocap_id].copy()
        return self.data.qpos[self.qpos_adr : self.qpos_adr + 3].copy()

    def _set_pallet_pose(self, pos: np.ndarray) -> None:
        if self.mocap_id >= 0:
            self.data.mocap_pos[self.mocap_id] = pos
            self.data.mocap_quat[self.mocap_id] = PALLET_QUAT
        else:
            self.data.qpos[self.qpos_adr : self.qpos_adr + 3] = pos
            self.data.qpos[self.qpos_adr + 3 : self.qpos_adr + 7] = PALLET_QUAT
            self.data.qvel[self.dof_adr : self.dof_adr + 6] = 0.0

    def _log(self, message: str) -> None:
        print(f"[{self.data.time:8.3f}s] {message}")

    def _enter(self, state: str) -> None:
        self.state = state
        self.state_time = 0.0
        self._log(f"state -> {state}")

    def _begin_move(self, target_key: str, next_state: str) -> None:
        current = self._current_pallet_pos()
        target = self.site_pos[target_key]
        distance = float(np.linalg.norm(target - current))
        duration = max(MIN_MOVE_DURATION, distance / max(MOVE_SPEED, 1e-6)) * TIME_SCALE
        self.move = MoveSegment(current, target, duration)
        self.move_next_state = next_state
        self._enter(f"MOVE_TO_{target_key}")
        self._log(f"target {target_key}: {distance:.3f} m, {duration:.2f}s")

    def _dwell_done(self, state: str) -> bool:
        return self.state_time >= DWELL[state] * TIME_SCALE

    def step(self, dt: float) -> None:
        self.state_time += dt

        if self.state == "INIT":
            self._hide_all_stages()
            self._set_pallet_pose(self.site_pos["S0"])
            self._enter("S0_LOAD_BOTTOM_CASE")
            return

        if self.state.startswith("MOVE_TO_"):
            assert self.move is not None
            pos, done = self.move.step(dt)
            self._set_pallet_pose(pos)
            if done:
                self._set_pallet_pose(self.move.target)
                next_state = self.move_next_state
                self.move = None
                self.move_next_state = ""
                self._enter(next_state)
            return

        if self.state == "S0_LOAD_BOTTOM_CASE":
            if self.state_time >= 0.05:
                self._show_stage("s0")
            if self._dwell_done("S0_LOAD_BOTTOM_CASE"):
                self._begin_move("S1", "S1_PROCESS")
            return

        if self.state == "S1_PROCESS":
            if self._dwell_done("S1_PROCESS"):
                self._show_stage("s1")
                self._log("S1 complete: EPDM, battery, PCB, plate, screws/stabilizers visible")
                self._begin_move("S3_BUFFER", "S3_BUFFER_WAIT")
            return

        if self.state == "S3_BUFFER_WAIT":
            if self._dwell_done("S3_BUFFER_WAIT"):
                self._begin_move("S3", "S3_PROCESS")
            return

        if self.state == "S3_PROCESS":
            if self._dwell_done("S3_PROCESS"):
                self._show_stage("s3")
                self._log("S3 complete: switch field visible")
                self._begin_move("S3_EXIT", "S4_BUFFER_MOVE_READY")
            return

        if self.state == "S4_BUFFER_MOVE_READY":
            self._begin_move("S4_BUFFER", "S4_BUFFER_WAIT")
            return

        if self.state == "S4_BUFFER_WAIT":
            if self._dwell_done("S4_BUFFER_WAIT"):
                self._begin_move("S4", "S4_PROCESS")
            return

        if self.state == "S4_PROCESS":
            if self._dwell_done("S4_PROCESS"):
                self._show_stage("s4")
                self._log("S4 complete: keycaps visible")
                self._begin_move("S4_EXIT", "S5_MOVE_READY")
            return

        if self.state == "S5_MOVE_READY":
            self._begin_move("S5", "S5_LASER_OVERVIEW")
            return

        if self.state == "S5_LASER_OVERVIEW":
            if self._dwell_done("S5_LASER_OVERVIEW"):
                self._begin_move("S6", "S6_PACKAGING_OVERVIEW")
            return

        if self.state == "S6_PACKAGING_OVERVIEW":
            if self._dwell_done("S6_PACKAGING_OVERVIEW"):
                self._begin_move("S7", "S7_EXIT_PAUSE")
            return

        if self.state == "S7_EXIT_PAUSE":
            if self._dwell_done("S7_EXIT_PAUSE"):
                self._enter("DONE")
            return

        if self.state == "DONE" and LOOP_DEMO:
            self._enter("INIT")


def main() -> None:
    xml_path = model_path()
    print(f"Loading model: {xml_path}")
    print(
        f"TIME_SCALE={TIME_SCALE}; MOVE_SPEED={MOVE_SPEED}; REALTIME={int(REALTIME)}; LOOP={int(LOOP_DEMO)}"
    )

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    controller = OverviewController(model, data)

    print("Overview path checkpoints:")
    for key, pos in controller.site_pos.items():
        print(f"  {key:9s}: [{pos[0]: .3f}, {pos[1]: .3f}, {pos[2]: .3f}]")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 10.0
        viewer.cam.azimuth = -65
        viewer.cam.elevation = -35
        viewer.cam.lookat[:] = np.array([5.8, -0.15, 0.75])

        while viewer.is_running():
            tic = time.time()
            dt = model.opt.timestep

            controller.step(dt)
            # Kinematic/mocap body: forward first so the viewer sees the commanded pose.
            mujoco.mj_forward(model, data)
            viewer.sync()
            # Step after sync only to advance MuJoCo time. The mocap pallet will not fall.
            mujoco.mj_step(model, data)

            if REALTIME:
                sleep_time = max(0.0, model.opt.timestep - (time.time() - tic))
                if sleep_time > 0.0:
                    time.sleep(sleep_time)


if __name__ == "__main__":
    main()
