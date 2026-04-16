"""
test_head_z.py — Verify head Z position actuator stroke

Tests head_1_drive and head_2_drive independently:
  1. Drive head 1 down to -0.015m (insertion depth), hold, retract
  2. Drive head 2 down to -0.015m, hold, retract
  3. Drive both simultaneously down, retract

Prints actual joint position each phase so you can confirm:
  - Head reaches commanded depth (within ~1mm)
  - Head retracts cleanly to 0.0
  - No oscillation / runaway

Run from repo root:
    mjpython controllers/test_head_z.py
"""

import mujoco
import mujoco.viewer
import numpy as np
import time

MODEL_PATH  = "models/stations/station3.xml"
TIMESTEP    = 0.002   # must match <option timestep="..."/> in XML

INSERT_M    = -0.015  # metres — 15mm insertion depth
RETRACT_M   =  0.000  # metres — fully retracted

MOVE_STEPS  = 200     # steps to drive down or up (~0.4 s real-time)
HOLD_STEPS  = 100     # steps to hold at position and read result
SETTLE_STEPS = 60


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def act_id(m, name):
    aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if aid < 0:
        raise ValueError(f"Actuator not found: '{name}'")
    return aid


def jnt_qadr(m, name):
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise ValueError(f"Joint not found: '{name}'")
    return m.jnt_qposadr[jid]


def read_head_pos(m, d, label):
    """Print current Z position of both heads."""
    qa1 = jnt_qadr(m, "head_1_z")
    qa2 = jnt_qadr(m, "head_2_z")
    print(f"  [{label}] head_1_z = {d.qpos[qa1]*1000:+.2f} mm   "
          f"head_2_z = {d.qpos[qa2]*1000:+.2f} mm")


def sim_step(m, d, v):
    t0 = time.perf_counter()
    mujoco.mj_step(m, d)
    v.sync()
    rem = TIMESTEP - (time.perf_counter() - t0)
    if rem > 0:
        time.sleep(rem)


def drive_to(m, d, v, targets: dict, steps: int, label: str):
    """Smooth-step drive actuators to target values (metres)."""
    aids   = {n: act_id(m, n) for n in targets}
    starts = {n: d.ctrl[aids[n]] for n in targets}
    for i in range(steps):
        t = smoothstep((i + 1) / steps)
        for n, tgt in targets.items():
            d.ctrl[aids[n]] = starts[n] + (tgt - starts[n]) * t
        sim_step(m, d, v)
    print(f"  ✓ {label}")


def hold(m, d, v, steps: int):
    for _ in range(steps):
        sim_step(m, d, v)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_head1_solo(m, d, v):
    print("\n── TEST 1: Head 1 solo stroke ──────────────────────────")
    read_head_pos(m, d, "start")

    drive_to(m, d, v, {"head_1_drive": INSERT_M}, MOVE_STEPS, "head 1 down")
    hold(m, d, v, HOLD_STEPS)
    read_head_pos(m, d, f"after down (target {INSERT_M*1000:.0f}mm)")

    drive_to(m, d, v, {"head_1_drive": RETRACT_M}, MOVE_STEPS, "head 1 up")
    hold(m, d, v, HOLD_STEPS)
    read_head_pos(m, d, "after retract (target 0mm)")


def test_head2_solo(m, d, v):
    print("\n── TEST 2: Head 2 solo stroke ──────────────────────────")
    read_head_pos(m, d, "start")

    drive_to(m, d, v, {"head_2_drive": INSERT_M}, MOVE_STEPS, "head 2 down")
    hold(m, d, v, HOLD_STEPS)
    read_head_pos(m, d, f"after down (target {INSERT_M*1000:.0f}mm)")

    drive_to(m, d, v, {"head_2_drive": RETRACT_M}, MOVE_STEPS, "head 2 up")
    hold(m, d, v, HOLD_STEPS)
    read_head_pos(m, d, "after retract (target 0mm)")


def test_both_simultaneous(m, d, v):
    print("\n── TEST 3: Both heads simultaneous ─────────────────────")
    read_head_pos(m, d, "start")

    drive_to(m, d, v,
             {"head_1_drive": INSERT_M, "head_2_drive": INSERT_M},
             MOVE_STEPS, "both down")
    hold(m, d, v, HOLD_STEPS)
    read_head_pos(m, d, f"after down (target {INSERT_M*1000:.0f}mm)")

    drive_to(m, d, v,
             {"head_1_drive": RETRACT_M, "head_2_drive": RETRACT_M},
             MOVE_STEPS, "both up")
    hold(m, d, v, HOLD_STEPS)
    read_head_pos(m, d, "after retract (target 0mm)")


def test_full_range(m, d, v):
    """Drive to joint limit (-150mm) to confirm no runaway."""
    print("\n── TEST 4: Full range to joint limit ───────────────────")
    drive_to(m, d, v, {"head_1_drive": -0.150}, MOVE_STEPS * 3, "head 1 → -150mm")
    hold(m, d, v, HOLD_STEPS)
    read_head_pos(m, d, "at joint limit")
    drive_to(m, d, v, {"head_1_drive": RETRACT_M}, MOVE_STEPS * 3, "head 1 → 0mm")
    hold(m, d, v, HOLD_STEPS)
    read_head_pos(m, d, "after retract")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading: {MODEL_PATH}")
    m = mujoco.MjModel.from_xml_path(MODEL_PATH)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)

    # Confirm actuator types loaded correctly
    for name in ("head_1_drive", "head_2_drive"):
        aid = act_id(m, name)
        print(f"  {name}: type={m.actuator_dyntype[aid]}  "
              f"gaintype={m.actuator_gaintype[aid]}  "
              f"ctrlrange=[{m.actuator_ctrlrange[aid,0]:.3f}, "
              f"{m.actuator_ctrlrange[aid,1]:.3f}]")

    with mujoco.viewer.launch_passive(m, d) as v:
        v.cam.lookat[:] = [0.0, 0.05, 0.90]
        v.cam.distance  = 0.6
        v.cam.elevation = -15
        v.cam.azimuth   = 160

        # Settle at home
        hold(m, d, v, SETTLE_STEPS)

        test_head1_solo(m, d, v)
        test_head2_solo(m, d, v)
        test_both_simultaneous(m, d, v)
        test_full_range(m, d, v)

        print("\n✓ All tests complete. Close window to exit.")
        while v.is_running():
            mujoco.mj_step(m, d)
            v.sync()


if __name__ == "__main__":
    main()
