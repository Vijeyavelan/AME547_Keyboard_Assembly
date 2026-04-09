"""
test_feeder.py — Step 1: Tray feeder validation

Loads station3.xml and continuously advances the 16 real switch bodies
(tray_sw_1 through tray_sw_16) along the tray X axis toward the pick
position at x=0.  When the front switch reaches x=0 it is "consumed"
(teleported back to the reload position at the far end) and all remaining
switches advance one slot.  This runs forever so you can visually confirm:
  - Switches appear correctly in the tray (mesh geometry)
  - Advance animation is smooth
  - Reload cycle looks right

Coordinate recap:
  tray_feeder body is at world pos (0, 0.330, 0).
  Slot positions are LOCAL to tray_feeder:
    slot 1 (pick end) : x =  0.0000  → world x = 0.000
    slot 2            : x = -0.0191
    ...
    slot 16 (far end) : x = -0.2858
  Y local = -0.030  → world y = 0.300
  Z local =  0.834  (tray top face, switch bottom sits here)

  Freejoint qpos layout (7 values each):
    [0:3] = world position (x, y, z)
    [3:7] = quaternion (w, x, y, z)

Run from repo root:
    mjpython controllers/test_feeder.py
"""

import mujoco
import mujoco.viewer
import numpy as np
import time

MODEL_PATH = "models/stations/station3.xml"

# ── Tray geometry (world coordinates) ────────────────────────────────────────
# tray_feeder body offset: world y = 0.330, x = 0, z = 0
TRAY_WORLD_Y   = 0.330 + (-0.030)   # = 0.300  (local y=-0.030 + body y=0.330)
TRAY_WORLD_Z   = 0.8261             # tray top face = plate top face (levelled)

# 16 slot X positions in LOCAL tray_feeder frame (= world X since body x=0)
SLOT_X = [
     0.0000, -0.0191, -0.0381, -0.0572, -0.0762, -0.0953,
    -0.1143, -0.1333, -0.1524, -0.1715, -0.1905, -0.2096,
    -0.2286, -0.2477, -0.2667, -0.2858,
]
N_SLOTS   = len(SLOT_X)
PITCH     = abs(SLOT_X[1] - SLOT_X[0])   # ≈ 0.01905m

# Pick position is slot 0 (x=0). Reload position is one pitch beyond slot 15.
RELOAD_X  = SLOT_X[-1] - PITCH    # park newly reloaded switch here, then advance

# ── Advance animation ─────────────────────────────────────────────────────────
ADVANCE_STEPS = 350    # steps to slide one pitch — matches column 180° rotation time
DWELL_STEPS   = 100    # steps to pause at pick position before consuming


def get_jnt_qadr(m, name):
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise ValueError(f"Joint not found: '{name}'")
    return m.jnt_qposadr[jid], m.jnt_dofadr[jid]


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def set_switch_pos(d, qa, da, x, y, z):
    """Teleport a switch body to (x, y, z), identity quaternion, zero velocity."""
    d.qpos[qa + 0] = x
    d.qpos[qa + 1] = y
    d.qpos[qa + 2] = z
    d.qpos[qa + 3] = 1.0   # w
    d.qpos[qa + 4] = 0.0   # x
    d.qpos[qa + 5] = 0.0   # y
    d.qpos[qa + 6] = 0.0   # z
    d.qvel[da:da + 6] = 0.0


def main():
    print(f"Loading: {MODEL_PATH}")
    m = mujoco.MjModel.from_xml_path(MODEL_PATH)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)

    # Resolve all 16 tray switch freejoints
    print("Resolving tray switch joints...")
    slots = []   # list of (qa, da) in slot order 1..16
    for n in range(1, N_SLOTS + 1):
        qa, da = get_jnt_qadr(m, f"tray_swj_{n}")
        slots.append((qa, da))
        print(f"  tray_swj_{n}: qadr={qa}")

    # Place all switches at their initial tray positions
    print("\nPlacing switches in tray...")
    for i, (qa, da) in enumerate(slots):
        set_switch_pos(d, qa, da, SLOT_X[i], TRAY_WORLD_Y, TRAY_WORLD_Z)
    mujoco.mj_forward(m, d)

    # active_slots: list of (slot_index_in_SLOT_X, switch_index_in_slots[])
    # We track which physical switch body is in which slot position.
    # Start: switch body 0 is at SLOT_X[0], body 1 at SLOT_X[1], etc.
    # slot_assignment[i] = which slot position (0..N-1) body i is currently at
    # current_x[i] = current world X of body i
    current_x = [SLOT_X[i] for i in range(N_SLOTS)]

    with mujoco.viewer.launch_passive(m, d) as v:
        v.cam.lookat[:] = [-0.14, 0.30, 0.86]
        v.cam.distance  = 0.6
        v.cam.elevation = -20
        v.cam.azimuth   = 90

        cycle = 0

        def lock_all():
            """Pin every switch's Y, Z, and quaternion every step — prevents gravity drift."""
            for qa, da in slots:
                d.qpos[qa + 1] = TRAY_WORLD_Y
                d.qpos[qa + 2] = TRAY_WORLD_Z
                d.qpos[qa + 3] = 1.0
                d.qpos[qa + 4] = 0.0
                d.qpos[qa + 5] = 0.0
                d.qpos[qa + 6] = 0.0
                d.qvel[da:da + 6] = 0.0

        def sim_step():
            """One physics step with real-time pacing and gravity lock."""
            lock_all()
            t0 = time.perf_counter()
            mujoco.mj_step(m, d)
            v.sync()
            rem = 0.002 - (time.perf_counter() - t0)
            if rem > 0:
                time.sleep(rem)

        while v.is_running():
            cycle += 1
            print(f"\n[Cycle {cycle}] Advancing...")

            start_x  = current_x[:]
            target_x = [x + PITCH for x in start_x]

            # Smooth-step advance
            for step in range(ADVANCE_STEPS):
                if not v.is_running():
                    break
                t = smoothstep((step + 1) / ADVANCE_STEPS)
                for i, (qa, _) in enumerate(slots):
                    d.qpos[qa] = start_x[i] + (target_x[i] - start_x[i]) * t
                sim_step()

            current_x = target_x[:]

            # Dwell at pick position
            for _ in range(DWELL_STEPS):
                if not v.is_running():
                    break
                sim_step()

            # Consume: teleport front switch to last slot X (no gap), rotate queue
            qa0, da0 = slots[0]
            d.qpos[qa0]     = SLOT_X[-1]
            d.qpos[qa0 + 1] = TRAY_WORLD_Y
            d.qpos[qa0 + 2] = TRAY_WORLD_Z
            d.qpos[qa0 + 3] = 1.0
            d.qpos[qa0 + 4] = 0.0
            d.qpos[qa0 + 5] = 0.0
            d.qpos[qa0 + 6] = 0.0
            d.qvel[da0:da0 + 6] = 0.0
            current_x[0] = SLOT_X[-1]

            slots.append(slots.pop(0))
            current_x.append(current_x.pop(0))

            mujoco.mj_forward(m, d)
            v.sync()
            print(f"  [Cycle {cycle} done]")


if __name__ == "__main__":
    main()