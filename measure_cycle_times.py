"""
measure_cycle_times.py
======================
Headless (no viewer) cycle time measurement for all stations.
Imports each station's assembly sequence function, wraps it with
CycleTimer, and prints the full cell analysis at the end.

Run with:
    mjpython measure_cycle_times.py

Each station controller needs minor edits — see INTEGRATION GUIDE below.
Results are saved to logs/cycle_times.json.

──────────────────────────────────────────────────────────────────────────
INTEGRATION GUIDE — Where to add timer.mark() in each controller
──────────────────────────────────────────────────────────────────────────

STATION 1 (S1 — UR5e structural assembly, 5 parts):
────────────────────────────────────────────────────
Add these marks in your part loop (after each phase completes):

    timer.mark(data, "Belt advance — EPDM foam")
    # ... pick move steps ...
    timer.mark(data, "Pick — EPDM foam")
    # ... carry + place move steps ...
    timer.mark(data, "Place — EPDM foam")

    # PCB has extra phases:
    timer.mark(data, "Pick — PCB")
    timer.mark(data, "PCB tilt sequence")
    timer.mark(data, "JST dwell")
    timer.mark(data, "PCB place (tilt-back + descend)")

Recommended ref_key for print_report(): "pick_place_fast" / "pick_place_slow"


STATION 3 (S3 — switch insertion, 84 switches):
─────────────────────────────────────────────────
Wrap the per-column or per-switch loop:

    # Outside the column loop — mark tray advance
    timer.mark(data, f"Tray advance — col {col}")

    # Inside the switch loop — mark each insertion
    timer.mark(data, f"Switch insert {i+1:03d}/{N_SWITCHES}")

    # Optionally collapse to per-column for cleaner output:
    timer.mark(data, f"Column {col} — {switches_in_col} switches")

Recommended ref_key for print_report(): "switch_fast" / "switch_slow"


STATION 4 (S4 — keycap installation, 84 keycaps):
───────────────────────────────────────────────────
Same pattern as S3:

    timer.mark(data, f"Tray advance — keycap col {col}")
    timer.mark(data, f"Keycap install {i+1:03d}/{N_KEYCAPS}")

    # Or per-row for the width-aware rows:
    timer.mark(data, f"Row {row} — {n} keycaps (width={w}u)")

Recommended ref_key for print_report(): "keycap_fast" / "keycap_slow"

──────────────────────────────────────────────────────────────────────────
PATTERN — Minimal edit to an existing controller
──────────────────────────────────────────────────────────────────────────

Before the controller's main loop, add:

    from cycle_timer import CycleTimer
    timer = CycleTimer("Station 1")   # or 3, 4
    timer.start(data)

After each phase's step loop, add:

    timer.mark(data, "Pick — EPDM foam")

At the very end (before viewer closes or after last step):

    timer.finish(data)
    timer.print_report(ref_key="pick_place_fast")

──────────────────────────────────────────────────────────────────────────
"""

import mujoco
import numpy as np
from cycle_timer import CycleTimer, AssemblyCellTimer

# ── You will replace these imports with your actual controller modules ──
# from controllers.station1 import run_station1
# from controllers.station3 import run_station3
# from controllers.station4 import run_station4

cell = AssemblyCellTimer()


# ─────────────────────────────────────────────────────────────
# HEADLESS RUNNER TEMPLATE
# Copy this pattern for each station. Replace the XML path and
# the step loop body with your actual controller logic.
# ─────────────────────────────────────────────────────────────

def run_station_headless(xml_path: str, station_name: str,
                         controller_fn, ref_key: str) -> CycleTimer:
    """
    Load a station model headlessly, run the controller function,
    return the populated CycleTimer.

    controller_fn signature:
        def controller_fn(m, d, timer: CycleTimer) -> None
    """
    print(f"\nLoading {station_name} model: {xml_path}")
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)

    timer = CycleTimer(station_name)
    timer.start(d)

    controller_fn(m, d, timer)

    timer.finish(d)
    timer.print_report(ref_key=ref_key)
    return timer


# ─────────────────────────────────────────────────────────────
# EXAMPLE: Stub controllers (replace with your real ones)
# These measure how long your EXISTING step loops actually take
# in sim time — no logic changes needed, just timer.mark() calls
# ─────────────────────────────────────────────────────────────

def stub_s1(m, d, timer: CycleTimer):
    """
    Replace this body with your actual S1 assembly sequence.
    Keep timer.mark() calls at the phase boundaries.
    """
    PARTS = [
        ("EPDM Foam",      "pick_place"),
        ("Battery",        "pick_place"),
        ("PCB",            "pick_place_pcb"),
        ("Sound Foam",     "pick_place"),
        ("Aluminum Plate", "pick_place"),
    ]
    for part_name, _ in PARTS:
        # Simulate belt advance (replace with your BeltSlide.advance() call)
        for _ in range(350):          # ADVANCE_STEPS
            mujoco.mj_step(m, d)
        timer.mark(d, f"Belt advance — {part_name}")

        # Simulate pick (replace with your IK pick loop)
        for _ in range(300):
            mujoco.mj_step(m, d)
        timer.mark(d, f"Pick — {part_name}")

        # Simulate place (replace with your carry + place loop)
        for _ in range(400):
            mujoco.mj_step(m, d)
        timer.mark(d, f"Place — {part_name}")

        if part_name == "PCB":
            # Extra: tilt sequence + JST dwell
            for _ in range(60):       # tilt slerp
                mujoco.mj_step(m, d)
            timer.mark(d, "PCB tilt sequence")
            for _ in range(2500):     # 5s JST dwell @ 0.002s timestep
                mujoco.mj_step(m, d)
            timer.mark(d, "JST dwell (5s)")


def stub_s3(m, d, timer: CycleTimer):
    """Replace with your actual S3 switch insertion loop."""
    N_SWITCHES = 84
    N_COLS = 15   # approximate column count for a 75% layout

    for col in range(N_COLS):
        for _ in range(350):          # tray advance between columns
            mujoco.mj_step(m, d)
        timer.mark(d, f"Tray advance — col {col+1}")

        switches_this_col = N_SWITCHES // N_COLS + (1 if col < N_SWITCHES % N_COLS else 0)
        for sw in range(switches_this_col):
            for _ in range(200):      # descend + insert + retract
                mujoco.mj_step(m, d)
            timer.mark(d, f"Switch insert col{col+1} sw{sw+1}")


def stub_s4(m, d, timer: CycleTimer):
    """Replace with your actual S4 keycap installation loop."""
    N_KEYCAPS = 84
    N_COLS = 15

    for col in range(N_COLS):
        for _ in range(350):
            mujoco.mj_step(m, d)
        timer.mark(d, f"Tray advance — col {col+1}")

        keycaps_this_col = N_KEYCAPS // N_COLS + (1 if col < N_KEYCAPS % N_COLS else 0)
        for kc in range(keycaps_this_col):
            for _ in range(150):      # press-fit is faster than switch insert
                mujoco.mj_step(m, d)
            timer.mark(d, f"Keycap install col{col+1} kc{kc+1}")


# ─────────────────────────────────────────────────────────────
# MAIN — run all stations, then print cell analysis
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Replace paths with your actual station XML paths ──
    configs = [
        ("models/station1/station1.xml", "Station 1", stub_s1, "pick_place_fast"),
        ("models/station3/station3.xml", "Station 3", stub_s3, "switch_fast"),
        ("models/station4/station4.xml", "Station 4", stub_s4, "keycap_fast"),
    ]

    for xml_path, name, fn, ref in configs:
        try:
            t = run_station_headless(xml_path, name, fn, ref)
            cell.add(t)
        except Exception as e:
            print(f"\n  ⚠ Could not run {name}: {e}")
            print(f"    (Add your real controller function and XML path above)\n")

    if cell.stations:
        cell.print_cell_report()
        cell.save_json("logs/cycle_times.json")
    else:
        print("\n  No stations completed — add your controller functions to measure_cycle_times.py")
