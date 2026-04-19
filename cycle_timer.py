"""
cycle_timer.py — Simulation cycle time instrumentation for AME547 keyboard assembly.

Tracks MuJoCo sim time (data.time) per operation and per station, then produces
a formatted report with real-world believability checks.

Usage:
    from cycle_timer import CycleTimer

    timer = CycleTimer("Station 1")
    timer.start(data)

    # ... your pick move ...
    timer.mark(data, "Pick EPDM foam")

    # ... your place move ...
    timer.mark(data, "Place EPDM foam")

    timer.finish(data)
    timer.print_report()
"""

import time
from dataclasses import dataclass, field
from typing import Optional
import json


# ---------------------------------------------------------------------------
# Real-world reference times (seconds) for believability checks
# These are conservative UR5e / pneumatic gantry benchmarks from literature
# ---------------------------------------------------------------------------
REAL_WORLD_REFS = {
    # UR5e pick-and-place (small part, ~300mm travel): 3–6s
    "pick_place_fast":  3.0,
    "pick_place_slow":  6.0,
    # Screw driving (M2/M3, pneumatic driver): 2–4s per screw
    "screw_fast":       2.0,
    "screw_slow":       4.0,
    # Switch hot-swap insertion (compliant tool, 5mm stroke): 0.4–0.8s
    "switch_fast":      0.4,
    "switch_slow":      0.8,
    # Keycap press-fit installation: 0.3–0.6s
    "keycap_fast":      0.3,
    "keycap_slow":      0.6,
    # Stabilizer press (6 pins, sequential): 1.5–3.0s total
    "stabilizer_fast":  1.5,
    "stabilizer_slow":  3.0,
    # Belt/tray advance (part staging): 1–2s
    "belt_fast":        1.0,
    "belt_slow":        2.0,
}


@dataclass
class Operation:
    name: str
    sim_start: float    # data.time at start
    sim_end: float      # data.time at end
    wall_start: float   # time.perf_counter() at start
    wall_end: float     # time.perf_counter() at end

    @property
    def sim_duration(self) -> float:
        return self.sim_end - self.sim_start

    @property
    def wall_duration(self) -> float:
        return self.wall_end - self.wall_start


@dataclass
class CycleTimer:
    station_name: str
    ops: list = field(default_factory=list)

    _sim_start: float = field(default=0.0, init=False)
    _wall_start: float = field(default=0.0, init=False)
    _last_sim: float = field(default=0.0, init=False)
    _last_wall: float = field(default=0.0, init=False)
    _running: bool = field(default=False, init=False)
    _sim_total: float = field(default=0.0, init=False)
    _wall_total: float = field(default=0.0, init=False)

    def start(self, data) -> None:
        """Call once at the very beginning of the station sequence."""
        self._sim_start  = data.time
        self._wall_start = time.perf_counter()
        self._last_sim   = data.time
        self._last_wall  = time.perf_counter()
        self._running    = True
        print(f"\n{'='*60}")
        print(f"  {self.station_name} — timing started")
        print(f"  MuJoCo sim time at start: {data.time:.4f}s")
        print(f"{'='*60}")

    def mark(self, data, op_name: str) -> Operation:
        """
        Call after each discrete operation completes.
        Example: after gripper lifts off, after placement, after belt advance.
        """
        assert self._running, "Call start() before mark()"
        now_sim  = data.time
        now_wall = time.perf_counter()

        op = Operation(
            name=op_name,
            sim_start=self._last_sim,
            sim_end=now_sim,
            wall_start=self._last_wall,
            wall_end=now_wall,
        )
        self.ops.append(op)

        self._last_sim  = now_sim
        self._last_wall = now_wall

        print(f"  ✓ {op_name:<40} "
              f"sim={op.sim_duration:6.2f}s  "
              f"wall={op.wall_duration:5.2f}s")
        return op

    def finish(self, data) -> None:
        """Call at the very end of the station sequence."""
        assert self._running, "Call start() before finish()"
        self._sim_total  = data.time - self._sim_start
        self._wall_total = time.perf_counter() - self._wall_start
        self._running = False
        print(f"{'='*60}")
        print(f"  {self.station_name} — timing complete")

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_report(self, ref_key: Optional[str] = None) -> None:
        """
        Print a full breakdown table plus believability check.

        ref_key: key from REAL_WORLD_REFS to check per-op sim time against,
                 e.g. "pick_place_fast" / "pick_place_slow".
                 If None, no per-op check is performed.
        """
        print(f"\n{'─'*65}")
        print(f"  CYCLE TIME REPORT — {self.station_name}")
        print(f"{'─'*65}")
        print(f"  {'Operation':<42} {'Sim (s)':>8} {'Wall (s)':>9}")
        print(f"  {'─'*42} {'─'*8} {'─'*9}")

        for op in self.ops:
            flag = ""
            if ref_key:
                lo = REAL_WORLD_REFS.get(f"{ref_key}_fast", 0)
                hi = REAL_WORLD_REFS.get(f"{ref_key}_slow", 999)
                if op.sim_duration < lo:
                    flag = " ⚠ TOO FAST"
                elif op.sim_duration > hi * 3:
                    flag = " ⚠ TOO SLOW"
            print(f"  {op.name:<42} {op.sim_duration:>8.3f} {op.wall_duration:>9.3f}{flag}")

        print(f"  {'─'*42} {'─'*8} {'─'*9}")
        print(f"  {'STATION TOTAL':<42} {self._sim_total:>8.3f} {self._wall_total:>9.3f}")
        print(f"{'─'*65}")
        print(f"  Sim speed ratio: {self._wall_total / self._sim_total:.2f}x "
              f"({'faster' if self._wall_total < self._sim_total else 'slower'} than real-time)")
        print()

    def to_dict(self) -> dict:
        """Serialisable summary for cross-station aggregation."""
        return {
            "station": self.station_name,
            "sim_total": round(self._sim_total, 4),
            "wall_total": round(self._wall_total, 4),
            "ops": [
                {
                    "name": op.name,
                    "sim": round(op.sim_duration, 4),
                    "wall": round(op.wall_duration, 4),
                }
                for op in self.ops
            ],
        }


# ---------------------------------------------------------------------------
# Multi-station aggregator
# ---------------------------------------------------------------------------

class AssemblyCellTimer:
    """
    Collects CycleTimer results from all stations and prints the
    full cell analysis: bottleneck identification + S2 budget.
    """

    def __init__(self):
        self.stations: list[CycleTimer] = []

    def add(self, timer: CycleTimer) -> None:
        self.stations.append(timer)

    def print_cell_report(self) -> None:
        totals = {t.station_name: t._sim_total for t in self.stations}
        max_time = max(totals.values())
        bottleneck = max(totals, key=totals.get)

        print(f"\n{'═'*65}")
        print(f"  ASSEMBLY CELL — CYCLE TIME ANALYSIS")
        print(f"{'═'*65}")
        print(f"  {'Station':<20} {'Sim Total (s)':>14} {'% of max':>10} {'Status':>12}")
        print(f"  {'─'*20} {'─'*14} {'─'*10} {'─'*12}")

        for name, t in totals.items():
            pct = (t / max_time) * 100
            status = "◀ BOTTLENECK" if name == bottleneck else "OK"
            print(f"  {name:<20} {t:>14.2f} {pct:>9.1f}% {status:>12}")

        print(f"{'─'*65}")

        # S2 budget
        s1 = totals.get("Station 1", 0)
        s3 = totals.get("Station 3", 0)
        s2_budget = max(s1, s3)
        print(f"\n  S2 DESIGN BUDGET")
        print(f"  ├─ S1 cycle time : {s1:.2f}s")
        print(f"  ├─ S3 cycle time : {s3:.2f}s")
        print(f"  ├─ S2 must finish in ≤ {s2_budget:.2f}s to avoid being bottleneck")
        print()

        # S2 feasibility breakdown
        n_screws = 4
        n_stab_pins = 6
        screw_budget_each = (s2_budget * 0.6) / n_screws   # 60% of budget for screws
        stab_budget_each  = (s2_budget * 0.3) / n_stab_pins  # 30% for stabilizers
        # 10% for gantry travel overhead

        print(f"  S2 OPERATION BUDGET (to stay within {s2_budget:.1f}s)")
        print(f"  ├─ Gantry travel overhead (10%)  : {s2_budget * 0.10:.2f}s")
        print(f"  ├─ Screw driving × {n_screws}  (60%)     : {s2_budget * 0.60:.2f}s total  "
              f"→ {screw_budget_each:.2f}s/screw")
        print(f"  └─ Stabilizer press × {n_stab_pins} (30%)   : {s2_budget * 0.30:.2f}s total  "
              f"→ {stab_budget_each:.2f}s/pin")

        # Believability check against real-world refs
        print(f"\n  REAL-WORLD BELIEVABILITY CHECK")
        lo_screw, hi_screw = REAL_WORLD_REFS["screw_fast"], REAL_WORLD_REFS["screw_slow"]
        lo_stab,  hi_stab  = REAL_WORLD_REFS["stabilizer_fast"] / n_stab_pins, \
                              REAL_WORLD_REFS["stabilizer_slow"] / n_stab_pins

        def check(val, lo, hi, label):
            if val < lo:
                verdict = f"⚠ UNREALISTICALLY FAST (min {lo:.2f}s)"
            elif val > hi:
                verdict = f"⚠ SLOW but feasible (benchmark: {lo:.2f}–{hi:.2f}s)"
            else:
                verdict = f"✓ Realistic ({lo:.2f}–{hi:.2f}s benchmark)"
            print(f"  {label:<35} {verdict}")

        check(screw_budget_each, lo_screw, hi_screw, f"Time/screw ({screw_budget_each:.2f}s)")
        check(stab_budget_each,  lo_stab,  hi_stab,  f"Time/stab pin ({stab_budget_each:.2f}s)")

        print(f"{'═'*65}\n")

    def save_json(self, path: str = "logs/cycle_times.json") -> None:
        import os, json
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "stations": [t.to_dict() for t in self.stations],
            "cell_total_sim": sum(t._sim_total for t in self.stations),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Saved cycle time data → {path}")
