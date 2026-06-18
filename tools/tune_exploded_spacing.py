#!/usr/bin/env python3
"""Tune Z spacing for keyboard_exploded_view_v3.xml.
Edit only BASE_Z and STEP below, then run:
  python tools/tune_exploded_spacing.py keyboard_exploded_view_v3.xml
"""
from pathlib import Path
import re
import sys

BASE_Z = 0.050   # bottom-case layer height
STEP   = 0.035   # one-line spacing control between layers

LAYERS = [
    "layer_01_bottom_case",
    "layer_02_epdm_foam",
    "layer_03_battery",
    "layer_04_pcb",
    "layer_05_aluminum_plate",
    "layer_06_stabilizers",
    "layer_07_screws",
    "layer_08_switches",
    "layer_09_keycaps",
]

def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "tools/keyboard_exploded_view.xml")
    text = path.read_text()
    for i, layer in enumerate(LAYERS):
        z = BASE_Z + i * STEP
        pattern = rf'(<body name="{re.escape(layer)}" pos=")0 0 [^"]+("?>)'
        text, n = re.subn(pattern, rf'\g<1>0 0 {z:.3f}\2', text)
        if n != 1:
            raise RuntimeError(f"Could not update {layer}; matches found: {n}")
    path.write_text(text)
    print(f"Updated {path} with BASE_Z={BASE_Z:.3f}, STEP={STEP:.3f}")

if __name__ == "__main__":
    main()
