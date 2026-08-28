#!/usr/bin/env python
"""Pick the better 150M base from the one step both arms trained to under an identical recipe.

Prints the winning run name on stdout (the queue reads it) and the numbers on stderr (the log
keeps them). Missing data falls back to the ProtSent-V2 base: that is the parsimonious prior --
every 35M comparison in this campaign has favoured it -- so an absent measurement must not be
what promotes the vanilla arm.
"""
import csv
import sys
from pathlib import Path

VANILLA, PROTSENTV2 = "late-r2-esm2-150m", "late-r2-protsentv2-150m"
MARK, LEVEL = "@4000", "superfamily"

curve = Path(sys.argv[1])
scores = {}
if curve.exists():
    for r in csv.DictReader(curve.open()):
        if r["level"] == LEVEL and r["model"].endswith(MARK):
            scores[r["model"].split("@")[0]] = float(r["eligible_MAP"])

v, p = scores.get(VANILLA), scores.get(PROTSENTV2)
print(f"gate: {LEVEL} eligible_MAP {MARK} -- vanilla={v} protsentv2={p}", file=sys.stderr)
if v is None or p is None:
    print(f"gate: incomplete data, falling back to the parsimonious prior ({PROTSENTV2})",
          file=sys.stderr)
    print(PROTSENTV2)
else:
    print(f"gate: margin {v - p:+.4f} in favour of {'vanilla' if v > p else 'protsentv2'}",
          file=sys.stderr)
    print(VANILLA if v > p else PROTSENTV2)
