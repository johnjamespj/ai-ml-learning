#!/usr/bin/env python3
"""Validate the landmark-paper notebook track."""
from __future__ import annotations
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PAPER_DIR=ROOT/"papers"/"notebooks"

def main():
    nbs=sorted(PAPER_DIR.glob("*.ipynb"))
    problems=[]
    for p in nbs:
        try:
            nb=json.loads(p.read_text())
        except Exception as e:
            problems.append(f"{p.name}: invalid JSON: {e}")
            continue
        cells=nb.get("cells",[])
        code=[c for c in cells if c.get("cell_type")=="code"]
        text="\n".join("".join(c.get("source",[])) for c in cells if c.get("cell_type")=="markdown")
        if not code: problems.append(f"{p.name}: no code cells")
        for required in ["Scale gap","Before you read","Central claim","Ablation table","Defend the paper"]:
            if required not in text: problems.append(f"{p.name}: missing section {required!r}")
    print("paper notebooks:",len(nbs))
    if len(nbs)!=21: problems.append(f"expected 21 notebooks, found {len(nbs)}")
    if problems:
        print("\nProblems:")
        for x in problems: print(" -",x)
        return 1
    print("All 21 paper notebooks passed structural validation.")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
