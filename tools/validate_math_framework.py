#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def markdown_text(nb):
    return "\n".join("".join(c.get("source",[])) for c in nb.get("cells",[]) if c.get("cell_type")=="markdown")

def check_group(paths, expected, require_math_section):
    problems=[]
    paths=sorted(paths)
    if len(paths)!=expected:
        problems.append(f"expected {expected}, found {len(paths)}")
    for p in paths:
        try: nb=json.loads(p.read_text())
        except Exception as e:
            problems.append(f"{p}: invalid JSON: {e}"); continue
        text=markdown_text(nb)
        if require_math_section and "## Mathematical Framework" not in text:
            problems.append(f"{p}: missing Mathematical Framework section")
        if not any(c.get("cell_type")=="code" for c in nb.get("cells",[])):
            problems.append(f"{p}: no code cells")
    return problems

def main():
    lesson=list((ROOT/"lessons").glob("*/lesson.ipynb"))
    papers=list((ROOT/"papers"/"notebooks").glob("*.ipynb"))
    maths=list((ROOT/"math").glob("*.ipynb"))
    problems=[]
    problems += ["lessons: "+x for x in check_group(lesson,78,True)]
    problems += ["papers: "+x for x in check_group(papers,21,True)]
    problems += ["math: "+x for x in check_group(maths,17,False)]
    print(f"lesson notebooks: {len(lesson)}")
    print(f"paper notebooks: {len(papers)}")
    print(f"math notebooks: {len(maths)}")
    if problems:
        print("\nProblems:")
        for p in problems: print(" -",p)
        return 1
    print("Mathematical framework is structurally wired into the full course.")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
