#!/usr/bin/env python3
"""Validate that every lesson is a usable Jupyter notebook."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LESSONS = ROOT / "lessons"

def main() -> int:
    dirs = sorted(p for p in LESSONS.iterdir() if p.is_dir())
    problems = []
    notebooks = []
    for d in dirs:
        nb_path = d / "lesson.ipynb"
        if not nb_path.exists():
            problems.append(f"{d.name}: missing lesson.ipynb")
            continue
        notebooks.append(nb_path)
        try:
            nb = json.loads(nb_path.read_text())
        except Exception as e:
            problems.append(f"{d.name}: invalid JSON: {e}")
            continue
        if nb.get("nbformat") != 4:
            problems.append(f"{d.name}: expected nbformat 4")
        cells = nb.get("cells", [])
        code = [c for c in cells if c.get("cell_type") == "code"]
        markdown = "\n".join("".join(c.get("source", [])) for c in cells if c.get("cell_type") == "markdown")
        if not code:
            problems.append(f"{d.name}: no executable code cells")
        if "Runnable activity" not in markdown:
            problems.append(f"{d.name}: missing runnable activity section")
    print(f"lesson directories: {len(dirs)}")
    print(f"valid notebook files found: {len(notebooks)}")
    if problems:
        print("\nProblems:")
        for p in problems:
            print(" -", p)
        return 1
    if len(notebooks) != 78:
        print(f"Expected 78 notebooks, found {len(notebooks)}")
        return 1
    print("All 78 lesson notebooks passed structural validation.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
