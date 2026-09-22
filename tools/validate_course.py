"""Structural, code-cell syntax, inventory and local-link checks; not execution."""
from __future__ import annotations
import argparse,json,re
from pathlib import Path
from urllib.parse import unquote
import nbformat
from IPython.core.inputtransformer2 import TransformerManager
ROOT=Path(__file__).resolve().parents[1]


def main():
    errors=[];inventory={}
    groups={'lessons':(list((ROOT/'lessons').glob('*/lesson.ipynb')),78),
            'math':(list((ROOT/'math').glob('*.ipynb')),17),
            'papers':(list((ROOT/'papers/notebooks').glob('*.ipynb')),21),
            'exams':(list((ROOT/'student/exams').glob('*/exam.ipynb')),6),
            'problem_sets':(list((ROOT/'math/problem_sets').glob('*.ipynb')),8),
            'capstones':(list((ROOT/'capstones/rf_detection').glob('*.ipynb')),1)}
    transformer=TransformerManager()
    def links(path,text):
        for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)',text):
            target=target.split()[0].split('#')[0]
            if not target or re.match(r'^[a-zA-Z]+:',target): continue
            resolved=(path.parent/unquote(target)).resolve()
            if not resolved.exists(): errors.append(f'{path.relative_to(ROOT)}: broken link {target}')
    for group,(paths,expected) in groups.items():
        inventory[group]=len(paths)
        if len(paths)!=expected: errors.append(f'{group}: expected {expected}, found {len(paths)}')
        for path in paths:
            try:
                nb=nbformat.read(path,as_version=4);nbformat.validate(nb)
                ids=[c.get('id') for c in nb.cells]
                if None in ids or len(set(ids))!=len(ids): errors.append(f'{path}: missing/duplicate cell IDs')
                if group in ('lessons','papers','math') and not nb.metadata.get('course',{}).get('hardening_version'):
                    errors.append(f'{path}: missing explicit runtime setup')
                for i,cell in enumerate(nb.cells):
                    if cell.cell_type=='code':
                        compile(transformer.transform_cell(cell.source),f'{path}:cell{i}','exec')
                    elif cell.cell_type=='markdown': links(path,cell.source)
            except Exception as exc: errors.append(f'{path}: {type(exc).__name__}: {exc}')
    for path in ROOT.rglob('*.md'):
        if any(part in ('work','artifacts','results','.git','.venv') for part in path.parts):continue
        links(path,path.read_text())
    tracks=json.loads((ROOT/'course_tracks.json').read_text())
    selections=tracks['core']+tracks['radar']+tracks['advanced']
    if sorted(selections)!=list(range(78)) or len(set(selections))!=78: errors.append('Track assignment must cover each lesson exactly once')
    report={'scope':'structure/syntax/links only','inventory':inventory,'errors':errors}
    out=ROOT/'artifacts/structure.json';out.parent.mkdir(exist_ok=True);out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    return bool(errors)
if __name__=='__main__': raise SystemExit(main())
