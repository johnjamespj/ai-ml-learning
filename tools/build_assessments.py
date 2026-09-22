"""Materialize six runnable exam notebooks and starter modules without answers."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import nbformat as nbf
from assessment.specs import EXAMS


def build():
    for key,spec in EXAMS.items():
        target=ROOT/'student/exams'/key
        target.mkdir(parents=True,exist_ok=True)
        starter=target/'submission.py'
        if not starter.exists(): starter.write_text(spec['starter'])
        cells=[nbf.v4.new_markdown_cell(f'# Exam {key[:2]}: {spec["title"]}\n\nPrerequisites: {spec["prerequisites"]}\n\n100 points. Attempt before reading instructor material. Numerical checks are not a mastery certificate.'),
               nbf.v4.new_code_cell("from pathlib import Path\nimport sys, os\nROOT=Path(os.environ.get('COURSE_ROOT', next((str(p) for p in [Path.cwd(),*Path.cwd().parents] if (p/'assessment').is_dir()), '.')))\nsys.path.insert(0,str(ROOT))\nimport numpy as np\nRUN_CHECKS=False\nprint('NOT GRADED: implement the functions and set RUN_CHECKS=True.')")]
        for part,weight in [('derive',20),('code',30),('debug',15),('design',20),('defend',15)]:
            cells.append(nbf.v4.new_markdown_cell(f'## {part.title()} ({weight} points)\n\n{spec[part]}\n\nWrite your answer below.'))
            cells.append(nbf.v4.new_code_cell(spec['starter']) if part=='code' else nbf.v4.new_markdown_cell('**Your answer:**'))
        cells.extend([nbf.v4.new_markdown_cell('## Numerical self-check\n\nThe full assessment still requires human review of reasoning, design and defense.'),
                      nbf.v4.new_code_cell(f"if RUN_CHECKS:\n    from assessment.checks import check_submission\n    print(check_submission('{key}', globals()))\nelse:\n    print('NOT GRADED: checks have not run.')")])
        nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'assessment':{'id':key,'status':'not_graded','points':100}})
        nbf.write(nb,target/'exam.ipynb')
    print(f'Built {len(EXAMS)} exam notebooks; starter files were not overwritten.')
if __name__=='__main__': build()
