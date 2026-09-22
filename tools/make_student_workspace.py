"""Copy starter work without importing or copying reference solutions."""
from pathlib import Path
import argparse,re,shutil
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,default=ROOT/'work')
    a=p.parse_args()
    if a.output.exists(): p.error('Destination exists; choose a new directory to preserve your work.')
    shutil.copytree(ROOT/'student/exams',a.output/'exams')
    count=0
    for lab in sorted((ROOT/'labs').glob('lab*')):
        if not (lab/'starter.py').is_file(): continue
        target=a.output/'labs'/lab.name; target.mkdir(parents=True)
        shutil.copy(lab/'starter.py',target/'solution.py')
        if (lab/'README.md').is_file(): shutil.copy(lab/'README.md',target/'README.md')
        test=(lab/'test_lab.py').read_text()
        loader=("from pathlib import Path\n"
                "import importlib.util\n"
                "_spec=importlib.util.spec_from_file_location('student_lab',Path(__file__).with_name('solution.py'))\n"
                "m=importlib.util.module_from_spec(_spec)\n_spec.loader.exec_module(m)")
        test,n=re.subn(r'^m\s*=\s*importlib\.import_module\([^\n]+\)',lambda _:loader,test,flags=re.M)
        if n!=1: raise ValueError(f'Unexpected test loader in {lab}')
        (target/'test_lab.py').write_text(test); count+=1
    (a.output/'README.md').write_text('# Student workspace\n\nThese are intentionally unfinished starters, not reference answers.\n\nRun `python -m pytest work/labs/lab01_linear_regression/test_lab.py -q`.\nUse the exam notebooks or grade_submission.py. A reference-suite pass does not complete your work.\n')
    print(f'Created {count} independent lab starters and 6 exams at {a.output}')
if __name__=='__main__': main()
