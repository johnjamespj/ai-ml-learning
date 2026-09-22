"""Run coding checks; only an explicit human rubric can establish course mastery."""
from pathlib import Path
import argparse, importlib.util, json, sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from assessment.checks import check_submission

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exam',required=True); p.add_argument('--submission',type=Path,required=True)
    p.add_argument('--review',type=Path); p.add_argument('--output',type=Path,default=Path('grading.json'))
    a=p.parse_args()
    # This executes a local learner module, just as pytest does. Do not grade untrusted submissions on a privileged host.
    spec=importlib.util.spec_from_file_location('learner_submission',a.submission)
    module=importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        report=check_submission(a.exam,vars(module))
        if a.review:
            review=json.loads(a.review.read_text())
            weights={'derive':20,'code':30,'debug':15,'design':20,'defend':15}
            points=review['points']
            if set(points)!=set(weights): raise ValueError('Rubric needs all five components')
            if not review.get('reviewer') or not review.get('date'): raise ValueError('Reviewer and date are required')
            if any(not 0<=points[k]<=v for k,v in weights.items()): raise ValueError('Invalid rubric points')
            passed=sum(points.values())>=80 and all(points[k]>=.5*v for k,v in weights.items())
            passed=passed and review.get('no_critical_errors') is True
            report.update(human_review=review,mastery='passed' if passed else 'revision_required')
    except Exception as exc:
        report={'exam':a.exam,'coding_checks':'failed','mastery':'not_assessed','error':f'{type(exc).__name__}: {exc}'}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
    return int(report['coding_checks']!='passed' or report.get('mastery')=='revision_required')
if __name__=='__main__': raise SystemExit(main())
