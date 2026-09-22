import csv,json
import pytest
from coursekit.experiments import Experiment,run_trials


def test_artifacts_do_not_invent_completed_ablations(tmp_path,monkeypatch):
    monkeypatch.setenv('COURSE_RESULTS_DIR',str(tmp_path))
    a=Experiment('test',{'seed':2});a.finish({'loss':.5})
    assert json.loads((a.directory/'run_status.json').read_text())['interpretation']=='pending'
    assert 'not_run' in (a.directory/'ablation.csv').read_text()
    assert json.loads((a.directory/'diagnostics.json').read_text())['loss']==.5


def test_paired_trials_are_explicit_and_isolated(tmp_path,monkeypatch):
    monkeypatch.setenv('COURSE_RESULTS_DIR',str(tmp_path))
    a=Experiment('paired',{'seeds':[0,1]});b=Experiment('paired',{})
    assert a.directory!=b.directory
    run_trials(a,{'base':lambda seed:{'accuracy':.5+seed*.1},'ablated':lambda seed:{'accuracy':.4+seed*.1}},[0,1])
    rows=list(csv.DictReader((a.directory/'ablation.csv').open()))
    assert len(rows)==4 and all(r['status']=='measured' for r in rows)
    with pytest.raises(ValueError): a.log('bad',0,{'loss':float('nan')})
