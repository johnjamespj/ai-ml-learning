"""One-time, reviewable repair of notebook conversion defects.

The execution runner never changes code. This script edits source notebooks;
its exact diff is reviewed and committed. Existing worked examples are preserved.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import textwrap
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
VERSION=1
BOOTSTRAP='''from pathlib import Path
import os, sys
ROOT = Path(os.environ.get('COURSE_ROOT', next((str(p) for p in [Path.cwd(), *Path.cwd().parents] if (p / 'coursekit').is_dir()), '.'))).resolve()
if not (ROOT / 'coursekit').is_dir():
    raise RuntimeError('Open this notebook in the cloned ai-ml-learning repository.')
sys.path.insert(0, str(ROOT))
from coursekit.runtime import configure
SEED = 0
configure(SEED)
'''
SKLEARN='''import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
X, y = make_classification(n_samples=500, n_features=6, n_informative=4, random_state=SEED)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, stratify=y, random_state=SEED)
'''
TORCH='''import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
torch.manual_seed(SEED)
X_train = torch.randn(160, 2)
y_train = (X_train[:, 0] + .5 * X_train[:, 1] > 0).long()
X_val = torch.randn(60, 2)
y_val = (X_val[:, 0] + .5 * X_val[:, 1] > 0).long()
device = torch.device('cpu')
model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 2)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=.01)
loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)
'''
SETUP={
11:SKLEARN,12:SKLEARN,13:SKLEARN,
15:SKLEARN+'''from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
''',
18:'''import numpy as np
rng = np.random.default_rng(SEED)
X = np.concatenate([rng.normal(size=(400, 2)), rng.normal(5, .3, size=(10, 2))])
''',
21:'''import numpy as np
rng = np.random.default_rng(SEED)
X = rng.normal(size=(12, 2)); y = rng.normal(size=(12, 1))
W1 = rng.normal(size=(2, 4)); W2 = rng.normal(size=(4, 1))
b1 = np.zeros(4); b2 = np.zeros(1)
Z1 = X @ W1 + b1; A1 = np.maximum(0, Z1); Z2 = A1 @ W2 + b2
''',
22:'''from sklearn.datasets import make_moons
from labs.lab04_numpy_mlp.solution import init_params, forward, backward, cross_entropy
# This is the worked example. Implement your own version in the student lab.
X, y = make_moons(n_samples=200, noise=.2, random_state=SEED)
params = init_params(2, 16, 2, seed=SEED)
epochs = 300; learning_rate = .05
''',
26:TORCH,27:TORCH,28:TORCH,
29:TORCH+'''for epoch in range(10):
    for xb, yb in loader:
        optimizer.zero_grad(); loss = loss_fn(model(xb), yb)
        loss.backward(); optimizer.step()
model.eval()
with torch.no_grad():
    val_loss = float(loss_fn(model(X_val), y_val))
''',
30:TORCH+'''loss_fn(model(X_train), y_train).backward()
''',
52:'''import numpy as np
import pandas as pd
rng = np.random.default_rng(SEED)
df = pd.DataFrame({'value': np.sin(np.arange(100)/10) + rng.normal(0,.1,100)})
''',
54:'''import numpy as np
from scipy.integrate import trapezoid
fs=2000
_t=np.arange(4000)/fs
x=np.sin(2*np.pi*200*_t)+np.random.default_rng(SEED).normal(0,.2,len(_t))
f1, f2 = 150, 250
''',
61:TORCH+'''xb, yb = next(iter(loader))
''',
64:'''import numpy as np
rng = np.random.default_rng(SEED)
y_prob = rng.uniform(.01, .99, 2000)
y_true = rng.binomial(1, y_prob)
'''
}


def main():
    counts={'notebooks':0,'setup_repairs':0}
    paths=sorted(list((ROOT/'lessons').glob('*/lesson.ipynb'))+list((ROOT/'math').glob('*.ipynb'))+
                 list((ROOT/'papers/notebooks').glob('*.ipynb'))+list((ROOT/'notebooks').glob('*.ipynb')))
    for path in paths:
        nb=nbf.read(path,as_version=4)
        if nb.metadata.get('course',{}).get('hardening_version')==VERSION:
            continue
        course=nb.metadata.setdefault('course',{})
        course['hardening_version']=VERSION
        course.setdefault('optional_paths',[])
        first=next((i for i,c in enumerate(nb.cells) if c.cell_type=='code'),len(nb.cells))
        setup=''
        if path.parent.parent.name=='lessons':
            number=int(path.parent.name[:2]); setup=SETUP.get(number,'')
            if setup: counts['setup_repairs']+=1
            for cell in nb.cells:
                if cell.cell_type!='code': continue
                s=cell.source
                if number==0 and s.startswith('%%bash'):
                    cell.cell_type='markdown'; cell.pop('execution_count',None);cell.pop('outputs',None)
                    cell.source=('## Environment creation is a terminal operation\n\nDo not launch a second Jupyter server from a running kernel. From the repository root, run:\n\n'
                        '```bash\npython3 -m venv .venv\nsource .venv/bin/activate\npython -m pip install -r requirements/cpu.txt\npython -m ipykernel install --user --name ai-ml-course\njupyter lab\n```\n\nLinux CPU instructions. Choose platform-specific PyTorch wheels on macOS or CUDA.')
                    continue
                if number==21: s=s.replace('2 * (Z2 - y) / len(X)','2 * (Z2 - y) / Z2.size')
                if number==22: s=s.replace('logits, cache = forward','probs, cache = forward').replace('cross_entropy(logits, y)','cross_entropy(probs, y)')
                if number==42:
                    if 'from_pretrained' in s:
                        # Keep the original example but make downloading an explicit opt-in.
                        raw=s[s.index('try:\n')+5:s.index('\nexcept Exception')]
                        raw=textwrap.dedent(raw)
                        s="if os.environ.get('COURSE_RUN_DOWNLOADS') == '1':\n"+textwrap.indent(raw,'    ')+"\nelse:\n    print('NOT RUN: pretrained weights require explicit COURSE_RUN_DOWNLOADS=1 and optional dependencies.')"
                    elif 'BertConfig' in s:
                        raw=s[s.index('try:\n')+5:s.index('\nexcept Exception')]
                        s="import importlib.util\nif importlib.util.find_spec('transformers') is not None:\n"+textwrap.indent(textwrap.dedent(raw),'    ')+"\nelse:\n    print('NOT RUN: optional Transformers package is absent. No pretrained-model result is claimed.')"
                    course['optional_paths']=['Pretrained Hugging Face model download','Optional Transformers random-architecture inspection']
                if number==52: s=s.replace('df["value"].rolling(10).mean()', 'df["value"].shift(1).rolling(10).mean()')
                if number==54: s=s.replace('np.trapz(', 'trapezoid(')
                if number==61: s=s.replace('device_type="cuda", dtype=torch.float16', 'device_type=xb.device.type, dtype=torch.bfloat16, enabled=xb.device.type == "cuda"')
                if number==34: s=s.replace('np.sin(2*np.pi*(100 + 200*t)*t)', 'signal.chirp(t, f0=100, f1=500, t1=2, method="linear")').replace('np.sin(2*np.pi*(100+200*t)*t)','signal.chirp(t, f0=100, f1=500, t1=2, method="linear")')
                if number==60: s=s.replace('edges[0]-=1e-9; edges[-1]+=1e-9','edges[0]=-np.inf; edges[-1]=np.inf')
                cell.source=s
            if number in (28,61): course['optional_paths']=['CUDA acceleration and mixed-precision hardware performance']
            if number==59: course['optional_paths']=['Actual Docker image build and container execution']
            if number==62: course['optional_paths']=['Actual multi-process or multi-GPU distributed training']
            if number==52:
                nb.cells.append(nbf.v4.new_markdown_cell('### Forecast timestamp convention\n\nTo predict value at time t, every feature must depend only on times before t. The rolling mean above is shifted by one step; including value[t] would leak the current target.'))
            if number==31:
                nb.cells.append(nbf.v4.new_markdown_cell('### Convolution convention\n\n`torch.nn.Conv2d` computes cross-correlation, without reversing the learned kernel. Mathematical convolution reverses the kernel. Learning the kernel makes the naming convention operationally harmless, but it matters when checking a fixed hand-calculated filter.'))
        nb.cells.insert(first,nbf.v4.new_code_cell(BOOTSTRAP+setup))
        if path.parent.name=='notebooks' and path.parent.parent.name=='papers':
            key=path.stem
            init=("from coursekit.experiments import Experiment\n"
                  f"experiment = Experiment('paper-{key}', {{'bootstrap_seed': SEED, 'scope': 'educational mechanism demonstration', 'note': 'Original notebook may use additional explicit seeds; source hash records the exact experiment.'}}, source='{path.relative_to(ROOT).as_posix()}')\n"
                  "experiment.capture_figures()")
            nb.cells.insert(first+1,nbf.v4.new_code_cell(init))
            nb.cells.append(nbf.v4.new_markdown_cell('## Evidence export\n\nFigures and numeric diagnostics are captured. Explicit metrics use `experiment.log(variant, seed, metrics)`. Use `run_trials` for paired-seed ablations. An empty metrics table or `not_run` ablation is incomplete evidence, not success. Interpretations remain your work.'))
            nb.cells.append(nbf.v4.new_code_cell("print('Evidence directory:', experiment.finish(globals()))"))
        if path.parent==ROOT/'notebooks':
            for cell in nb.cells:
                if cell.cell_type=='code':
                    cell.source=cell.source.replace('import starter as lab','import solution as lab').replace('import starter as attn','import solution as attn')
            nb.cells.insert(1,nbf.v4.new_markdown_cell('**Worked reference demonstration.** This notebook is not evidence you completed the starter. Use `tools/make_student_workspace.py` for your independent implementation and tests.'))
        for i,cell in enumerate(nb.cells):
            cell['id']=hashlib.sha256(f'{path.relative_to(ROOT)}:{i}:{cell.source}'.encode()).hexdigest()[:12]
            if cell.cell_type=='code': cell.outputs=[];cell.execution_count=None
        nbf.validate(nb)
        nbf.write(nb,path);counts['notebooks']+=1
    print(json.dumps(counts,indent=2))
if __name__=='__main__': main()
