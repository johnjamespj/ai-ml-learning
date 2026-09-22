"""Repair research-method examples without asserting that an ablation must win."""
from pathlib import Path
import hashlib,re,textwrap
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]


def main():
    for filename in ['09_resnet.ipynb','10_transformer.ipynb','15_ddpm.ipynb','13_simclr.ipynb','11_bert.ipynb']:
        path=ROOT/'papers/notebooks'/filename;nb=nbf.read(path,as_version=4)
        course=nb.metadata.setdefault('course',{})
        if course.get('methods_review')==1: continue
        course['methods_review']=1
        for cell in nb.cells:
            if cell.cell_type!='code': continue
            s=cell.source
            if filename=='09_resnet.ipynb':
                s=s.replace('self.net=nn.Sequential(nn.Linear(d,d),nn.ReLU())',
                            'self.net=nn.Sequential(nn.Linear(d,d),nn.ReLU(),nn.Linear(d,d))')
                s=s.replace('def forward(self,x): return self.net(x)','def forward(self,x): return torch.relu(self.net(x))')
            if filename=='15_ddpm.ipynb':
                s=s.replace('T=50','T=100').replace('torch.linspace(1e-4,.02,T)','torch.linspace(1e-4,.2,T)')
                s=s.replace('abar=torch.cumprod(alphas,0)','abar=torch.cumprod(alphas,0)\nassert abar[-1] < 1e-3, "Terminal distribution must approach the sampling prior"\nabar_prev=torch.cat([torch.ones(1),abar[:-1]])\nposterior_var=betas*(1-abar_prev)/(1-abar)')
                s=s.replace('[0,10,30,49]','[0,20,60,99]')
                s=s.replace('torch.sqrt(betas[ti])*torch.randn_like(x)','torch.sqrt(posterior_var[ti])*torch.randn_like(x)')
            if filename=='13_simclr.ipynb':
                s=s.replace('ids=torch.randint(0,len(X),(256,))','ids=torch.randperm(len(X))[:256]')
                start=s.find('def loss_fn(')
                end=s.find('\nfor step',start)
                if start>=0 and end>=0:
                    s=s[:start]+'''def loss_fn(z1,z2,temp=.2):
    # Full 2N NT-Xent: each anchor has one positive and 2N-2 negatives.
    z=torch.cat([nn.functional.normalize(z1,dim=1),nn.functional.normalize(z2,dim=1)],dim=0)
    n=len(z1); logits=z@z.T/temp
    logits=logits.masked_fill(torch.eye(2*n,dtype=torch.bool,device=z.device),float('-inf'))
    target=(torch.arange(2*n,device=z.device)+n)%(2*n)
    return nn.functional.cross_entropy(logits,target)
''' + s[end:]
            if filename=='11_bert.ipynb' and 'from transformers import BertConfig' in s:
                body=textwrap.dedent(s[s.index('try:\n')+5:s.index('\nexcept Exception')])
                s="import importlib.util\nif importlib.util.find_spec('transformers') is not None:\n"+textwrap.indent(body,'    ')+"\nelse:\n    print('NOT RUN: optional Transformers architecture inspection requires its package.')"
                course['optional_paths']=['Optional Hugging Face architecture inspection']
            if filename=='10_transformer.ipynb' and 'label=((seq[:,0]' in s:
                s=s.replace('label=((seq[:,0]==seq[:,-1])).long()',
                    'label=(torch.arange(N)%2).long()\nseq[:,-1]=torch.where(label.bool(),seq[:,0],(seq[:,0]+torch.randint(1,VOC,(N,)))%VOC)')
                s=s.replace('tr=perm[:1500]; te=perm[1500:]','tr=perm[:1400]; va=perm[1400:1700]; te=perm[1700:]')
                s=s.replace('def train(use_pos):','trained_models={}\ndef train(use_pos):')
                s=s.replace('opt.zero_grad(); loss=ce(m(seq[tr])','m.train(); opt.zero_grad(); loss=ce(m(seq[tr])')
                s=s.replace('with torch.no_grad(): acc=(m(seq[te]).argmax(1)==label[te])',
                            'm.eval()\n        with torch.no_grad(): acc=(m(seq[va]).argmax(1)==label[va])')
                s=s.replace('return np.array(hist)','trained_models[use_pos]=m\n    return np.array(hist)')
                s=s.replace('plt.ylabel("test accuracy")','plt.ylabel("validation accuracy")')
                s += '''
# Only after the predeclared training runs: one untouched-test comparison.
final_test={}
for setting, fitted in trained_models.items():
    fitted.eval()
    with torch.no_grad(): final_test[str(setting)]=float((fitted(seq[te]).argmax(1)==label[te]).float().mean())
print('Final test, with/without position:',final_test)
print('Majority baseline:',max(float(label[te].float().mean()),1-float(label[te].float().mean())))
print('Small-scale optimization may not reproduce a positive positional advantage. Report the observed result, not the expected story.')
'''
            cell.source=s
        nb.cells.append(nbf.v4.new_markdown_cell('## Methodology note\n\nThis is a bounded mechanism experiment, not an original-benchmark reproduction. A run that completes is not evidence that the paper claim was reproduced. Report actual baseline comparisons, uncertainty and failed ablations.'))
        for i,c in enumerate(nb.cells):
            c.id=hashlib.sha256(f'{filename}:{i}:{c.source}'.encode()).hexdigest()[:12]
            if c.cell_type=='code': c.outputs=[];c.execution_count=None
        nbf.write(nb,path)
    print('Reviewed ResNet capacity matching, Transformer evaluation, DDPM terminal prior, SimCLR negatives, and optional BERT imports.')
if __name__=='__main__':main()
