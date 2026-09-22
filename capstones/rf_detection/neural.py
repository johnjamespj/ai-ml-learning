"""Optional small CPU/GPU models; validation selects checkpoints, never test data."""
from __future__ import annotations
import copy
import numpy as np
import torch
from torch import nn


def spectrograms(iq: np.ndarray) -> torch.Tensor:
    x=torch.tensor(iq,dtype=torch.complex64)
    z=torch.stft(x,n_fft=16,hop_length=8,win_length=16,window=torch.hann_window(16),
                 center=True,return_complex=True,onesided=False)
    return torch.log1p(z.abs().square()/16)


class SpectrogramCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(1,8,3,padding=1),nn.ReLU(),nn.Conv2d(8,8,3,padding=1),
                               nn.ReLU(),nn.AdaptiveAvgPool2d((4,4)),nn.Flatten(),nn.Linear(128,1))
    def forward(self,x): return self.net(x[:,None]).squeeze(1)


class TemporalTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.project=nn.Linear(16,16)
        self.position=nn.Parameter(torch.randn(1,17,16)*.01)
        block=nn.TransformerEncoderLayer(16,2,32,dropout=0.,batch_first=True)
        self.encoder=nn.TransformerEncoder(block,1)
        self.head=nn.Linear(16,1)
    def forward(self,x):
        x=self.project(x.transpose(1,2))+self.position
        return self.head(self.encoder(x).mean(1)).squeeze(1)


def fit(kind: str, train, val, *, seed=0, epochs=12):
    torch.manual_seed(seed);torch.set_num_threads(1)
    model={'spectrogram_cnn':SpectrogramCNN,'temporal_transformer':TemporalTransformer}[kind]()
    x=spectrograms(train.iq); y=torch.tensor(train.y,dtype=torch.float32)
    xv=spectrograms(val.iq);yv=torch.tensor(val.y,dtype=torch.float32)
    opt=torch.optim.Adam(model.parameters(),lr=.003)
    objective=nn.BCEWithLogitsLoss();best=float('inf');best_state=None;history=[];stale=0
    for epoch in range(epochs):
        model.train();order=torch.randperm(len(x));total=0.
        for batch in order.split(64):
            opt.zero_grad();loss=objective(model(x[batch]),y[batch]);loss.backward();opt.step()
            total+=float(loss.detach())*len(batch)
        model.eval()
        with torch.no_grad(): validation=float(objective(model(xv),yv))
        history.append({'epoch':epoch,'train_loss':total/len(x),'validation_loss':validation})
        if validation<best:
            best=validation;best_state=copy.deepcopy(model.state_dict());stale=0
        else: stale+=1
        if stale>=3: break
    model.load_state_dict(best_state);model.eval()
    def score(iq):
        with torch.no_grad(): return model(spectrograms(iq)).cpu().numpy()
    return score,history,model
