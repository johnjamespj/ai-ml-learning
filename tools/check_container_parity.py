"""Check real HTTP/container inference against the exported offline model."""
from pathlib import Path
import argparse,json,sys,time,urllib.request,urllib.error
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from capstones.rf_detection.inference import load_artifact,predict

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url',default='http://127.0.0.1:8000');p.add_argument('--artifact',type=Path,required=True)
    p.add_argument('--request',type=Path,required=True);a=p.parse_args()
    for _ in range(60):
        try:
            with urllib.request.urlopen(a.url+'/health',timeout=2) as r:
                if json.load(r)['status']=='ready': break
        except (OSError,ValueError,KeyError): time.sleep(.5)
    else: raise RuntimeError('Container did not become ready')
    req=json.loads(a.request.read_text());q=np.asarray(req['iq']);iq=q[:,0]+1j*q[:,1]
    payload=load_artifact(a.artifact);prob,label=predict(payload,iq)
    durations=[]
    for _ in range(30):
        request=urllib.request.Request(a.url+'/predict',data=json.dumps(req).encode(),headers={'Content-Type':'application/json'})
        start=time.perf_counter()
        with urllib.request.urlopen(request,timeout=5) as r: answer=json.load(r)
        durations.append((time.perf_counter()-start)*1000)
        assert abs(answer['probability']-prob[0])<1e-10
        assert answer['signal_present']==bool(label[0])
        assert answer['model_version']==payload['sha256'][:12]
    report={'parity':'passed','requests':30,'http_p50_ms':float(np.quantile(durations,.5)),
            'http_p95_ms':float(np.quantile(durations,.95)),'scope':'localhost Docker HTTP, not production network'}
    out=ROOT/'artifacts/container-parity.json';out.parent.mkdir(exist_ok=True);out.write_text(json.dumps(report,indent=2));print(report)
if __name__=='__main__':main()
