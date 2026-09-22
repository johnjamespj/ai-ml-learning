"""A small validated inference API. It refuses to start without a valid artifact."""
from contextlib import asynccontextmanager
import os
from pathlib import Path
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, FiniteFloat
from .inference import load_artifact,predict


class IQRequest(BaseModel):
    iq: list[tuple[FiniteFloat,FiniteFloat]] = Field(min_length=128,max_length=128)


def create_app(model_path: str | Path) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        app.state.payload=load_artifact(model_path)
        yield
    service=FastAPI(title='Educational RF detector',lifespan=lifespan)
    @service.get('/health')
    def health():
        return {'status':'ready','model_version':service.state.payload['sha256'][:12],
                'scope':'synthetic educational detector'}
    @service.post('/predict')
    def infer(request: IQRequest):
        z=np.asarray(request.iq,float);iq=z[:,0]+1j*z[:,1]
        try: probability,label=predict(service.state.payload,iq)
        except ValueError as exc: raise HTTPException(status_code=422,detail=str(exc)) from exc
        return {'signal_present':bool(label[0]),'probability':float(probability[0]),
                'threshold':service.state.payload['model']['threshold'],
                'model_version':service.state.payload['sha256'][:12]}
    return service

app=create_app(os.environ.get('MODEL_PATH','results/rf-capstone/model.json'))
