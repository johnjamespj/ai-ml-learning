# Lesson 58: Model serving with FastAPI

Training is only half of an ML system.

## Inference API
```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Input(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(inp: Input):
    # preprocess -> model -> postprocess
    return {"prediction": 0}
```

## Production concerns
- input validation
- model loading
- preprocessing consistency
- batching
- latency
- concurrency
- error handling
- versioning
- observability

## Exercise
Serve one trained classifier. Add:
- /health
- /predict
- model version in response
- request validation
- unit tests

## Important
The exact same preprocessing used during training must be applied at inference.
