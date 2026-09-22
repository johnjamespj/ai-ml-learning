# Python AI/ML Toolbox

Use this file as the map from concepts to tools.

| Tool | Learn it for |
|---|---|
| Python | language, modules, environments, packaging |
| NumPy | arrays, linear algebra, vectorization, from-scratch ML |
| pandas | tabular data, cleaning, joins, feature preparation |
| Matplotlib | visualization and diagnostics |
| SciPy | statistics, optimization, signal processing |
| SymPy | symbolic math and derivative checking |
| scikit-learn | classical ML, preprocessing, pipelines, metrics, CV |
| PyTorch | neural networks, autograd, GPU training |
| torchvision | image datasets/models/transforms |
| Hugging Face Transformers | pretrained Transformers and LLM workflows |
| Hugging Face Datasets | dataset loading/processing |
| PEFT | LoRA and parameter-efficient fine-tuning |
| Gymnasium | reinforcement-learning environments |
| FastAPI | model-serving APIs |
| Pydantic | validated API schemas |
| pytest | automated tests |
| Docker | reproducible deployment |
| MLflow / W&B concepts | experiment tracking |
| Optuna concepts | hyperparameter optimization |
| SHAP concepts | model explanations |
| FAISS/vector DB concepts | efficient embedding search |

## Principle
Learn the abstraction underneath each library first.

Examples:
- NumPy gradient descent before `sklearn.fit()`
- manual backprop before PyTorch autograd
- raw attention before Transformer convenience APIs
- brute-force vector similarity before a vector database

## Suggested install layers

### Core
```bash
pip install numpy pandas matplotlib scipy sympy scikit-learn jupyterlab pytest
```

### Deep learning
```bash
pip install torch torchvision
```

### LLM
```bash
pip install transformers datasets accelerate peft
```

### RL and serving
```bash
pip install gymnasium fastapi uvicorn
```

Install optional specialized tools only when you reach the lesson that needs them.
