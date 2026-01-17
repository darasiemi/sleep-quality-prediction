To initialize the dependency manager, uv
```bash
uv init
```
To remove the redundant main.py
```bash
rm main.py
```
To install libraries
```bash
uv add scikit-learn fastapi uvicorn pandas numpy
```
We also add a development dependency we won't need in production
```bash
uv add --dev requests
```

To run the training script
```bash
uv run python -m deployment.train
```
To run single prediction
```bash
 uv run python -m deployment.predict_one
```

To make the prediction endpoint
```bash
uv run uvicorn deployment.predict:app --host 0.0.0.0 --port 9696 --reload
```

To run the test
```bash
uv run python -m deployment.test
```

