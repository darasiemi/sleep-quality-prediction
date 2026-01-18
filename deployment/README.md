### Dependency Manager
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

### Docker
To build docker image
```bash
docker build -t sleep-quality-prediction:v1 .
```
To run container
```bash
docker run -it --rm -p 9696:9696 sleep-quality-prediction:v1
```

### Best Practices
To install black and isort for code formating
```bash
uv add --dev black isort pre-commit
```
To run the formatting
```bash
uv run black .
```
```bash
uv run isort .
```

To install precommit hooks
```bash
uv run pre-commit install
```
To run precommits
```bash
uv run pre-commit run --all-files
```

### Kubernetes Deployment
Ensure you have kind and kubectl installed
```bash
kind create cluster --name sleep-project
```
Verify the cluster is running
```bash
kubectl cluster-info
```
```bash
kubectl get nodes
```
Loading image into kind
Kind clusters run in Docker, so they can't access images from your local Docker daemon by default. We need to load the image into Kind.
```bash
kind load docker-image sleep-quality-prediction:v1 --name sleep-project
```
To create resources using the yaml file
```bash
kubectl apply -f deployment.yaml
```
To get pods

```bash
kubectl get pods
```
To see what is happening in the pod
```bash
kubectl describe pod <pod>
```
To create service

```bash
kubectl apply -f service.yaml
```
To get services
```bash
kubectl get services
```
Our kind cluster is not configured for NodePort, so it won't work. We don't really need this for testing things locally, so let's just use a quick fix: Use kubectl port-forward.
```bash
kubectl port-forward service/sleep-quality-prediction 30080:9696
```
Check the health endpoint
```bash
curl http://localhost:30080/health
```
Now we change the endpoint URL i.e `BASE_URL` in test.py to `"http://localhost:30080"`,
```bash
uv run python -m deployment.test


```
<!-- To create horizontal pod autoscaler
```bash
kubectl apply -f hpa.yaml
```

To check how many deployments you have
```bash
kubectl get deployments
``` -->

