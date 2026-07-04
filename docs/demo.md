# Capsule — Demo Guide

## What this demo proves

- Framework auto-detection identifies PyTorch models correctly
- Dockerfile generation produces a valid container definition
- ONNX optimisation runs and reduces model size
- MinIO model registry stores and retrieves versioned artifacts
- Helm chart generation produces valid Kubernetes manifests
- CLI commands work end-to-end (package, list, deploy, status, rollback)
- Test suite passes without K3s or Docker

---

## Prerequisites

```bash
pip install -r requirements.txt
pip install -e .
brew install helm
docker compose up minio registry -d
```

---

## Demo Commands

### 1. Train the demo fraud detection model

```bash
python examples/fraud_detector/train_model.py
```

Expected: `model.pt` saved in `examples/fraud_detector/`.

### 2. Package the model

```bash
capsule package --manifest examples/fraud_detector/capsule.yaml
```

Expected output:
```
[capsule] Detected framework: pytorch
[capsule] Generating Dockerfile...
[capsule] Building Docker image: fraud-detector:1.0
[capsule] ONNX optimisation: model.pt → model.onnx (size: -35%)
[capsule] Pushing to MinIO registry: fraud-detector:1.0
[capsule] Package complete: fraud-detector:1.0
```

> ⚠️ ONNX size reduction is approximate and varies by model.

### 3. List packages

```bash
capsule list
```

Expected:
```
Name              Version  Framework  Status    Created
fraud-detector    1.0      pytorch    packaged  2024-01-01 12:00:00
```

### 4. Deploy to K3s (requires K3s installed)

```bash
# K3s must be running
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml

capsule deploy fraud-detector:1.0
```

Expected:
```
[capsule] Generating Helm chart...
[capsule] Deploying fraud-detector:1.0 to K3s...
[capsule] Waiting for rollout...
[capsule] Deployment complete: 1/1 pods running
```

> ⚠️ **K3s deployment note**: K3s deployment is implemented but end-to-end local demo verification is pending. The `capsule package` and `capsule list` commands work without K3s.

### 5. Check deployment status

```bash
capsule status fraud-detector
```

### 6. Deploy v2 with canary

```bash
# Train v2 (modify training script slightly)
capsule package --manifest examples/fraud_detector/capsule.yaml  # bump version

capsule deploy fraud-detector:2.0 --canary 10
# 90% traffic → v1, 10% traffic → v2
```

### 7. Rollback

```bash
capsule rollback fraud-detector --yes
```

### 8. Run tests (no K3s or Docker needed)

```bash
pytest tests/ -v
```

---

## Expected Output Summary

| Check | Expected |
|-------|----------|
| `train_model.py` | model.pt saved |
| `capsule package` | Docker image built, artifact in MinIO |
| `capsule list` | Package visible in history |
| `capsule deploy` | Pod running in K3s (requires K3s) |
| `capsule status` | Pod health, canary %, version shown |
| `capsule rollback` | Traffic shifted back to previous version |
| `pytest tests/ -v` | All 40+ tests pass |

---

## Known Limitations

- K3s deployment requires K3s installed and running. All non-deploy commands work without K3s.
- Docker is required for image building. Without Docker, `capsule package` fails at the build step.
- ONNX optimisation size reduction is approximate.
- Canary requires Traefik ingress (default in K3s).
- Screenshot of `capsule status` Rich table: pending.
