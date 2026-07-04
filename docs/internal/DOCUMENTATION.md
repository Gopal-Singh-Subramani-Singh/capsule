# Capsule — Container-Native ML Model Deployment Platform

> Package, optimise, deploy, and manage ML models on Kubernetes with a single CLI.

---

## Table of Contents

1. [What Is Capsule](#1-what-is-capsule)
2. [Architecture](#2-architecture)
3. [Project Structure](#3-project-structure)
4. [Prerequisites](#4-prerequisites)
5. [How to Run — Step by Step](#5-how-to-run--step-by-step)
6. [CLI Reference](#6-cli-reference)
7. [capsule.yaml Reference](#7-capsuleyaml-reference)
8. [How ONNX Optimisation Works](#8-how-onnx-optimisation-works)
9. [How Canary Deployment Works](#9-how-canary-deployment-works)
10. [Generated Model Server](#10-generated-model-server)
11. [K3s Setup](#11-k3s-setup)
12. [Configuration Reference](#12-configuration-reference)
13. [Monitoring](#13-monitoring)
14. [Port Reference](#14-port-reference)
15. [Running Tests](#15-running-tests)
16. [Troubleshooting](#16-troubleshooting)

---

## 1. What Is Capsule

Capsule is a container-native ML model deployment platform that takes a raw model file and handles everything from containerisation through production traffic management.

**The four core commands:**

| Command | What it does |
|---|---|
| `capsule package` | Auto-detects framework, generates Dockerfile, builds Docker image, ONNX-optimises (~35% size reduction), pushes to local registry |
| `capsule deploy` | Generates a Helm chart, deploys to K3s, supports canary traffic splits |
| `capsule status` | Shows pod health, canary weight, version, and recent events in a Rich table |
| `capsule rollback` | Switches back to the previous stable version in one command |

**Why Capsule exists:** The gap between a trained model file and a production Kubernetes deployment involves a lot of undifferentiated work — writing Dockerfiles, choosing base images, generating Helm charts, wiring up health probes, configuring Prometheus metrics, and building rollback logic. Capsule automates all of it.

**What Capsule is not:** a managed cloud service, a training platform, or a feature store. It sits squarely in the serving layer.


---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        capsule CLI (Typer)                       │
│   package │ deploy │ status │ rollback │ list │ watch │ audit    │
└────┬──────┴───┬────┴────────┴──────────┴──────┴───────┴─────────┘
     │          │
     ▼          ▼
┌──────────┐  ┌──────────────────────────────────────────────────┐
│ Packager │  │                   Deployer                        │
│          │  │  Helm chart gen → K3s deploy → canary weights    │
│ • detect │  └──────────────┬─────────────────┬────────────────┘
│   fwk    │                 │                 │
│ • gen    │         ┌───────┘          ┌──────┘
│   Docker-│         ▼                  ▼
│   file   │  ┌─────────────┐  ┌────────────────┐
│ • ONNX   │  │  K8sClient  │  │ HelmChart      │
│   opt    │  │  (k8s SDK)  │  │ Generator      │
│ • Docker │  │  pod status │  │ Chart.yaml     │
│   build  │  │  rollout    │  │ values.yaml    │
│ • push   │  │  wait       │  │ deployment.yaml│
└────┬─────┘  └─────────────┘  └────────────────┘
     │
     ▼
┌──────────────────────────┐    ┌──────────────────────────────┐
│   ModelRegistry (MinIO)  │    │  CanaryController (async)    │
│   • model artifacts      │    │  • polls Prometheus /30s     │
│   • image tags           │    │  • N failures → rollback     │
│   • image digests        │    │  • M clean windows → promote │
│   • Dockerfiles          │    └──────────────────────────────┘
└──────────────────────────┘
     │
     ▼
┌──────────────────────────┐
│   ManifestStore (SQLite) │
│   ~/.capsule/capsule.db  │
│   • packages table       │
│   • deployments table    │
│   • events table         │
│   • audit_log table      │
└──────────────────────────┘
```

**Data flow for `capsule package`:**

```
model.pt / .pkl / .onnx
       │
       ▼
  detect_framework()          ← extension-based detection
       │
       ▼
  ONNX optimisation           ← PyTorch → torch.onnx.export → INT8 quantise
       │
       ▼
  generate Dockerfile         ← Jinja2 template, non-root user, HEALTHCHECK
  generate server.py          ← FastAPI + prometheus-client, framework-specific
  generate requirements.txt   ← minimal, pinned versions
       │
       ▼
  docker build + push         ← to localhost:5001 registry
       │
       ▼
  MinIO push                  ← model artifact + image tag + digest stored
       │
       ▼
  SQLite save_package()       ← package record + audit log entry
```

**Data flow for `capsule deploy`:**

```
name:version argument
       │
       ▼
  registry.get_deploy_ref()   ← prefers sha256 digest (immutable) over tag
       │
       ▼
  HelmChartGenerator.generate()  ← Chart.yaml, values.yaml, deployment+service
       │
       ▼
  helm upgrade --install --atomic ← auto-rolls back on failure
       │
       ▼
  k8s.wait_for_rollout()      ← waits up to 120s
       │
       ▼
  SQLite save_deployment()    ← deployment record + audit log
```


---

## 3. Project Structure

```
capsule/
├── capsule/                        # Main Python package
│   ├── __init__.py
│   ├── cli.py                      # Typer CLI: package, deploy, status, rollback,
│   │                               #   list, watch, audit
│   ├── packager.py                 # Dockerfile generation + Docker build + ONNX opt
│   ├── registry.py                 # MinIO model registry client (boto3)
│   ├── deployer.py                 # Helm chart gen + K3s deploy + rollback
│   ├── canary.py                   # Async canary controller + auto-rollback watcher
│   ├── server_template.py          # Generated FastAPI model server (3 flavours)
│   ├── manifest.py                 # capsule.yaml parser + SQLite ManifestStore
│   ├── onnx_optimizer.py           # PyTorch→ONNX export + INT8 quantisation
│   ├── k8s_client.py               # Kubernetes Python client wrappers
│   ├── helm.py                     # Helm chart template generator (Jinja2)
│   ├── detector.py                 # ML framework auto-detector
│   └── models.py                   # All Pydantic v2 schemas
│
├── config/
│   └── defaults.yaml               # All tuneable defaults (registry, k8s, canary…)
│
├── examples/
│   ├── fraud_detector/
│   │   ├── capsule.yaml            # v1.0 manifest
│   │   ├── capsule_v2.yaml         # v2.0 manifest (canary demo)
│   │   ├── fraud_model_module.pt   # Pre-trained TorchScript model
│   │   ├── fraud_model_v2_module.pt
│   │   └── train_model.py          # Training script (generates the .pt files)
│   └── sentiment_classifier/
│       ├── capsule.yaml            # scikit-learn manifest
│       └── train_model.py
│
├── tests/
│   ├── conftest.py                 # Shared pytest fixtures (mocks, tmp dirs)
│   ├── test_detector.py
│   ├── test_packager.py
│   ├── test_onnx_optimizer.py
│   ├── test_registry.py
│   ├── test_deployer.py
│   ├── test_canary.py
│   ├── test_manifest.py
│   └── test_integration.py
│
├── dashboards/
│   └── capsule.json                # Grafana dashboard definition
│
├── .github/workflows/capsule-ci.yml
├── docker-compose.yml              # MinIO, Docker registry, Prometheus, Grafana
├── prometheus.yml                  # Prometheus scrape config
├── requirements.txt
└── pyproject.toml                  # Entry point: capsule → capsule.cli:app
```

### Key module responsibilities

| Module | Responsibility |
|---|---|
| `cli.py` | User-facing Typer commands; wires components together; Rich output |
| `packager.py` | Dockerfile + server + requirements generation; Docker build/push; ONNX pipeline |
| `deployer.py` | Helm chart deploy/rollback via subprocess; K8s rollout wait; status aggregation |
| `canary.py` | Async Prometheus polling loop; failure counting; injectable rollback/promote callbacks |
| `manifest.py` | `load_manifest()` YAML parser; `ManifestStore` SQLite WAL store |
| `registry.py` | MinIO `boto3` client: model artifacts, image tags, digests, Dockerfiles |
| `helm.py` | Jinja2 Chart.yaml / values.yaml / deployment / service template generation |
| `detector.py` | Extension → `Framework` enum mapping; base image and pip package selection |
| `onnx_optimizer.py` | `torch.onnx.export` → graph optimisation → `quantize_dynamic` INT8 |
| `server_template.py` | Three FastAPI server templates (ONNX, PyTorch, scikit-learn) |
| `models.py` | All Pydantic v2 models: `CapsuleManifest`, `PackageResult`, `DeploymentRecord`, etc. |
| `k8s_client.py` | `kubernetes` Python SDK wrappers: namespace, pods, events, rollout wait |


---

## 4. Prerequisites

### Required

| Tool | Version | Install |
|---|---|---|
| Python | 3.11+ | `brew install python@3.11` or [python.org](https://python.org) |
| Docker Desktop or OrbStack | Latest | [docker.com](https://www.docker.com/products/docker-desktop/) / [orbstack.dev](https://orbstack.dev) |
| Helm | 3.x | `brew install helm` |

### Optional (for Kubernetes deploys)

| Tool | Notes |
|---|---|
| K3s | Lightweight Kubernetes — see [K3s Setup](#11-k3s-setup) |
| kubectl | `brew install kubectl` — for manual inspection |

### Python dependencies

All dependencies are pinned in `requirements.txt`. Key ones:

```
typer==0.12.5          # CLI framework
rich==13.9.2           # Terminal UI
pydantic==2.9.2        # Data validation / schemas
docker==7.1.0          # Docker SDK for build/push
boto3==1.35.40         # MinIO / S3 client
kubernetes==31.0.0     # K8s Python client
onnx==1.17.0           # ONNX model format
onnxruntime==1.19.2    # ONNX inference + quantisation
torch==2.4.0           # PyTorch (for export)
httpx==0.27.2          # Async HTTP (canary Prometheus queries)
structlog==24.4.0      # Structured logging
```


---

## 5. How to Run — Step by Step

### Step 1 — Clone and install

```bash
git clone <repo-url>
cd Capsule

pip install -r requirements.txt
pip install -e .
```

Verify the CLI is available:

```bash
capsule --help
```

### Step 2 — Start infrastructure

MinIO (model registry), a local Docker registry, Prometheus, and Grafana are all defined in `docker-compose.yml`.

```bash
# Minimum required for packaging and local testing:
docker compose up minio registry -d

# Full observability stack:
docker compose up -d
```

Wait for MinIO to become healthy (takes ~5 seconds):

```bash
docker compose ps
```

All four services should show `healthy` or `running`.

### Step 3 — Run the test suite (no K3s or Docker required)

The test suite mocks all external dependencies (Docker, MinIO, K8s). Run it to confirm your environment is set up correctly:

```bash
cd Capsule
pytest tests/ -v
```

Expected output: 40+ tests passing.

### Step 4 — Train the demo models

The example models are already committed as `.pt` files, so this step is optional. Run it if you want to retrain them:

```bash
# Fraud detector (PyTorch, 2-class MLP)
python examples/fraud_detector/train_model.py

# Sentiment classifier (scikit-learn, LogisticRegression)
python examples/sentiment_classifier/train_model.py
```

### Step 5 — Package a model

```bash
# Package the fraud detector (auto-detects PyTorch, runs ONNX optimisation)
capsule package --manifest examples/fraud_detector/capsule.yaml
```

Expected output:

```
Capsule Package

  Model:      fraud-detector:1.0
  Model path: examples/fraud_detector/fraud_model_module.pt

✓ Packaged fraud-detector:1.0
  Image:     localhost:5001/fraud-detector:1.0
  Framework: onnx
  Size:      4.2 MB → 1.1 MB (-73%)
  Build time: 38.4s
```

To skip ONNX optimisation:

```bash
capsule package --manifest examples/fraud_detector/capsule.yaml --no-onnx
```

To build without pushing to the registry:

```bash
capsule package --manifest examples/fraud_detector/capsule.yaml --no-push
```

### Step 6 — List packaged models

```bash
capsule list
```

### Step 7 — Deploy (requires K3s)

Full traffic deploy:

```bash
capsule deploy fraud-detector:1.0
```

Canary deploy at 10% traffic:

```bash
# First package v2
capsule package --manifest examples/fraud_detector/capsule_v2.yaml

# Deploy v2 as canary (10% of traffic routes to v2, 90% to v1)
capsule deploy fraud-detector:2.0 --canary 10
```

### Step 8 — Check status

```bash
capsule status fraud-detector
```

This shows a Rich panel with:
- Overall status (RUNNING / CANARY / FAILED / ROLLED_BACK)
- Stable version and canary version
- Per-pod table: name, phase, readiness, restart count, age, version
- Last 5 recent events

### Step 9 — Watch a canary (auto-rollback/promote)

```bash
capsule watch fraud-detector \
  --threshold 0.05 \
  --windows 10 \
  --failures 2 \
  --interval 30
```

This starts an async loop that queries Prometheus every 30 seconds. If error rate exceeds 5% for 2 consecutive windows, it triggers an automatic rollback. After 10 clean windows, it signals promotion to full traffic.

### Step 10 — Rollback

```bash
# Interactive confirmation
capsule rollback fraud-detector

# Non-interactive (CI/CD)
capsule rollback fraud-detector --yes
```

### Step 11 — View audit log

```bash
capsule audit
capsule audit --limit 10
```


---

## 6. CLI Reference

All commands share the config path `config/defaults.yaml` which is loaded automatically. Override any value there or via environment variables.

---

### `capsule package`

Package a model: detect framework, build Docker image, optimise, push to registry.

```
capsule package [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--manifest`, `-m` | `capsule.yaml` | Path to the capsule.yaml manifest |
| `--no-onnx` | off | Skip ONNX optimisation and INT8 quantisation |
| `--no-push` | off | Build the Docker image but do not push it or run Docker |

**What it does internally:**

1. Loads and validates the `capsule.yaml` manifest
2. Detects the ML framework from the model file extension (or uses `framework:` override)
3. Runs ONNX optimisation if the framework is PyTorch or ONNX and `--no-onnx` is not set
4. Creates a temporary build directory with: `Dockerfile`, `server.py`, `requirements.txt`, model artifact
5. Calls `docker build` then `docker push`
6. Stores the model artifact in MinIO
7. Records the `image_tag`, `image_digest`, size stats, and build time in SQLite

**Examples:**

```bash
capsule package --manifest examples/fraud_detector/capsule.yaml
capsule package --manifest examples/fraud_detector/capsule.yaml --no-onnx
capsule package -m capsule.yaml --no-push
```

---

### `capsule deploy`

Deploy a packaged model to K3s with optional canary traffic splitting.

```
capsule deploy NAME:VERSION [OPTIONS]
```

| Argument | Description |
|---|---|
| `NAME:VERSION` | Deployment name and version, e.g. `fraud-detector:2.0` |

| Option | Default | Description |
|---|---|---|
| `--canary`, `-c` | `0` | Canary traffic percentage (0 = full traffic to new version) |
| `--manifest`, `-m` | `capsule.yaml` | Manifest path (used for resource/health specs if present) |
| `--namespace`, `-n` | `capsule` | Kubernetes namespace |

**What it does internally:**

1. Looks up the image tag/digest for the given name:version in MinIO
2. Generates a Helm chart (Chart.yaml, values.yaml, deployment.yaml, service.yaml)
3. Runs `helm upgrade --install --atomic --wait --timeout 120s`
4. Waits up to 120s for the rollout to complete
5. Saves the deployment record (status: `running` or `canary`) to SQLite

When `--canary N` is provided, the stable deployment stays active and the Helm chart configures `canary.weight: N`.

**Examples:**

```bash
capsule deploy fraud-detector:1.0
capsule deploy fraud-detector:2.0 --canary 10
capsule deploy fraud-detector:2.0 --canary 25 --namespace production
```

---

### `capsule status`

Show detailed deployment status for a named model.

```
capsule status NAME
```

Output includes:
- A Rich panel showing overall status, stable version, canary version, canary traffic weight
- A table of pods: name, phase, readiness (✓/✗), restart count, age, version
- Up to 5 recent events (combined from K8s events and the SQLite event log)

**Example:**

```bash
capsule status fraud-detector
```

---

### `capsule rollback`

Roll back a deployment to the previous stable version.

```
capsule rollback NAME [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--yes`, `-y` | off | Skip interactive confirmation prompt |

**What it does internally:**

1. Looks up the current deployment version from SQLite
2. Finds the last `RUNNING` version for the same model (skips `CANARY` and `ROLLED_BACK` entries)
3. Resolves the deploy reference — prefers the pinned `sha256:` digest over the mutable image tag
4. Runs `helm rollback <release> --wait --timeout 120s`
5. Saves a new `ROLLED_BACK` deployment record

**Examples:**

```bash
capsule rollback fraud-detector
capsule rollback fraud-detector --yes
```

---

### `capsule list`

List all packaged models in the registry.

```
capsule list [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `NAME` (positional, optional) | — | Filter by model name |
| `--limit`, `-l` | `50` | Maximum rows to show |

Output columns: Name, Version, Framework, Size (MB), ONNX (✓/—), Digest (truncated), Build time (s), Packaged At.

**Examples:**

```bash
capsule list
capsule list fraud-detector
capsule list --limit 10
```

---

### `capsule watch`

Watch a canary deployment and trigger auto-rollback or auto-promote based on Prometheus error rate.

```
capsule watch NAME [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--interval`, `-i` | `30` | Prometheus poll interval in seconds |
| `--threshold`, `-t` | `0.05` | Error rate threshold for rollback (0.0–1.0) |
| `--windows`, `-w` | `10` | Number of consecutive healthy windows before promote |
| `--failures`, `-f` | `2` | Number of consecutive failing windows before rollback |
| `--prometheus`, `-p` | `http://localhost:9090` | Prometheus base URL |

The watch loop runs until Ctrl+C, auto-rollback, or auto-promote. It will not start if the deployment is not currently in canary mode.

**Example:**

```bash
capsule watch fraud-detector --threshold 0.05 --windows 10 --failures 2
```

---

### `capsule audit`

Show the audit log of all package and deploy operations.

```
capsule audit [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--limit`, `-l` | `30` | Number of entries to show |

Output columns: Time, Action, Target, Detail, Actor.


---

## 7. capsule.yaml Reference

Every model needs a `capsule.yaml` manifest. Capsule validates it using Pydantic before any operation.

### Minimal example

```yaml
name: my-model
version: "1.0"
model_path: model.pt
```

### Full example (fraud-detector v1.0)

```yaml
name: fraud-detector
version: "1.0"
framework: pytorch          # optional — auto-detected from model_path extension
model_path: fraud_model_module.pt
python_version: "3.11"
port: 8080

resources:
  cpu_request: "100m"
  cpu_limit: "500m"
  memory_request: "256Mi"
  memory_limit: "1Gi"
  replicas: 2

health:
  path: /health
  port: 8080
  initial_delay_seconds: 15

canary:
  enabled: true
  initial_weight: 10
  auto_promote: true
  error_rate_threshold: 0.05

labels:
  team: "ml-platform"
  domain: "fraud"
```

### Field reference

#### Top-level fields

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `name` | string | ✓ | — | Lowercase DNS label. Letters, digits, hyphens. 2–63 chars. Must match `^[a-z0-9][a-z0-9\-]{0,61}[a-z0-9]$` |
| `version` | string | ✓ | — | Version string, e.g. `"1.0"`, `"2.1.3"` |
| `model_path` | string | ✓ | — | Relative (to manifest) or absolute path to the model file |
| `framework` | string | — | auto-detected | `pytorch`, `onnx`, `sklearn`, or `generic` |
| `python_version` | string | — | `"3.11"` | Python version for the Docker base image |
| `port` | int | — | `8080` | Port the model server listens on (1–65535) |
| `requirements` | list[string] | — | `[]` | Extra pip packages to include in the Docker image |
| `env` | map[string, string] | — | `{}` | Environment variables injected into the container. Do not put secrets here — use K8s Secrets |
| `labels` | map[string, string] | — | `{}` | Kubernetes labels applied to the deployment |

#### `resources` block

| Field | Type | Default | Description |
|---|---|---|---|
| `cpu_request` | string | `"100m"` | Kubernetes CPU request |
| `cpu_limit` | string | `"500m"` | Kubernetes CPU limit |
| `memory_request` | string | `"256Mi"` | Kubernetes memory request |
| `memory_limit` | string | `"1Gi"` | Kubernetes memory limit |
| `replicas` | int | `1` | Number of pod replicas (must be ≥ 1) |

#### `health` block

| Field | Type | Default | Description |
|---|---|---|---|
| `path` | string | `"/health"` | HTTP path for readiness and liveness probes |
| `port` | int | `8080` | Port for health probes |
| `initial_delay_seconds` | int | `10` | Seconds to wait before first probe |
| `period_seconds` | int | `10` | Probe interval |
| `failure_threshold` | int | `3` | Failures before pod is marked unhealthy |

#### `canary` block

| Field | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `false` | Whether canary is configured for this model |
| `initial_weight` | int | `10` | Default canary traffic percentage (0–100) when deploying with `--canary` |
| `auto_promote` | bool | `true` | Signal intent to auto-promote (actual promotion requires `capsule watch`) |
| `error_rate_threshold` | float | `0.05` | Error rate above which a window is counted as a failure (0 < x ≤ 1) |

### Framework detection rules

When `framework` is omitted, Capsule detects it from the file extension:

| Extension | Framework |
|---|---|
| `.pt`, `.pth`, `.torchscript` | `pytorch` |
| `.onnx` | `onnx` |
| `.pkl`, `.pickle`, `.joblib` | `sklearn` |
| anything else | `generic` |

For `.pkl` files, Capsule also tries content-based detection by unpickling the object and checking its module.


---

## 8. How ONNX Optimisation Works

### Why bother

ONNX optimisation reduces model artifact size (typically 30–75%), speeds up inference via graph-level operator fusion, and enables INT8 weight quantisation which cuts memory bandwidth during serving.

### The pipeline

```
model.pt / model.pth
      │
      ▼  (1) Load model
   torch.load()  ← tries state_dict checkpoint first
   torch.jit.load()  ← TorchScript fallback
      │
      ▼  (2) Probe input shape
   try dummy inputs: size 10, 32, 64, 128, 256
      │
      ▼  (3) Export to ONNX (opset 17)
   torch.onnx.export(model, dummy_input, raw.onnx)
   onnx.checker.check_model()  ← validates graph
      │
      ▼  (4) INT8 quantisation
   quantize_dynamic(raw.onnx, optimised.onnx, weight_type=QInt8)
      │
      ▼  (5) Compute stats
   size_reduction = (1 - opt_size / orig_size) * 100
      │
      ▼
   optimised.onnx  →  used for Docker build
   raw.onnx  →  deleted
```

### For existing ONNX models

If the model is already in ONNX format, Capsule runs only step 4 (quantisation). Steps 1–3 are skipped.

### Fallback behaviour

If ONNX conversion fails for any reason (unsupported ops, shape inference failure, etc.), Capsule falls back to the original model file. The `--no-onnx` flag bypasses the entire pipeline.

### Stats reported

After optimisation, Capsule reports:

- `original_size_mb` — original model file size
- `optimised_size_mb` — post-quantisation size
- `size_reduction_pct` — percentage reduction
- `quantised` — whether INT8 quantisation was applied

These are stored in SQLite and shown in `capsule list`.

### Generated ONNX server

When a PyTorch model is successfully converted to ONNX, the deployed server uses the ONNX Runtime instead of PyTorch, which has lower memory and CPU overhead:

```python
providers = [p for p in ["CoreMLExecutionProvider", "CPUExecutionProvider"]
             if p in ort.get_available_providers()]
session = ort.InferenceSession(MODEL_PATH, providers=providers)
```

On Apple Silicon, it automatically uses CoreML when available.

### Disabling per-model

Add `--no-onnx` flag at package time, or set `onnx.enabled: false` in `config/defaults.yaml` to disable globally.


---

## 9. How Canary Deployment Works

### What canary means in Capsule

When you deploy with `--canary N`, the stable version keeps serving `(100 - N)%` of traffic while the new version handles `N%`. The `CanaryController` monitors the new version's error rate and decides whether to promote or roll back automatically.

### Traffic split

The Helm chart sets `canary.weight: N` in `values.yaml`. Traefik (K3s's default ingress) routes traffic based on this weight.

### The monitor loop

`capsule watch` starts an async `CanaryController` that runs the following loop every `--interval` seconds (default 30s):

```
┌─ every interval ─────────────────────────────────────────────┐
│                                                               │
│  query Prometheus:                                            │
│    sum(rate(model_requests_total{job="capsule-NAME",          │
│             status="error"}[5m]))                            │
│    / sum(rate(model_requests_total{job="capsule-NAME"}[5m]))  │
│                                                               │
│  if result is NaN or no data → skip (no traffic yet)          │
│  if result is None (Prometheus unreachable)                   │
│    → fallback: GET /health → 0.0 if 200, 1.0 otherwise        │
│                                                               │
│  if error_rate > threshold:                                   │
│    failure_count += 1                                         │
│    success_count = 0                                          │
│    if failure_count >= consecutive_failures:                  │
│      → AUTO-ROLLBACK                                          │
│  else:                                                        │
│    failure_count = 0                                          │
│    success_count += 1                                         │
│    if success_count >= auto_promote_windows:                  │
│      → AUTO-PROMOTE                                           │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

### Default thresholds

| Parameter | Default | CLI flag |
|---|---|---|
| Error rate threshold | 5% | `--threshold 0.05` |
| Failures before rollback | 2 windows | `--failures 2` |
| Clean windows before promote | 10 windows | `--windows 10` |
| Poll interval | 30s | `--interval 30` |

With defaults: a canary that produces >5% errors for 60 consecutive seconds triggers rollback. A canary that stays under 5% for 5 minutes promotes automatically.

### Auto-rollback

When triggered, the controller:
1. Logs a `rollback` event to the SQLite events table
2. Calls the injected `auto_rollback_fn` (which calls `deployer.rollback()`)
3. Stops the monitor loop

### Auto-promote

When triggered, the controller:
1. Logs a `promote` event to SQLite
2. Calls the injected `auto_promote_fn` (which prints the command to re-deploy at 100%)
3. Stops the monitor loop

> Note: auto-promote does not automatically re-deploy at 100%. It signals the intent and prints the command. This is intentional — production promotion should be explicit.

### Prometheus query scope

The query is scoped to a specific job label (`job="capsule-NAME"`) to avoid aggregating error rates across multiple models sharing the same namespace.

### Resilience features

- **No-traffic guard**: if Prometheus returns NaN or empty results (the canary hasn't received any requests yet), the window is skipped rather than counted as a failure
- **Monitor error isolation**: if the Prometheus query itself throws 3 consecutive exceptions, the loop marks itself as degraded and stops rather than silently looping
- **Health fallback**: if Prometheus is unreachable, the controller falls back to checking the `/health` endpoint directly

### Manual workflow (without `capsule watch`)

```bash
# 1. Deploy canary
capsule deploy fraud-detector:2.0 --canary 10

# 2. Monitor manually
capsule status fraud-detector

# 3. Rollback if needed
capsule rollback fraud-detector --yes

# 4. Or promote manually (re-deploy at full traffic)
capsule deploy fraud-detector:2.0
```


---

## 10. Generated Model Server

Each packaged model gets an auto-generated FastAPI server (`server.py`) embedded in the Docker image. The server is framework-specific.

### Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Readiness/liveness probe. Returns `503` if model not yet loaded |
| `/predict` | POST | Run inference. Returns `{"output": [...], "latency_ms": 12.3}` |
| `/metrics` | GET | Prometheus metrics in text format |

### Predict request format

```json
{
  "inputs": [[1.2, 3.4, 5.6], [7.8, 9.0, 1.2]]
}
```

- `inputs` must be a list (batch of rows)
- Maximum batch size: 1024 rows (configurable via `MAX_BATCH` env var)
- Maximum request body: 10 MB (configurable via `MAX_BODY_BYTES` env var)

### Predict response format

```json
{
  "output": [[0.95], [0.12]],
  "latency_ms": 3.21
}
```

### Prometheus metrics

Each server emits two metrics:

```
# HELP model_requests_total Total requests
# TYPE model_requests_total counter
model_requests_total{status="ok"} 1234
model_requests_total{status="error"} 5

# HELP model_request_duration_seconds Request latency
# TYPE model_request_duration_seconds histogram
model_request_duration_seconds_bucket{le="0.005"} 800
...
```

These are what `capsule watch` queries via Prometheus.

### Security features in generated servers

- Request body size cap (413 before parsing)
- Batch size cap (422 for oversized batches)
- Input type validation (`ValueError`/`TypeError` → 422)
- Graceful SIGTERM/SIGINT handling (marks server unhealthy before exit)
- Non-root user in Docker (`uid 1000`, `capsule:capsule`)
- scikit-learn server verifies model file SHA-256 digest at startup if `MODEL_DIGEST` env var is set

### Framework-specific server behaviour

| Framework | Model load | Inference call |
|---|---|---|
| ONNX | `ort.InferenceSession()`, prefers CoreML on Apple Silicon | `session.run(None, {input_name: np_array})` |
| PyTorch | `torch.jit.load()`, `model.eval()` | `torch.no_grad(); model(tensor)` |
| scikit-learn | `pickle.load()`, optional digest check | `model.predict(np_array)` |


---

## 11. K3s Setup

K3s is only needed if you want to actually deploy to Kubernetes. The test suite and `capsule package` work without it.

### Install K3s (macOS via Multipass or Lima)

K3s runs natively on Linux. On macOS, use a lightweight VM:

```bash
# Option A: Lima (recommended for macOS)
brew install lima
limactl start --name=k3s template://k3s
export KUBECONFIG=$(limactl shell k3s -- cat /etc/rancher/k3s/k3s.yaml | sed "s/127.0.0.1/$(limactl list k3s --format '{{.IP}}')/g" > /tmp/k3s.yaml && echo /tmp/k3s.yaml)

# Option B: On a Linux machine or VM directly
curl -sfL https://get.k3s.io | sh -
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
sudo chmod 644 /etc/rancher/k3s/k3s.yaml
```

### Verify cluster access

```bash
kubectl get nodes
kubectl cluster-info
```

### Create the capsule namespace

Capsule creates the namespace automatically via `kubectl.ensure_namespace()` on first deploy, but you can do it manually:

```bash
kubectl create namespace capsule
```

### Configure the Docker registry as insecure

K3s needs to pull images from `localhost:5001` (the local registry started by docker-compose). Add a registries config:

```bash
# On the K3s node:
sudo mkdir -p /etc/rancher/k3s
cat <<EOF | sudo tee /etc/rancher/k3s/registries.yaml
mirrors:
  "localhost:5001":
    endpoint:
      - "http://localhost:5001"
EOF
sudo systemctl restart k3s
```

### Verify Helm can reach K3s

```bash
helm version
helm list --namespace capsule
```

### Helm release naming

Capsule uses the pattern `capsule-{model-name}` as the Helm release name. So `fraud-detector` becomes the release `capsule-fraud-detector` in namespace `capsule`.

To inspect a deployed release manually:

```bash
helm status capsule-fraud-detector --namespace capsule
helm history capsule-fraud-detector --namespace capsule
kubectl get pods --namespace capsule -l app=capsule-fraud-detector
```


---

## 12. Configuration Reference

All defaults live in `config/defaults.yaml`. The CLI loads this file automatically at startup. Override values by editing the file directly.

```yaml
# config/defaults.yaml

registry:
  endpoint: "http://localhost:9000"     # MinIO API endpoint
  access_key: "minioadmin"              # MinIO access key
  secret_key: "minioadmin"              # MinIO secret key
  bucket: "capsule-models"             # S3 bucket for model artifacts
  secure: false                         # Use HTTPS? (false for local dev)

k8s:
  namespace: "capsule"                  # Kubernetes namespace for all deployments
  context: "default"                    # kubectl context name
  kubeconfig: null                      # Explicit kubeconfig path; null = ~/.kube/config

docker:
  registry: "localhost:5001"            # Docker registry to push images to
  platform: "linux/arm64"              # Build platform (arm64 for M-series Macs)
  build_timeout: 300                    # Docker build timeout in seconds

helm:
  chart_dir: "/tmp/capsule-charts"      # Where generated Helm charts are written
  release_prefix: "capsule"            # Helm release name prefix

canary:
  initial_weight: 10                    # Default canary % if not set in capsule.yaml
  monitor_interval_seconds: 30          # How often the canary watcher polls Prometheus
  error_rate_threshold: 0.05            # Error rate above which a window is bad
  consecutive_failures: 2               # Bad windows before auto-rollback
  auto_promote_windows: 10              # Good windows before auto-promote

onnx:
  enabled: true                         # Enable ONNX optimisation by default
  quantization: true                    # Enable INT8 quantisation
  validation_tolerance: 0.01            # Max allowed output delta after quantisation
  max_model_size_mb: 2000               # Models above this size skip ONNX (not yet enforced)

server:
  port: 8080                            # Default server port
  workers: 1                            # Uvicorn workers
  health_path: "/health"
  predict_path: "/predict"
  metrics_path: "/metrics"

prometheus:
  enabled: true
  scrape_port: 8080                     # Port Prometheus scrapes for /metrics
```

### SQLite database location

The manifest store writes to `~/.capsule/capsule.db` by default. Override with the `CAPSULE_DB_PATH` environment variable:

```bash
export CAPSULE_DB_PATH=/path/to/custom/capsule.db
```

### Environment variables

| Variable | Description |
|---|---|
| `CAPSULE_DB_PATH` | Override SQLite database path |
| `MAX_BODY_BYTES` | Override max request body in generated servers (default 10485760) |
| `MAX_BATCH` | Override max batch size in generated servers (default 1024) |
| `MODEL_PATH` | Path to model file inside container |
| `FRAMEWORK` | Framework name inside container |
| `PORT` | Server port inside container |
| `MODEL_DIGEST` | SHA-256 digest for model integrity check (scikit-learn servers) |


---

## 13. Monitoring

### Start the observability stack

```bash
docker compose up prometheus grafana -d
```

### Prometheus

URL: http://localhost:9090

Prometheus scrapes `/metrics` from each deployed model server. The scrape configuration is in `prometheus.yml`.

Useful queries:

```promql
# Request rate for fraud-detector
rate(model_requests_total{job="capsule-fraud-detector"}[5m])

# Error rate
sum(rate(model_requests_total{job="capsule-fraud-detector", status="error"}[5m]))
/ sum(rate(model_requests_total{job="capsule-fraud-detector"}[5m]))

# p99 latency
histogram_quantile(0.99, rate(model_request_duration_seconds_bucket{job="capsule-fraud-detector"}[5m]))

# Total requests
sum(model_requests_total{job="capsule-fraud-detector"})
```

### Grafana

URL: http://localhost:3000
Credentials: `admin` / `capsule`

A pre-built dashboard is included at `dashboards/capsule.json`. Import it via:

1. Open Grafana → Dashboards → Import
2. Upload `dashboards/capsule.json`
3. Select the Prometheus datasource

The dashboard shows request rate, error rate, latency percentiles, and canary traffic split.

### MinIO Console

URL: http://localhost:9001
Credentials: `minioadmin` / `minioadmin`

Browse model artifacts stored under `s3://capsule-models/models/{name}/{version}/`.


---

## 14. Port Reference

| Service | Port | URL | Credentials |
|---|---|---|---|
| MinIO API | 9000 | http://localhost:9000 | minioadmin / minioadmin |
| MinIO Console | 9001 | http://localhost:9001 | minioadmin / minioadmin |
| Docker Registry | 5001 | http://localhost:5001 | none |
| Prometheus | 9090 | http://localhost:9090 | none |
| Grafana | 3000 | http://localhost:3000 | admin / capsule |
| Model server (default) | 8080 | http://localhost:8080 | none |

Port 5001 maps to container port 5000 (the Docker Registry standard port).

---

## 15. Running Tests

The test suite uses pytest with mocks for all external dependencies. No running Docker, MinIO, or K3s instance is required.

```bash
cd Capsule
pytest tests/ -v
```

Run a specific test file:

```bash
pytest tests/test_canary.py -v
pytest tests/test_onnx_optimizer.py -v
```

Run a specific test:

```bash
pytest tests/test_deployer.py::test_deploy_success -v
```

### Test files

| File | What it covers |
|---|---|
| `test_detector.py` | Framework detection from extensions and content |
| `test_packager.py` | Dockerfile generation, ONNX pipeline, build/push flow |
| `test_onnx_optimizer.py` | PyTorch→ONNX export, quantisation, size stats |
| `test_registry.py` | MinIO push/pull for models, tags, digests |
| `test_deployer.py` | Helm upgrade/rollback, rollout wait, status aggregation |
| `test_canary.py` | Prometheus polling, failure counting, auto-rollback/promote |
| `test_manifest.py` | capsule.yaml loading, SQLite CRUD, audit log |
| `test_integration.py` | End-to-end package → deploy → rollback flow |

### Async tests

`pytest-asyncio` is configured with `asyncio_mode = "auto"` in `pyproject.toml`. All `async def test_*` functions run automatically in an asyncio event loop.


---

## 16. Troubleshooting

### `capsule: command not found`

The package is not installed. Run:

```bash
pip install -e .
```

Check that your Python environment's bin is on PATH:

```bash
which capsule
python -m capsule --help  # alternative
```

---

### `Model not found: examples/fraud_detector/fraud_model_module.pt`

The model file is resolved relative to the manifest location. Either:
- Run `capsule package` from the project root, or
- Use an absolute path in `model_path`, or
- Pass `--manifest` with the full path to the `capsule.yaml`

```bash
# From repo root
capsule package --manifest examples/fraud_detector/capsule.yaml
```

---

### `No image found for fraud-detector:1.0`

This means `capsule package` either wasn't run or the image wasn't pushed. Check:

```bash
capsule list                    # is the package recorded?
curl http://localhost:5001/v2/fraud-detector/tags/list  # is the image in the registry?
```

If the registry is down:

```bash
docker compose up registry -d
capsule package --manifest examples/fraud_detector/capsule.yaml
```

---

### `helm upgrade failed` / `helm binary not on PATH`

Install Helm:

```bash
brew install helm
helm version
```

If Helm is installed but `capsule deploy` still warns `helm binary not on PATH`, ensure your shell PATH includes the Homebrew bin directory:

```bash
export PATH="/opt/homebrew/bin:$PATH"
```

---

### Docker build fails: `platform linux/arm64`

The Packager builds for `linux/arm64` by default (M-series Macs). If you're on x86 or targeting a different architecture, update `config/defaults.yaml`:

```yaml
docker:
  platform: "linux/amd64"
```

---

### ONNX optimisation fails silently

Check structured logs for `onnx_optimizer.load_failed` or `onnx_optimizer.export_failed`. Common causes:

- Model uses ops not supported by `torch.onnx.export` — use `--no-onnx`
- Model requires a fixed input shape that can't be probed automatically — add a custom `dummy_input` to your training script and export ONNX manually before packaging
- `onnxruntime` quantisation doesn't support the operator set — Capsule falls back to the unquantised ONNX model

---

### `capsule status` shows `PENDING` after deploy

The K8s rollout may still be in progress, or the pod is failing its health probe. Check:

```bash
kubectl get pods --namespace capsule
kubectl describe pod <pod-name> --namespace capsule
kubectl logs <pod-name> --namespace capsule
```

Common causes:
- Wrong `initial_delay_seconds` — increase it in `capsule.yaml`
- Image pull failure — verify the registry is running and the image tag exists
- OOMKill — increase `memory_limit` in `resources`

---

### `capsule rollback` fails: `No previous RUNNING version found`

Rollback targets the last deployment with `status = RUNNING`. If the previous deploy also failed or was a canary, there's no safe target. You can:

1. Deploy a known-good version manually: `capsule deploy name:good-version`
2. Use `helm rollback capsule-{name} --namespace capsule` directly to roll Helm back by revision number

---

### `capsule watch` exits immediately with `not in canary mode`

The deployment must be in canary mode (deployed with `--canary N > 0`) before you run `capsule watch`. Deploy with a canary weight first:

```bash
capsule deploy fraud-detector:2.0 --canary 10
capsule watch fraud-detector
```

---

### MinIO connection refused

```bash
docker compose up minio -d
docker compose ps minio
```

If the container is running but connection fails, wait for the health check:

```bash
docker compose logs minio
```

---

### SQLite database locked

Capsule uses WAL journal mode with a 5s busy timeout. If you see `database is locked`, it likely means another `capsule` process is running concurrently. Check:

```bash
ps aux | grep capsule
```

Or override the DB path to use a fresh database:

```bash
export CAPSULE_DB_PATH=/tmp/capsule-test.db
```

---

### Tests fail with `ImportError: torch`

The test suite requires PyTorch. Install from `requirements.txt`:

```bash
pip install -r requirements.txt
```

If you want a lighter test environment (no torch), the tests that require it are isolated in `test_onnx_optimizer.py` and can be skipped:

```bash
pytest tests/ -v --ignore=tests/test_onnx_optimizer.py
```

---

*Capsule — built for ML teams who want Kubernetes-grade deployments without the boilerplate.*
