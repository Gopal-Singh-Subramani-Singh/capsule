from __future__ import annotations
import os
import tempfile
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from capsule.models import (
    CapsuleManifest, PackageResult, DeploymentRecord,
    DeploymentStatus, Framework, ResourceSpec
)


@pytest.fixture
def sample_manifest():
    return CapsuleManifest(
        name="test-model",
        version="1.0",
        framework=Framework.PYTORCH,
        model_path="/tmp/test_model.pt",
        port=8080,
        resources=ResourceSpec(replicas=1),
    )


@pytest.fixture
def onnx_manifest():
    return CapsuleManifest(
        name="onnx-model",
        version="2.0",
        framework=Framework.ONNX,
        model_path="/tmp/test_model.onnx",
        port=8080,
    )


@pytest.fixture
def sklearn_manifest():
    return CapsuleManifest(
        name="sklearn-model",
        version="1.5",
        framework=Framework.SKLEARN,
        model_path="/tmp/model.pkl",
        port=8080,
    )


@pytest.fixture
def sample_package_result():
    return PackageResult(
        name="test-model",
        version="1.0",
        framework=Framework.PYTORCH,
        image_tag="localhost:5001/test-model:1.0",
        image_digest="sha256:abc123",
        registry_path="s3://capsule-models/models/test-model/1.0/",
        original_size_mb=50.0,
        optimised_size_mb=30.0,
        onnx_optimised=True,
        size_reduction_pct=40.0,
        build_seconds=45.0,
    )


@pytest.fixture
def sample_deployment():
    return DeploymentRecord(
        name="test-model",
        version="1.0",
        image_tag="localhost:5001/test-model:1.0",
        image_digest="sha256:abc123",
        namespace="capsule",
        status=DeploymentStatus.RUNNING,
        canary_weight=0,
    )


@pytest.fixture
def tmp_store(tmp_path):
    from capsule.manifest import ManifestStore
    return ManifestStore(db_path=str(tmp_path / "test.db"))


@pytest.fixture
def mock_registry():
    registry = MagicMock()
    registry.push_model.return_value = "models/test-model/1.0/model.pt"
    registry.push_image_tag.return_value = None
    registry.push_image_digest.return_value = None
    registry.get_image_tag.return_value = "localhost:5001/test-model:1.0"
    registry.get_image_digest.return_value = "sha256:abc123"
    registry.get_deploy_ref.return_value = "localhost:5001/test-model@sha256:abc123"
    registry.model_exists.return_value = True
    registry.list_versions.return_value = ["1.0", "2.0"]
    return registry


@pytest.fixture
def mock_k8s():
    k8s = MagicMock()
    k8s.available = False
    k8s.ensure_namespace.return_value = None
    k8s.get_deployment.return_value = {"replicas": 1, "ready_replicas": 1}
    k8s.get_pods.return_value = []
    k8s.wait_for_rollout.return_value = True
    k8s.get_events.return_value = []
    return k8s
