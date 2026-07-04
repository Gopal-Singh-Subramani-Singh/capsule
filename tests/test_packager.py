from __future__ import annotations
import pytest
from unittest.mock import MagicMock, patch
from capsule.models import Framework


def test_packager_init(mock_registry, tmp_store):
    from capsule.packager import Packager
    p = Packager(
        registry=mock_registry,
        store=tmp_store,
        docker_registry="localhost:5001",
        onnx_enabled=True,
    )
    assert p._onnx_enabled is True
    assert p._docker_registry == "localhost:5001"


def test_dockerfile_template_renders(sample_manifest):
    from capsule.packager import _DOCKERFILE_TEMPLATE
    from jinja2 import Environment, BaseLoader
    env = Environment(loader=BaseLoader())
    result = env.from_string(_DOCKERFILE_TEMPLATE).render(
        base_image="python:3.11-slim",
        model_filename="model.pt",
        framework="pytorch",
        port=8080,
    )
    assert "FROM python:3.11-slim" in result
    assert "model.pt" in result
    assert "EXPOSE 8080" in result


def test_package_fails_missing_model(mock_registry, tmp_store, sample_manifest):
    from capsule.packager import Packager
    sample_manifest.model_path = "/absolutely/nonexistent/model.pt"
    p = Packager(registry=mock_registry, store=tmp_store, onnx_enabled=False)
    with pytest.raises(FileNotFoundError):
        p.package(sample_manifest)


def test_server_code_generated_pytorch(sample_manifest):
    from capsule.server_template import generate_server_code
    from capsule.models import Framework
    code = generate_server_code(Framework.PYTORCH, sample_manifest)
    assert "FastAPI" in code
    assert "torch" in code
    assert "/health" in code
    assert "/predict" in code
    assert "/metrics" in code


def test_server_code_generated_onnx(onnx_manifest):
    from capsule.server_template import generate_server_code
    from capsule.models import Framework
    code = generate_server_code(Framework.ONNX, onnx_manifest)
    assert "onnxruntime" in code
    assert "/predict" in code


def test_server_code_generated_sklearn(sklearn_manifest):
    from capsule.server_template import generate_server_code
    from capsule.models import Framework
    code = generate_server_code(Framework.SKLEARN, sklearn_manifest)
    assert "pickle" in code or "joblib" in code
    assert "/predict" in code
