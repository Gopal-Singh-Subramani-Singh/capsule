from __future__ import annotations
import pytest
import os
import tempfile


def test_full_manifest_parse_and_validate():
    import yaml
    import tempfile
    from capsule.manifest import load_manifest

    manifest_yaml = """
name: integration-test
version: "3.0"
model_path: /tmp/model.pt
port: 8080
resources:
  cpu_request: "200m"
  memory_request: "512Mi"
canary:
  enabled: true
  initial_weight: 20
"""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        f.write(manifest_yaml)
        tmp_path = f.name

    try:
        manifest = load_manifest(tmp_path)
        assert manifest.name == "integration-test"
        assert manifest.version == "3.0"
        assert manifest.resources.cpu_request == "200m"
        assert manifest.canary.initial_weight == 20
    finally:
        os.unlink(tmp_path)


def test_framework_detection_and_server_generation():
    from capsule.detector import detect_framework
    from capsule.server_template import generate_server_code
    from capsule.models import CapsuleManifest, Framework

    for path, expected_fw in [
        ("model.pt", Framework.PYTORCH),
        ("model.onnx", Framework.ONNX),
        ("model.pkl", Framework.SKLEARN),
    ]:
        fw = detect_framework(path)
        assert fw == expected_fw

        manifest = CapsuleManifest(
            name="test", version="1.0", model_path=path
        )
        code = generate_server_code(fw, manifest)
        assert "/health" in code
        assert "/predict" in code
        assert "/metrics" in code


def test_store_full_lifecycle(tmp_store, sample_package_result, sample_deployment):
    # Package
    tmp_store.save_package(sample_package_result)
    packages = tmp_store.list_packages("test-model")
    assert len(packages) == 1

    # Deploy v1
    tmp_store.save_deployment(sample_deployment)
    latest = tmp_store.get_latest_deployment("test-model")
    assert latest.version == "1.0"

    # Deploy v2
    import time
    time.sleep(0.01)
    from capsule.models import DeploymentRecord, DeploymentStatus
    v2 = DeploymentRecord(
        name="test-model", version="2.0",
        image_tag="img:2.0", namespace="capsule",
        status=DeploymentStatus.RUNNING,
    )
    tmp_store.save_deployment(v2)
    prev = tmp_store.get_previous_version("test-model", "2.0")
    assert prev == "1.0"


def test_helm_chart_renders_correctly():
    from capsule.helm import HelmChartGenerator
    from capsule.models import CapsuleManifest
    import tempfile

    manifest = CapsuleManifest(
        name="mymodel", version="1.0", model_path="m.pt", port=8080
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        gen = HelmChartGenerator(chart_dir=tmpdir)
        chart_path = gen.generate(
            manifest,
            image_tag="localhost:5001/mymodel:1.0",
            canary_weight=0,
        )
        import os
        assert os.path.exists(os.path.join(chart_path, "Chart.yaml"))
        assert os.path.exists(os.path.join(chart_path, "values.yaml"))
        assert os.path.exists(
            os.path.join(chart_path, "templates", "deployment.yaml")
        )
