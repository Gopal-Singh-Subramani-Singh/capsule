from __future__ import annotations
import pytest
import tempfile
import os
from capsule.models import Framework, DeploymentStatus


def test_save_and_load_package(tmp_store, sample_package_result):
    tmp_store.save_package(sample_package_result)
    packages = tmp_store.list_packages("test-model")
    assert len(packages) == 1
    assert packages[0]["name"] == "test-model"


def test_save_and_get_deployment(tmp_store, sample_deployment):
    tmp_store.save_deployment(sample_deployment)
    latest = tmp_store.get_latest_deployment("test-model")
    assert latest is not None
    assert latest.version == "1.0"
    assert latest.status == DeploymentStatus.RUNNING


def test_get_previous_version(tmp_store):
    from capsule.models import DeploymentRecord, DeploymentStatus
    r1 = DeploymentRecord(
        name="m", version="1.0",
        image_tag="img:1.0", namespace="capsule",
        status=DeploymentStatus.RUNNING,
    )
    r2 = DeploymentRecord(
        name="m", version="2.0",
        image_tag="img:2.0", namespace="capsule",
        status=DeploymentStatus.RUNNING,
    )
    import time
    tmp_store.save_deployment(r1)
    time.sleep(0.01)
    tmp_store.save_deployment(r2)
    prev = tmp_store.get_previous_version("m", "2.0")
    assert prev == "1.0"


def test_no_previous_version(tmp_store, sample_deployment):
    tmp_store.save_deployment(sample_deployment)
    prev = tmp_store.get_previous_version("test-model", "1.0")
    assert prev is None


def test_log_and_get_events(tmp_store):
    from capsule.models import CanaryEvent
    event = CanaryEvent(
        deployment_name="test-model",
        event_type="check",
        canary_weight=10,
        error_rate=0.02,
        message="All good",
    )
    tmp_store.log_event(event)
    events = tmp_store.get_events("test-model")
    assert len(events) == 1
    assert events[0]["event_type"] == "check"


def test_load_manifest_from_file(tmp_path):
    import yaml
    from capsule.manifest import load_manifest
    manifest_data = {
        "name": "test", "version": "1.0", "model_path": "model.pt"
    }
    p = tmp_path / "capsule.yaml"
    p.write_text(yaml.dump(manifest_data))
    manifest = load_manifest(str(p))
    assert manifest.name == "test"
    assert manifest.version == "1.0"


def test_load_manifest_missing_raises():
    from capsule.manifest import load_manifest
    with pytest.raises(FileNotFoundError):
        load_manifest("/nonexistent/capsule.yaml")
