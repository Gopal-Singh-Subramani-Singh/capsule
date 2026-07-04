from __future__ import annotations
import pytest
from unittest.mock import MagicMock, patch
from capsule.models import DeploymentStatus


def test_deployer_init(mock_registry, tmp_store, mock_k8s):
    from capsule.deployer import Deployer
    from capsule.helm import HelmChartGenerator
    helm = HelmChartGenerator("/tmp/test-charts")
    d = Deployer(
        registry=mock_registry, store=tmp_store,
        k8s=mock_k8s, helm_gen=helm,
    )
    assert d._namespace == "capsule"


def test_deploy_creates_record(
    mock_registry, tmp_store, mock_k8s, sample_manifest
):
    from capsule.deployer import Deployer
    from capsule.helm import HelmChartGenerator

    helm = MagicMock()
    helm.generate.return_value = "/tmp/fake-chart"
    d = Deployer(
        registry=mock_registry, store=tmp_store,
        k8s=mock_k8s, helm_gen=helm,
    )

    with patch.object(d, "_helm_upgrade"):
        record = d.deploy(
            sample_manifest,
            image_tag="localhost:5001/test-model:1.0",
            canary_weight=0,
        )

    assert record.name == "test-model"
    assert record.version == "1.0"
    assert record.status == DeploymentStatus.RUNNING


def test_deploy_canary_sets_status(
    mock_registry, tmp_store, mock_k8s, sample_manifest
):
    from capsule.deployer import Deployer
    from capsule.helm import HelmChartGenerator

    helm = MagicMock()
    helm.generate.return_value = "/tmp/fake-chart"
    d = Deployer(
        registry=mock_registry, store=tmp_store,
        k8s=mock_k8s, helm_gen=helm,
    )

    with patch.object(d, "_helm_upgrade"):
        record = d.deploy(
            sample_manifest,
            image_tag="localhost:5001/test-model:2.0",
            canary_weight=10,
        )
    assert record.status == DeploymentStatus.CANARY
    assert record.canary_weight == 10


def test_rollback_no_previous_raises(mock_registry, tmp_store, mock_k8s):
    from capsule.deployer import Deployer
    helm = MagicMock()
    d = Deployer(
        registry=mock_registry, store=tmp_store,
        k8s=mock_k8s, helm_gen=helm,
    )
    with pytest.raises(ValueError, match="No deployment"):
        d.rollback("nonexistent-model")
