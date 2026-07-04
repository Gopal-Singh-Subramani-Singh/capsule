from __future__ import annotations
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_canary_starts_and_stops(tmp_store, mock_k8s):
    from capsule.canary import CanaryController
    ctrl = CanaryController(
        store=tmp_store,
        k8s=mock_k8s,
        deployment_name="test",
        canary_service_url="http://localhost:8080",
        monitor_interval_seconds=999,
    )
    await ctrl.start()
    assert ctrl._running is True
    await ctrl.stop()
    assert ctrl._running is False


@pytest.mark.asyncio
async def test_canary_evaluate_healthy(tmp_store, mock_k8s, sample_deployment):
    from capsule.canary import CanaryController
    tmp_store.save_deployment(sample_deployment)
    ctrl = CanaryController(
        store=tmp_store, k8s=mock_k8s,
        deployment_name="test-model",
        canary_service_url="http://localhost:8080",
        error_rate_threshold=0.05,
    )
    await ctrl._evaluate(0.01)
    assert ctrl._failure_count == 0
    assert ctrl._success_count == 1


@pytest.mark.asyncio
async def test_canary_evaluate_degraded(tmp_store, mock_k8s, sample_deployment):
    from capsule.canary import CanaryController
    tmp_store.save_deployment(sample_deployment)
    ctrl = CanaryController(
        store=tmp_store, k8s=mock_k8s,
        deployment_name="test-model",
        canary_service_url="http://localhost:8080",
        error_rate_threshold=0.05,
        consecutive_failures=2,
    )
    await ctrl._evaluate(0.10)
    assert ctrl._failure_count == 1
    assert ctrl._success_count == 0


@pytest.mark.asyncio
async def test_canary_auto_rollback_on_consecutive_failures(
    tmp_store, mock_k8s, sample_deployment
):
    from capsule.canary import CanaryController
    tmp_store.save_deployment(sample_deployment)
    ctrl = CanaryController(
        store=tmp_store, k8s=mock_k8s,
        deployment_name="test-model",
        canary_service_url="http://localhost:8080",
        error_rate_threshold=0.05,
        consecutive_failures=2,
    )
    await ctrl._evaluate(0.20)
    await ctrl._evaluate(0.20)
    assert ctrl._running is False  # stopped after auto-rollback


@pytest.mark.asyncio
async def test_canary_health_check_fallback(tmp_store, mock_k8s):
    from capsule.canary import CanaryController
    ctrl = CanaryController(
        store=tmp_store, k8s=mock_k8s,
        deployment_name="test",
        canary_service_url="http://localhost:99999",
    )
    rate = await ctrl._get_error_rate()
    assert isinstance(rate, float)
    assert 0.0 <= rate <= 1.0
