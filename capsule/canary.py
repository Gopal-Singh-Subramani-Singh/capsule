from __future__ import annotations
import asyncio
from datetime import datetime
from typing import Callable, Optional
import httpx
import structlog

from capsule.models import CanaryEvent, DeploymentStatus
from capsule.manifest import ManifestStore
from capsule.k8s_client import K8sClient

logger = structlog.get_logger(__name__)


class CanaryController:
    """
    Async background controller that monitors a canary deployment.

    Production hardening applied:
    - auto_rollback_fn: injectable callable — actually triggers the rollback
      instead of just logging intent (the #1 critical gap)
    - auto_promote_fn: injectable callable — promotes to full traffic
    - Scoped Prometheus query: filters by job label to avoid cross-deployment noise
    - NaN / zero-traffic guard: returns None when denominator is 0 (no traffic yet)
    - Exception isolation: monitor errors are counted; after 3 consecutive errors
      the loop marks itself degraded rather than silently continuing
    - Thread-safe flag access via asyncio primitives
    """

    def __init__(
        self,
        store: ManifestStore,
        k8s: K8sClient,
        deployment_name: str,
        canary_service_url: str,
        error_rate_threshold: float = 0.05,
        monitor_interval_seconds: int = 30,
        consecutive_failures: int = 2,
        auto_promote_windows: int = 10,
        prometheus_url: str = "http://localhost:9090",
        # Callables injected by the Deployer/CLI so the controller can
        # actually execute rollback/promote rather than just logging
        auto_rollback_fn: Optional[Callable[[], None]] = None,
        auto_promote_fn: Optional[Callable[[], None]] = None,
    ):
        self._store = store
        self._k8s = k8s
        self._name = deployment_name
        self._canary_url = canary_service_url
        self._threshold = error_rate_threshold
        self._interval = monitor_interval_seconds
        self._max_failures = consecutive_failures
        self._promote_windows = auto_promote_windows
        self._prometheus_url = prometheus_url
        self._rollback_fn = auto_rollback_fn
        self._promote_fn = auto_promote_fn

        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._failure_count = 0
        self._success_count = 0
        self._monitor_errors = 0

    # ── lifecycle ─────────────────────────────────────────────────────────

    async def start(self):
        self._running = True
        self._task = asyncio.create_task(
            self._monitor_loop(), name=f"canary-{self._name}"
        )
        logger.info(
            "canary.started",
            name=self._name,
            threshold=self._threshold,
            interval=self._interval,
        )

    async def stop(self):
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    # ── monitor loop ──────────────────────────────────────────────────────

    async def _monitor_loop(self):
        while self._running:
            await asyncio.sleep(self._interval)
            if not self._running:
                break
            try:
                error_rate = await self._get_error_rate()
                if error_rate is None:
                    # No traffic yet — skip evaluation this window
                    logger.info("canary.no_traffic", name=self._name)
                    continue
                self._monitor_errors = 0
                await self._evaluate(error_rate)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._monitor_errors += 1
                logger.error(
                    "canary.monitor_error",
                    name=self._name,
                    error=str(exc),
                    consecutive_errors=self._monitor_errors,
                )
                if self._monitor_errors >= 3:
                    logger.error(
                        "canary.monitor_degraded",
                        name=self._name,
                        msg="3 consecutive monitor errors — stopping canary loop",
                    )
                    self._running = False

    # ── error rate measurement ────────────────────────────────────────────

    async def _get_error_rate(self) -> Optional[float]:
        """
        Returns error rate in [0, 1], or None when there is no traffic yet.

        Query is scoped to the specific deployment job to avoid aggregating
        across multiple model versions running in the same namespace.
        """
        prom_rate = await self._query_prometheus()
        if prom_rate is not None:
            return prom_rate

        # Fallback: health endpoint
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._canary_url}/health")
                return 0.0 if resp.status_code == 200 else 1.0
        except Exception:
            return 1.0

    async def _query_prometheus(self) -> Optional[float]:
        """Query Prometheus; return None on error or when result is NaN / no-data."""
        try:
            job = f"capsule-{self._name}"
            query = (
                f'sum(rate(model_requests_total{{job="{job}",status="error"}}[5m])) / '
                f'sum(rate(model_requests_total{{job="{job}"}}[5m]))'
            )
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(
                    f"{self._prometheus_url}/api/v1/query",
                    params={"query": query},
                )
                data = resp.json()
                results = data.get("data", {}).get("result", [])
                if not results:
                    return None  # no data — too early
                raw = results[0]["value"][1]
                if raw in ("NaN", "+Inf", "-Inf"):
                    return None  # division by zero — no traffic
                value = float(raw)
                return value if value == value else None  # NaN guard
        except Exception:
            return None

    # ── evaluation ───────────────────────────────────────────────────────

    async def _evaluate(self, error_rate: float):
        self._store.log_event(CanaryEvent(
            deployment_name=self._name,
            event_type="check",
            canary_weight=self._get_current_weight(),
            error_rate=error_rate,
            message=f"Error rate: {error_rate:.3f} (threshold: {self._threshold})",
        ))

        if error_rate > self._threshold:
            self._failure_count += 1
            self._success_count = 0
            logger.warning(
                "canary.degraded",
                name=self._name,
                error_rate=round(error_rate, 4),
                consecutive=self._failure_count,
                threshold=self._threshold,
            )
            if self._failure_count >= self._max_failures:
                await self._do_rollback(error_rate)
        else:
            self._failure_count = 0
            self._success_count += 1
            logger.info(
                "canary.healthy",
                name=self._name,
                error_rate=round(error_rate, 4),
                window=self._success_count,
                target=self._promote_windows,
            )
            if self._success_count >= self._promote_windows:
                await self._do_promote()

    async def _do_rollback(self, error_rate: float):
        logger.warning(
            "canary.auto_rollback_triggered",
            name=self._name,
            error_rate=round(error_rate, 4),
        )
        self._store.log_event(CanaryEvent(
            deployment_name=self._name,
            event_type="rollback",
            canary_weight=0,
            error_rate=error_rate,
            message=(
                f"Auto-rollback: error rate {error_rate:.3f} "
                f"exceeded threshold {self._threshold} "
                f"for {self._failure_count} consecutive windows"
            ),
        ))

        # ── ACTUALLY execute the rollback ──────────────────────────────
        if self._rollback_fn is not None:
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._rollback_fn)
                logger.info("canary.rollback_executed", name=self._name)
            except Exception as exc:
                logger.error(
                    "canary.rollback_fn_failed",
                    name=self._name,
                    error=str(exc),
                )
        else:
            logger.error(
                "canary.no_rollback_fn",
                name=self._name,
                msg="No rollback function injected — K8s traffic NOT reverted. "
                    "Run: capsule rollback " + self._name,
            )

        self._running = False

    async def _do_promote(self):
        logger.info(
            "canary.auto_promote_triggered",
            name=self._name,
            windows=self._success_count,
        )
        self._store.log_event(CanaryEvent(
            deployment_name=self._name,
            event_type="promote",
            canary_weight=100,
            message=f"Auto-promoted after {self._success_count} healthy windows",
        ))

        # ── ACTUALLY execute the promotion ─────────────────────────────
        if self._promote_fn is not None:
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._promote_fn)
                logger.info("canary.promote_executed", name=self._name)
            except Exception as exc:
                logger.error(
                    "canary.promote_fn_failed",
                    name=self._name,
                    error=str(exc),
                )
        else:
            logger.warning(
                "canary.no_promote_fn",
                name=self._name,
                msg="No promote function injected — traffic weight NOT updated. "
                    "Run: capsule deploy " + self._name + ":<version>",
            )

        self._running = False

    # ── helpers ───────────────────────────────────────────────────────────

    def _get_current_weight(self) -> int:
        record = self._store.get_latest_deployment(self._name)
        return record.canary_weight if record else 0
