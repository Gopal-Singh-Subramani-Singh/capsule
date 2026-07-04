from __future__ import annotations
import os
import subprocess
import time
from typing import Optional
import structlog

from capsule.models import (
    CapsuleManifest, DeploymentRecord, DeploymentStatus, StatusReport, PodStatus
)
from capsule.helm import HelmChartGenerator
from capsule.k8s_client import K8sClient
from capsule.registry import ModelRegistry
from capsule.manifest import ManifestStore

logger = structlog.get_logger(__name__)


class HelmError(RuntimeError):
    """Raised when a Helm operation returns non-zero or times out."""


class Deployer:
    def __init__(
        self,
        registry: ModelRegistry,
        store: ManifestStore,
        k8s: K8sClient,
        helm_gen: HelmChartGenerator,
        namespace: str = "capsule",
    ):
        self._registry = registry
        self._store = store
        self._k8s = k8s
        self._helm = helm_gen
        self._namespace = namespace

    # ── deploy ────────────────────────────────────────────────────────────

    def deploy(
        self,
        manifest: CapsuleManifest,
        image_tag: str,
        canary_weight: int = 0,
    ) -> DeploymentRecord:
        """Deploy a model to K3s using Helm. Raises HelmError on failure."""
        release_name = f"capsule-{manifest.name}"

        stable_version: Optional[str] = None
        if canary_weight > 0:
            existing = self._store.get_latest_deployment(manifest.name)
            if existing and existing.status == DeploymentStatus.RUNNING:
                stable_version = existing.version

        # Resolve deploy reference — prefer pinned digest
        deploy_ref = self._registry.get_deploy_ref(manifest.name, manifest.version) or image_tag

        record = DeploymentRecord(
            name=manifest.name,
            version=manifest.version,
            image_tag=image_tag,
            image_digest=self._registry.get_image_digest(manifest.name, manifest.version),
            namespace=self._namespace,
            canary_weight=canary_weight,
            stable_version=stable_version,
            status=DeploymentStatus.PENDING,
            deployed_by=os.environ.get("USER", "unknown"),
        )
        self._store.save_deployment(record)

        self._k8s.ensure_namespace()

        chart_path = self._helm.generate(
            manifest,
            image_tag=deploy_ref,
            canary_weight=canary_weight,
            release_name=release_name,
        )

        # _helm_upgrade raises HelmError on failure — status stays PENDING
        self._helm_upgrade(release_name, chart_path)

        ready = self._k8s.wait_for_rollout(release_name, timeout_seconds=120)
        if not ready:
            record.status = DeploymentStatus.FAILED
            self._store.save_deployment(record)
            raise HelmError(
                f"Rollout timed out for {manifest.name}:{manifest.version}. "
                "Pods may still be starting — check `capsule status`."
            )

        record.status = (
            DeploymentStatus.RUNNING if canary_weight == 0
            else DeploymentStatus.CANARY
        )
        self._store.save_deployment(record)

        logger.info(
            "deployer.deployed",
            name=manifest.name,
            version=manifest.version,
            deploy_ref=deploy_ref,
            canary_weight=canary_weight,
        )
        return record

    # ── rollback ──────────────────────────────────────────────────────────

    def rollback(self, name: str) -> dict:
        """
        Roll back to the last RUNNING version.

        Uses the content-addressable digest when available so we always
        pull the exact binary that was originally verified, even if the tag
        has been overwritten since.
        """
        t0 = time.monotonic()
        current = self._store.get_latest_deployment(name)
        if not current:
            raise ValueError(f"No deployment found for '{name}'")

        target_version = self._store.get_previous_version(name, current.version)
        if not target_version:
            raise ValueError(
                f"No previous RUNNING version found for '{name}'. "
                "Cannot determine a safe rollback target."
            )

        deploy_ref = self._registry.get_deploy_ref(name, target_version)
        if not deploy_ref:
            raise ValueError(
                f"No image reference found for {name}:{target_version}. "
                "The registry may be offline or the version was never pushed."
            )

        release_name = f"capsule-{name}"
        self._helm_rollback(release_name)

        ready = self._k8s.wait_for_rollout(release_name, timeout_seconds=120)
        duration = time.monotonic() - t0

        record = DeploymentRecord(
            name=name,
            version=target_version,
            image_tag=self._registry.get_image_tag(name, target_version) or deploy_ref,
            image_digest=self._registry.get_image_digest(name, target_version),
            namespace=self._namespace,
            status=DeploymentStatus.ROLLED_BACK,
            deployed_by=os.environ.get("USER", "unknown"),
        )
        self._store.save_deployment(record)

        logger.info(
            "deployer.rollback_complete",
            name=name,
            from_version=current.version,
            to_version=target_version,
            deploy_ref=deploy_ref,
            success=ready,
        )
        return {
            "name": name,
            "rolled_back_from": current.version,
            "rolled_back_to": target_version,
            "deploy_ref": deploy_ref,
            "success": ready,
            "duration_seconds": round(duration, 1),
        }

    # ── status ────────────────────────────────────────────────────────────

    def get_status(self, name: str) -> StatusReport:
        release_name = f"capsule-{name}"
        record = self._store.get_latest_deployment(name)
        pods_raw = self._k8s.get_pods(release_name)
        k8s_events = self._k8s.get_events(release_name)
        store_events = self._store.get_events(name, limit=5)

        pods = [
            PodStatus(
                name=p["name"],
                phase=p["phase"],
                ready=p["ready"],
                restarts=p["restarts"],
                age_seconds=p["age_seconds"],
                version=p["version"],
            )
            for p in pods_raw
        ]

        all_events = k8s_events + [e["message"] for e in store_events]

        return StatusReport(
            name=name,
            namespace=self._namespace,
            status=record.status if record else DeploymentStatus.PENDING,
            stable_version=record.stable_version if record else None,
            canary_version=(
                record.version
                if record and record.canary_weight > 0
                else None
            ),
            canary_weight=record.canary_weight if record else 0,
            pods=pods,
            events=all_events[:10],
            uptime_seconds=pods[0].age_seconds if pods else None,
        )

    # ── helm wrappers ─────────────────────────────────────────────────────

    def _helm_upgrade(self, release_name: str, chart_path: str):
        cmd = [
            "helm", "upgrade", "--install",
            release_name, chart_path,
            "--namespace", self._namespace,
            "--create-namespace",
            "--wait", "--timeout", "120s",
            "--atomic",   # rolls back automatically on failure
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=150
            )
            if result.returncode != 0:
                raise HelmError(
                    f"helm upgrade failed for '{release_name}': "
                    + result.stderr[:800]
                )
            logger.info("deployer.helm_upgrade_ok", release=release_name)
        except FileNotFoundError:
            logger.warning(
                "deployer.helm_not_found",
                msg="helm binary not on PATH — K8s deploy skipped",
            )
        except subprocess.TimeoutExpired:
            raise HelmError(
                f"helm upgrade timed out after 150s for release '{release_name}'"
            )

    def _helm_rollback(self, release_name: str):
        cmd = [
            "helm", "rollback", release_name,
            "--namespace", self._namespace,
            "--wait", "--timeout", "120s",
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=150
            )
            if result.returncode != 0:
                raise HelmError(
                    f"helm rollback failed for '{release_name}': "
                    + result.stderr[:800]
                )
            logger.info("deployer.helm_rollback_ok", release=release_name)
        except FileNotFoundError:
            logger.warning("deployer.helm_not_found", msg="helm binary not on PATH")
        except subprocess.TimeoutExpired:
            raise HelmError(
                f"helm rollback timed out after 150s for release '{release_name}'"
            )
