from __future__ import annotations
import time
from typing import List, Optional, Dict, Any
import structlog

logger = structlog.get_logger(__name__)


class K8sClient:
    """
    Kubernetes Python client wrapper.
    Gracefully degrades when not connected (for testing).
    """

    def __init__(self, namespace: str = "capsule", kubeconfig: Optional[str] = None):
        self.namespace = namespace
        self._available = False
        self._apps_v1 = None
        self._core_v1 = None
        self._custom = None
        self._try_connect(kubeconfig)

    def _try_connect(self, kubeconfig: Optional[str] = None):
        try:
            from kubernetes import client, config
            if kubeconfig:
                config.load_kube_config(config_file=kubeconfig)
            else:
                try:
                    config.load_incluster_config()
                except Exception:
                    config.load_kube_config()
            self._apps_v1 = client.AppsV1Api()
            self._core_v1 = client.CoreV1Api()
            self._custom = client.CustomObjectsApi()
            self._available = True
            logger.info("k8s_client.connected", namespace=self.namespace)
        except Exception as exc:
            logger.warning("k8s_client.unavailable", error=str(exc))

    def ensure_namespace(self):
        if not self._available:
            return
        from kubernetes import client
        try:
            self._core_v1.read_namespace(self.namespace)
        except Exception:
            ns = client.V1Namespace(
                metadata=client.V1ObjectMeta(name=self.namespace)
            )
            self._core_v1.create_namespace(ns)
            logger.info("k8s_client.namespace_created", namespace=self.namespace)

    def get_deployment(self, name: str) -> Optional[Dict]:
        if not self._available:
            return None
        try:
            dep = self._apps_v1.read_namespaced_deployment(
                name=name, namespace=self.namespace
            )
            return {
                "name": dep.metadata.name,
                "replicas": dep.spec.replicas,
                "ready_replicas": dep.status.ready_replicas or 0,
                "image": dep.spec.template.spec.containers[0].image,
            }
        except Exception:
            return None

    def get_pods(self, name: str) -> List[Dict]:
        if not self._available:
            return []
        try:
            pods = self._core_v1.list_namespaced_pod(
                namespace=self.namespace,
                label_selector=f"app={name}",
            )
            result = []
            now = time.time()
            for pod in pods.items:
                ready = all(
                    cs.ready
                    for cs in (pod.status.container_statuses or [])
                )
                restarts = sum(
                    cs.restart_count
                    for cs in (pod.status.container_statuses or [])
                )
                created = pod.metadata.creation_timestamp
                age = (now - created.timestamp()) if created else 0
                result.append({
                    "name": pod.metadata.name,
                    "phase": pod.status.phase or "Unknown",
                    "ready": ready,
                    "restarts": restarts,
                    "age_seconds": age,
                    "version": pod.metadata.labels.get("version", "unknown"),
                })
            return result
        except Exception as exc:
            logger.warning("k8s_client.get_pods_error", error=str(exc))
            return []

    def wait_for_rollout(
        self,
        name: str,
        timeout_seconds: int = 120,
        poll_interval: int = 5,
    ) -> bool:
        if not self._available:
            return True
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            dep = self.get_deployment(name)
            if dep and dep.get("ready_replicas", 0) >= dep.get("replicas", 1):
                logger.info("k8s_client.rollout_complete", name=name)
                return True
            time.sleep(poll_interval)
        logger.warning("k8s_client.rollout_timeout", name=name)
        return False

    def scale_deployment(self, name: str, replicas: int):
        if not self._available:
            return
        from kubernetes import client
        patch = {"spec": {"replicas": replicas}}
        self._apps_v1.patch_namespaced_deployment_scale(
            name=name, namespace=self.namespace, body=patch
        )
        logger.info("k8s_client.scaled", name=name, replicas=replicas)

    def get_events(self, name: str, limit: int = 10) -> List[str]:
        if not self._available:
            return []
        try:
            events = self._core_v1.list_namespaced_event(
                namespace=self.namespace,
                field_selector=f"involvedObject.name={name}",
            )
            sorted_events = sorted(
                events.items,
                key=lambda e: e.last_timestamp or e.event_time,
                reverse=True,
            )
            return [
                f"{e.reason}: {e.message}"
                for e in sorted_events[:limit]
                if e.message
            ]
        except Exception:
            return []
