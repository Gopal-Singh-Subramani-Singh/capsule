from __future__ import annotations
import os
import tempfile
from pathlib import Path
from typing import Optional
from jinja2 import Environment, BaseLoader
import structlog

from capsule.models import CapsuleManifest

logger = structlog.get_logger(__name__)

_CHART_YAML = """apiVersion: v2
name: {{ name }}
description: Capsule deployment for {{ name }}
type: application
version: 0.1.0
appVersion: "{{ version }}"
"""

_VALUES_YAML = """replicaCount: {{ replicas }}
image:
  repository: {{ image_repo }}
  tag: "{{ image_tag }}"
  pullPolicy: Always
service:
  type: ClusterIP
  port: {{ port }}
resources:
  requests:
    cpu: {{ cpu_request }}
    memory: {{ memory_request }}
  limits:
    cpu: {{ cpu_limit }}
    memory: {{ memory_limit }}
healthCheck:
  path: {{ health_path }}
  port: {{ port }}
  initialDelaySeconds: {{ initial_delay }}
  periodSeconds: 10
canary:
  enabled: {{ canary_enabled }}
  weight: {{ canary_weight }}
"""

_DEPLOYMENT_YAML = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ "{{" }} .Release.Name {{ "}}" }}
  namespace: {{ "{{" }} .Release.Namespace {{ "}}" }}
  labels:
    app: {{ "{{" }} .Release.Name {{ "}}" }}
    version: {{ "{{" }} .Values.image.tag {{ "}}" }}
    managed-by: capsule
spec:
  replicas: {{ "{{" }} .Values.replicaCount {{ "}}" }}
  selector:
    matchLabels:
      app: {{ "{{" }} .Release.Name {{ "}}" }}
  template:
    metadata:
      labels:
        app: {{ "{{" }} .Release.Name {{ "}}" }}
        version: {{ "{{" }} .Values.image.tag {{ "}}" }}
    spec:
      containers:
        - name: model-server
          image: "{{ "{{" }} .Values.image.repository {{ "}}" }}:{{ "{{" }} .Values.image.tag {{ "}}" }}"
          imagePullPolicy: {{ "{{" }} .Values.image.pullPolicy {{ "}}" }}
          ports:
            - containerPort: {{ "{{" }} .Values.service.port {{ "}}" }}
          resources:
            requests:
              cpu: {{ "{{" }} .Values.resources.requests.cpu {{ "}}" }}
              memory: {{ "{{" }} .Values.resources.requests.memory {{ "}}" }}
            limits:
              cpu: {{ "{{" }} .Values.resources.limits.cpu {{ "}}" }}
              memory: {{ "{{" }} .Values.resources.limits.memory {{ "}}" }}
          readinessProbe:
            httpGet:
              path: {{ "{{" }} .Values.healthCheck.path {{ "}}" }}
              port: {{ "{{" }} .Values.healthCheck.port {{ "}}" }}
            initialDelaySeconds: {{ "{{" }} .Values.healthCheck.initialDelaySeconds {{ "}}" }}
            periodSeconds: {{ "{{" }} .Values.healthCheck.periodSeconds {{ "}}" }}
          livenessProbe:
            httpGet:
              path: {{ "{{" }} .Values.healthCheck.path {{ "}}" }}
              port: {{ "{{" }} .Values.healthCheck.port {{ "}}" }}
            initialDelaySeconds: 30
            periodSeconds: 30
"""

_SERVICE_YAML = """apiVersion: v1
kind: Service
metadata:
  name: {{ "{{" }} .Release.Name {{ "}}" }}
  namespace: {{ "{{" }} .Release.Namespace {{ "}}" }}
  labels:
    app: {{ "{{" }} .Release.Name {{ "}}" }}
    managed-by: capsule
spec:
  selector:
    app: {{ "{{" }} .Release.Name {{ "}}" }}
  ports:
    - protocol: TCP
      port: {{ "{{" }} .Values.service.port {{ "}}" }}
      targetPort: {{ "{{" }} .Values.service.port {{ "}}" }}
  type: {{ "{{" }} .Values.service.type {{ "}}" }}
"""


class HelmChartGenerator:
    def __init__(self, chart_dir: str = "/tmp/capsule-charts"):
        self.chart_dir = Path(chart_dir)
        self._env = Environment(loader=BaseLoader())

    def generate(
        self,
        manifest: CapsuleManifest,
        image_tag: str,
        canary_weight: int = 0,
        release_name: Optional[str] = None,
    ) -> str:
        """Generate a Helm chart and return the chart directory path."""
        name = release_name or f"capsule-{manifest.name}"
        chart_path = self.chart_dir / name
        templates_path = chart_path / "templates"
        templates_path.mkdir(parents=True, exist_ok=True)

        image_parts = image_tag.rsplit(":", 1)
        image_repo = image_parts[0]
        img_tag = image_parts[1] if len(image_parts) > 1 else "latest"

        ctx = {
            "name": manifest.name,
            "version": manifest.version,
            "replicas": manifest.resources.replicas,
            "image_repo": image_repo,
            "image_tag": img_tag,
            "port": manifest.port,
            "cpu_request": manifest.resources.cpu_request,
            "cpu_limit": manifest.resources.cpu_limit,
            "memory_request": manifest.resources.memory_request,
            "memory_limit": manifest.resources.memory_limit,
            "health_path": manifest.health.path,
            "initial_delay": manifest.health.initial_delay_seconds,
            "canary_enabled": str(canary_weight > 0).lower(),
            "canary_weight": canary_weight,
        }

        def render(tmpl_str: str) -> str:
            return self._env.from_string(tmpl_str).render(**ctx)

        (chart_path / "Chart.yaml").write_text(render(_CHART_YAML))
        (chart_path / "values.yaml").write_text(render(_VALUES_YAML))
        (templates_path / "deployment.yaml").write_text(_DEPLOYMENT_YAML)
        (templates_path / "service.yaml").write_text(_SERVICE_YAML)

        logger.info(
            "helm.chart_generated",
            name=name,
            path=str(chart_path),
            canary_weight=canary_weight,
        )
        return str(chart_path)
