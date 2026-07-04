from __future__ import annotations
import hashlib
import os
import time
import tempfile
import shutil
from pathlib import Path
from typing import Optional
import structlog
from jinja2 import Environment, BaseLoader

from capsule.models import CapsuleManifest, PackageResult, Framework
from capsule.detector import detect_framework, get_framework_packages, get_base_image
from capsule.registry import ModelRegistry
from capsule.manifest import ManifestStore
from capsule.onnx_optimizer import optimize_pytorch_to_onnx, optimize_onnx_model
from capsule.server_template import generate_server_code

logger = structlog.get_logger(__name__)

# ── Dockerfile template ────────────────────────────────────────────────────────
# Uses a non-root user (uid 1000) for security
_DOCKERFILE_TEMPLATE = """FROM {{ base_image }}

WORKDIR /app

# System deps — minimal; curl for HEALTHCHECK only
RUN apt-get update && apt-get install -y --no-install-recommends curl \\
    && rm -rf /var/lib/apt/lists/*

# Non-root runtime user
RUN groupadd -g 1000 capsule && useradd -u 1000 -g capsule -s /sbin/nologin capsule

# Python deps — installed as root before switching user
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Model artifact
COPY {{ model_filename }} /app/model/{{ model_filename }}

# Server
COPY server.py .

RUN chown -R capsule:capsule /app

USER capsule

ENV MODEL_PATH=/app/model/{{ model_filename }}
ENV FRAMEWORK={{ framework }}
ENV PORT={{ port }}

EXPOSE {{ port }}

HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=3 \\
  CMD curl -f http://localhost:{{ port }}/health || exit 1

CMD ["python", "server.py"]
"""

_REQUIREMENTS_TEMPLATE = """\
fastapi==0.115.0
uvicorn[standard]==0.30.6
prometheus-client==0.21.0
{{ extra_packages }}
"""


class Packager:
    def __init__(
        self,
        registry: ModelRegistry,
        store: ManifestStore,
        docker_registry: str = "localhost:5001",
        onnx_enabled: bool = True,
    ):
        self._registry = registry
        self._store = store
        self._docker_registry = docker_registry
        self._onnx_enabled = onnx_enabled

    def package(
        self,
        manifest: CapsuleManifest,
        push_image: bool = True,
        manifest_dir: Optional[str] = None,
    ) -> PackageResult:
        t0 = time.monotonic()

        # Resolve model path relative to manifest location
        model_path = manifest.model_path
        if not os.path.isabs(model_path) and manifest_dir:
            model_path = os.path.join(manifest_dir, model_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        framework = manifest.framework or detect_framework(model_path)
        logger.info(
            "packager.start",
            name=manifest.name,
            version=manifest.version,
            framework=framework.value,
        )

        orig_size_mb = os.path.getsize(model_path) / (1024 ** 2)
        optimised_path = model_path
        onnx_optimised = False
        opt_size_mb = None

        # ONNX optimisation
        if self._onnx_enabled:
            if framework == Framework.PYTORCH:
                ok, optimised_path, stats = optimize_pytorch_to_onnx(model_path)
                if ok:
                    onnx_optimised = True
                    opt_size_mb = stats.get("optimised_size_mb")
                    framework = Framework.ONNX
                    logger.info("packager.onnx_optimised", stats=stats)
            elif framework == Framework.ONNX:
                ok, optimised_path, stats = optimize_onnx_model(model_path)
                if ok:
                    onnx_optimised = True
                    opt_size_mb = stats.get("optimised_size_mb")

        with tempfile.TemporaryDirectory() as build_dir:
            model_filename = os.path.basename(optimised_path)
            shutil.copy2(optimised_path, os.path.join(build_dir, model_filename))

            server_code = generate_server_code(framework, manifest)
            with open(os.path.join(build_dir, "server.py"), "w") as f:
                f.write(server_code)

            extra = "\n".join(
                get_framework_packages(framework) + manifest.requirements
            )
            reqs = _REQUIREMENTS_TEMPLATE.replace("{{ extra_packages }}", extra).strip()
            with open(os.path.join(build_dir, "requirements.txt"), "w") as f:
                f.write(reqs)

            env = Environment(loader=BaseLoader())
            dockerfile_content = env.from_string(_DOCKERFILE_TEMPLATE).render(
                base_image=get_base_image(framework, manifest.python_version),
                model_filename=model_filename,
                framework=framework.value,
                port=manifest.port,
            )
            with open(os.path.join(build_dir, "Dockerfile"), "w") as f:
                f.write(dockerfile_content)

            self._registry.push_dockerfile(manifest.name, manifest.version, dockerfile_content)

            image_tag = f"{self._docker_registry}/{manifest.name}:{manifest.version}"
            image_digest: Optional[str] = None

            if push_image:
                image_digest = self._build_and_push(build_dir, image_tag)
                # Store the immutable digest in the registry for rollback pinning
                if image_digest:
                    self._registry.push_image_digest(
                        manifest.name, manifest.version, image_digest
                    )
            else:
                logger.info("packager.skipping_docker_build", tag=image_tag)

        self._registry.push_model(
            manifest.name, manifest.version, optimised_path, framework.value
        )
        self._registry.push_image_tag(manifest.name, manifest.version, image_tag)

        build_seconds = time.monotonic() - t0
        size_reduction = (
            (1 - opt_size_mb / orig_size_mb) * 100
            if (opt_size_mb and orig_size_mb)
            else None
        )

        result = PackageResult(
            name=manifest.name,
            version=manifest.version,
            framework=framework,
            image_tag=image_tag,
            image_digest=image_digest,
            registry_path=f"s3://capsule-models/models/{manifest.name}/{manifest.version}/",
            original_size_mb=max(round(orig_size_mb, 3), 0.001),
            optimised_size_mb=opt_size_mb,
            onnx_optimised=onnx_optimised,
            size_reduction_pct=round(size_reduction, 1) if size_reduction else None,
            build_seconds=round(build_seconds, 1),
        )
        self._store.save_package(result)
        logger.info(
            "packager.complete",
            name=manifest.name,
            version=manifest.version,
            image_tag=image_tag,
            image_digest=image_digest,
            build_seconds=round(build_seconds, 1),
        )
        return result

    def _build_and_push(self, build_dir: str, tag: str) -> Optional[str]:
        """Build the Docker image, push it, and return its sha256 digest."""
        try:
            import docker
            client = docker.from_env()
            logger.info("packager.building_image", tag=tag)

            _, build_logs = client.images.build(
                path=build_dir,
                tag=tag,
                rm=True,
                platform="linux/arm64",
            )
            for chunk in build_logs:
                if "error" in chunk:
                    raise RuntimeError(chunk["error"].strip())

            logger.info("packager.image_built", tag=tag)
            client.images.push(tag)
            logger.info("packager.image_pushed", tag=tag)

            # Resolve digest
            digest = self._resolve_digest(client, tag)
            if digest:
                logger.info("packager.image_digest", tag=tag, digest=digest)
            return digest

        except ImportError:
            logger.warning("packager.docker_sdk_missing")
            return None
        except Exception as exc:
            logger.error("packager.build_failed", error=str(exc), tag=tag)
            raise

    @staticmethod
    def _resolve_digest(client, tag: str) -> Optional[str]:
        """Return the repo digest (sha256:...) for a just-pushed image."""
        try:
            image = client.images.get(tag)
            digests = image.attrs.get("RepoDigests", [])
            for d in digests:
                if "@sha256:" in d:
                    return d.split("@")[-1]
        except Exception:
            pass
        return None
