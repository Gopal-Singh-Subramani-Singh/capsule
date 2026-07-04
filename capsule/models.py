from __future__ import annotations
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Any, Dict, List, Optional, Literal
from enum import Enum
from datetime import datetime
import uuid


class Framework(str, Enum):
    PYTORCH = "pytorch"
    ONNX = "onnx"
    SKLEARN = "sklearn"
    GENERIC = "generic"


class DeploymentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    CANARY = "canary"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class ResourceSpec(BaseModel):
    cpu_request: str = "100m"
    cpu_limit: str = "500m"
    memory_request: str = "256Mi"
    memory_limit: str = "1Gi"
    replicas: int = 1

    @field_validator("replicas")
    @classmethod
    def replicas_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("replicas must be >= 1")
        return v


class HealthProbeSpec(BaseModel):
    path: str = "/health"
    port: int = 8080
    initial_delay_seconds: int = 10
    period_seconds: int = 10
    failure_threshold: int = 3


class CanarySpec(BaseModel):
    enabled: bool = False
    initial_weight: int = 10
    auto_promote: bool = True
    error_rate_threshold: float = 0.05

    @field_validator("initial_weight")
    @classmethod
    def weight_range(cls, v: int) -> int:
        if not 0 <= v <= 100:
            raise ValueError("initial_weight must be 0–100")
        return v

    @field_validator("error_rate_threshold")
    @classmethod
    def threshold_range(cls, v: float) -> float:
        if not 0.0 < v <= 1.0:
            raise ValueError("error_rate_threshold must be in (0, 1]")
        return v


class CapsuleManifest(BaseModel):
    """Represents a capsule.yaml file."""
    model_config = ConfigDict(protected_namespaces=())

    name: str
    version: str
    framework: Optional[Framework] = None
    model_path: str
    requirements: List[str] = []
    python_version: str = "3.11"
    port: int = 8080
    resources: ResourceSpec = Field(default_factory=ResourceSpec)
    health: HealthProbeSpec = Field(default_factory=HealthProbeSpec)
    canary: CanarySpec = Field(default_factory=CanarySpec)
    # env values must NOT contain secrets — use K8s Secrets or external injection
    env: Dict[str, str] = {}
    labels: Dict[str, str] = {}
    metadata: Dict[str, Any] = {}

    @field_validator("name")
    @classmethod
    def name_slug(cls, v: str) -> str:
        import re
        if not re.match(r"^[a-z0-9][a-z0-9\-]{0,61}[a-z0-9]$", v):
            raise ValueError(
                "name must be a lowercase DNS label: letters, digits, hyphens, 2–63 chars"
            )
        return v

    @field_validator("port")
    @classmethod
    def port_range(cls, v: int) -> int:
        if not 1 <= v <= 65535:
            raise ValueError("port must be 1–65535")
        return v


class PackageResult(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    name: str
    version: str
    framework: Framework
    image_tag: str
    # Content-addressable digest — sha256:... — stored after push
    image_digest: Optional[str] = None
    registry_path: str
    original_size_mb: float
    optimised_size_mb: Optional[float] = None
    onnx_optimised: bool = False
    size_reduction_pct: Optional[float] = None
    build_seconds: float
    packaged_at: datetime = Field(default_factory=datetime.utcnow)


class DeploymentRecord(BaseModel):
    deployment_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str
    version: str
    image_tag: str
    image_digest: Optional[str] = None
    namespace: str
    status: DeploymentStatus = DeploymentStatus.PENDING
    canary_weight: int = 0
    stable_version: Optional[str] = None
    deployed_by: Optional[str] = None
    deployed_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    # NOTE: events list is in-memory only; durable events go to the events table
    events: List[str] = []


class PodStatus(BaseModel):
    name: str
    phase: str
    ready: bool
    restarts: int
    age_seconds: float
    version: str


class StatusReport(BaseModel):
    name: str
    namespace: str
    status: DeploymentStatus
    stable_version: Optional[str]
    canary_version: Optional[str]
    canary_weight: int
    pods: List[PodStatus]
    events: List[str]
    uptime_seconds: Optional[float]


class RollbackResult(BaseModel):
    name: str
    rolled_back_from: str
    rolled_back_to: str
    success: bool
    message: str
    duration_seconds: float


class CanaryEvent(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    deployment_name: str
    event_type: Literal["check", "promote", "rollback", "error"]
    canary_weight: int
    error_rate: Optional[float] = None
    message: str
