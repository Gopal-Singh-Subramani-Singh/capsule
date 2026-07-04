from __future__ import annotations
import os
from pathlib import Path
from typing import List, Optional
import structlog

from capsule.models import Framework

logger = structlog.get_logger(__name__)

FRAMEWORK_SIGNATURES = {
    Framework.PYTORCH: [".pt", ".pth", ".torchscript"],
    Framework.ONNX: [".onnx"],
    Framework.SKLEARN: [".pkl", ".joblib", ".pickle"],
}


def detect_framework(model_path: str) -> Framework:
    """
    Detect the ML framework from the model file extension.
    Falls back to generic if unknown.
    """
    path = Path(model_path)
    suffix = path.suffix.lower()

    for framework, extensions in FRAMEWORK_SIGNATURES.items():
        if suffix in extensions:
            logger.info(
                "detector.framework_detected",
                path=model_path,
                framework=framework.value,
                extension=suffix,
            )
            return framework

    # Try content-based detection for .pkl files
    if suffix in (".pkl", ".pickle"):
        try:
            import pickle
            with open(model_path, "rb") as f:
                obj = pickle.load(f)
            module = type(obj).__module__
            if "sklearn" in module:
                return Framework.SKLEARN
            elif "torch" in module:
                return Framework.PYTORCH
        except Exception:
            pass

    logger.info(
        "detector.framework_unknown",
        path=model_path,
        suffix=suffix,
        fallback="generic",
    )
    return Framework.GENERIC


def get_base_image(framework: Framework, python_version: str = "3.11") -> str:
    """Return appropriate Docker base image for a framework."""
    base_map = {
        Framework.PYTORCH: f"python:{python_version}-slim",
        Framework.ONNX: f"python:{python_version}-slim",
        Framework.SKLEARN: f"python:{python_version}-slim",
        Framework.GENERIC: f"python:{python_version}-slim",
    }
    return base_map.get(framework, f"python:{python_version}-slim")


def get_framework_packages(framework: Framework) -> List[str]:
    """Return pip packages needed for each framework."""
    packages = {
        Framework.PYTORCH: ["torch", "torchvision", "fastapi", "uvicorn", "prometheus-client"],
        Framework.ONNX: ["onnxruntime", "numpy", "fastapi", "uvicorn", "prometheus-client"],
        Framework.SKLEARN: ["scikit-learn", "numpy", "pandas", "fastapi", "uvicorn", "prometheus-client"],
        Framework.GENERIC: ["fastapi", "uvicorn", "prometheus-client"],
    }
    return packages.get(framework, [])
