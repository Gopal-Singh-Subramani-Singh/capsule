from __future__ import annotations
import os
import tempfile
import pytest
from capsule.detector import detect_framework, get_base_image, get_framework_packages
from capsule.models import Framework


def test_detects_pytorch_pt():
    assert detect_framework("/models/fraud.pt") == Framework.PYTORCH


def test_detects_pytorch_pth():
    assert detect_framework("/models/bert.pth") == Framework.PYTORCH


def test_detects_onnx():
    assert detect_framework("/models/model.onnx") == Framework.ONNX


def test_detects_sklearn_pkl():
    assert detect_framework("/models/classifier.pkl") == Framework.SKLEARN


def test_detects_sklearn_joblib():
    assert detect_framework("/models/pipeline.joblib") == Framework.SKLEARN


def test_unknown_returns_generic():
    assert detect_framework("/models/mystery.xyz") == Framework.GENERIC


def test_get_base_image_returns_string():
    img = get_base_image(Framework.PYTORCH, "3.11")
    assert "python" in img
    assert "3.11" in img


def test_get_framework_packages_pytorch():
    pkgs = get_framework_packages(Framework.PYTORCH)
    assert "torch" in pkgs
    assert "fastapi" in pkgs


def test_get_framework_packages_onnx():
    pkgs = get_framework_packages(Framework.ONNX)
    assert "onnxruntime" in pkgs


def test_get_framework_packages_sklearn():
    pkgs = get_framework_packages(Framework.SKLEARN)
    assert "scikit-learn" in pkgs
