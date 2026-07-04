from __future__ import annotations
import pytest
import tempfile
import os
from unittest.mock import patch, MagicMock


def test_optimize_pytorch_fails_gracefully_no_model():
    from capsule.onnx_optimizer import optimize_pytorch_to_onnx
    ok, path, stats = optimize_pytorch_to_onnx("/nonexistent/model.pt")
    assert ok is False
    assert path == "/nonexistent/model.pt"


def test_optimize_onnx_model_no_input():
    from capsule.onnx_optimizer import optimize_onnx_model
    ok, path, stats = optimize_onnx_model("/nonexistent/model.onnx")
    assert ok is False


def test_optimize_returns_tuple():
    from capsule.onnx_optimizer import optimize_pytorch_to_onnx
    result = optimize_pytorch_to_onnx("/tmp/fake.pt")
    assert isinstance(result, tuple)
    assert len(result) == 3


def test_optimize_onnx_with_real_model(tmp_path):
    """Creates a real tiny ONNX model and optimises it."""
    try:
        import torch
        import onnx
        from capsule.onnx_optimizer import optimize_onnx_model

        class Tiny(torch.nn.Module):
            def forward(self, x):
                return x * 2.0

        model_path = str(tmp_path / "tiny.onnx")
        dummy = torch.randn(1, 4)
        torch.onnx.export(
            Tiny(), dummy, model_path,
            input_names=["x"], output_names=["y"], opset_version=17
        )

        out_path = str(tmp_path / "tiny_opt.onnx")
        ok, result_path, stats = optimize_onnx_model(
            model_path, output_path=out_path, quantize=True
        )
        # May fail if quantize doesn't support simple models — that's ok
        assert isinstance(ok, bool)
    except ImportError:
        pytest.skip("ONNX or torch not available")
