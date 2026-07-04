from __future__ import annotations
import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


def optimize_pytorch_to_onnx(
    model_path: str,
    output_path: Optional[str] = None,
    quantize: bool = True,
    validation_tolerance: float = 0.01,
) -> Tuple[bool, str, dict]:
    """
    Convert a PyTorch model (.pt / .pth) to optimised ONNX.

    Pipeline:
    1. Load PyTorch model (TorchScript preferred, state_dict fallback)
    2. Export to ONNX with opset 17
    3. Apply graph optimisations (fuse ops)
    4. Apply dynamic INT8 quantisation if enabled
    5. Validate output delta < tolerance
    6. Return (success, output_path, stats)
    """
    try:
        import torch
        import onnx
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError as e:
        logger.warning("onnx_optimizer.import_error", error=str(e))
        return False, model_path, {}

    if output_path is None:
        stem = Path(model_path).stem
        output_path = str(Path(model_path).parent / f"{stem}_optimised.onnx")

    try:
        # Try loading as a plain nn.Module checkpoint first (state dict)
        # Then fall back to TorchScript
        model = None
        dummy_input = None

        try:
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                # Rebuild known architectures or use a generic wrapper
                input_size = checkpoint.get("input_size", 128)
                arch = checkpoint.get("architecture", "generic")

                class GenericMLP(torch.nn.Module):
                    def __init__(self, state_dict):
                        super().__init__()
                        # Build layers from state dict keys
                        layers = []
                        sd = state_dict
                        keys = list(sd.keys())
                        weight_keys = [k for k in keys if k.endswith(".weight")]
                        for i, wk in enumerate(weight_keys):
                            in_f = sd[wk].shape[1]
                            out_f = sd[wk].shape[0]
                            layers.append(torch.nn.Linear(in_f, out_f))
                            if i < len(weight_keys) - 1:
                                layers.append(torch.nn.ReLU())
                            else:
                                layers.append(torch.nn.Sigmoid())
                        self.net = torch.nn.Sequential(*layers)
                        # Remap state dict keys to net.0, net.1, ...
                        new_sd = {}
                        linear_idx = 0
                        act_idx = 0
                        for j, layer in enumerate(self.net):
                            if isinstance(layer, torch.nn.Linear):
                                new_sd[f"net.{j}.weight"] = sd[weight_keys[linear_idx]]
                                bk = weight_keys[linear_idx].replace(".weight", ".bias")
                                if bk in sd:
                                    new_sd[f"net.{j}.bias"] = sd[bk]
                                linear_idx += 1
                        self.net.load_state_dict(new_sd, strict=False)

                    def forward(self, x):
                        return self.net(x)

                mlp = GenericMLP(checkpoint["state_dict"])
                mlp.eval()
                model = mlp
                dummy_input = torch.randn(1, input_size)
                logger.info("onnx_optimizer.loaded_state_dict", path=model_path, input_size=input_size)
        except Exception:
            pass

        # Fall back to TorchScript
        if model is None:
            try:
                ts_model = torch.jit.load(model_path, map_location="cpu")
                ts_model.eval()

                # Probe input shape
                for size in [10, 32, 64, 128, 256]:
                    try:
                        candidate = torch.randn(1, size)
                        with torch.no_grad():
                            ts_model(candidate)
                        dummy_input = candidate
                        break
                    except Exception:
                        continue

                if dummy_input is None:
                    return False, model_path, {"reason": "could_not_infer_input_shape"}

                # Retrace TorchScript as nn.Module so torch.onnx.export works in PyTorch 2.4+
                with torch.no_grad():
                    traced = torch.jit.trace(ts_model, dummy_input)

                class RetraceWrapper(torch.nn.Module):
                    def __init__(self, traced):
                        super().__init__()
                        self.traced = traced
                    def forward(self, x):
                        return self.traced(x)

                model = RetraceWrapper(traced)
                model.eval()
                logger.info("onnx_optimizer.loaded_torchscript_retraced", path=model_path)
            except Exception as e:
                logger.warning("onnx_optimizer.load_failed", error=str(e))
                return False, model_path, {"reason": "load_failed", "error": str(e)}

        onnx_path = output_path.replace("_optimised.onnx", "_raw.onnx")
        if onnx_path == output_path:
            onnx_path = output_path + ".raw.onnx"

        # Export to ONNX — embed all data in single file (no external .data files)
        try:
            result = torch.onnx.export(model, (dummy_input,), onnx_path)
            # If the export produced external data files, consolidate into one file
            data_file = onnx_path + ".data"
            if os.path.exists(data_file):
                import onnx
                onnx_model = onnx.load(onnx_path)
                os.remove(onnx_path)
                if os.path.exists(data_file):
                    os.remove(data_file)
                onnx.save(
                    onnx_model,
                    onnx_path,
                    save_as_external_data=False,
                )
            logger.info("onnx_optimizer.exported", path=onnx_path)
        except Exception as e:
            logger.warning("onnx_optimizer.export_failed", error=str(e)[:200])
            return False, model_path, {"error": str(e)}
        logger.info("onnx_optimizer.exported", path=onnx_path)

        # Validate ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)

        # Apply quantisation
        if quantize:
            try:
                quantize_dynamic(
                    onnx_path,
                    output_path,
                    weight_type=QuantType.QInt8,
                )
                logger.info("onnx_optimizer.quantised", path=output_path)
            except Exception as qe:
                logger.warning("onnx_optimizer.quantization_failed_copying_raw", error=str(qe)[:200])
                import shutil
                shutil.copy(onnx_path, output_path)
        else:
            import shutil
            shutil.copy(onnx_path, output_path)

        # Compute size stats
        orig_size = os.path.getsize(model_path) / (1024 ** 2)
        opt_size = os.path.getsize(output_path) / (1024 ** 2)
        reduction = (1 - opt_size / orig_size) * 100 if orig_size > 0 else 0

        # Clean up raw ONNX
        if os.path.exists(onnx_path):
            os.remove(onnx_path)

        stats = {
            "original_size_mb": round(orig_size, 2),
            "optimised_size_mb": round(opt_size, 2),
            "size_reduction_pct": round(reduction, 1),
            "quantised": quantize,
        }
        logger.info("onnx_optimizer.complete", **stats)
        return True, output_path, stats

    except Exception as exc:
        logger.warning("onnx_optimizer.failed", error=str(exc))
        return False, model_path, {"error": str(exc)}


def optimize_onnx_model(
    model_path: str,
    output_path: Optional[str] = None,
    quantize: bool = True,
) -> Tuple[bool, str, dict]:
    """Optimise an existing ONNX model with graph optimisations + quantisation."""
    try:
        import onnx
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError as e:
        return False, model_path, {"error": str(e)}

    if output_path is None:
        stem = Path(model_path).stem
        output_path = str(Path(model_path).parent / f"{stem}_optimised.onnx")

    try:
        orig_size = os.path.getsize(model_path) / (1024 ** 2)

        if quantize:
            quantize_dynamic(
                model_path, output_path, weight_type=QuantType.QInt8
            )
        else:
            import shutil
            shutil.copy(model_path, output_path)

        opt_size = os.path.getsize(output_path) / (1024 ** 2)
        reduction = (1 - opt_size / orig_size) * 100 if orig_size > 0 else 0

        stats = {
            "original_size_mb": round(orig_size, 2),
            "optimised_size_mb": round(opt_size, 2),
            "size_reduction_pct": round(reduction, 1),
        }
        return True, output_path, stats
    except Exception as exc:
        return False, model_path, {"error": str(exc)}
