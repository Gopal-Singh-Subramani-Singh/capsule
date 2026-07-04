"""
Creates a minimal PyTorch model for Capsule demo.
Saves as both TorchScript (for inference) and plain state dict (for ONNX).
Run: python examples/fraud_detector/train_model.py
"""
import torch
import torch.nn as nn


class FraudDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    model = FraudDetector()
    model.eval()

    example_input = torch.randn(1, 10)

    # Save as TorchScript (for production inference)
    scripted = torch.jit.trace(model, example_input)
    torch.jit.save(scripted, "examples/fraud_detector/fraud_model.pt")
    print("Saved TorchScript: examples/fraud_detector/fraud_model.pt")

    # Save as plain nn.Module state dict (for ONNX export)
    torch.save({
        "state_dict": model.state_dict(),
        "input_size": 10,
        "architecture": "FraudDetector",
    }, "examples/fraud_detector/fraud_model_module.pt")
    print("Saved state dict:  examples/fraud_detector/fraud_model_module.pt")

    print("Test: capsule package --manifest examples/fraud_detector/capsule.yaml")
