import torch
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

class IrisClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )

    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    # 1. Load and prep real data
    iris = load_iris()
    X, y = iris.data, iris.target
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    # 2. Train actual model
    model = IrisClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    
    print("Training Iris PyTorch Model...")
    for epoch in range(150):
        optimizer.zero_grad()
        out = model(X_tensor)
        loss = criterion(out, y_tensor)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")
            
    model.eval()
    
    # 3. Save as plain nn.Module state dict (for ONNX export)
    torch.save({
        "state_dict": model.state_dict(),
        "input_size": 4,
        "architecture": "IrisClassifier",
    }, "examples/iris_classifier/iris_model_module.pt")
    
    print("Model trained and saved: examples/iris_classifier/iris_model_module.pt")
    print("Test: capsule package --manifest examples/iris_classifier/capsule.yaml")
