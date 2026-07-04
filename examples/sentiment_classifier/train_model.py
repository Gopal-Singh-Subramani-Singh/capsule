"""
Creates a minimal scikit-learn sentiment classifier for Capsule demo.
Run: python examples/sentiment_classifier/train_model.py
"""
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression


if __name__ == "__main__":
    # Simulate TF-IDF feature vectors (1000 features)
    rng = np.random.default_rng(42)
    X_train = rng.random((200, 1000))
    y_train = rng.integers(0, 2, size=200)

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    output_path = "examples/sentiment_classifier/sentiment_model.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Saved: {output_path}")
    print("Test: capsule package --manifest examples/sentiment_classifier/capsule.yaml")
