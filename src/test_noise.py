
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix


class PointCloudClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.point_features = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )

    def forward(self, points):
        features = self.point_features(points)
        global_features = torch.max(features, dim=1).values
        return self.classifier(global_features)


def predict(model, point_clouds):
    points = torch.tensor(point_clouds, dtype=torch.float32)

    with torch.no_grad():
        logits = model(points)
        predictions = logits.argmax(dim=1).numpy()

    return predictions


def main():
    # Load the test set
    data = np.load("data/splits.npz")
    X_test = data["X_test"]
    y_test = data["y_test"]

    # Load the trained model
    model = PointCloudClassifier()

    state_dict = torch.load(
        "models/pointcloud_classifier.pth",
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.eval()

    # Evaluate multiple noise levels
    noise_levels = [0.00, 0.02, 0.05]
    rng = np.random.default_rng(123)

    print("NOISE ROBUSTNESS TEST")
    print("-" * 35)

    for sigma in noise_levels:
        noise = rng.normal(
            loc=0.0,
            scale=sigma,
            size=X_test.shape
        )

        noisy_clouds = X_test + noise
        predictions = predict(model, noisy_clouds)

        accuracy = accuracy_score(y_test, predictions)

        print(f"\nNoise standard deviation: {sigma:.2f}")
        print(f"Accuracy: {accuracy * 100:.2f}%")

        print("Confusion matrix:")
        print(confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ))


if __name__ == "__main__":
    main()
