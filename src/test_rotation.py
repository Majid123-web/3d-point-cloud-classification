
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix


# Define the same model architecture used during training
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


def random_rotation_matrix(rng):
    """Create a random 3D rotation matrix."""

    angles = rng.uniform(0, 2 * np.pi, size=3)
    ax, ay, az = angles

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(ax), -np.sin(ax)],
        [0, np.sin(ax), np.cos(ax)]
    ])

    Ry = np.array([
        [np.cos(ay), 0, np.sin(ay)],
        [0, 1, 0],
        [-np.sin(ay), 0, np.cos(ay)]
    ])

    Rz = np.array([
        [np.cos(az), -np.sin(az), 0],
        [np.sin(az), np.cos(az), 0],
        [0, 0, 1]
    ])

    return Rz @ Ry @ Rx


def rotate_point_clouds(point_clouds, seed=123):
    """Rotate each point cloud independently."""

    rng = np.random.default_rng(seed)
    rotated_clouds = np.empty_like(point_clouds)

    for i, points in enumerate(point_clouds):
        rotation = random_rotation_matrix(rng)

        # Apply the same rotation to all points in one object
        rotated_clouds[i] = points @ rotation.T

    return rotated_clouds


def predict(model, point_clouds):
    """Predict the class of each point cloud."""

    points = torch.tensor(point_clouds, dtype=torch.float32)

    with torch.no_grad():
        logits = model(points)
        predictions = logits.argmax(dim=1).numpy()

    return predictions


def main():
    # Load the held-out test set
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

    # Evaluate the original, unrotated point clouds
    original_predictions = predict(model, X_test)
    original_accuracy = accuracy_score(
        y_test, original_predictions
    )

    # Rotate each test object randomly in 3D
    X_rotated = rotate_point_clouds(X_test)
    rotated_predictions = predict(model, X_rotated)
    rotated_accuracy = accuracy_score(
        y_test, rotated_predictions
    )

    print("ROTATION ROBUSTNESS TEST")
    print("-" * 35)

    print(f"Number of test samples: {len(y_test)}")
    print(f"Original accuracy: {original_accuracy * 100:.2f}%")
    print(f"Rotated accuracy:  {rotated_accuracy * 100:.2f}%")

    print("\nConfusion matrix for rotated point clouds:")
    print(confusion_matrix(
        y_test, rotated_predictions, labels=[0, 1]
    ))

    print("\nClass order: 0 = Cube, 1 = Sphere")


if __name__ == "__main__":
    main()
