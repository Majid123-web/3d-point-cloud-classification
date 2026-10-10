
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix


import torch.nn as nn


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



def rotate_clouds_z(points, angle_degrees):
    """Rotate point clouds around the z-axis."""
    angle = np.radians(angle_degrees)

    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0,              0,             1]
    ])

    return points @ rotation_matrix.T


def main():
    data = np.load("data/splits.npz")
    X_test = data["X_test"]
    y_test = data["y_test"]

    model = PointCloudClassifier()
    model.load_state_dict(
        torch.load(
            "models/pointcloud_classifier.pth",
            map_location="cpu",
            weights_only=True
        )
    )
    model.eval()

    angles = [0, 15, 30, 45, 60, 90]

    print("ROTATION ANGLE ROBUSTNESS TEST")
    print("-" * 40)
    print(f"Test samples: {len(X_test)}")

    results = []

    for angle in angles:
        rotated = rotate_clouds_z(X_test, angle)

        with torch.no_grad():
            inputs = torch.tensor(rotated, dtype=torch.float32)
            outputs = model(inputs)
            predictions = outputs.argmax(dim=1).numpy()

        accuracy = accuracy_score(y_test, predictions) * 100
        results.append((angle, accuracy))

        print(f"Rotation: {angle:3d} degrees | Accuracy: {accuracy:.2f}%")

        if angle == 90:
            print("\nConfusion matrix at 90 degrees:")
            print(confusion_matrix(y_test, predictions))

    np.savetxt(
        "models/rotation_angle_results.csv",
        np.array(results),
        delimiter=",",
        header="angle_degrees,accuracy_percent",
        comments="",
        fmt=["%d", "%.4f"]
    )

    print("\nResults saved to models/rotation_angle_results.csv")


if __name__ == "__main__":
    main()
