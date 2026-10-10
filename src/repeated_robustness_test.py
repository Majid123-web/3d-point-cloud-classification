import csv
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


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
        predictions = model(points).argmax(dim=1).numpy()

    return predictions


def random_rotation_matrix(rng):
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


def rotate_clouds(point_clouds, rng):
    rotated = np.empty_like(point_clouds)

    for i, points in enumerate(point_clouds):
        rotation = random_rotation_matrix(rng)
        rotated[i] = points @ rotation.T

    return rotated


def main():
    # Reproducible experiments
    rng = np.random.default_rng(2026)

    # Load held-out test data
    data = np.load("data/splits.npz")
    X_test = data["X_test"]
    y_test = data["y_test"]

    # Load the saved model
    model = PointCloudClassifier()
    state_dict = torch.load(
        "models/pointcloud_classifier.pth",
        map_location="cpu",
        weights_only=True
    )
    model.load_state_dict(state_dict)
    model.eval()

    num_trials = 30
    noise_levels = [0.02, 0.05]

    # Baseline accuracy
    baseline_predictions = predict(model, X_test)
    baseline_accuracy = accuracy_score(y_test, baseline_predictions)

    rotation_accuracies = []
    noise_results = {level: [] for level in noise_levels}
    trial_results = []

    # Repeat rotation tests
    for trial in range(1, num_trials + 1):
        rotated = rotate_clouds(X_test, rng)
        predictions = predict(model, rotated)
        accuracy = accuracy_score(y_test, predictions) * 100

        rotation_accuracies.append(accuracy)

        trial_results.append({
            "condition": "rotation",
            "noise_std": 0.0,
            "trial": trial,
            "accuracy_percent": accuracy
        })

    # Repeat noise tests
    for level in noise_levels:
        for trial in range(1, num_trials + 1):
            noise = rng.normal(
                loc=0.0,
                scale=level,
                size=X_test.shape
            )

            noisy_clouds = X_test + noise
            predictions = predict(model, noisy_clouds)
            accuracy = accuracy_score(y_test, predictions) * 100

            noise_results[level].append(accuracy)

            trial_results.append({
                "condition": "gaussian_noise",
                "noise_std": level,
                "trial": trial,
                "accuracy_percent": accuracy
            })


    print("REPEATED ROBUSTNESS EXPERIMENTS")
    print("-" * 45)
    print(f"Test samples: {len(y_test)}")
    print(f"Trials per condition: {num_trials}")
    print(f"Baseline accuracy: {baseline_accuracy * 100:.2f}%")

    print("\nRotation:")
    print(f"Mean accuracy: {np.mean(rotation_accuracies):.2f}%")
    print(f"Standard deviation: {np.std(rotation_accuracies):.2f}%")
    print(f"Minimum accuracy: {np.min(rotation_accuracies):.2f}%")
    print(f"Maximum accuracy: {np.max(rotation_accuracies):.2f}%")

    print("\nGaussian noise:")
    for level, scores in noise_results.items():
        print(f"\nNoise standard deviation = {level:.2f}")
        print(f"Mean accuracy: {np.mean(scores):.2f}%")
        print(f"Standard deviation: {np.std(scores):.2f}%")
        print(f"Minimum accuracy: {np.min(scores):.2f}%")
        print(f"Maximum accuracy: {np.max(scores):.2f}%")

    # Plot mean accuracies with standard-deviation error bars
    conditions = ["Original", "Rotation"] + [
        f"Noise {level:.2f}" for level in noise_levels
    ]

    means = [baseline_accuracy * 100, np.mean(rotation_accuracies)]
    errors = [0, np.std(rotation_accuracies)]

    for level in noise_levels:
        scores = noise_results[level]
        means.append(np.mean(scores))
        errors.append(np.std(scores))

    plt.figure(figsize=(9, 5))
    plt.bar(
        conditions,
        means,
        yerr=errors,
        capsize=5
    )
    plt.ylim(0, 110)
    plt.ylabel("Mean accuracy (%)")
    plt.title("Repeated Robustness Tests")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    plt.savefig(
        "models/repeated_robustness_results.png",
        dpi=300,
        bbox_inches="tight"
    )

    print("\nChart saved to models/repeated_robustness_results.png")
    
    # Save individual trial results to CSV
    with open(
        "models/robustness_trial_results.csv",
        "w",
        newline=""
    ) as file:
        fieldnames = [
            "condition",
            "noise_std",
            "trial",
            "accuracy_percent"
        ]

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trial_results)

    print("Trial results saved to models/robustness_trial_results.csv")

    plt.show()


if __name__ == "__main__":
    main()
