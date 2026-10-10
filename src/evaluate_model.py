
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    accuracy_score
)

# Recreate the same model architecture
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


# Load the test dataset
data = np.load("data/splits.npz")

X_test = torch.tensor(data["X_test"], dtype=torch.float32)
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

# Make predictions
with torch.no_grad():
    logits = model(X_test)
    predictions = logits.argmax(dim=1).numpy()

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, predictions)
cm = confusion_matrix(y_test, predictions, labels=[0, 1])

print("Test Accuracy:", f"{accuracy * 100:.2f}%")
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Cube", "Sphere"],
        zero_division=0
    )
)

print("Confusion Matrix:")
print(cm)

# Visualize the confusion matrix
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Cube", "Sphere"]
)

display.plot(cmap="Blues", values_format="d")
plt.title("Test Set Confusion Matrix")
plt.tight_layout()
plt.show()
