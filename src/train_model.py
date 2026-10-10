
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load the prepared dataset
data = np.load("data/splits.npz")

X_train = torch.tensor(data["X_train"], dtype=torch.float32)
y_train = torch.tensor(data["y_train"], dtype=torch.long)

X_val = torch.tensor(data["X_val"], dtype=torch.float32)
y_val = torch.tensor(data["y_val"], dtype=torch.long)

X_test = torch.tensor(data["X_test"], dtype=torch.float32)
y_test = torch.tensor(data["y_test"], dtype=torch.long)

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=16,
    shuffle=True
)

# PointNet-inspired model
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
        # points shape: (batch, 128, 3)
        features = self.point_features(points)

        # Aggregate features across all points
        global_features = torch.max(features, dim=1).values

        # Return two class scores for each object
        return self.classifier(global_features)


model = PointCloudClassifier().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# Training loop
EPOCHS = 50

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for points, labels in train_loader:
        points = points.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        predictions = model(points)
        loss = criterion(predictions, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * points.size(0)

    average_loss = total_loss / len(X_train)

    # Evaluate on validation data
    model.eval()

    with torch.no_grad():
        val_predictions = model(X_val.to(device))
        val_predicted_classes = val_predictions.argmax(dim=1)
        val_accuracy = (
            val_predicted_classes.cpu() == y_val
        ).float().mean().item()

    if (epoch + 1) % 5 == 0:
        print(
            f"Epoch [{epoch + 1:02d}/{EPOCHS}] "
            f"Loss: {average_loss:.4f} "
            f"Validation Accuracy: {val_accuracy:.4f}"
        )

# Final test evaluation
model.eval()

with torch.no_grad():
    test_logits = model(X_test.to(device))
    test_predictions = test_logits.argmax(dim=1).cpu()

test_accuracy = (test_predictions == y_test).float().mean().item()

print("\nTraining completed!")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Accuracy (%): {test_accuracy * 100:.2f}%")

# Save the trained model
import os
os.makedirs("models", exist_ok=True)

torch.save(
    model.cpu().state_dict(),
    "models/pointcloud_classifier.pth"
)

print("Model saved to models/pointcloud_classifier.pth")
