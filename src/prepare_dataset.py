
import numpy as np
from sklearn.model_selection import train_test_split

# Load the dataset
X = np.load("data/pointclouds.npy")
y = np.load("data/labels.npy")

print("Original dataset:", X.shape)
print("Original labels:", y.shape)

# Split 15% for the test set
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,
    y,
    test_size=0.15,
    random_state=42,
    stratify=y
)

# Split the remaining 85% to obtain 70% training
# and 15% validation from the original dataset
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.15 / 0.85,
    random_state=42,
    stratify=y_train_val
)

# Save each split
np.savez(
    "data/splits.npz",
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    X_test=X_test,
    y_test=y_test
)

print("\nDataset split completed!")
print("Training set:", X_train.shape, y_train.shape)
print("Validation set:", X_val.shape, y_val.shape)
print("Testing set:", X_test.shape, y_test.shape)

print("\nTraining class counts:", np.bincount(y_train))
print("Validation class counts:", np.bincount(y_val))
print("Testing class counts:", np.bincount(y_test))

print("\nSaved dataset splits to data/splits.npz")
