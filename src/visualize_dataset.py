
import numpy as np
import matplotlib.pyplot as plt

# Load the saved dataset
X = np.load("data/pointclouds.npy")
y = np.load("data/labels.npy")

print("Dataset shape:", X.shape)
print("Labels shape:", y.shape)

# Find one cube and one sphere
cube_index = np.where(y == 0)[0][0]
sphere_index = np.where(y == 1)[0][0]

cube = X[cube_index]
sphere = X[sphere_index]

# Create two 3D plots
fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(121, projection="3d")
ax1.scatter(
    cube[:, 0], cube[:, 1], cube[:, 2],
    s=15
)
ax1.set_title("Class 0: Cube")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")

ax2 = fig.add_subplot(122, projection="3d")
ax2.scatter(
    sphere[:, 0], sphere[:, 1], sphere[:, 2],
    s=15
)
ax2.set_title("Class 1: Sphere")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_zlabel("Z")

plt.tight_layout()
plt.show()
