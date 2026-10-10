
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate 500 random 3D points
np.random.seed(42)

points = np.random.randn(500, 3)

# Step 2: Normalize points to form a spherical point cloud
norms = np.linalg.norm(points, axis=1, keepdims=True)
points = points / norms

# Step 3: Calculate the centroid
centroid = np.mean(points, axis=0)

# Step 4: Display basic information
print("Number of points:", points.shape[0])
print("Dimensions:", points.shape[1])
print("Centroid:", centroid)

# Step 5: Visualize the point cloud
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(
    points[:, 0],
    points[:, 1],
    points[:, 2],
    s=12,
    alpha=0.7
)

ax.scatter(
    centroid[0],
    centroid[1],
    centroid[2],
    color="red",
    s=100,
    label="Centroid"
)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Spherical Point Cloud (500 Points)")
ax.legend()

plt.show()
