
import os
import numpy as np

# Dataset settings
NUM_SAMPLES = 100
NUM_POINTS = 128

# Reproducibility: generate the same dataset each time
np.random.seed(42)


def generate_cube():
    """Generate points on the six surfaces of a cube."""
    points = np.random.uniform(-1, 1, size=(NUM_POINTS, 3))

    # Select one coordinate axis for each point
    axes = np.random.randint(0, 3, size=NUM_POINTS)

    # Put the selected coordinate on either cube face
    sides = np.random.choice([-1.0, 1.0], size=NUM_POINTS)

    points[np.arange(NUM_POINTS), axes] = sides

    return points


def generate_sphere():
    """Generate points uniformly distributed on a sphere surface."""
    points = np.random.normal(size=(NUM_POINTS, 3))

    # Normalize each point to make its distance from the origin equal to 1
    lengths = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / lengths

    return points


def create_dataset():
    """Create and save cube and sphere point-cloud samples."""
    X = []
    y = []

    # Class 0: cubes
    for _ in range(NUM_SAMPLES):
        X.append(generate_cube())
        y.append(0)

    # Class 1: spheres
    for _ in range(NUM_SAMPLES):
        X.append(generate_sphere())
        y.append(1)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    # Shuffle samples so classes are mixed
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    os.makedirs("data", exist_ok=True)

    np.save("data/pointclouds.npy", X)
    np.save("data/labels.npy", y)

    print("Surface dataset created successfully!")
    print("Point clouds shape:", X.shape)
    print("Labels shape:", y.shape)
    print("Cube samples:", np.sum(y == 0))
    print("Sphere samples:", np.sum(y == 1))
    print("Saved in: data/")


if __name__ == "__main__":
    create_dataset()
