# 3D Point-Cloud Classification Using Deep Learning

A deep learning project for classifying 3D point clouds into **Cube** and **Sphere** classes, with additional experiments to evaluate model robustness under 3D rotations and Gaussian noise.

## Project Overview

This project explores how a neural network can learn geometric features from 3D point-cloud data. Each object is represented by a collection of 3D points, where every point has three coordinates: \(x\), \(y\), and \(z\).

The project includes synthetic dataset generation, data preparation, model training, evaluation, and robustness testing.

## Objectives

- Generate synthetic 3D point clouds for cubes and spheres.
- Prepare training, validation, and testing datasets.
- Build a PointNet-inspired neural network classifier using PyTorch.
- Evaluate classification performance using accuracy, a confusion matrix, and a classification report.
- Investigate robustness under random 3D rotations and Gaussian coordinate noise.
- Record repeated experimental results and visualize model performance.

## Technologies Used

- Python
- PyTorch
- NumPy
- Pandas
- Matplotlib
- scikit-learn

## Dataset

The project uses a synthetic dataset containing **200 point clouds**:

| Class | Number of point clouds | Label |
|---|---:|---:|
| Cube | 100 | 0 |
| Sphere | 100 | 1 |
| **Total** | **200** | — |

Each point cloud contains 128 points, with three coordinates per point.

The dataset is divided into three subsets using stratified splitting:

| Subset | Number of point clouds |
|---|---:|
| Training | 139 |
| Validation | 31 |
| Testing | 30 |

The dataset is synthetic and is intended for experimentation and educational research.

## Model Architecture

The project implements a PointNet-inspired classifier in PyTorch.

The network processes each 3D point independently, extracts point-level features, and applies max pooling to obtain a global feature representation. A fully connected classification head then predicts one of the two classes.

Main components:

1. Input: 3D coordinates \((x,y,z)\).
2. Point feature layers: 3 → 64 → 128, with ReLU activations.
3. Global max pooling across the points.
4. Classification head: 128 → 64 → 2, with ReLU activation and dropout.
5. Output: scores for the Cube and Sphere classes.

This is a simplified PointNet-inspired architecture, not a full reproduction of the original PointNet model.

## Training Configuration

- Framework: PyTorch
- Optimizer: Adam
- Learning rate: 0.001
- Loss function: Cross-entropy
- Training epochs: 50
- Batch size: 16

The trained model is saved at:

`models/pointcloud_classifier.pth`

## Experimental Results

### Baseline Classification

The trained model achieved the following results on the held-out test set:

| Metric | Result |
|---|---:|
| Test accuracy | 100% |
| Correct predictions | 30 / 30 |
| Incorrect predictions | 0 |

Confusion matrix:

```text
[[15  0]
 [ 0 15]]
```

These results apply to the current synthetic dataset and should not be interpreted as evidence of generalization to real-world 3D scans.

### Robustness Evaluation

Repeated perturbation experiments were conducted on the same 30 test point clouds.

| Experiment | Mean accuracy | Standard deviation |
|---|---:|---:|
| Baseline | 100.00% | — |
| Random 3D rotations | 98.33% | 2.55 percentage points |
| Gaussian noise, σ = 0.02 | 100.00% | 0.00 percentage points |
| Gaussian noise, σ = 0.05 | 100.00% | 0.00 percentage points |

The rotation results are based on 30 repeated perturbation trials. The noise experiments also use 30 trials per noise level. These are repeated tests of the same held-out objects, not independent test datasets.

### Fixed Rotation Angles

A separate experiment applied rotations around the z-axis:

| Rotation angle | Accuracy |
|---|---:|
| 0° | 100% |
| 15° | 100% |
| 30° | 100% |
| 45° | 100% |
| 60° | 100% |
| 90° | 100% |

These results are specific to the tested rotation axis, model, and synthetic test set.

## Project Structure

```text
3d-point-cloud-classification/
├── data/
│   ├── pointclouds.npy
│   ├── labels.npy
│   └── splits.npz
├── models/
│   ├── pointcloud_classifier.pth
│   ├── robustness_results.png
│   ├── repeated_robustness_results.png
│   ├── robustness_trial_results.csv
│   └── rotation_angle_results.csv
├── src/
│   ├── pointcloud_intro.py
│   ├── create_dataset.py
│   ├── visualize_dataset.py
│   ├── prepare_dataset.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── test_rotation.py
│   ├── test_noise.py
│   ├── plot_robustness_results.py
│   ├── repeated_robustness_test.py
│   └── test_rotation_angles.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/Majid123-web/3d-point-cloud-classification.git
cd 3d-point-cloud-classification
```

Create and activate a virtual environment.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Running the Project

Run commands from the project root.

To generate the synthetic dataset and prepare the data:

```bash
python src/create_dataset.py
python src/prepare_dataset.py
```

Train the classifier:

```bash
python src/train_model.py
```

Evaluate the trained model:

```bash
python src/evaluate_model.py
```

Run the robustness experiments:

```bash
python src/test_rotation.py
python src/test_noise.py
python src/repeated_robustness_test.py
python src/test_rotation_angles.py
python src/plot_robustness_results.py
```

**Note:** These commands assume each script uses the paths and filenames described in this README. The dataset-generation and training steps can overwrite existing files. The provided dataset, trained weights, and experimental results are already included in the repository.

## Limitations and Future Work

- The dataset is small and synthetically generated.
- Only two geometric classes are considered.
- The model is a simplified PointNet-inspired implementation.
- Robustness experiments reuse the same held-out test objects.
- Further testing is needed on larger, real-world point-cloud datasets.

Potential extensions include using ModelNet40 or another public 3D dataset, implementing the original PointNet architecture, adding more object classes, testing rotations around multiple axes, and comparing against alternative deep learning architectures.

## Author

**Majid Khan**  
Mathematics Graduate | Gold Medalist | Deep Learning and Scientific Computing Enthusiast

GitHub: https://github.com/Majid123-web

## License

No license has been specified yet. Please check the repository license before reusing or redistributing this project's code or data.