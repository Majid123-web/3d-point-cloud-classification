
import matplotlib.pyplot as plt

# Results from your completed experiments
conditions = [
    "Original",
    "Rotation",
    "Noise 0.02",
    "Noise 0.05"
]

accuracies = [100.00, 96.67, 100.00, 100.00]

# Create the bar chart
plt.figure(figsize=(9, 5))

bars = plt.bar(
    conditions,
    accuracies,
    width=0.6
)

plt.ylim(0, 110)
plt.ylabel("Accuracy (%)")
plt.xlabel("Test condition")
plt.title("3D Point-Cloud Classifier: Robustness Results")

# Display the accuracy above each bar
for bar, accuracy in zip(bars, accuracies):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{accuracy:.2f}%",
        ha="center",
        va="bottom"
    )

plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()

# Save the figure for your project report
plt.savefig(
    "models/robustness_results.png",
    dpi=300,
    bbox_inches="tight"
)

print("Robustness chart saved to models/robustness_results.png")

plt.show()
