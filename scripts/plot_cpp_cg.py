import re

import matplotlib.pyplot as plt
import numpy as np

# Extract path values from output.txt
path = []
with open("output.txt", "r") as f:
    for line in f:
        if "x: [" in line:
            x_values = re.findall(r"x: \[(.*?), (.*?)\]", line)
            path.append([float(x) for x in x_values[0]])

path = np.array(path)

# Estimate the number of evaluations
num_evaluations_grad = len(path)  # Based on the number of lines in the output
num_evaluations_func = num_evaluations_grad  # Use gradient evaluations as a proxy

# Create meshgrid
x = np.linspace(-2, 4, 400)
y = np.linspace(-2, 8, 400)
X, Y = np.meshgrid(x, y)
Z = (1.0 - X) ** 2 + 100.0 * (Y - X**2) ** 2

# Plot
plt.figure(figsize=(12, 9))
plt.contourf(X, Y, Z, 50, cmap="viridis", alpha=0.6)
plt.colorbar()

# Plot path and arrows
for i in range(len(path) - 1):
    plt.arrow(
        path[i, 0],
        path[i, 1],
        path[i + 1, 0] - path[i, 0],
        path[i + 1, 1] - path[i, 1],
        head_width=0.05,
        head_length=0.1,
        fc="red",
        ec="red",
    )

# Mark start, end, and true minimum
plt.scatter(path[0, 0], path[0, 1], marker="o", color="blue", label="Start", s=100)
plt.scatter(path[-1, 0], path[-1, 1], marker="x", color="red", label="End", s=100)
plt.scatter(1, 1, marker="*", color="yellow", s=150, label="True Minimum")

# Display number of evaluations
plt.text(
    1.5,
    -1.5,
    f"Function evaluations: {num_evaluations_func}\n"
    f"Gradient evaluations: {num_evaluations_grad}",
    bbox=dict(facecolor="white", alpha=0.5),
)

plt.legend(loc="upper left")
plt.title("Optimization Trajectory on the Rosenbrock function")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()
