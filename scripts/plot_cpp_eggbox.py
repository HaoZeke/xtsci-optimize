import re

import matplotlib.pyplot as plt
import numpy as np


# Define the eggbox function
def eggbox(x, y):
    return np.cos(x) * np.cos(y) * np.exp(-((x / np.pi) ** 2) - (y / np.pi) ** 2)


# Extract path values from output.txt
path = []
best_positions = []
with open("output.txt", "r") as f:
    for line in f:
        if "Best position:" in line:
            x_values = re.findall(r"Best position: \[(.*?), (.*?)\]", line)
            best_positions.append([float(x) for x in x_values[0]])
        elif "New position for" in line:
            x_values = re.findall(r"New position for \d+: \[(.*?), (.*?)\]", line)
            path.append([float(x) for x in x_values[0]])
        elif "Number of iterations:" in line:
            num_iterations = int(
                re.search(r"Number of iterations: (\d+)", line).group(1)
            )
        elif "Number of function evaluations:" in line:
            num_evaluations_func = int(
                re.search(r"Number of function evaluations: (\d+)", line).group(1)
            )
        elif "Number of gradient evaluations:" in line:
            num_evaluations_grad = int(
                re.search(r"Number of gradient evaluations: (\d+)", line).group(1)
            )

path = np.array(path)
best_positions = np.array(best_positions)

# Create meshgrid
x = np.linspace(-512, 613, 400)
y = np.linspace(-512, 613, 400)
X, Y = np.meshgrid(x, y)
Z = -(Y + 47) * np.sin(np.sqrt(abs(X / 2 + Y + 47))) - X * np.sin(
    np.sqrt(abs(X - Y - 47))
)  # Eggholder function

# # Animate
# fig, ax = plt.subplots(figsize=(12, 9))
# ax.contourf(X, Y, Z, 50, cmap="viridis", alpha=0.6)
# minima = np.array([512, 404.2319])
# ax.scatter(minima[0], minima[1], marker="*",
#            color="yellow", s=150, label="True Minima")
# line, = ax.plot([], [], 'o-', color="blue", alpha=0.7)  # Initial empty line

# # Init function to set the initial state of the animation
# def init():
#     line.set_data([], [])
#     return line,

# # Update function for each frame of the animation
# def update(frame):
#     line.set_data(path[:frame, 0], path[:frame, 1])  # Update line data
# if frame < len(best_positions):
#     ax.scatter(
#         best_positions[frame, 0], best_positions[frame, 1],
#                marker="x", color="red", s=100,
#                label=f"Best Position at iteration {frame}"
#     )
#     return line,

# ani = FuncAnimation(fig, update, frames=len(path)+1,
#                     init_func=init, blit=True, repeat=False)

# plt.tight_layout()
# plt.show()

# Plot
plt.figure(figsize=(12, 9))
plt.contourf(X, Y, Z, 50, cmap="viridis", alpha=0.6)
plt.colorbar()

# Plot path and best positions
plt.scatter(path[:, 0], path[:, 1], marker="o", color="blue", s=100)
plt.scatter(
    best_positions[:, 0],
    best_positions[:, 1],
    marker="x",
    color="red",
    s=100,
    label="Best Position",
)
minima = np.array([512, 404.2319])
plt.scatter(
    minima[0],
    minima[1],
    marker="*",
    color="yellow",
    s=150,
    label="True Minima",
)

# Display number of evaluations
plt.text(
    -500,
    -580,
    f"Iterations: {num_iterations}\n"
    f"Function evaluations: {num_evaluations_func}\n"
    f"Gradient evaluations: {num_evaluations_grad}",
    bbox=dict(facecolor="white", alpha=0.5),
)
plt.legend(loc="upper left")
title_text = (
    "Optimization Trajectory on the Eggbox function\n"
    "Using Particle Swarm Optimization"
)
plt.title(title_text, fontsize=12, pad=15)
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()
