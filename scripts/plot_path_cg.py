import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize


def rosenbrock(x):
    return (1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2


def rosenbrock_gradient(x):
    dfdx0 = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] ** 2)
    dfdx1 = 200.0 * (x[1] - x[0] ** 2)
    return np.array([dfdx0, dfdx1])


# Store path
path = []


def callback(x):
    print(x)
    path.append(np.array(x))


res = minimize(
    fun=rosenbrock,
    x0=[-1.0, 2.0],
    jac=rosenbrock_gradient,
    method="CG",
    callback=callback,
    options={"gtol": 1e-6, "disp": True},
)
path = np.array(path)
# Create meshgrid
x = np.linspace(-2, 2, 400)
y = np.linspace(-2, 2, 400)
X, Y = np.meshgrid(x, y)
Z = (1.0 - X) ** 2 + 100.0 * (Y - X**2) ** 2

# Plot
plt.figure(figsize=(12, 9))
plt.contourf(X, Y, Z, 50, cmap="viridis", alpha=0.6)
plt.colorbar()

offset_factor = 0.1  # Offset for text labels
# Plot path, direction arrows, and step markers with offset
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

    # Adding an offset for the text labels to prevent overlap
    # x_offset = offset_factor if path[i, 0] < path[i + 1, 0] else -offset_factor
    # y_offset = offset_factor if path[i, 1] < path[i + 1, 1] else -offset_factor

    # plt.text(path[i, 0] + x_offset, path[i, 1] + y_offset, str(i), fontsize=10,
    # ha='center', va='center', color='white', bbox=dict(facecolor='blue', alpha=0.5))

# Mark start, end, and true minimum
plt.scatter(path[0, 0], path[0, 1], marker="o", color="blue", label="Start", s=100)
plt.scatter(path[-1, 0], path[-1, 1], marker="x", color="red", label="End", s=100)
plt.scatter(1, 1, marker="*", color="yellow", s=150, label="True Minimum")

# Display number of evaluations
num_evaluations_func = res.nfev  # Number of function evaluations
num_evaluations_grad = (
    res.njev
)  # Number of gradient evaluations (might need to adjust attribute based on solver)
plt.text(
    1.5,
    -1.5,
    f"Function evaluations: {num_evaluations_func}\n"
    f"Gradient evaluations: {num_evaluations_grad}",
    bbox=dict(facecolor="white", alpha=0.5),
)

plt.legend(loc="upper left")
plt.title("Path, Direction, and Steps taken by CG on the Rosenbrock function")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()
