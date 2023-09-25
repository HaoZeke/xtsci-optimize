import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize


def himmelblau(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


def himmelblau_gradient(x):
    dfdx0 = 4.0 * x[0] * (x[0] ** 2 + x[1] - 11) + 2.0 * (x[0] + x[1] ** 2 - 7)
    dfdx1 = 2.0 * (x[0] ** 2 + x[1] - 11) + 4.0 * x[1] * (x[0] + x[1] ** 2 - 7)
    return np.array([dfdx0, dfdx1])


path = []


def callback(x):
    path.append(np.array(x))


res = minimize(
    fun=himmelblau,
    x0=[0, 0],  # Initial guess
    jac=himmelblau_gradient,
    method="CG",
    callback=callback,
    options={"gtol": 1e-6, "disp": True},
)

path = np.array(path)

x = np.linspace(-5, 5, 400)
y = np.linspace(-5, 5, 400)
X, Y = np.meshgrid(x, y)
Z = (X**2 + Y - 11) ** 2 + (X + Y**2 - 7) ** 2  # Himmelblau function

plt.figure(figsize=(12, 9))
plt.contourf(X, Y, Z, 50, cmap="viridis", alpha=0.6)
plt.colorbar()

for i in range(len(path) - 1):
    plt.arrow(
        path[i, 0],
        path[i, 1],
        path[i + 1, 0] - path[i, 0],
        path[i + 1, 1] - path[i, 1],
        head_width=0.15,
        head_length=0.2,
        fc="red",
        ec="red",
    )

plt.scatter(path[0, 0], path[0, 1], marker="o", color="blue", label="Start", s=100)
plt.scatter(path[-1, 0], path[-1, 1], marker="x", color="red", label="End", s=100)
minima = np.array(
    [[3, 2], [-2.805118, 3.131312], [-3.779310, -3.283186], [3.584428, -1.848126]]
)
plt.scatter(
    minima[:, 0], minima[:, 1], marker="*", color="yellow", s=150, label="True Minima"
)

num_evaluations_func = res.nfev
num_evaluations_grad = res.njev
plt.text(
    -1.5,
    -4.5,
    f"Function evaluations: {num_evaluations_func}\n"
    f"Gradient evaluations: {num_evaluations_grad}",
    bbox=dict(facecolor="white", alpha=0.5),
)

plt.legend(loc="upper left")
plt.title("Path taken by CG on the Himmelblau function")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()
