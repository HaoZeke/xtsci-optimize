import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize


def eggholder(x):
    return -(x[1] + 47) * np.sin(np.sqrt(abs(x[0] / 2 + x[1] + 47))) - x[0] * np.sin(
        np.sqrt(abs(x[0] - x[1] - 47))
    )


def eggholder_gradient(x):
    dfdx0 = (
        -np.sin(np.sqrt(abs(x[0] / 2 + x[1] + 47)))
        - (
            x[0]
            * np.cos(np.sqrt(abs(x[0] / 2 + x[1] + 47)))
            / np.sqrt(abs(x[0] / 2 + x[1] + 47))
        )
        - np.sin(np.sqrt(abs(x[0] - x[1] - 47)))
        + (
            x[0]
            * np.cos(np.sqrt(abs(x[0] - x[1] - 47)))
            / np.sqrt(abs(x[0] - x[1] - 47))
        )
    )

    dfdx1 = (
        -np.cos(np.sqrt(abs(x[0] / 2 + x[1] + 47)))
        - (
            (x[1] + 47)
            * np.cos(np.sqrt(abs(x[0] / 2 + x[1] + 47)))
            / np.sqrt(abs(x[0] / 2 + x[1] + 47))
        )
        + np.cos(np.sqrt(abs(x[0] - x[1] - 47)))
        - (
            (x[1] + 47)
            * np.cos(np.sqrt(abs(x[0] - x[1] - 47)))
            / np.sqrt(abs(x[0] - x[1] - 47))
        )
    )

    return np.array([dfdx0, dfdx1])


path = []


def callback(x):
    path.append(np.array(x))


res = minimize(
    fun=eggholder,
    x0=[0, 0],  # Initial guess
    jac=eggholder_gradient,
    method="CG",
    callback=callback,
    options={"gtol": 1e-6, "disp": True},
)

path = np.array(path)

x = np.linspace(-512, 613, 400)
y = np.linspace(-512, 613, 400)
X, Y = np.meshgrid(x, y)
Z = -(Y + 47) * np.sin(np.sqrt(abs(X / 2 + Y + 47))) - X * np.sin(
    np.sqrt(abs(X - Y - 47))
)  # Eggholder function

plt.figure(figsize=(12, 9))
plt.contourf(X, Y, Z, 50, cmap="viridis", alpha=0.6)
plt.colorbar()

for i in range(len(path) - 1):
    plt.arrow(
        path[i, 0],
        path[i, 1],
        path[i + 1, 0] - path[i, 0],
        path[i + 1, 1] - path[i, 1],
        head_width=15,
        head_length=20,
        fc="red",
        ec="red",
    )

plt.scatter(path[0, 0], path[0, 1], marker="o", color="blue", label="Start", s=100)
plt.scatter(path[-1, 0], path[-1, 1], marker="x", color="red", label="End", s=100)
minima = np.array([512, 404.2319])
plt.scatter(
    minima[0],
    minima[1],
    marker="*",
    color="yellow",
    s=150,
    label="True Minima",
)

num_evaluations_func = res.nfev
num_evaluations_grad = res.njev
plt.text(
    -400,
    -400,
    f"Function evaluations: {num_evaluations_func}\n"
    f"Gradient evaluations: {num_evaluations_grad}",
    bbox=dict(facecolor="white", alpha=0.5),
)

plt.legend(loc="upper left")
plt.title("Path taken by CG on the Eggholder function")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()
