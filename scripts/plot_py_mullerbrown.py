import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize


def muller_brown(x):
    A = [-200, -100, -170, 15]
    a = [-1, -1, -6.5, 0.7]
    b = [0, 0, 11, 0.6]
    c = [-10, -10, -6.5, 0.7]
    x0 = [1, 0, -0.5, -1]
    y0 = [0, 0.5, 1.5, 1]

    return sum(
        A[i]
        * np.exp(
            a[i] * (x[0] - x0[i]) ** 2
            + b[i] * (x[0] - x0[i]) * (x[1] - y0[i])
            + c[i] * (x[1] - y0[i]) ** 2
        )
        for i in range(4)
    )


def muller_brown_gradient(x):
    A = [-200, -100, -170, 15]
    a = [-1, -1, -6.5, 0.7]
    b = [0, 0, 11, 0.6]
    c = [-10, -10, -6.5, 0.7]
    x0 = [1, 0, -0.5, -1]
    y0 = [0, 0.5, 1.5, 1]

    dfdx = 0
    dfdy = 0
    for i in range(4):
        dfdx += (
            A[i]
            * (2 * a[i] * (x[0] - x0[i]) + b[i] * (x[1] - y0[i]))
            * np.exp(
                a[i] * (x[0] - x0[i]) ** 2
                + b[i] * (x[0] - x0[i]) * (x[1] - y0[i])
                + c[i] * (x[1] - y0[i]) ** 2
            )
        )
        dfdy += (
            A[i]
            * (b[i] * (x[0] - x0[i]) + 2 * c[i] * (x[1] - y0[i]))
            * np.exp(
                a[i] * (x[0] - x0[i]) ** 2
                + b[i] * (x[0] - x0[i]) * (x[1] - y0[i])
                + c[i] * (x[1] - y0[i]) ** 2
            )
        )
    return np.array([dfdx, dfdy])


path = []


def callback(x):
    print(x)
    path.append(np.array(x))


res = minimize(
    fun=muller_brown,
    x0=[0, 0],  # Initial guess
    jac=muller_brown_gradient,
    method="CG",
    callback=callback,
    options={"gtol": 1e-6, "disp": True},
)

path = np.array(path)

x = np.linspace(-1.5, 1.2, 400)
y = np.linspace(-0.2, 2.0, 400)
X, Y = np.meshgrid(x, y)
Z = muller_brown([X, Y])

plt.figure(figsize=(12, 9))
plt.contourf(X, Y, Z, 50, cmap="viridis", alpha=0.6)
plt.colorbar()

for i in range(len(path) - 1):
    plt.arrow(
        path[i, 0],
        path[i, 1],
        path[i + 1, 0] - path[i, 0],
        path[i + 1, 1] - path[i, 1],
        head_width=0.03,
        head_length=0.05,
        fc="red",
        ec="red",
    )

plt.scatter(path[0, 0], path[0, 1], marker="o", color="blue", label="Start", s=100)
plt.scatter(path[-1, 0], path[-1, 1], marker="x", color="red", label="End", s=100)

# Known minima for the Müller-Brown potential
minima = np.array([[-0.558, 1.442], [-0.050, 0.466], [0.623, 0.028]])
plt.scatter(
    minima[:, 0], minima[:, 1], marker="*", color="yellow", s=150, label="True Minima"
)
# Known saddles for the Müller-Brown potential
saddle = np.array([[0.212, 0.293], [-0.822, 0.624]])
plt.scatter(
    saddle[:, 0], saddle[:, 1], marker="*", color="white", s=150, label="Saddle"
)

num_evaluations_func = res.nfev
num_evaluations_grad = res.njev
plt.text(
    -1.3,
    -0.1,
    f"Function evaluations: {num_evaluations_func}\n"
    f"Gradient evaluations: {num_evaluations_grad}",
    bbox=dict(facecolor="white", alpha=0.5),
)

plt.legend(loc="upper left")
plt.title("Path taken by CG on the Müller-Brown potential")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()
