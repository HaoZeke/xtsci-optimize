import argparse
import re

import matplotlib.pyplot as plt
import numpy as np


def main(step_size_method, line_search_method, minimize_method):
    # Initialize counters
    num_iterations = 0
    num_evaluations_func = 0
    num_evaluations_grad = 0

    # Extract path values from output.txt
    path = []
    with open("output.txt", "r") as f:
        for line in f:
            if "x: [" in line:
                x_values = re.findall(r"x: \[(.*?), (.*?)\]", line)
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

    # Create meshgrid
    x = np.linspace(-5, 5, 400)
    y = np.linspace(-5, 5, 400)
    X, Y = np.meshgrid(x, y)
    Z = (X**2 + Y - 11) ** 2 + (X + Y**2 - 7) ** 2  # Himmelblau function

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
            head_width=0.3,
            head_length=0.5,
            fc="red",
            ec="red",
        )

    # Mark start and end
    plt.scatter(path[0, 0], path[0, 1], marker="o", color="blue", label="Start", s=100)
    plt.scatter(path[-1, 0], path[-1, 1], marker="x", color="red", label="End", s=100)
    minima = np.array(
        [[3, 2], [-2.805118, 3.131312], [-3.779310, -3.283186], [3.584428, -1.848126]]
    )
    plt.scatter(
        minima[:, 0],
        minima[:, 1],
        marker="*",
        color="yellow",
        s=150,
        label="True Minima",
    )

    # Display number of evaluations
    plt.text(
        -5.5,
        -5.8,
        f"Iterations: {num_iterations}\n"
        f"Function evaluations: {num_evaluations_func}\n"
        f"Gradient evaluations: {num_evaluations_grad}",
        bbox=dict(facecolor="white", alpha=0.5),
    )

    plt.legend(loc="upper left")
    title_text = (
        f"Optimization Trajectory on the Himmelblau function\n"
        f"StepSize: {step_size_method} | "
        f"Line Search: {line_search_method} | "
        f"Minimize: {minimize_method}"
    )
    plt.title(title_text, fontsize=12, pad=15)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optimization Trajectory on the Himmelblau function."
    )

    # Add arguments
    parser.add_argument(
        "--step-size-method",
        type=str,
        required=True,
        help="Specify the step size method.",
    )
    parser.add_argument(
        "--line-search-method",
        type=str,
        required=True,
        help="Specify the line search method.",
    )
    parser.add_argument(
        "--minimize-method",
        type=str,
        required=True,
        help="Specify the minimize method.",
    )

    args = parser.parse_args()

    main(args.step_size_method, args.line_search_method, args.minimize_method)
