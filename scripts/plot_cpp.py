import argparse
import re

import matplotlib.pyplot as plt
import numpy as np
from cmcrameri import cm


def load_trajectory_from_txt(filename):
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

    return np.array(path), {
        "niter": num_iterations,
        "nfev": num_evaluations_func,
        "njev": num_evaluations_grad,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Load and plot grid data from an NPZ file and overlay a trajectory."
    )
    parser.add_argument(
        "--gridfile",
        type=str,
        required=True,
        help="The path to the NPZ file containing grid data.",
    )
    parser.add_argument(
        "--trajectoryfile",
        type=str,
        required=True,
        help="The path to the TXT file containing trajectory data.",
    )
    args = parser.parse_args()

    # Load grid data
    grid = np.load(args.gridfile)
    X, Y, Z = grid["x"], grid["y"], grid["z"]

    # Load trajectory data
    path, ntime = load_trajectory_from_txt(args.trajectoryfile)

    # Plotting
    plt.figure(figsize=(12, 9))
    plt.contourf(X, Y, Z, 50, cmap=cm.batlow, alpha=0.8)
    plt.colorbar()

    # Plot trajectory on top
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
    plt.scatter(path[-1, 0], path[-1, 1], marker="x", color="black", label="End", s=100)
    # TODO: Add true minimum
    # plt.scatter(1, 1, marker="*", color="yellow", s=150, label="True Minimum")

    # Display number of evaluations
    plt.text(
        -1.5,
        -1.8,
        f"Iterations: {ntime.get('niter')}\n"
        f"Function evaluations: {ntime.get('nfev')}\n"
        f"Gradient evaluations: {ntime.get('njev')}",
        bbox=dict(facecolor="white", alpha=0.5),
    )

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
