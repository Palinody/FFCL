import matplotlib.pyplot as plt
import os
import numpy as np


def read_dataset(filepath: str):
    return np.loadtxt(filepath, dtype=np.float32, delimiter=" ")


root_folder = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "dual_tree_traversal/"
)


def plot(root_folder, split_index_fn, brute_force_time_fn, dual_tree_search_time_fn):
    split_index_path = root_folder + split_index_fn
    brute_force_time_path = root_folder + brute_force_time_fn
    dual_tree_search_time_path = root_folder + dual_tree_search_time_fn

    split_index_vector = read_dataset(split_index_path)
    brute_force_time_vector = read_dataset(brute_force_time_path)
    dual_tree_search_time_vector = read_dataset(dual_tree_search_time_path)

    # Determine the maximum number of data points we can use from split_index
    max_len = min(
        len(split_index_vector),
        len(dual_tree_search_time_vector),
        len(brute_force_time_vector),
    )

    # Create figure and axis objects
    fig, (ax1, ax_diff) = plt.subplots(
        2, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [3, 1]}
    )

    # Plotting the dual tree search times on the primary axis
    ax1.plot(
        split_index_vector[:max_len],
        dual_tree_search_time_vector[:max_len],
        label=f"Dual Tree Search Time.\nTotal: {np.sum(dual_tree_search_time_vector):.2f} (s)",
        marker="o",
        color="red",
    )
    ax1.set_xlabel("Split Index")
    ax1.set_ylabel("Dual Tree Search Time (s)", color="red")
    ax1.tick_params(axis="y", labelcolor="red")

    # Create a secondary axis and plot brute force search times
    ax2 = ax1.twinx()
    ax2.plot(
        split_index_vector[:max_len],
        brute_force_time_vector[:max_len],
        label=f"Brute Force Time.\nTotal: {np.sum(brute_force_time_vector):.2f} (s)",
        marker="x",
        color="blue",
    )
    ax2.set_ylabel("Brute Force Time (s)", color="blue")
    ax2.tick_params(axis="y", labelcolor="blue")

    plt.title("Comparison of Search Times")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    # Plot the difference between brute force and dual tree search times in the lower plot
    time_diff = (
        brute_force_time_vector[:max_len] - dual_tree_search_time_vector[:max_len]
    )

    # Define epsilon for float equality
    epsilon = 0.001

    # Split points into above zero, below zero, and approximately zero
    above_zero = time_diff > epsilon
    below_zero = time_diff < -epsilon
    approx_zero = np.abs(time_diff) <= epsilon

    # Plot points above zero in green
    ax_diff.scatter(
        split_index_vector[:max_len][above_zero],
        time_diff[above_zero],
        label="Better (Brute Force > Dual Tree Search)",
        color="green",
        marker="s",
    )

    # Plot points below zero in red
    ax_diff.scatter(
        split_index_vector[:max_len][below_zero],
        time_diff[below_zero],
        label="Worse (Brute Force < Dual Tree Search)",
        color="red",
        marker="s",
    )

    # Plot points approximately equal to zero in blue
    ax_diff.scatter(
        split_index_vector[:max_len][approx_zero],
        time_diff[approx_zero],
        label=f"Approx. same (|Diff| <= {epsilon})",
        color="blue",
        marker="s",
    )

    ax_diff.set_xlabel("Split Index")
    ax_diff.set_ylabel("Time Difference (s)")
    ax_diff.axhline(0, color="black", linewidth=1, linestyle="--")

    ax_diff.legend(loc="upper left")

    plt.tight_layout()
    plt.show()


plot(
    root_folder, "split_index.txt", "brute_force_time.txt", "dual_tree_search_time.txt"
)
plot(
    root_folder,
    "split_index_union_find.txt",
    "brute_force_time_union_find.txt",
    "dual_tree_search_time_union_find.txt",
)
