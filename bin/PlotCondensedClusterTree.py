import numpy as np
from matplotlib import pyplot as plt

import os

import py_helpers.IO as IO


def plot_predictions(datapath, filename):
    n_features = IO.n_features_in_txt_file(datapath + "inputs/" + filename + ".txt")

    data = IO.auto_decode(
        datapath + "inputs/" + filename + ".txt",
        dtype=np.float32,
        n_features=n_features,
    )

    predictions = IO.auto_decode(
        datapath + "predictions/" + filename + ".txt", dtype=np.int32, n_features=1
    )

    # Separate the points based on their predictions
    noise_points = data[predictions == 0]

    plt.scatter(
        noise_points[:, 0],
        noise_points[:, 1],
        color="black",
        marker="x",
        label="Noise",
    )

    # Assign a unique color to each cluster index
    unique_predictions = np.unique(predictions)

    colors = plt.cm.inferno(np.linspace(0, 1, len(unique_predictions)))

    for label, color in zip(unique_predictions, colors):
        if label > 0:
            cluster_points_label = data[predictions == label]
            plt.scatter(
                cluster_points_label[:, 0],
                cluster_points_label[:, 1],
                color=color,
                label=f"Cluster {label}",
            )

    plt.title(filename)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()


def main():
    """noisy_circles, noisy_moons, varied, aniso, blobs, no_structure, unbalanced_blobs"""
    root_folder = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "single_linkage_cluster_tree/"
    )

    file_names = os.listdir(root_folder)

    datapath = r"./clustering/"

    for filename in file_names:
        stem = os.path.splitext(filename)[0]

        print(stem)

        plot_predictions(datapath, stem)


if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf)

    main()
